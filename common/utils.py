import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from keras import backend as K, Input, Model
from keras.layers import Dense, Reshape, Activation
from sqlalchemy import create_engine
from tqdm.auto import tqdm
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalTrueColorFormatter

from global_config import config

def print_code(source_code):
    print(highlight(source_code, PythonLexer(), TerminalTrueColorFormatter(style='autumn')))


def get_serann_keras_model(source_code, clear_session=False):
    if clear_session:
        K.clear_session()

    K.set_floatx('float16')
    K.set_epsilon(1e-4)

    X_layer = X_input = Input(shape=[28, 28, 1])
    g_layer = g_input = Input(shape=(100, 1))

    exec(source_code)

    last_layer = locals()['con']

    y_hat = Dense(10)(Reshape((1, -1))(last_layer))
    y_hat = Activation('softmax', name='classification_output')(y_hat)

    g_rep = Dense(100)(Reshape((1, -1))(last_layer))
    g_rep = Activation('sigmoid', name='replication_output')(g_rep)

    model = Model(inputs=[X_input, g_input], outputs=[y_hat, g_rep])

    lb = locals()['loss_balance']

    def replication_fidelity(gt, pred):
        pred = K.clip(K.round(pred), 0, 1)
        return K.mean(K.sum(K.cast(pred == gt, 'float16'), axis=1))

    model.compile(loss={'classification_output': 'categorical_crossentropy', 'replication_output': 'mse'},
                  metrics={'classification_output': 'categorical_accuracy', 'replication_output': replication_fidelity},
                  loss_weights={'classification_output': lb, 'replication_output': 1 - lb},
                  optimizer='adam')

    return model


def load_experiment_results(results_name, evaluations_name=None, cache_invalidate=False, **evaluations_params):

    cache_path = Path(config['data_cache_dir']) / 'experiment_results' / f'{results_name}.pkl'

    if not cache_invalidate and cache_path.is_file():
        results = pd.read_pickle(cache_path)

    else:
        print('Loading from SQL db')
        db_path = Path(config['experiment_results_dir']) / f'{results_name}.sqlite'
        conn = create_engine(f'sqlite:////{db_path}').connect()

        all_columns = pd.read_sql('PRAGMA table_info("srann")', con=conn)['name']

        genotypes = pd.read_sql(f'select id, genotype from srann', con=conn).set_index('id')
        source_codes = pd.read_sql(f'select distinct genotype, source_code from srann', con=conn).set_index('genotype')

        genotypes['logical_genotype'] = genotypes['genotype']
        logical_genotypes = genotypes.set_index('genotype')['logical_genotype'].drop_duplicates().map(
            lambda x: np.array(eval(x)).astype(int))

        genotype_hex = logical_genotypes.map(lambda x: hex(int(re.sub('\D', '', str(x)), 2))).rename('genotype_hex')

        columns = set(all_columns) - {'source_code', 'genotype'}

        df = pd.read_sql(f'select {", ".join(columns)} from srann', con=conn).set_index('id')

        results = df.join(genotypes['logical_genotype'].rename('str_genotype'))
        results = results.join(source_codes['source_code'], on='str_genotype')
        results = results.join(logical_genotypes.rename('genotype'), on='str_genotype')

        if 'gnotype_hex' not in results.columns:
            results = results.join(genotype_hex, on='str_genotype').drop('str_genotype', axis=1)

        results['is_mutant'] = results['genotype_hamming_distance_from_parent'] > 0

        descendants = {}
        pairs = results.dropna(subset=['parent_id']).iloc[::-1]['parent_id']

        for _id, parent_id in zip(pairs.index, pairs.values):
            descendants[parent_id] = descendants.get(parent_id, 0) + descendants.get(_id, 0) + 1

        results['descendants'] = pd.Series(descendants, index=results.index).fillna(0)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_pickle(cache_path, compression=None)

    if evaluations_name is not None:

        evaluations = load_experiment_evaluations(evaluations_name, results, cache_invalidate,
                                                  **evaluations_params).drop('id', axis=1)
        results = results.join(evaluations.drop_duplicates(subset=['genotype_hex']).set_index('genotype_hex'),
                               on='genotype_hex', how='left')

    results = results.join(results.rename(columns={c: f'parent_{c}' for c in results.columns}), on='parent_id')

    results.loc[results['parent_id'].isna(), 'parent_id'] = 'experiment_ancestor_id'
    results.loc[results['parent_id'].isna(), 'parent_genotype_hex'] = 'experiment_ancestor_genotype_hex'

    return results


def load_experiment_evaluations(evaluations_name, experiment_results, cache_invalidate=False, selection_intensity=1,
                                raw_data=False):

    cache_path = Path(config['data_cache_dir']) / 'deep_evaluations' / f'{evaluations_name}.pkl'

    if not cache_invalidate and cache_path.is_file():
        return pd.read_pickle(cache_path)

    if raw_data:
        sample_path = f'{config["deep_evaluations_dir"]}/{evaluations_name}.pkl'

        with open(sample_path, 'rb') as f:
            raw_evaluations = pickle.load(f)

        rows = []

        for serann_id, evaluations in raw_evaluations.items():
            ca = np.mean(evaluations[0]['classification_accuracy'])

            m, f = list(zip(*evaluations[0]['mutation_rate'].items()))
            mutation_rate = np.sum(np.array(m) * np.array(f)) / np.sum(f)

            s, f = list(zip(*evaluations[0]['offspring_survival'].items()))
            offspring_viability = np.sum(np.array(s) * np.array(f)) / np.sum(f)

            rows.append({
                'id': serann_id,
                'classification_accuracy': ca,
                'mutation_rate': mutation_rate,
                'offspring_viability': offspring_viability
            })

        evaluations = pd.DataFrame(rows)

    else:
        sample_path = f'{config["deep_evaluations_dir"]}/{evaluations_name}.csv'
        evaluations = pd.read_csv(sample_path).rename(columns={'serann_id': 'id'})

    evaluations = evaluations[evaluations['visited'] == True].drop('visited', axis=1)

    results = experiment_results.loc[evaluations["id"].values, ['absolute_fitness', 'generation', 'genotype_hex']]
    evaluations = evaluations.merge(results, on='id').rename(columns={
        'classification_accuracy': 'm_classification_accuracy',
        'mutation_rate': 'm_mutation_rate'
    })

    evaluations['m_absolute_fertility'] = evaluations['m_classification_accuracy'].fillna(0) ** selection_intensity
    evaluations['m_offspring_survival'] = evaluations['offspring_viability'].fillna(0)

    absolute_fecundity_sum_by_gen = experiment_results.groupby('generation')['absolute_fitness'].apply(
        lambda x: np.sum(x.fillna(0) ** selection_intensity))
    evaluations = evaluations.join(absolute_fecundity_sum_by_gen.rename('absolute_fertility_sum'), on='generation')

    # 𝑓 = 𝐹^𝜆 / ∑𝐹𝑗^𝜆
    evaluations['m_relative_fertility'] = evaluations['m_absolute_fertility'] / evaluations['absolute_fertility_sum']

    pop_size = experiment_results.groupby('generation')['genotype'].count().max()

    # 𝑤(𝑡) = 𝑉⋅𝑓⋅𝑁
    evaluations['m_relative_fitness'] = evaluations['m_offspring_survival'] * evaluations['m_relative_fertility'] * pop_size

    # 𝑊(𝑡) = 𝑉⋅𝐹
    evaluations['m_absolute_fitness'] = evaluations['m_offspring_survival'] * evaluations['m_absolute_fertility']

    mask_slice = experiment_results.loc[evaluations['id'].values]
    mask = (mask_slice['is_valid'] == False) | (mask_slice['is_overweight'] == True)
    evaluations.loc[mask.values, 'm_relative_fitness'] = 0
    evaluations.loc[mask.values, 'm_absolute_fitness'] = 0

    columns = ['id', 'genotype_hex', 'm_classification_accuracy', 'm_mutation_rate', 'm_offspring_survival',
               'm_relative_fertility', 'm_absolute_fertility', 'm_absolute_fitness', 'm_relative_fitness']

    evaluations = evaluations[columns]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    evaluations.to_pickle(cache_path, compression=None)

    return evaluations


def load_experiment_data(path, deep_evaluations_path=None, force_download=False, cache_invalidate=False):

    results_cache_path = Path(config['data_cache_dir']) / Path(path).name

    if deep_evaluations_path is not None:
        deep_evaluations_cache_path = config['data_cache_dir'] / Path(deep_evaluations_path).name

    if (force_download is False and cache_invalidate is False) and results_cache_path.is_file():

        results = pd.read_pickle(results_cache_path)

        if deep_evaluations_path is not None:
            evaluations = pd.read_pickle(deep_evaluations_cache_path)
            columns = ['classification_accuracy', 'mutation_rate', 'offspring_viability', 'fecundity',
                       'normed_fecundity', 'fitness', 'genotype_hex']
            return results.join(
                evaluations[columns].drop_duplicates(subset=['genotype_hex']).set_index('genotype_hex'),
                on='genotype_hex', how='left')

        return results

    conn = create_engine(f'sqlite:////{path}').connect()

    all_columns = pd.read_sql('PRAGMA table_info("srann")', con=conn)['name']

    genotypes = pd.read_sql(f'select id, genotype from srann', con=conn).set_index('id')
    source_codes = pd.read_sql(f'select distinct genotype, source_code from srann', con=conn).set_index('genotype')

    genotypes['logical_genotype'] = genotypes['genotype']
    logical_genotypes = genotypes.set_index('genotype')['logical_genotype'].drop_duplicates().map(
        lambda x: np.array(eval(x)).astype(int))

    genotype_hex = logical_genotypes.map(lambda x: hex(int(re.sub('\D', '', str(x)), 2))).rename('genotype_hex')

    columns = set(all_columns) - {'source_code', 'genotype'}

    df = pd.read_sql(f'select {", ".join(columns)} from srann', con=conn).set_index('id')

    results = df.join(genotypes['logical_genotype'].rename('str_genotype'))
    results = results.join(source_codes['source_code'], on='str_genotype')
    results = results.join(logical_genotypes.rename('genotype'), on='str_genotype')
    results = results.join(genotype_hex, on='str_genotype').drop('str_genotype', axis=1)

    results[results['parent_id'].isna(), 'parent_id'] = 'experiment_ancestor_id'
    results[results['parent_id'].isna(), 'parent_genotype_hex'] = 'experiment_ancestor_genotype_hex'

    results['is_mutant'] = results['genotype_hamming_distance_from_parent'] > 0

    descendants = {}
    pairs = results.dropna(subset=['parent_id']).iloc[::-1]['parent_id']

    for _id, parent_id in zip(pairs.index, pairs.values):
        descendants[parent_id] = descendants.get(parent_id, 0) + descendants.get(_id, 0) + 1

    results['descendants'] = pd.Series(descendants, index=results.index).fillna(0)

    results_cache_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_pickle(results_cache_path, compression=None)

    return results


    if hasattr(df['genotype'], 'parallel_map'):
        df['genotype'] = df['genotype'].parallel_map(lambda x: np.array(eval(x)).astype(int))
    else:
        df['genotype'] = df['genotype'].map(lambda x: np.array(eval(x)).astype(int))

    if 'genotype_hex' not in df.columns:
        df['total_layers'] = df['classification_layers'] + df['replication_layers'] + df['merged_layers']

        if hasattr(df['genotype'], 'parallel_map'):
            df['genotype_hex'] = df['genotype'].parallel_map(lambda x: hex(int(re.sub('\D', '', str(x)), 2)))
        else:
            df['genotype_hex'] = df['genotype'].map(lambda x: hex(int(re.sub('\D', '', str(x)), 2)))

        df['is_mutant'] = df['genotype_hamming_distance_from_parent'] > 0

    df = df.join(df.rename(columns={c: f'parent_{c}' for c in df.columns}), on='parent_id')

    df.to_pickle(results_cache_path, compression=None)

    return df


def load_deep_evaluations(sample_name, experiment_id, selection_intensity=1):

    sample_path = f'{config["deep_evaluations_dir"]}/{sample_name}.csv'
    db_path = f'{config["experiment_results_dir"]}/{experiment_id}.sqlite'

    evaluations = pd.read_csv(sample_path)
    evaluations = evaluations[evaluations['visited'] == True].drop('visited', axis=1)

    ids = '", "'.join(evaluations["serann_id"].values)

    conn = create_engine(f'sqlite:////{db_path}').connect()
    experiment_df = pd.read_sql(f'select * from srann where id in ("{ids}")', con=conn)
    experiment_df['genotype'] = experiment_df['genotype'].map(lambda x: np.array(eval(x)).astype(int))
    experiment_df['genotype_hex'] = experiment_df['genotype'].map(lambda x: hex(int(re.sub('\D', '', str(x)), 2)))

    evaluations = evaluations.merge(experiment_df, left_on='serann_id', right_on='id')#.dropna(subset=['parent_id'])

    parent_ids = '", "'.join(evaluations["parent_id"].dropna().values)

    parents_info = pd.read_sql(f'select * from srann where id in ("{parent_ids}")', con=conn).drop('parent_id', axis=1)
    parents_info['genotype'] = parents_info['genotype'].map(lambda x: np.array(eval(x)).astype(int))
    parents_info['genotype_hex'] = parents_info['genotype'].map(lambda x: hex(int(re.sub('\D', '', str(x)), 2)))
    parents_info.columns = ['parent_' + column for column in parents_info.columns]
    evaluations = evaluations.merge(parents_info, on='parent_id', how='left')

    evaluations['absolute_fecundity'] = evaluations['classification_accuracy'].fillna(0)
    evaluations['offspring_viability'] = evaluations['offspring_viability'].fillna(0)

    conn.connection.create_function("power", 2, lambda x, y: x ** y)

    # ∑𝐹𝑗^𝜆
    sql = f'''
        SELECT generation, SUM(power(IFNULL(absolute_fitness, 0), {selection_intensity})) AS absolute_fecundity_sum
        FROM srann
        GROUP BY generation
    '''

    absolute_fecundity_sum_by_gen = pd.read_sql(sql, con=conn).set_index('generation')['absolute_fecundity_sum']
    evaluations = evaluations.join(absolute_fecundity_sum_by_gen, on='generation')

    # 𝑓𝑖 = 𝐹^𝜆 / ∑𝐹𝑗^𝜆
    evaluations['fecundity'] = (evaluations['absolute_fecundity'] ** selection_intensity) / evaluations['absolute_fecundity_sum']

    population_size = pd.read_sql('select COUNT(*) from srann where generation = 0', con=conn).iloc[0, 0]
    evaluations['normed_fecundity'] = population_size * evaluations['fecundity']

    # 𝑤(𝑡) = 𝑉⋅𝑓⋅𝑁
    evaluations['fitness'] = evaluations['offspring_viability'] * evaluations['fecundity'] * population_size

    evaluations.loc[(evaluations['is_valid'] == False) | (evaluations['is_overweight'] == True), 'fitness'] = 0

    return evaluations


def prepare_muller_plot_data(df, ancestor_id=None, frequency_threshold=0.3, return_identity_map=False):
    df = df.copy()

    ancestor_id = ancestor_id or df[df['generation'] == 0].iloc[0].name
    df = df.reset_index()
    df.loc[df['generation'] == 1, 'parent_id'] = ancestor_id
    df.loc[df['generation'] == 0, 'id'] = ancestor_id
    df = df.drop_duplicates(subset='id').set_index('id')

    population_size = len(df[df['generation'] == 1])

    offspring_counts = df.groupby('parent_id')['genotype'].count()
    offspring_counts = offspring_counts.reindex(df.index, fill_value=0)
    df['offspring_counts'] = offspring_counts

    df = df.groupby('genotype_hex').filter(lambda x: x['offspring_counts'].max() > 0)

    df['identity'] = 1
    df['parent_identity'] = None

    def update(x):
        df.loc[x.index, 'identity'] = df['identity'].max() + 1

    for generation, data in tqdm(df.groupby('generation'), total=df['generation'].nunique()):

        if generation == 0:
            continue

        same = data[data['genotype_hex'] == data['parent_genotype_hex']].index
        parent_identities = df.loc[df.loc[same, 'parent_id'], 'identity'].values
        df.loc[same, 'identity'] = parent_identities
        df.loc[same, 'parent_identity'] = parent_identities

        different = data.index.difference(same)

        parent_identities = df.loc[df.loc[different, 'parent_id'], 'identity'].values
        df.loc[different, 'parent_identity'] = parent_identities
        # New clones must share the same parent identity and the same genotype to have the same identity (IBS)
        data.loc[different].groupby([parent_identities, 'genotype_hex']).apply(update)

    max_descendants_frequency = df.groupby('identity').apply(
        lambda x: x.groupby('generation')['genotype'].count().max())

    grouped_by_parent = df.sort_values('parent_identity', ascending=False).groupby('parent_identity', sort=False)

    for parent_identity, offspring in tqdm(grouped_by_parent, total=df['parent_identity'].nunique()):
        candidates = [parent_identity] + list(offspring['identity'].unique())
        max_descendants_frequency.loc[parent_identity] = max(max_descendants_frequency.loc[candidates])

    dominant_clones = max_descendants_frequency[max_descendants_frequency > frequency_threshold * population_size]

    dominant_rows = df[df['identity'].isin(dominant_clones.index)]
    populations_df = dominant_rows.groupby(['generation', 'identity'])['genotype'].count()

    multiindex = pd.MultiIndex.from_product([dominant_rows['generation'].unique(), dominant_clones.index],
                                            names=['generation', 'identity'])
    populations_df = populations_df.reindex(multiindex, fill_value=0)

    background_clone = 0

    for i in dominant_rows['generation'].unique():
        populations_df.loc[(i, background_clone)] = population_size - populations_df.loc[i].sum()

    populations_df = populations_df.sort_index().reset_index()
    populations_df.columns = ['Generation', 'Identity', 'Population']

    adjacency_df = dominant_rows.groupby('identity')['parent_identity'].min().reset_index().rename(
        columns={'identity': 'Identity', 'parent_identity': 'Parent'})
    adjacency_df.loc[0, 'Parent'] = 0
    adjacency_df = adjacency_df[adjacency_df['Identity'] != adjacency_df['Parent']]
    adjacency_df = adjacency_df[['Parent', 'Identity']]

    if return_identity_map:
        return populations_df, adjacency_df, df['identity']

    return populations_df, adjacency_df