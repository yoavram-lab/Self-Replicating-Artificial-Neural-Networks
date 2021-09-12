from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, cdist

from evolutionary_experiment.config import config

from edlib import align

from evolutionary_experiment.logic.ribosomal_autoencoder import RibosomalAutoencoder


def levenshtein(a, b, *args, **kwargs):
    return align(a, b)['editDistance']

def hamming(a, b, *args, **kwargs):
    return np.not_equal(a, b).sum(axis=-1) / np.shape(a)[-1]


class Experiment(object):

    def __init__(self, experiment_id, serann_dataset, workers_pool, experiment_db, parameters, start_generation=0):

        self._experiment_db = experiment_db
        self._parameters = parameters

        self._serann_dataset = serann_dataset

        self._id = experiment_id

        self._start_generation = start_generation

        self._genetic_encoder = RibosomalAutoencoder(self._parameters['max_tokens'])

        self._experiment_db.save_execution_info(datetime.now(), self._parameters)

        self._workers_pool = workers_pool

    def execute(self):

        print(f'Experiment {self._id} has started')
        print(f'Using {self._workers_pool.get_workers_count()} GPU workers')

        current_generation = self._get_first_generation()
        offspring_pool_size = self._parameters['initial_offspring_pool_size']
        print('Initial offspring pool size:', offspring_pool_size)

        for generation_number in range(self._start_generation, self._parameters['num_generations']):

            generation_start_time = datetime.now()

            print(f'### Generation {generation_number} execution has started ###')

            current_generation['experiment_id'] = self._id
            current_generation['generation'] = generation_number
            current_generation['num_offspring'] = 0

            print('Starting training and replication')
            models_info, offspring_by_id, time_measurements = self._learn_and_replicate(current_generation,
                                                                                        offspring_pool_size)
            current_generation = current_generation.join(models_info)

            print('Calculating fecundity scores')
            current_generation['absolute_fertility'], current_generation['relative_fertility'] = \
                self._get_fertility(current_generation['classification_validation_accuracy'])

            print('Sampling offspring counts')
            if current_generation['is_valid'].sum() > 0:
                current_generation['num_offspring'] = np.random.multinomial(self._parameters['num_seranns'],
                                                                            current_generation['relative_fertility'])

            valid_serann = current_generation[current_generation['is_valid'] & ~current_generation['is_overweight'].astype(bool)]

            print('Extracting source codes statistics')
            source_code_stats = self._extract_layer_counts(valid_serann['source_code'])
            current_generation = current_generation.join(source_code_stats, how='left')

            print('Saving generation information to the experiment DB')
            self._report_generation_statistics(current_generation, generation_number, time_measurements,
                                               generation_start_time)

            print('Saving SeRANN records to the experiment DB')
            self._experiment_db.save_seranns_info(current_generation)

            if len(valid_serann) == 0:
                print('No valid SeRANNs left! stopping...')
                break

            print('Applying offspring selection')
            next_generation = self._select_offspring(current_generation, offspring_by_id)

            print(f'Generation {generation_number} execution is done')

            offspring_pool_size = current_generation['num_offspring'].max() * self._parameters['offspring_pool_size_factor']
            print('Offspring pool size was updated to:', offspring_pool_size)
            current_generation = next_generation

            self._print_generation_time(datetime.now() - generation_start_time)

    def _print_generation_time(self, generation_time):
        hours = generation_time.seconds // 3600
        minutes = (generation_time.seconds // 60) % 60
        seconds = generation_time.seconds % 60

        print(f'\033[1mGeneration execution time: {hours:02d}:{minutes:02d}:{seconds:02d}\033[0m')

    def _get_fertility(self, classification_performance):

        absolute_fertility = classification_performance ** self._parameters['selection_pressure']
        unsafe_relative_fertility = absolute_fertility / np.sum(absolute_fertility)
        relative_fertility = np.nan_to_num(unsafe_relative_fertility)
        relative_fertility /= relative_fertility.sum()

        return absolute_fertility, relative_fertility

    def _load_last_generation_from_db(self):

        last_generation = self._experiment_db.get_serann_by_generation(self._start_generation - 1)

        print('Replicating the last generation')
        valid_mask = (last_generation['is_valid'] == True) & (last_generation['is_overweight'] == False)
        valid_serann = last_generation[valid_mask]

        if len(valid_serann) == 0 or valid_serann['num_offspring'].sum() == 0:
            raise Exception('No valid SeRANNs left in the last generation!')

        print('Retraining and replicating the last generation')
        pool_size = self._parameters['initial_offspring_pool_size']
        print('Initial offspring pool size:', pool_size)
        models_info, offspring_by_id, time_measurements = self._learn_and_replicate(last_generation, pool_size)

        return self._select_offspring(last_generation, offspring_by_id)

    def _get_first_generation(self):

        if self._start_generation > 0:
            return self._load_last_generation_from_db()

        ancestor_genotype = self._parameters.get('ancestor_genotype')

        if ancestor_genotype is None:
            serann_sample = np.random.randint(0, len(self._serann_dataset), self._parameters['num_seranns'])
            genotype_by_id = {str(uuid4()): self._serann_dataset[s] for s in serann_sample}

        else:
            ancestor_genotype = np.array(ancestor_genotype)
            genotype_by_id = {str(uuid4()): ancestor_genotype for _ in range(self._parameters['num_seranns'])}

        current_generation = pd.Series(genotype_by_id).rename('genotype').rename_axis('id').to_frame()
        genotypes = np.array(current_generation['genotype'].values.tolist())
        current_generation['source_code'] = self._genetic_encoder.decode_to_string(genotypes)
        current_generation['parent_id'] = None
        current_generation['genotype_euclidean_distance_from_parent'] = np.NaN
        current_generation['genotype_hamming_distance_from_parent'] = np.NaN
        current_generation['source_code_levenshtein_distance_from_parent'] = np.NaN

        return current_generation

    def _learn_and_replicate(self, current_generation, num_offspring):

        time_measurements = {}
        model_infos = []
        offspring_by_id = {}

        available_gpus = self._workers_pool.get_workers_count()

        if len(current_generation) / available_gpus < config['max_serann_per_gpu']:
            batch_index = np.arange(len(current_generation)) % available_gpus

        else:
            num_of_batches = np.ceil(len(current_generation) / config['max_serann_per_gpu'])
            batch_index = np.arange(len(current_generation)) % num_of_batches

        batch_by_id = pd.Series(batch_index, index=current_generation.index).astype(int)
        total_batches = batch_by_id.nunique()

        jobs = [{'args': [data, num_offspring]} for _, data in current_generation.groupby(batch_by_id)]

        for i, result in enumerate(self._workers_pool.imap_unordered(jobs), 1):
            print(f'{i} of {total_batches} batches completed')
            model_infos.append(result['models_info'])
            offspring_by_id.update(result.get('offspring_by_id', {}))

            time_measurements.setdefault('learning_times', []).append(result['learning_time'])
            time_measurements.setdefault('replication_times', []).append(result['replication_time'])

        return pd.concat(model_infos), offspring_by_id, time_measurements

    def _select_offspring(self, current_generation, offspring_by_id):

        selection_strategy = {
            'random': self._random_offspring_selection,
            'best': self._best_offspring_selection
        }.get(self._parameters['offspring_selection_strategy'])

        if selection_strategy is None:
            raise Exception('Unknown offspring selection strategy')

        offspring_ids = []
        parent_ids = []
        genotypes = []

        for parent_id, serann_offspring in offspring_by_id.items():
            parent_genotype = current_generation.loc[parent_id, 'genotype']
            num_offspring = current_generation.loc[parent_id, 'num_offspring']

            serann_offspring = np.round(np.clip(serann_offspring, 0, 1))
            selected_genotypes = selection_strategy(parent_genotype, num_offspring, serann_offspring)
            selected_genotypes = self._probabilistic_proofreading(parent_genotype, selected_genotypes)

            offspring_ids += [str(uuid4()) for _ in range(num_offspring)]
            parent_ids += [parent_id] * num_offspring
            genotypes += list(selected_genotypes)

        offspring_genotypes = np.array(genotypes)
        source_code = self._genetic_encoder.decode_to_string(offspring_genotypes)

        next_generation = pd.DataFrame({'genotype': genotypes, 'parent_id': parent_ids, 'source_code': source_code},
                                       index=offspring_ids).rename_axis('id')

        parent_genotypes = np.array(current_generation.loc[parent_ids, 'genotype'].values.tolist())

        euclidean_from_parent = np.sqrt(np.sum(np.power(offspring_genotypes - parent_genotypes, 2), axis=1))
        next_generation['genotype_euclidean_distance_from_parent'] = euclidean_from_parent

        hamming_from_parent = (parent_genotypes != offspring_genotypes).sum(axis=1) / self._parameters['genotype_size']
        next_generation['genotype_hamming_distance_from_parent'] = hamming_from_parent

        def levenshtein_distance_from_parent(x):
            return levenshtein(x['source_code'], current_generation.loc[x['parent_id'], 'source_code'])

        levenshtein_from_parent = next_generation.apply(levenshtein_distance_from_parent, axis=1)
        next_generation['source_code_levenshtein_distance_from_parent'] = levenshtein_from_parent

        return next_generation

    def get_genotype_stats(self, serann_info):

        genotypes = np.array(serann_info['genotype'].values.tolist())
        distribution = serann_info['genotype'].astype(str).value_counts(normalize=True)

        return {
            'mean_pairwise_euclidean_distance': np.mean(pdist(genotypes)),
            'mean_pairwise_hamming_distance': np.mean(pdist(genotypes, metric=hamming)),
            'mean_euclidean_distance_from_parent': serann_info['genotype_euclidean_distance_from_parent'].mean(),
            'mean_hamming_distance_from_parent': serann_info['genotype_hamming_distance_from_parent'].mean(),
            'shannon_index': -np.sum(distribution * np.log(distribution)),
            'nucleotide_diversity': cdist(genotypes, genotypes, metric=hamming).sum(),
            'species_richness': serann_info['genotype'].astype(str).nunique()
        }

    def get_source_code_stats(self, serann_info):

        source_codes = np.expand_dims(serann_info['source_code'], -1)
        distribution = serann_info['source_code'].value_counts(normalize=True)

        return {
            # 'median_pairwise_levenshtein_distance': np.median(pdist(source_codes,
            #                                                         metric=lambda a, b: levenshtein(a[0], b[0]))),
            'median_levenshtein_distance_from_parent':
                serann_info['source_code_levenshtein_distance_from_parent'].median(),
            'shannon_index': -np.sum(distribution * np.log(distribution)),
            'species_richness': serann_info['source_code'].nunique()
        }

    def _report_generation_statistics(self, serann_info, generation_number, time_measurements, start_time):

        survived = serann_info['is_valid'] & ~serann_info['is_overweight'].astype(bool)

        generation_info = {
            'experiment_id': self._id,
            'generation': generation_number,
            'start_time': start_time,
            'survival_rate': survived.mean(),
            'overweight_rate': serann_info['is_overweight'].mean(),
            'invalid_rate': 1 - serann_info['is_valid'].mean(),
            'mean_parameters_count': serann_info['parameters_count'].mean(),
            'mean_absolute_fertility': serann_info['absolute_fertility'].mean(),
            'absolute_fertility_std': serann_info['absolute_fertility'].std(),
            'mean_loss_balance': serann_info['loss_balance'].mean(),
            'mean_classification_validation_accuracy': serann_info['classification_validation_accuracy'].mean(),
            'mean_classification_training_accuracy': serann_info['classification_training_accuracy'].mean(),
            'mean_classification_test_accuracy': serann_info['classification_test_accuracy'].mean(),
            'max_classification_test_accuracy': serann_info['classification_test_accuracy'].max(),
            'mean_replication_mse': serann_info['replication_mse'].mean(),
            'learning_time_seconds': np.mean(time_measurements['learning_times']),
            'replication_time_seconds': np.mean(time_measurements['replication_times']),
            'total_time_seconds': (datetime.now() - start_time).total_seconds(),
            'mean_classification_layers': serann_info['classification_layers'].mean(),
            'mean_replication_layers': serann_info['replication_layers'].mean(),
            'mean_merged_layers': serann_info['merged_layers'].mean()
        }

        genotype_stats = {f'genotype_{k}': v for k, v in self.get_genotype_stats(serann_info).items()}
        source_code_stats = {f'source_code_{k}': v for k, v in self.get_source_code_stats(serann_info).items()}

        generation_info.update(genotype_stats)
        generation_info.update(source_code_stats)

        self._experiment_db.save_generation_info(generation_info)

    def _random_offspring_selection(self, _, num_offspring, offspring_genotypes):
        return offspring_genotypes[np.random.permutation(len(offspring_genotypes))[:num_offspring]]

    def _best_offspring_selection(self, parent_genotype, num_offspring, offspring_genotypes):
        dists = hamming(parent_genotype, offspring_genotypes)
        return offspring_genotypes[np.argsort(dists)[:num_offspring]]

    def _probabilistic_proofreading(self, parent_genotype, offspring_genotypes):

        ec_factor = self._parameters['error_correction_probability']
        offspring_genotypes = offspring_genotypes.copy()
        rows, cols = np.where(offspring_genotypes != parent_genotype)
        fixed_loci = np.random.permutation(len(rows))[:int(ec_factor * len(rows))]
        rows, cols = rows[fixed_loci], cols[fixed_loci]
        offspring_genotypes[rows, cols] = parent_genotype[cols]

        return offspring_genotypes

    def _extract_layer_counts(self, serann_source_codes):

        return pd.DataFrame({
            'classification_layers': serann_source_codes.map(lambda x: x.count('X_layer=')),
            'replication_layers': serann_source_codes.map(lambda x: x.count('g_layer=')),
            'merged_layers': serann_source_codes.map(lambda x: x.count('con='))
        })


