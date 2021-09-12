import argparse
import json
import os
import time

import pandas as pd
from pathlib import Path
import numpy as np

from distributed_computing.api import start_head_node, start_node
from distributed_computing.api import Pool as WorkersPool

from serann_evaluation.logic import SampleDeepEvaluator, SerannEvaluationWorker
from common.utils import load_experiment_results
from evolutionary_experiment.config import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

SAMPLERS = {}


def prepare_workers_pool(server_address, server_port, password):

    if server_address == 'localhost':
        workers_terminate, _ = start_head_node(password, server_port)

    else:
        workers_terminate, _ = start_node(password, server_address, server_port, False)

    return workers_terminate


class MetaSampler(type):
    def __new__(mcs, name, bases, attrs):

        cls = super().__new__(mcs, name, bases, attrs)

        if name != 'SamplerInterface':
            SAMPLERS[attrs['name']] = cls

        return cls


class SamplerInterface(object, metaclass=MetaSampler):

    def sample(self, df, parameters):
        raise NotImplemented


class MutantsSampler(SamplerInterface):

    name = 'mutants'

    def sample(self, df, parameters):

        mask = ~df['parent_id'].isna() & df['genotype_hex'].ne(df['parent_genotype_hex'])

        sample = df[mask].sample(parameters['total_samples'])

        sample = sample.sample(frac=1).assign(temp_index=np.arange(len(sample)))
        parents = df.loc[sample['parent_id']].assign(temp_index=np.arange(len(sample)))

        return pd.concat([sample, parents]).sort_values('temp_index').index


class UniqueGenotypesSampler(SamplerInterface):

    name = 'unique_genotypes'

    def sample(self, df, parameters):

        all_unique = df.sample(frac=1).drop_duplicates(subset=['genotype_hex'])

        if parameters['total_samples'] > -1:
            return all_unique.iloc[:parameters['total_samples']].index
        else:
            return all_unique.index


class UniqueSourceCodesSampler(SamplerInterface):

    name = 'unique_source_codes'

    def sample(self, df, parameters):

        all_unique = df.sample(frac=1).drop_duplicates(subset=['source_code'])

        if parameters['total_samples'] > -1:
            return all_unique.iloc[:parameters['total_samples']].index
        else:
            return all_unique.index


class GeneralSampler(SamplerInterface):

    name = 'general'

    def sample(self, df, parameters):

        mask = ((df['generation'] % parameters['generation_step'] == 0) & (df['generation'] != 0)) | (df['generation'] == 1)

        if parameters['samples_per_generation'] > 0:
            sample = df[mask].groupby('generation').apply(lambda x: x.sample(parameters['samples_per_generation']))
            sample = sample.reset_index('generation', drop=True)
        else:
            sample = df[mask].sample(parameters['total_samples'])

        return sample.sample(frac=1).index


def update_workers_with_cached_evaluations(evaluations_path, df, workers_pool):

    sample = pd.read_csv(evaluations_path)
    visited = sample[sample['visited'] == True]
    visited_ids = df.index.intersection(visited['serann_id'].drop_duplicates().values)

    keys = df.loc[visited_ids, 'genotype_hex']

    def get_record(x):
        x = x.drop_duplicates(subset=['proofreading_strength'])

        return {
            'classification_accuracy': x['classification_accuracy'].mean(),
            'mutation_rate': {r['proofreading_strength']: r['mutation_rate'] for _, r in x.iterrows()},
            'offspring_viability': {r['proofreading_strength']: r['offspring_viability'] for _, r in x.iterrows()}
        }

    evaluations_cache = visited.set_index('serann_id').groupby(keys).apply(get_record).to_dict()
    workers_pool.update_workers({'args': [evaluations_cache]})


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--parameters", required=True, help='Parameters file path')
    parser.add_argument("-n", "--output-name", required=True, help='Evaluation file name')
    parser.add_argument("-c", "--cached-evaluations", required=False, help='Use evaluations from this file as cache')
    parser.add_argument("-s", "--no-shared-cache", default=False, action='store_true', help='Don\'t share cache between'
                                                                                            ' workers')
    parser.add_argument("-w", "--workers-pool-port", required=False, type=int, help='Workers pool server port')
    parser.add_argument("-o", "--workers-pool-password", required=False, help='Workers pool server password')
    parser.add_argument("-a", "--workers-pool-address", required=False, help='Workers pool server address')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    password = args.workers_pool_password or str(np.random.randint(1e+9)).zfill(9)
    port = args.workers_pool_port or config['workers_pool_port']
    address = args.workers_pool_address or 'localhost'

    workers_terminate = prepare_workers_pool(address, port, password)

    time.sleep(10)

    with open(args.parameters) as f:
        parameters = json.load(f)

    print('Loading data')

    df = load_experiment_results(parameters["experiment_id"])

    print('Sampling examples to evaluate')

    if parameters['sampler'] not in SAMPLERS:
        raise Exception('Unknown sampler')

    sample = SAMPLERS[parameters['sampler']]().sample(df, parameters)

    sample = sample.drop_duplicates()

    print('Creating workers pool')
    worker_params = [parameters['experiment_params'], parameters['num_evaluations'],
                     parameters['replications_per_evaluation']]

    workers_pool = WorkersPool(SerannEvaluationWorker, {'args': worker_params}, password, address, port,
                               job_timeout=800, min_gpu_memory_required=8000, min_workers=1)

    if args.cached_evaluations is not None:
        update_workers_with_cached_evaluations(args.cached_evaluations, df, workers_pool)

    output_path = Path(config['serann_evaluations_dir']) / (args.output_name + '.pkl')

    sampler_evaluator = SampleDeepEvaluator(df, str(output_path), sample, workers_pool,
                                            shared_cache=not args.no_shared_cache)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    sampler_evaluator.run()

    workers_terminate()






