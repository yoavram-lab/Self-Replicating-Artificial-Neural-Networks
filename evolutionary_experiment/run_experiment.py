import argparse
import json
import time
import os
import socket
from pathlib import Path
from uuid import uuid4

import numpy as np

from distributed_computing.api import start_head_node, start_node
from distributed_computing.api import Pool as WorkersPool
from evolutionary_experiment.config import config
from evolutionary_experiment.logic.experiment import Experiment
from evolutionary_experiment.logic.experiment_db import ExperimentDB
from evolutionary_experiment.logic.experiment_worker import EvolutionaryExperimentWorker


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--parameters", required=False, help='Experiment parameters file path')
    parser.add_argument("-r", "--resume-experiment-id", required=False, help='Experiment ID to resume')
    parser.add_argument("-s", "--random-seed", required=False, help='Override random seed')
    parser.add_argument("-w", "--workers-pool-port", required=False, type=int, help='Workers pool server port')
    parser.add_argument("-o", "--workers-pool-password", required=False, help='Workers pool server password')
    parser.add_argument("-a", "--workers-pool-address", required=False, help='Workers pool server address')

    return parser.parse_args()


def prepare_workers_pool(server_address, server_port, password):

    if server_address == 'localhost':
        workers_terminate, _ = start_head_node(password, server_port)

    else:
        workers_terminate, _ = start_node(password, server_address, server_port, False)

    return workers_terminate


if __name__ == '__main__':

    args = get_args()

    password = args.workers_pool_password or str(np.random.randint(1e+9)).zfill(9)
    port = args.workers_pool_port or config['workers_pool_port']
    address = args.workers_pool_address or 'localhost'

    workers_terminate = prepare_workers_pool(address, port, password)

    time.sleep(10)

    np.random.seed(int(args.random_seed or config['random_seed']))

    Path(config['experiment_results_dir']).mkdir(exist_ok=True, parents=True)

    serann_dataset = np.load(config['encodings_dataset_path'])['encodings']

    experiment_info = {}

    if args.resume_experiment_id is None:
        experiment_id = str(uuid4())
        db_path = str(Path(config['experiment_results_dir'], f'{experiment_id}.sqlite'))

        with open(args.parameters, 'r') as f:
            experiment_info = {
                'parameters': json.load(f),
                'experiment_db': ExperimentDB(db_path)
            }

    else:
        experiment_id = args.resume_experiment_id
        db_path = str(Path(config['experiment_results_dir'], f'{args.resume_experiment_id}.sqlite'))
        experiment_db = ExperimentDB(db_path)
        start_generation = experiment_db.get_generations_count()

        if args.parameters is not None:
            with open(args.parameters, 'r') as f:
                experiment_parameters = json.load(f)
        else:
            info = experiment_db.get_last_execution_info()
            experiment_parameters = dict(info.drop('start_time'))
            if start_generation == experiment_parameters['num_generations']:
                experiment_parameters['num_generations'] *= 2

        experiment_info = {
            'parameters': experiment_parameters,
            'experiment_db': experiment_db,
            'start_generation': start_generation
        }

    experiment_info['parameters']['host_name'] = socket.gethostname()
    experiment_info['parameters']['encodings_dataset'] = Path(config['encodings_dataset_path']).stem
    experiment_info['parameters']['tokens_vocabulary'] = Path(config['vocabulary_path']).stem
    experiment_info['parameters']['ribosomal_autoencoder'] = Path(config['ribosomal_autoencoder_path']).stem
    experiment_info['parameters']['random_seed'] = int(args.random_seed or config['random_seed'])

    required_workers = np.ceil(experiment_info["parameters"]['num_seranns'] / config['max_serann_per_gpu'])

    workers_pool_args = (EvolutionaryExperimentWorker, {'args': [experiment_info["parameters"]]},
                         password, address, port, config['worker_pool_job_timeout'], 6000,
                         required_workers)

    print(f'Waiting for at least {required_workers:.0f} pool workers to initialize.')

    with WorkersPool(*workers_pool_args) as p:
        experiment = Experiment(experiment_id, serann_dataset, p, **experiment_info)
        experiment.execute()

    time.sleep(5)

    workers_terminate()
