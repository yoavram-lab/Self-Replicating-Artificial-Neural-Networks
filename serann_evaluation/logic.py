import ast
import pickle
import re
import shutil
from collections import Counter
from pathlib import Path
import tensorflow.keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
from distributed_computing.api import WorkerInterface

from common.logic import probabilistic_proofreading, build_serann_model, get_serann_training_data
from evolutionary_experiment.logic.ribosomal_autoencoder import RibosomalAutoencoder


class SampleDeepEvaluator(object):

    def __init__(self, experiment_df, output_path, sample_ids, workers_pool, shared_cache=True):

        K.set_floatx('float16')
        K.set_epsilon(1e-4)

        self._workers_pool = workers_pool
        print(f'Available workers: {self._workers_pool.get_workers_count()}')

        self._exp_df = experiment_df
        self._output_path = output_path
        self._shared_cache = shared_cache

        self._sample_size = min(len(sample_ids), len(self._exp_df))

        if Path(output_path).is_file():
            print('Resuming existing SeRANNs sample evaluation')
            with open(output_path, 'rb') as f:
                sample = pickle.load(f)
            self.use_existing_sample(sample, self._exp_df, workers_pool)

        else:
            print('Sampling SeRANNs for evaluation')
            self._sample_metadata = self._exp_df.loc[sample_ids]
            self._results = {id_: {} for id_ in sample_ids}

    def use_existing_sample(self, sample, mutants, workers_pool):

        evaluations_cache = {k: v for k, v in sample.items() if len(v) > 0}
        workers_pool.update_workers({'args': [evaluations_cache]})

        self._results = sample
        self._sample_metadata = mutants.loc[list(sample)]

        path = Path(self._output_path)
        shutil.copy(path, path.with_name(path.stem + '_backup.csv'))

    def run(self):

        remaining_ids = [id_ for id_, r in self._results.items() if len(r) == 0]
        remaining = self._sample_metadata.loc[remaining_ids]

        jobs = [{'args': row} for row in remaining.iterrows()]

        durations = []
        tb = pd.to_datetime('now')

        for i, (serann_id, results) in enumerate(self._workers_pool.imap_unordered(jobs), 1):
            genotype_hex = remaining.at[serann_id, 'genotype_hex']

            if self._shared_cache:
                self._workers_pool.update_workers({'args': [{genotype_hex: results}]})

            self._results[serann_id] = {
                'classification_accuracy': results['classification_accuracy'],
                'mutation_rate': results['mutation_rate'],
                'offspring_survival': results['offspring_survival']
            }

            durations.append(pd.to_datetime('now') - tb)
            avg_duration = pd.Series(durations[-200:]).mean()
            duration_str = f'{avg_duration.seconds // 60:02d}:{avg_duration.seconds % 60:02d}'
            print(f'{i}/{len(jobs)} jobs done. Average evaluation time: {duration_str}.')

            if i % 100 == 0 or i == len(jobs):
                print('Saving to disk.')
                with open(self._output_path, 'wb') as f:
                    pickle.dump(self._results, f)

            tb = pd.to_datetime('now')


class SerannEvaluator(object):

    def __init__(self, experiment_params, num_evaluations=50, replications_per_evaluation=100, ribosomal_ae=None):

        K.set_floatx('float16')
        K.set_epsilon(1e-4)

        self._source_code_validity_cache = {}
        self._experiment_params = experiment_params

        from evolutionary_experiment.config import config

        self._train_x, self._train_y, self._classification_x_test = get_serann_training_data(config['encodings_dataset_path'],
                                                                                             experiment_params['num_classification_classes'])
        self._num_evaluations = num_evaluations
        self._replications_per_evaluation = replications_per_evaluation

        if ribosomal_ae is None:
            self._ribosomal_ae = RibosomalAutoencoder(self._experiment_params['max_tokens'])
        else:
            self._ribosomal_ae = ribosomal_ae

    def get_evaluation_model(self, source_code, loss_balance, evaluations):

        K.clear_session()

        from tensorflow.keras.layers import Input
        from tensorflow.keras.models import Model

        X_input = Input(shape=[*self._experiment_params['classification_image_dimensions'], 1])
        g_input = Input(shape=(self._experiment_params['genotype_size'], 1))

        outputs = sum([list(build_serann_model(source_code, X_input, g_input, i)[:2]) for i in range(evaluations)], [])

        model = Model(inputs=[X_input, g_input], outputs=outputs)

        losses, metrics, loss_weights = {}, {}, {}

        for j in range(evaluations):
            losses[f'classification_output_{j}'] = 'categorical_crossentropy'
            losses[f'replication_output_{j}'] = 'mse'
            metrics[f'classification_output_{j}'] = 'categorical_accuracy'
            loss_weights[f'classification_output_{j}'] = loss_balance
            loss_weights[f'replication_output_{j}'] = 1 - loss_balance

        model.compile(loss=losses, loss_weights=loss_weights, metrics=metrics,
                      optimizer=tf.keras.optimizers.Adam(1e-3, epsilon=1e-4))
        return model

    def is_valid_serann(self, source_code, loss_balance):

        try:
            ast.parse(source_code)
        except SyntaxError:
            return False

        try:
            self.get_evaluation_model(source_code, loss_balance, 1)
            return True

        except Exception as e:
            return False

    def evaluate(self, genotype, source_code=None, loss_balance=None):

        genotype = np.array(genotype)
        genotype_hex = hex(int(re.sub('\D', '', str(genotype)), 2))

        print('Evaluating genotype: ', genotype_hex)

        if source_code is None:
            source_code = self._ribosomal_ae.decode_to_string(genotype[None])[0]

        if loss_balance is None:
            try:
                loss_balance = float(re.search(r'loss_balance *= *([0-9]+\.[0-9]+?)(\s|$)', source_code).group(1))
            except AttributeError:
                print('Invalid/overweight SeRANN received. Returning empty result.')

                return {
                    'classification_accuracy': np.NaN,
                    'offspring_survival': {},
                    'mutation_rate': {},
                }

        classification_accuracies = []

        model = self.get_evaluation_model(source_code, loss_balance, self._num_evaluations)

        training_results = model.fit(self._train_x, self._train_y * self._num_evaluations,
                                     batch_size=self._experiment_params['training_batch_size'],
                                     epochs=self._experiment_params['training_epochs'], validation_split=0.05,
                                     verbose=0)

        for key, val in training_results.history.items():
            if key.startswith('val_') and key.endswith('_categorical_accuracy'):
                classification_accuracies.append(val[-1])

        replication_examples = np.repeat(genotype[None], self._replications_per_evaluation, axis=0)

        results = model.predict([self._classification_x_test[:self._replications_per_evaluation],
                                 np.expand_dims(replication_examples, 2)])

        replications = []

        for i in range(self._num_evaluations):
            for j in range(self._replications_per_evaluation):
                replications.append(results[i * 2 + 1][j].squeeze())

        genotypes = np.round(np.clip(replications, 0, 1))

        survival_evaluations, mutation_rate_evaluations = [], []

        genotypes = np.round(np.clip(genotypes, 0, 1))
        fixed_genotypes = probabilistic_proofreading(genotype, genotypes,
                                                     self._experiment_params['error_correction_probability'])

        source_codes = self._ribosomal_ae.decode_to_string(fixed_genotypes)

        for i in range(self._num_evaluations):

            micro_evaluations = {}

            for j in range(self._replications_per_evaluation):
                s = source_codes[i * self._replications_per_evaluation + j]

                is_valid = self._source_code_validity_cache.get(s) or int(self.is_valid_serann(s, loss_balance))
                micro_evaluations[is_valid] = micro_evaluations.get(is_valid, 0) + 1

                self._source_code_validity_cache[s] = is_valid

            survival_evaluations.append(micro_evaluations)

            mutations = np.sum(fixed_genotypes[i: i + self._replications_per_evaluation] != genotype, axis=1)
            mutation_rates = mutations / len(genotype)
            mutation_rate_evaluations.append(dict(Counter(mutation_rates)))

        return {
            'classification_accuracy': classification_accuracies,
            'mutation_rate': mutation_rate_evaluations,
            'offspring_survival': survival_evaluations
        }


class SerannEvaluationWorker(SerannEvaluator, WorkerInterface):

    def __init__(self, *args, evaluation_cache=True, **kwargs):
        super(SerannEvaluationWorker, self).__init__(*args, **kwargs)

        if evaluation_cache is True:
            self._evaluations_cache = {}
        else:
            self._evaluations_cache = None

        print('Evaluator initialization is done.')

    def run(self, serann_id, serann_info):

        if self._evaluations_cache is not None and serann_info['genotype_hex'] in self._evaluations_cache:
            evaluation_results = self._evaluations_cache[serann_info['genotype_hex']]

        elif not serann_info['is_valid'] or serann_info['is_overweight']:
            evaluation_results = {
                'classification_accuracy': np.NaN,
                'offspring_survival': {},
                'mutation_rate': {},
            }

            print('Invalid/overweight SeRANN received. Returning empty result.')

        else:
            print('Evaluating SeRANN. ', end='')
            evaluation_results = self.evaluate(serann_info['genotype'])
            print('Done.')

        if self._evaluations_cache is not None:
            self._evaluations_cache[serann_info['genotype_hex']] = evaluation_results

        return serann_id, evaluation_results

    def handle_update(self, update_data):
        if self._evaluations_cache is not None:
            self._evaluations_cache.update(update_data)
