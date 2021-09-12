from datetime import datetime
import os

from distributed_computing.api import WorkerInterface
from tensorflow.keras import Model
from tensorflow.keras.callbacks import Callback

# Can't use tensorflow.keras because of this memory leak:
# https://github.com/tensorflow/tensorflow/issues/35030
from common.logic import build_serann_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import tensorflow.keras.backend as K
import numpy as np
import pandas as pd


class EvolutionaryExperimentWorker(WorkerInterface):

    def __init__(self, experiment_params):

        import tensorflow as tf

        try:
            device = tf.config.get_visible_devices('GPU')[0]
            tf.config.experimental.set_memory_growth(device, True)
        except:
            pass

        self._parameters = experiment_params

        K.set_floatx('float16')
        K.set_epsilon(1e-4)

        from evolutionary_experiment.config import config

        self._serann_train = np.load(config['encodings_dataset_path'])['encodings']

        print('Worker initialization is done.')

    def run(self, seranns_info, num_replications):

        print('Processing ', len(seranns_info), 'SeRANNs')

        K.clear_session()

        main_inputs, models_info = self._build_keras_models(seranns_info)

        models_info['is_valid'] = ~models_info['loss_balance'].isna()
        models_info['is_overweight'] = False
        models_info['classification_validation_accuracy'] = np.NaN
        models_info['classification_training_accuracy'] = np.NaN
        models_info['classification_test_accuracy'] = np.NaN
        models_info['replication_mse'] = np.NaN
        models_info['is_overweight'] = models_info['parameters_count'] > self._parameters['max_serann_parameters']

        trainable_models = models_info[models_info['is_valid'] & ~models_info['is_overweight']]

        outputs = []
        outputs_by_id = {}

        for i, (serann_id, serann_info) in enumerate(trainable_models.iterrows()):
            outputs += serann_info['outputs']
            outputs_by_id[serann_id] = len(outputs) - 1

        joined_model = Model(inputs=main_inputs, outputs=outputs)

        losses, loss_weights, metrics = self._build_losses(trainable_models['loss_balance'])

        if len(losses) == 0:
            return {'learning_time': 0, 'replication_time': 0, 'models_info': models_info.drop('outputs', axis=1),
                    'offspring_by_id': {}}

        joined_model.compile(loss=losses, loss_weights=loss_weights, metrics=metrics,
                             # https://github.com/tensorflow/tensorflow/issues/7226#issuecomment-283195916
                             optimizer=tf.keras.optimizers.Adam(1e-3, epsilon=1e-4))

        train_data = self._build_dataset(len(trainable_models))
        joined_model, training_stats, training_time = self._learn(joined_model, train_data)

        print('Learning done')
        print('Test set evaluation started')
        test_evaluation = joined_model.evaluate(train_data['test_x'], train_data['test_y'], verbose=0)
        models_info = self._collect_accuracies(joined_model, models_info, training_stats,
                                               test_evaluation).drop('outputs', axis=1)

        learning_results = {'learning_time': training_time.total_seconds(), 'models_info': models_info}

        if num_replications > 0:
            replication_results = self._replicate(joined_model, models_info.join(seranns_info[['genotype']]),
                                                  outputs_by_id, num_replications)

            learning_results.update(replication_results)

        return learning_results

    def _learn(self, joined_model, train_data):

        epochs = self._parameters['training_epochs']

        class ProgressCallback(Callback):

            def on_train_begin(self, *args, **kwargs):
                print('Learning started')

            def on_epoch_end(self, epoch, logs, *args, **kwargs):
                print(f'Learning progress: {((epoch + 1) / epochs) * 100:.2f}%')

        def train(batch_size):
            time_before = datetime.now()
            training_stats = joined_model.fit(train_data['train_x'], train_data['train_y'],
                                              batch_size=batch_size, epochs=epochs,
                                              verbose=0, callbacks=[ProgressCallback()],
                                              validation_split=0.05)
            return training_stats, datetime.now() - time_before

        try:
            training_stats, training_time = train(self._parameters['training_batch_size'])

        except:  # ResourceExhaustedError - missing on AWS instances:
            print('ResourceExhaustedError. Trying a smaller batch size')
            training_stats, training_time = train(self._parameters['training_batch_size'] // 2)

        return joined_model, training_stats, training_time

    def _replicate(self, joined_model, seranns_info, outputs_by_id, num_replications):

        print('Num replications:', num_replications)

        offspring_by_id = {}
        time_before = datetime.now()

        valid_mask = seranns_info['is_valid'] & ~seranns_info['is_overweight']

        total_examples = valid_mask.sum() * num_replications
        repeat_mnist = np.ceil(total_examples / len(self._classification_x_test)).astype(int)

        genotypes = []
        index_by_serann = {}

        for serann_id, serann_info in seranns_info[seranns_info['is_valid'] & ~seranns_info['is_overweight']].iterrows():
            index_by_serann[serann_id] = len(genotypes) * num_replications
            genotypes.append(np.repeat([serann_info['genotype']], num_replications, axis=0))

        genotypes = np.vstack(genotypes)
        classification_examples = np.repeat(self._classification_x_test, repeat_mnist, axis=0)[:len(genotypes)]
        replication_results = joined_model.predict([classification_examples, np.expand_dims(genotypes, 2)])

        for serann_id, serann_info in seranns_info[valid_mask].iterrows():

            serann_results = replication_results[outputs_by_id[serann_id]]

            start_index = index_by_serann[serann_id]
            stop_index = start_index + num_replications

            offspring_by_id[serann_id] = serann_results[int(start_index):int(stop_index)][:, 0]

        print('Replication done')

        return {
            'replication_time': (datetime.now() - time_before).total_seconds(),
            'offspring_by_id': offspring_by_id
        }

    def _collect_accuracies(self, joined_model, models_info, training_stats, test_evaluation):

        for key, val in training_stats.history.items():

            if key.startswith('classification_output_') and key.endswith('_categorical_accuracy'):

                serann_id = key[len('classification_output_'):-len('_categorical_accuracy')]
                models_info.loc[serann_id, 'classification_training_accuracy'] = val[-1]
                models_info.loc[serann_id, 'is_valid'] = True

            elif key.startswith('val_classification_output_') and key.endswith('_categorical_accuracy'):

                serann_id = key[len('val_classification_output_'):-len('_categorical_accuracy')]
                models_info.loc[serann_id, 'classification_validation_accuracy'] = val[-1]

            elif key.startswith('val_replication_output_') and key.endswith('_loss'):
                serann_id = key[len('val_replication_output_'):-len('_loss')]
                models_info.loc[serann_id, 'replication_mse'] = val[-1]

        for key, val in zip(joined_model.metrics_names, test_evaluation):

            if key.startswith('classification_output_') and key.endswith('_categorical_accuracy'):

                serann_id = key[len('classification_output_'):-len('_categorical_accuracy')]
                models_info.loc[serann_id, 'classification_test_accuracy'] = val

        return models_info

    def _build_dataset(self, num_seranns):

        from tensorflow.keras.datasets import mnist
        from tensorflow.keras.utils import to_categorical

        mnist.load_data()
        classification_classes = self._parameters['num_classification_classes']

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        self._classification_x_train = np.expand_dims(x_train.astype('float32') / 255, axis=3)
        self._classification_x_test = np.expand_dims(x_test.astype('float32') / 255, axis=3)
        self._classification_y_train = np.expand_dims(to_categorical(y_train, classification_classes), axis=1)
        self._classification_y_test = np.expand_dims(to_categorical(y_test, classification_classes), axis=1)

        genotype_train = self._serann_train[:len(self._classification_x_train)]
        genotype_test = self._serann_train[len(self._classification_x_train):
                                          len(self._classification_x_train) + len(self._classification_x_test)]
        return {
            'train_x': [self._classification_x_train, np.expand_dims(genotype_train, 2)],
            'train_y': [self._classification_y_train, np.expand_dims(genotype_train, 1)] * num_seranns,
            'test_x': [self._classification_x_test, np.expand_dims(genotype_test, 2)],
            'test_y': [self._classification_y_test, np.expand_dims(genotype_test, 1)] * num_seranns
        }

    def _build_keras_models(self, seranns_info):

        from tensorflow.keras import Input, Model

        X_input = Input(shape=[*self._parameters['classification_image_dimensions'], 1])
        g_input = Input(shape=(self._parameters['genotype_size'], 1))

        results = pd.DataFrame(index=seranns_info.index).assign(outputs=None, parameters_count=np.NaN,
                                                                loss_balance=np.NaN)

        for serann_id, serann_info in seranns_info.iterrows():

            try:
                y_hat, g_rep, loss_balance = build_serann_model(serann_info['source_code'], X_input, g_input, serann_id,
                                                                self._parameters['num_classification_classes'],
                                                                self._parameters['genotype_size'])
            except Exception as e:
                continue

            results.at[serann_id, 'parameters_count'] = Model(inputs=[X_input, g_input],
                                                             outputs=[y_hat, g_rep]).count_params()
            results.at[serann_id, 'loss_balance'] = loss_balance
            results.at[serann_id, 'outputs'] = [y_hat, g_rep]

        return [X_input, g_input], results

    def _execute_network(self, source_code, serann_id, X_input, g_input, num_classification_classes, genotype_size):

        from tensorflow.keras.layers import Lambda, Dense, Reshape, concatenate, Conv1D, Conv2D, MaxPool2D, Activation
        # https://github.com/keras-team/keras/issues/9582#issuecomment-462277683
        from common.BatchNormalizationF16 import BatchNormalizationF16 as BatchNormalization

        X_layer, g_layer = X_input, g_input

        exec(source_code)

        last_layer = locals()['con']

        # Reshape is added to avoid exceptions when a skip is made over the concatenation layer
        y_hat = Dense(num_classification_classes)(Reshape((1, -1))(last_layer))
        y_hat = Activation('softmax', name=f'classification_output_{serann_id}')(y_hat)

        g_rep = Dense(genotype_size)(Reshape((1, -1))(last_layer))
        g_rep = Activation('sigmoid', name=f'replication_output_{serann_id}')(g_rep)

        return y_hat, g_rep, float(locals()['loss_balance'])

    def _build_losses(self, loss_balance_by_id):

        losses = {}
        loss_weights = {}
        metrics = {}

        for serann_id, loss_balance in loss_balance_by_id.iteritems():
            losses.update({f'classification_output_{serann_id}': 'categorical_crossentropy',
                           f'replication_output_{serann_id}': 'mse'})
            loss_weights.update({f'classification_output_{serann_id}': loss_balance,
                                 f'replication_output_{serann_id}': 1 - loss_balance})
            metrics[f'classification_output_{serann_id}'] = 'categorical_accuracy'

        return losses, loss_weights, metrics


