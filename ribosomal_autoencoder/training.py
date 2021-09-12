"""
Based on:
https://github.com/KarenUllrich/binary-VAE
"""

import argparse
import json
import os
from shutil import rmtree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
from ribosomal_autoencoder.model import ConcreteGAE

tfd = tfp.distributions

from ribosomal_autoencoder.config import config


def get_dataset(batch_size, train_test_ratio):

    dataset = np.load(config["token_sequences_dataset_path"])['sequences']

    split_point = int(len(dataset) * train_test_ratio)

    trainset = dataset[:split_point].astype('float32')
    testset = dataset[split_point:].astype('float32')

    train_dataset = tf.data.Dataset.from_tensor_slices(trainset).batch(batch_size).repeat().prefetch(10)
    test_dataset = tf.data.Dataset.from_tensor_slices(testset).batch(batch_size)

    return train_dataset, test_dataset


def compute_gradients(model, x, training_args):
    with tf.GradientTape() as tape:
        metrics = model.compute_loss(x, **training_args)

    return tape.gradient(metrics['loss'], model.trainable_variables), metrics


def apply_gradients(optimizer, gradients, variables):
    optimizer.apply_gradients(zip(gradients, variables))


def train(experiment_name, model, learning_rate, train_dataset, vocabulary, min_backup_interval):

    print('Starting training')
    min_loss = np.inf
    min_loss_batch = 0

    lr = tf.Variable(learning_rate, dtype='float64')
    optimizer = tf.keras.optimizers.Adam(lr)
    temperature_steps = int(5e+6)
    learning_rate_steps = int(1e+6)
    kld_weight_steps = int(1e+7)
    temperatures = np.logspace(np.log10(0.3), -3, temperature_steps)
    learning_rates = np.logspace(np.log10(3e-4), np.log10(2e-5), learning_rate_steps)
    kld_weights = np.linspace(0, 0.2, kld_weight_steps) ** 2

    for batch_number, batch_examples in train_dataset.enumerate(1):
        temperature = temperatures[min(batch_number, temperature_steps - 1)]
        learning_rate = learning_rates[min(batch_number, learning_rate_steps - 1)]
        kld_weight = kld_weights[min(batch_number, kld_weight_steps - 1)]

        if type(model) == ConcreteGAE:
            training_args = {'temperature': temperature, 'kld_weight': kld_weight}
        else:
            training_args = {}

        lr.assign(learning_rate)
        gradients, metrics = compute_gradients(model, batch_examples, training_args)
        apply_gradients(optimizer, gradients, model.trainable_variables)

        if batch_number % 10 == 0:
            print(f'Batch: {batch_number} | Loss: {metrics["loss"]:.5f} | NLL: {metrics["nll"]} | KL: {metrics["kld"]}')

        if batch_number % 50 == 0:
            z = model.encode(batch_examples[:1])
            sequence = model.decode(z).numpy()
            original = ''.join(vocabulary[batch_examples[0].numpy()]).rstrip('<PAD>')
            reconstruction = ''.join(vocabulary[sequence]).rstrip('<PAD>')
            print(f'\n\nOriginal: \n{original}\n\n')
            print(f'Genotype:\n{"".join([str(i) for i in z[0].numpy()])}\n\n')
            print(f'Reconstruction: \n{reconstruction}\n\n')

        if metrics["loss"] < min_loss and batch_number - min_loss_batch >= min_backup_interval:
            local_path = f'{config["ribosomal_autoencoders_dir"]}/{experiment_name}_b{batch_number}'
            print('Backing up to: ', local_path)
            tf.saved_model.save(model, local_path)
            prev_backup = f'{config["ribosomal_autoencoders_dir"]}/{experiment_name}_b{min_loss_batch}'

            if min_loss_batch > 0:
                rmtree(prev_backup)

            min_loss, min_loss_batch = metrics["loss"], batch_number


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--parameters", required=True, help='Experiment parameters file path')
    parser.add_argument("-n", "--name", required=True, help='Output model name')

    return parser.parse_args()


if __name__ == "__main__":

    args = get_args()

    with open(args.parameters) as f:
        parameters = json.load(f)

    vocabulary = pd.read_csv(config["vocabulary_path"])
    index_to_token = vocabulary.set_index('index')['token']

    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(1)
    tf.executing_eagerly()

    model = ConcreteGAE(parameters['genotype_size'], parameters['source_code_length'], len(vocabulary),
                        parameters['embedding_size'], parameters['genotype_alphabet_size'],
                        prior_temperature=parameters['prior_temperature'])

    train_dataset, test_dataset = get_dataset(parameters['batch_size'], parameters['train_test_ratio'])
    train(args.name, model, parameters['learning_rate'], train_dataset, index_to_token,
          parameters['min_backup_interval'])
