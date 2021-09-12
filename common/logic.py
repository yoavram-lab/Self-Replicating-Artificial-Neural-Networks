import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def probabilistic_proofreading(parent_genotype, offspring_genotypes, ec_factor):

    offspring_genotypes = offspring_genotypes.copy()
    rows, cols = np.where(offspring_genotypes != parent_genotype)
    fixed_loci = np.random.permutation(len(rows))[:int(ec_factor * len(rows))]
    rows, cols = rows[fixed_loci], cols[fixed_loci]
    offspring_genotypes[rows, cols] = parent_genotype[cols]

    return offspring_genotypes


def build_serann_model(source_code, x_input, g_input, name, num_classification_classes=10, genotype_size=100):

    from tensorflow.keras.layers import Lambda, Dense, Reshape, concatenate, Conv1D, Conv2D, MaxPool2D, Activation
    # https://github.com/keras-team/keras/issues/9582#issuecomment-462277683
    from common.BatchNormalizationF16 import BatchNormalizationF16 as BatchNormalization

    X_layer, g_layer = x_input, g_input

    exec(source_code)

    last_layer = locals()['con']

    y_hat = Dense(num_classification_classes)(Reshape((1, -1))(last_layer))
    y_hat = Activation('softmax', name=f'classification_output_{name}')(y_hat)

    g_rep = Dense(genotype_size)(Reshape((1, -1))(last_layer))
    g_rep = Activation('sigmoid', name=f'replication_output_{name}')(g_rep)

    return y_hat, g_rep, float(locals()['loss_balance'])


def get_serann_training_data(encodings_dataset_path, num_classification_classes):

    encoded_serann = np.load(encodings_dataset_path)['encodings']
    mnist.load_data()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    classification_x_train = np.expand_dims(x_train.astype('float32') / 255, axis=3)
    classification_y_train = np.expand_dims(to_categorical(y_train, num_classification_classes), axis=1)

    classification_x_test = np.expand_dims(x_test.astype('float32') / 255, axis=3)

    genotype_train = encoded_serann[:len(classification_x_train)]

    train_x = [classification_x_train, np.expand_dims(genotype_train, 2)]
    train_y = [classification_y_train, np.expand_dims(genotype_train, 1)]

    return train_x, train_y, classification_x_test