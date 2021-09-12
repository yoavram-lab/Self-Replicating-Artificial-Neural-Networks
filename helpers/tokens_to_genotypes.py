import argparse
from pathlib import Path

import keras
import keras.backend as K
import numpy as np
from tqdm.auto import tqdm

from global_config import config
from ribosomal_autoencoder.model import load_ribosomal_autoencoder


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset-name", required=True, help='Synthetic SeRANN dataset name')
    parser.add_argument("-r", "--riboae-name", required=True, help='Ribosomal autoencoder model name')
    parser.add_argument("-b", "--batch-size", default=256, type=int, help='Batch size for the ribosomal autoencoder '
                                                                          'inference')

    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()

    dataset_path = Path(config["token_sequences_dir"]) / (args.dataset_name + '.npz')
    dataset = np.load(dataset_path)['sequences']

    K.set_floatx('float16')
    K.set_epsilon(1e-4)

    encoder, decoder = load_ribosomal_autoencoder(Path(config["ribosomal_autoencoders_dir"]) / args.riboae_name)

    class DataGenerator(keras.utils.Sequence):

        def __init__(self, padded_examples, batch_size=32):
            self._examples = np.array(padded_examples).astype(np.uint8)
            self._batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self._examples) / self._batch_size))

        def __getitem__(self, index):
            return self._examples[index * self._batch_size:(index + 1) * self._batch_size]

    genotypes = []

    for batch in tqdm(DataGenerator(dataset, batch_size=args.batch_size)):
        encoded = encoder(dataset)
        genotypes.append(encoded)

    result_path = Path(config["encodings_datasets_dir"]) / f'{args.dataset_name}__{args.riboae_name}.npz'
    np.savez_compressed(result_path, encodings=np.vstack(genotypes))



