import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from global_config import config


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset-name", required=True, help='Synthetic SeRANN dataset name')
    parser.add_argument("-m", "--max-tokens", default=350, type=int, help='Maximum tokens sequence length')

    return parser.parse_args()


def tokenize(s, characters):

    if len(characters) == 0:
        try:
            int(s)
            return list(s)
        except:
            return [s]

    s = s.replace('\n\n', '\n').replace('\n\n', '\n').replace(' ', '')

    c = characters[0]

    splitted = sum([[ss, c] for ss in s.split(c) if s != ''], [])[:-1]

    return sum([tokenize(ss, characters[1:]) for ss in splitted], [])


if __name__ == '__main__':

    args = get_args()

    full_dataset = pd.read_csv(Path(config['synthetic_datasets_dir']) / (args.dataset_name + '.csv'), compression='zip')

    split_chars = ['\n', '=', '\'', '(', ')', '[', ']', ',', '.']

    def worker(slices):
        results = {}

        for index, s in slices:
            results[index] = tokenize(s, split_chars)

        return results

    results = {}
    inputs = []
    bulk_size = int(np.ceil(len(full_dataset) / 1000))

    for i in range(0, len(full_dataset), bulk_size):
        inputs.append(list(dict(full_dataset['code'].iloc[i:i + bulk_size]).items()))

    with mp.Pool() as p:
        for r in tqdm(p.imap_unordered(worker, inputs), total=len(inputs)):
            results.update(r)

    full_dataset['tokens'] = pd.Series(results)

    unique = set()

    for tokens in tqdm(list(full_dataset['tokens'])):
        unique.update(tokens)

    word2index = dict(zip(sorted(unique), range(len(unique))))
    index2word = {v: k for k, v in word2index.items()}

    word2index['<PAD>'] = len(word2index)
    index2word[word2index['<PAD>']] = '<PAD>'

    num_tokens = full_dataset['tokens'].map(len)
    full_dataset = full_dataset[num_tokens <= args.max_tokens]

    def to_numeric(r):
        return [word2index[t] for t in r] + [word2index['<PAD>']] * (args.max_tokens - len(r))

    numeric = full_dataset['tokens'].apply(to_numeric)
    mat = np.asmatrix(numeric.tolist())
    np.savez_compressed(Path(config['token_sequences_dir']) / (args.dataset_name + '.npz'), sequences=mat)

    vocab = pd.Series(word2index).rename_axis('token').rename('index').to_frame()
    vocab.to_csv(Path(config['vocabularies_dir']) / (args.dataset_name + '.csv'))
