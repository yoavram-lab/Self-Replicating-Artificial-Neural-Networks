import os

import numpy as np
import pandas as pd
from multiprocessing import Process, Queue

from evolutionary_experiment.config import config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class RibosomalAutoencoder(object):
    def __init__(self, max_serann_tokens, use_cpu=True):

        self._tokenizer = Tokenizer()
        self._max_serann_tokens = max_serann_tokens
        self._vocabulary = pd.read_csv(config['vocabulary_path'])

        self._model_name = config['ribosomal_autoencoder_path'].split(os.sep)[-1].split('.')[0]

        self._jobs_q, self._results_q = Queue(), Queue()

        self._ae_process = RiboAeProcess(self._jobs_q, self._results_q,
                                         config['ribosomal_autoencoder_path'], use_cpu)
        self._ae_process.daemon = True
        self._ae_process.start()

        self._word2index = self._vocabulary.set_index('token')['index']
        self._index2word = self._vocabulary.set_index('index')['token']

        self.PADDING_TOKEN = self._word2index.loc['<PAD>']

    def get_model_name(self):
        return self._model_name

    def remove_sequences_padding(self, sequences):
        lengths = sequences.shape[1] - np.sum(sequences == self._word2index.loc['<PAD>'], axis=1)
        return [sequences[i, :lengths[i]] for i in range(len(sequences))]

    def encode_sequence(self, serann_token_index_sequences):
        self._jobs_q.put(('encode', serann_token_index_sequences))
        result = self._results_q.get()

        if isinstance(result, Exception):
            raise result

        return result

    def encode_string(self, serann_codes):

        tokenized = [self._tokenizer(s)[:self._max_serann_tokens] for s in serann_codes]
        padded = np.ones((len(serann_codes), self._max_serann_tokens)) * self._word2index.loc['<PAD>']

        for i, t in enumerate(tokenized):
            padded[i, :len(t)] = self._word2index.loc[t]

        return self.encode_sequence(padded)

    def decode_to_sequence(self, encodded_serann_codes):
        self._jobs_q.put(('decode', encodded_serann_codes))

        result = self._results_q.get()

        if isinstance(result, Exception):
            raise result

        return result

    def sequence_to_string(self, serann_sequences):

        if type(serann_sequences) == pd.Series:
            serann_sequences = np.array(serann_sequences.values.tolist())

        words = np.array(self._index2word)[serann_sequences]

        codes = [''.join(serann_words).rstrip('<PAD>') for serann_words in words]

        return codes

    def decode_to_string(self, encodded_serann_codes):

        min_dists = self.decode_to_sequence(encodded_serann_codes)
        return self.sequence_to_string(min_dists)


class RiboAeProcess(Process):

    def __init__(self, jobs_q, results_q, model_path, use_cpu=False):
        super(RiboAeProcess, self).__init__()
        self._jobs_q = jobs_q
        self._results_q = results_q
        self._model_path = model_path
        self._use_cpu = use_cpu

    def run(self):

        if self._use_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        encode, decode = self._load_autoencoder()

        while True:
            job, data = self._jobs_q.get()

            try:
                result = {'encode': encode, 'decode': decode}[job](data)
            except Exception as e:
                result = e

            self._results_q.put(result)

    def _load_autoencoder(self):

        import tensorflow as tf
        raw = tf.saved_model.load(self._model_path)

        def encode(tokens):
            return np.argmax(raw.inference_net(tf.constant(tokens, dtype='float32'), False, None).numpy(), axis=-1)

        def decode(genotype):
            one_hot = np.eye(2)[genotype.astype(int)]
            return np.argmax(raw.generative_net(tf.constant(one_hot, dtype='float32'), False, None).numpy(), axis=-1)

        return encode, decode


class Tokenizer(object):

    def __init__(self):
        self._split_characters = ['\n', '=', '\'', '(', ')', '[', ']', ',', '.']

    def __call__(self, s, split_characters=None):

        if split_characters is None:
            split_characters = self._split_characters

        if len(split_characters) == 0:
            try:
                int(s)
                return list(s)
            except:
                return [s]

        s = s.replace('\n\n', '\n').replace('\n\n', '\n').replace(' ', '')

        c = split_characters[0]

        splitted = sum([[ss, c] for ss in s.split(c) if s != ''], [])[:-1]

        return sum([self(ss, split_characters[1:]) for ss in splitted], [])




