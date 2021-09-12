# Based on: https://github.com/KarenUllrich/binary-VAE
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

tfd = tfp.distributions


class GeneticAutoencoder(tf.keras.Model):

    def __init__(self, genotype_length, max_phenotype_length, vocabulary_size, embedding_dim, genotype_alphabet_size):
        super(GeneticAutoencoder, self).__init__()

        self._genotype_alphabet_size = genotype_alphabet_size

        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(max_phenotype_length,)),
                tf.keras.layers.Embedding(vocabulary_size, embedding_dim),
                tf.keras.layers.Reshape((max_phenotype_length, embedding_dim, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, 5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(16, 3),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(16, 3),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(genotype_length * genotype_alphabet_size),
                tf.keras.layers.Reshape((genotype_length, genotype_alphabet_size))
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(genotype_length, genotype_alphabet_size)),
                tf.keras.layers.Conv1D(32, 5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(max_phenotype_length * vocabulary_size),
                tf.keras.layers.Reshape((max_phenotype_length, vocabulary_size)),
                tf.keras.layers.BatchNormalization(),

                tf.keras.layers.Lambda(lambda x: tf.nn.log_softmax(x, axis=-1),
                                       output_shape=(max_phenotype_length, vocabulary_size))
            ]
        )

    def decode(self, z):
        logits = tf.squeeze(self.generative_net(tf.one_hot(z, self._genotype_alphabet_size)))
        return tf.argmax(logits, axis=-1)

    def _decode(self, x, z):
        logits = self.generative_net(z)

        observation_dist = tfd.Categorical(logits=logits)
        logpx_z = tf.reduce_sum(observation_dist.log_prob(x), axis=-1)
        return logpx_z


class ConcreteGAE(GeneticAutoencoder):

    def __init__(self, genotype_length, max_phenotype_length, vocabulary_size, embedding_dim, genotype_alphabet_size,
                 prior_temperature=0.1):

        super(ConcreteGAE, self).__init__(genotype_length, max_phenotype_length, vocabulary_size, embedding_dim,
                                          genotype_alphabet_size)

        probs = tf.ones((genotype_length, genotype_alphabet_size)) / genotype_alphabet_size
        self.prior = tfd.Gumbel(tf.math.log(probs) / prior_temperature, 1. / prior_temperature)

    def _encode(self, x, temperature=0.2):

        logits = self.inference_net(x)

        # The temperature adjusts the relaxation of the Concrete
        # distribution. We use,
        latent_dist = tfd.Gumbel(logits / temperature, float(1. / temperature))
        # instead of
        # tfd.RelaxedBernoulli(temperature=temperature, logits=logits).
        # Otherwise we run into underflow issues when computing the
        # log_prob. This has been explained in [2] Appendix C.3.2.

        logistic_samples = latent_dist.sample()
        return tf.nn.softmax(logistic_samples), latent_dist.log_prob(logistic_samples), self.prior.log_prob(logistic_samples)

    def compute_loss(self, x, temperature, kld_weight):
        # z ~ q(z|x), q(z|x), p(z)
        z, logqz_x, logpz = self._encode(x, temperature)
        # p(x|z)
        logpx_z = self._decode(x, z)

        # empirical KL divergence
        kl_div = tf.reduce_sum(logqz_x - logpz, axis=range(1, logqz_x.ndim))

        # we minimize the negative evidence lower bound (nelbo)
        nelbo = - tf.reduce_mean(logpx_z - kld_weight * kl_div)

        return {'loss': nelbo, 'nll': -tf.reduce_mean(logpx_z), 'kld': tf.reduce_mean(kl_div)}

    def encode(self, x):
        logits = self.inference_net(x)
        return tf.argmax(logits, axis=-1)


class DeterministicGAE(GeneticAutoencoder):

    def _encode(self, x):
        return tf.nn.softmax(self.inference_net(x), axis=-1)

    def compute_loss(self, x):
        z = self._encode(x)
        logpx_z = self._decode(x, z)

        return {'loss': -tf.reduce_mean(logpx_z)}

    def encode(self, x):
        logits = tf.nn.softmax(self.inference_net(x))
        return tf.argmax(logits, axis=-1)


def load_ribosomal_autoencoder(model_path):

    vae = tf.saved_model.load(str(model_path))

    def encode(tokens):
        return np.argmax(vae.inference_net(tf.constant(tokens, dtype='float32'), False, None).numpy(), axis=-1)

    def decode(genotype):
        one_hot = np.eye(2)[genotype.astype(int)]
        return np.argmax(vae.generative_net(tf.constant(one_hot, dtype='float32'), False, None).numpy(), axis=-1)

    return encode, decode