import tensorflow as tf
from keras import backend as K
from keras import layers, models, losses, callbacks


def vae_loss(x, x_decoded_mean, z_log_var, z_mean, original_dim):
    # Reconstruction loss
    xent_loss = original_dim * losses.binary_crossentropy(x, x_decoded_mean)
    # KL divergence
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss


def create_encoder(original_dim, latent_dim):
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(512, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    encoder = models.Model(inputs, [z_mean, z_log_var])
    return encoder


def create_decoder(latent_dim, original_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    h = layers.Dense(512, activation='relu')(latent_inputs)
    outputs = layers.Dense(original_dim, activation='sigmoid')(h)
    decoder = models.Model(latent_inputs, outputs)
    return decoder


class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var = self.encoder(inputs)
        z = layers.Lambda(self.sampling)([z_mean, z_log_var])
        reconstructed = self.decoder(z)
        # Add loss
        loss = vae_loss(inputs, reconstructed, z_log_var, z_mean, inputs.shape[-1])
        self.add_loss(loss)
        return reconstructed

    @staticmethod
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def build_vae(latent_dim=2, original_dim=1000):  # Example dimensions
    # Encoder
    inputs = layers.Input(shape=(original_dim,))
    h = layers.Dense(256, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_sigma = layers.Dense(latent_dim)(h)

    z = layers.Lambda(VAE.sampling)([z_mean, z_log_sigma])

    # Decoder
    decoder_h = layers.Dense(256, activation='relu')
    decoder_mean = layers.Dense(original_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = models.Model(inputs, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = models.Model(inputs, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = layers.Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = models.Model(decoder_input, _x_decoded_mean)

    # Loss
    xent_loss = original_dim * K.binary_crossentropy(inputs, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    return vae, encoder, generator
