import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Conv2D, Flatten


class Encoder(Layer):
    '''Encoder class:

        Parameters:
        inp_shape (tuple): Input shape
        latent_dim (int): Dimension of the latent space
        net (string): Neural network architecture. Choose from {"conv", "mlp", "clean_speech"}
        info (bool): if true print info at initialization
    '''

    def __init__(self,
                 inp_shape=(513, 1),
                 latent_dim=8,
                 net='clean_speech',
                 info=False,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        # Initialize Shape & Architecture
        self.inp_shape = inp_shape
        self.latent_dim = latent_dim
        self.net = net
        assert self.net in ['conv', 'mlp', 'clean_speech'], print(
            'Choose net from {"conv", "mlp", "clean_speech"}')

        # Initialize Layers
        self.flatten = Flatten()
        if self.net == 'conv':
            self.conv2d1 = Conv2D(filters=32, kernel_size=3,
                                  strides=(2, 2), activation='relu')
            self.conv2d2 = Conv2D(filters=64, kernel_size=3,
                                  strides=(2, 2), activation='relu')
        elif self.net == 'mlp':
            self.dense_hidden = Dense(512, activation='relu')
        elif self.net == 'clean_speech':
            self.dense_hidden = Dense(128, activation='tanh')
        self.dense_mu = Dense(self.latent_dim, name='mu')
        self.dense_log_var = Dense(self.latent_dim, name='log_var')

        if info == True:
            self.print_info()

    def __call__(self, x):
        if self.net == 'conv':
            cx = self.conv2d1(x)
            cx = self.conv2d2(cx)
            x = self.flatten(cx)
        elif self.net == 'clean_speech':
            x = self.flatten(x)
            x = self.complex_norm(x)
            x = self.dense_hidden(x)
        elif self.net == 'mlp':
            x = self.flatten(x)
            x = self.dense_hidden(x)
        mu = self.dense_mu(x)
        log_var = self.dense_log_var(x)
        return mu, log_var

    def print_info(self):
        print()
        print(self.net + '-Encoder')
        print(' - Input Shape: {}'.format(self.inp_shape))
        print(' - Latent Variable Space: {}'.format(self.latent_dim))


if __name__ == "__main__":
    pass
