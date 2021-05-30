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
                 hidden_dim=128,
                 activation='tanh',
                 info=False,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        # Initialize Shape & Architecture
        if type(inp_shape) == int:
            self.inp_shape = (inp_shape,)
        else:
            self.inp_shape = inp_shape
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Initialize Layers
        self.flatten = Flatten()
        self.dense_hidden = Dense(self.hidden_dim, activation=self.activation)

        self.dense_mu = Dense(self.latent_dim, name='mu_l')
        self.dense_log_var = Dense(self.latent_dim, name='log_var_l')

        if info == True:
            self.print_info()

    def __call__(self, x):
        x = self.flatten(x)
        x = self.dense_hidden(x)

        mu = self.dense_mu(x)
        log_var = self.dense_log_var(x)
        return mu, log_var

    def print_info(self):
        print()
        print('-Encoder')
        print(' - Output Shape: {}'.format(self.inp_shape))
        print(' - Hidden dimension: {}'.format(self.hidden_dim))
        print(' - Latent Variable Space: {}'.format(self.latent_dim))
        print(' - Activation-fct: {}'.format(self.activation))


if __name__ == "__main__":
    pass
