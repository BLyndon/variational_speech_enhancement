from tensorflow.keras.layers import Layer, Dense, Conv2DTranspose, Reshape


class Decoder(Layer):
    '''Decoder class:

        Parameters:
        latent_dim (int): Dimension of the latent space
        outp_shape (tuple): Output shape
        net (string): Neural network architecture. Choose from {"conv", "mlp", "clean_speech"}
        info (bool): if true print info at initialization
    '''

    def __init__(self,
                 latent_dim=8,
                 outp_shape=513,
                 hidden_dim=128,
                 activation='tanh',
                 info=False,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        # Initialize Shape & Architecture
        if type(outp_shape) == int:
            self.outp_shape = (outp_shape,)
        else:
            self.outp_shape = outp_shape
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        # Initialize Layers
        self.dense = Dense(self.hidden_dim, activation=self.activation)
        self.out = Dense(self.outp_shape[0], name='log_var_f')

        # Print Encoder-Decoder Info
        if info == True:
            self.print_info()

    def __call__(self, z):
        z = self.dense(z)
        x = self.out(z)
        return x

    def print_info(self):
        print()
        print('Decoder')
        print(' - Output dimension: {}'.format(self.outp_shape))
        print(' - Hidden dimension: {}'.format(self.hidden_dim))
        print(' - Latent Variable Space: {}'.format(self.latent_dim))
        print(' - Activation-fct: {}'.format(self.activation))


if __name__ == "__main__":
    pass
