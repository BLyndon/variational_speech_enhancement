from tensorflow.keras.layers import Layer, Dense, Conv2DTranspose, Reshape


class Decoder(Layer):
    '''Decoder class:

        Parameters:
        latent_dim (int): Dimension of the latent space
        outp_shape (tuple): input shape
        net (string): neural network architecture
        info (bool): print info at initialization
    '''

    def __init__(self,
                 latent_dim=2,
                 outp_shape=(28, 28, 1),
                 net='conv',
                 info=False,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        # Initialize Shape & Architecture
        self.outp_shape = outp_shape
        self.latent_dim = latent_dim
        self.net = net
        assert self.net == 'conv' or self.net == 'mlp', print(
            'Choose net from {"conv", "mlp"}')

        # Initialize Layers
        if self.net == 'conv':
            self.dense = Dense(7*7*32, activation='relu')
            self.reshape = Reshape((7, 7, 32))
            self.conv2dT1 = Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same', activation='relu')
            self.conv2dT2 = Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',  activation='relu')
            self.out = Conv2DTranspose(
                filters=1, kernel_size=3, padding='same', name='output')
        elif self.net == 'mlp':
            digit_len = self.outp_shape[0] * \
                self.outp_shape[1]*self.outp_shape[2]
            self.dense1 = Dense(512, activation='relu')
            self.dense2 = Dense(digit_len)
            self.out = Reshape(self.outp_shape)
        elif self.net == 'clean_speech':
            self.dense = Dense(128, activation='tanh')
            self.out = Dense(self.outp_shape[0])

        # Print Encoder-Decoder Info
        if info == True:
            self.print_info()

    def __call__(self, z):
        if self.net == 'conv':
            z = self.dense(z)
            z = self.reshape(z)
            cz = self.conv2dT1(z)
            cz = self.conv2dT2(cz)
            x = self.out(cz)
        elif self.net == 'mlp':
            z = self.dense1(z)
            z = self.dense2(z)
            x = self.out(z)
        elif self.net == 'clean_speech':
            z = self.dense(z)
            x = self.out(z)
        return x

    def print_info(self):
        print()
        print(self.net + '-Decoder')
        print(' - Latent Variable Space: {}'.format(self.latent_dim))
        print(' - Output Shape: {}'.format(self.outp_shape))


if __name__ == "__main__":
    pass
