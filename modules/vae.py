import time

from IPython import display

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

from plotter import Plotter


#######################################################################################
#    REPARAMETRIZATION TRICK                                                          #
#######################################################################################


class Sampling(Layer):
    ''' Reparametrization Trick:
    Apply reparametrization trick to lower the variance during training phase. Sample latent variable from normal distribution definde by the mean and the (log-)variance.

    Inputs
    inputs (list): list containing mean and the log variance
    '''

    def __call__(self, inputs):
        mean, log_var = inputs
        eps = tf.random.normal(shape=tf.shape(mean))
        return mean + tf.multiply(tf.exp(0.5 * log_var), eps)


#######################################################################################
#    VARIATIONAL AUTOENCODER                                                          #
#######################################################################################

class VAE(Model):
    ''' Variational Autoencoder

    Parameters:
    encoder (Encoder): encoder network
    decoder (Decoder): decoder network
    loss_fn (func): vae loss function
    '''

    def __init__(self, encoder, decoder, loss_fn):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        assert encoder.inp_shape == decoder.outp_shape, print(
            'Encoder input shape and decoder output shape need to be equal!')
        assert encoder.latent_dim == decoder.latent_dim, print(
            'Encoder and decoder latent space dimension need to be equal!')
        self.inp_shape = encoder.inp_shape
        self.latent_dim = encoder.latent_dim

        self.loss_fn = loss_fn

        self.print_vae_info()

        self.init_metrics()

        # Initialize Output
        self.plotter = Plotter()
        self.train_summary_writer = None
        self.test_summary_writer = None

        self.progress = 'Progress - Runtime {:.2f} s:\n'
        self.progress += 'Epoch {}/{}, '
        self.progress += 'Loss: {:.2f}, '
        self.progress += 'Test Loss: {:.2f}, '
        self.progress += 'Time: {:.2f} s'

    def __call__(self, x, logits=True):
        z, mean, log_var = self.encode(x)
        x_out = self.decode(z, logits=logits)
        return x_out, mean, log_var

    def print_vae_info(self):
        print()
        print('++++ VARIATIONAL AUTOENCODER ++++')
        self.encoder.print_info()
        self.decoder.print_info()

    def print_fit_info(self, epochs, lr):
        print()
        print('++++ FITTING ++++')
        print()
        print('- Epochs: {}'.format(epochs))
        print('- Learning Rate: {}'.format(lr))
        print('- Optimizer: Adam')
        print()
        print('Start Training...')

    def encode(self, x):
        mean, log_var = self.encoder(x)
        z = Sampling()([mean, log_var])
        return z, mean, log_var

    def decode(self, z, logits=True):
        x_logit = self.decoder(z)
        if logits:
            return x_logit
        return tf.math.sigmoid(x_logit)

    def init_metrics(self):
        print()
        print("Initialize Metrics:")
        self.history = []
        self.metrics_ = {}
        self.metrics_['train_loss'] = tf.keras.metrics.Mean(
            'NELBO_train', dtype=tf.float32)
        self.metrics_['test_loss'] = tf.keras.metrics.Mean(
            'NELBO_test', dtype=tf.float32)
        for key in self.metrics_.keys():
            print(" - " + key)

    def reset_metrics(self):
        print()
        print("Reset Metrics")
        for key in self.metrics_.keys():
            self.metrics_[key].reset_states()

    @tf.function
    def train_step(self, x, optimizer):
        with tf.GradientTape() as tape:
            x_out, mean, log_var = self.__call__(x, logits=True)
            loss = self.loss_fn(x, x_out, mean, log_var)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.metrics_['train_loss'](loss)

    def fit(self, train_ds, test_ds, epochs=20, lr=1e-4, plot_losses=False):
        self.optimizer = tf.keras.optimizers.Adam(lr)

        self.print_fit_info(epochs, lr)

        elapsed_time = 0
        for epoch in range(1, epochs+1):
            start_time = time.time()
            for train_x in train_ds:
                self.train_step(train_x, self.optimizer)
            end_time = time.time()
            if self.train_summary_writer != None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_[
                                      'train_loss'].result(), step=epoch)
                    tf.summary.scalar('accuracy', self.metrics_[
                                      'train_accuracy'].result(), step=epoch)
            for test_x in test_ds:
                x_out, mean, log_var = self.__call__(test_x, logits=True)
                loss = self.loss_fn(test_x, x_out, mean, log_var)
                self.metrics_['test_loss'](loss)
            if self.test_summary_writer != None:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('loss', self.metrics_[
                                      'test_loss'].result(), step=epoch)

            display.clear_output(wait=False)
            elapsed_time += end_time - start_time
            print(self.progress.format(elapsed_time,
                                       epoch,
                                       epochs,
                                       self.metrics_['train_loss'].result(),
                                       self.metrics_['test_loss'].result(),
                                       end_time - start_time))
            self.history.append(
                [self.metrics_['train_loss'].result(), self.metrics_['test_loss'].result()])

            if epoch > 1 and plot_losses == True:
                self.plotter.plot_losses(self)

            self.reset_metrics()
        print()
        print('---- FINISHED ----')


if __name__ == "__main__":
    pass
