import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Plotter:

    def plot_losses(self, model, figsize=(12, 6)):
        '''Plots the recorded losses and metrics.

        Inputs
        model (VAE): trained model with recorded history
        figsize (tuple): size of plot
        '''
        fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=False)

        fig.suptitle(
            'Encoder: {} -- Decoder: {}'.format(model.encoder.net, model.decoder.net))
        fig.tight_layout(pad=3.0)

        axs[0].set_xlabel('Epochs')
        axs[1].set_xlabel('Epochs')

        axs[0].set_ylabel('ELBO')
        axs[1].set_ylabel('Accuracy')

        train_loss = np.array(model.history)[:, 0]
        test_loss = np.array(model.history)[:, 2]
        train_acc = np.array(model.history)[:, 1]
        test_acc = np.array(model.history)[:, 3]

        loss_max_x = max(np.max(train_loss), np.max(test_loss))+5
        loss_min_x = min(np.min(train_loss), np.min(test_loss))-5
        acc_max_x = max(np.max(train_acc), np.max(test_acc))+1
        acc_min_x = min(np.min(train_acc), np.min(test_acc))-1

        axs[0].set_ylim(loss_min_x, loss_max_x)
        axs[1].set_ylim(acc_min_x, acc_max_x)

        axs[0].plot(train_loss, label='train data')
        axs[0].plot(test_loss, label='test data')
        axs[1].plot(train_acc, label='train data')
        axs[1].plot(test_acc, label='test data')

        axs[0].legend()
        axs[1].legend()

        plt.show()
