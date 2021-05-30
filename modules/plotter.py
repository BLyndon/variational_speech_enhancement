import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


class Plotter:

    def plot_losses(self, model, figsize=(6, 6)):
        '''Plots the recorded losses and metrics.

        Inputs
        model (VAE): trained model with recorded history
        figsize (tuple): size of plot
        '''
        fig, axs = plt.subplots(1, 1, figsize=figsize, sharey=False)

        fig.suptitle(
            'Clean speech VAE')
        fig.tight_layout(pad=3.0)

        axs.set_xlabel('Epochs')

        axs.set_ylabel('ELBO')

        train_loss = np.array(model.history)[:, 0]
        test_loss = np.array(model.history)[:, 1]

        loss_max_x = max(np.max(train_loss), np.max(test_loss))+5
        loss_min_x = min(np.min(train_loss), np.min(test_loss))-5

        axs.set_ylim(loss_min_x, loss_max_x)

        axs.plot(train_loss, label='train data')
        axs.plot(test_loss, label='test data')

        axs.legend()

        plt.show()
