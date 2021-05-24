import tensorflow as tf
import tensorflowprobability as tfp

from function_tk import var_mixture, target_log_prob_fn, cost_Q


class MC_EM:
    """
    Inference of the unsupervised model parameters from a Monte Carlo Expecation Maximization sheme.
    """

    def __init__(self,
                 vae, X_sq,
                 Theta=None, K=10,
                 dims=8, num_results=10,
                 num_burning_steps=30, step_size=0.01,
                 num_leapfrog_steps=3):

        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.vae = vae

        self.X_sq = X_sq
        self.F, self.N = self.X_sq.shape
        self.K = K
        self.Z = self.encoder(self.X_sq)[0]

        # Initialize training parameters
        if Theta == None:
            self.W = tf.math.abs(tf.keras.initializers.RandomNormal(
                mean=0., stddev=1.)(shape=(self.F, self.K)))
            self.H = tf.math.abs(tf.keras.initializers.RandomNormal(
                mean=0., stddev=1.)(shape=(self.K, self.N)))
            self.g = tf.ones((1, self.N))
        else:
            self.W, self.H, self.g = Theta
        self.ones_T = tf.ones((X_sq.shape(0), 1))

        # Set Monte Carlo parameters
        self.dims = dims
        self.num_results = num_results
        self.num_burning_steps = num_burning_steps

        self.hmc = tfp.mcmc.MetropolisHastings(
            tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps))

    def single_E_step(self):
        """
        - Initialize encoded noisy signal or current Z-value
        - Return var_out
        """
        self.Z = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            num_burnin_steps=self.num_burning_steps,
            current_state=self.Z,
            kernel=self.hmc)

        return tf.math.exp(self.decoder(self.Z)[1])

    def single_M_step(self, var_out):
        """
        Perform single parameter update
        """
        def update_W(XV_sq_sum, V_sum):
            XVH = tf.matmul(XV_sq_sum, tf.transpose(self.H))
            VH = tf.matmul(V_sum, tf.transpose(self.H))
            return self.W * tf.math.sqrt(XVH/VH)

        def update_H(XV_sq_sum, V_sum):
            WXV = tf.matmul(tf.transpose(self.W), XV_sq_sum)
            WV = tf.matmul(tf.transpose(self.W), V_sum)
            return self.H * tf.math.sqrt(WXV/WV)

        def update_g(XVV, VV):
            return tf.transpose(self.g) * tf.math.sqrt(self.ones_T * XVV / (self.ones_T * VV))

        V_x = var_mixture(self.W, self.H, self.g, var_out)
        V_sum = tf.reduce_mean(V_x, axis=-1)
        XV_sq_sum = self.X_sq * tf.reduce_mean(V_x**2, axis=-1)
        XVV = self.X_sq * tf.reduce_mean(var_out / V_x**2, axis=-1)
        VV = tf.reduce_mean(var_out/V_x, axis=-1)

        self.H = update_H(XV_sq_sum, V_sum)
        self.W = update_W(XV_sq_sum, V_sum)
        self.g = update_g(XVV, VV)

    def run_MC_EM(self, max_iter=1000, tol=1e-4, costs=[]):
        """
        Run Monte Carlo Expectation Maximization
        """
        var_out = tf.math.exp(self.decoder(self.Z))
        costs.append(cost_Q(self.W, self.H, self.g,
                            var_out, self.X_sq))
        for n in range(max_iter):
            var_out = self.single_E_step()
            self.single_M_step(var_out)

            costs.append(
                cost_Q(cost_Q(self.W, self.H, self.g, var_out, self.X_sq)))

            if abs(costs[n]-costs[n+1]) < tol:
                print("MC_EM converged after {} steps!".format(n+1))
                break


if __name__ == "__main__":
    pass
