from function_tk import var_mixture, target_log_prob_fn, cost_Q
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class MC_EM:
    """
    Inference of the unsupervised model parameters from a Monte Carlo Expecation Maximization sheme.
    """

    def __init__(self,
                 vae,
                 train_ds,
                 Theta=None,
                 K=10,
                 eps_sq=0.01,
                 num_results=10,
                 num_burning_steps=30,
                 num_leapfrog_steps=3):

        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.vae = vae

        # Training data
        self.train_ds = train_ds.unbatch()

        self.N = self.train_ds.cardinality().numpy()
        self.K = K
        for x_sq in self.train_ds.take(1):
            self.F = x_sq.shape[0]

        self.X = np.zeros((self.N, self.F))
        self.Z = np.zeros((self.N, self.K))

        for n, x_sq in enumerate(train_ds):
            self.X[n, :] = x_sq
        for n, x_sq in enumerate(train_ds):
            self.Z[n, :] = self.encoder(x_sq)

        # Initialize training parameters
        if Theta == None:
            self.W = tf.math.abs(tf.keras.initializers.RandomNormal(
                mean=0., stddev=1.)(shape=(self.F, self.K)))
            self.H = tf.math.abs(tf.keras.initializers.RandomNormal(
                mean=0., stddev=1.)(shape=(self.K, self.N)))
            self.g = tf.ones((1, self.N))
        else:
            self.W, self.H, self.g = Theta
        self.ones_T = tf.ones((self.F, 1))

        # Initialize Metropolos-Hastings parameters
        self.eps_sq = eps_sq
        self.num_results = num_results
        self.num_burning_steps = num_burning_steps
        self.num_leapfrog_steps = num_leapfrog_steps

    def single_E_step(self, x_sq, num_burning_steps=None, num_results=None):
        """
        - Initialize with encoded noisy signal or current Z-value
        - Return chain of var_out
        """
        def target_log_prob_fn(x_sq, z):
            var_out = tf.math.exp(self.vae(z)[0])

            loc = tf.zeros([self.K])
            scale_pr = tf.eye(self.K)
            scale_lh = var_mixture(self.W, self.H, self.g, var_out)

            prior = tfd.MultivariateNormalDiag(
                loc=loc,
                scale_identity_multiplier=scale_pr)
            likelihood = tfd.MultivariateNormalDiag(
                loc=loc,
                scale_identity_multiplier=scale_lh)

            return prior.log_prob(z) + likelihood.log_prob(x_sq)

        if num_burning_steps == None:
            num_burning_steps = self.num_burning_steps
        if num_results == None:
            num_results = self.num_results

        hmc = tfp.mcmc.MetropolisHastings(
            tfp.mcmc.UncalibratedHamiltonianMonteCarlo(
                target_log_prob_fn=lambda z: target_log_prob_fn(x_sq, z),
                step_size=self.eps_sq,
                num_leapfrog_steps=self.num_leapfrog_steps))

        z_chain = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=self.Z,
            num_burnin_steps=num_burning_steps,
            kernel=hmc)
        self.Z = z_chain[-1]

        return tf.math.exp(self.decoder(z_chain)[1])

    def single_M_step(self, x_sq, var_out):
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
        V_sum = tf.reduce_mean(V_x, axis=0)
        XV_sq_sum = x_sq * tf.reduce_mean(V_x**2, axis=0)
        XVV = x_sq * tf.reduce_mean(var_out / V_x**2, axis=0)
        VV = tf.reduce_mean(var_out/V_x, axis=0)

        H = update_H(XV_sq_sum, V_sum)
        W = update_W(XV_sq_sum, V_sum)
        g = update_g(XVV, VV)

        return W, H, g

    def run_MC_EM(self, max_iter=1000, tol=1e-4):
        """
        Run Monte Carlo Expectation Maximization
        """
        # costs = []

        # for n in range(max_iter):
        #     var_out_r_ds = self.train_ds.map(self.single_E_step)
        #     self.W, self.H, self.g = self.single_M_step(var_out_r_ds)

        #     costs.append(cost_Q(self.W, self.H, self.g, var_out, x_sq))

        #     if n > 0 and abs(costs[n]-costs[n+1]) < tol:
        #         print("MC_EM converged after {} steps!".format(n+1))
        #         break

    def S_reconst(self, X_complex_ds):
        def s_hat(xfn):
            z_samples = self.single_E_step()
            var_out = tf.math.exp(self.encoder(z_samples))

            V_s = self.g * var_out
            V_x = var_mixture(self.W, self.H, self.g, var_out)

            return tf.reduce_sum(V_s/V_x, axis=0) * xfn

        return X_complex_ds.map(s_hat)


if __name__ == "__main__":
    pass
