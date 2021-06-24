from function_tk import var_mixture, cost_Q
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
                 dX,
                 Theta=None,
                 K=10,
                 eps_sq=0.01,
                 num_results=10,
                 num_burnin=30,
                 num_leapfrog_steps=3,
                 print_info=True):

        self.encoder = vae.encoder
        self.decoder = vae.decoder
        self.vae = vae

        self.dX = dX

        self.N, self.K, self.L, self.F = self._init_shapes(
            self.encoder, self.dX, K)

        self.Z = self._init_Z(self.encoder, self.dX, self.N, self.L)

        if Theta == None:
            self.W, self.H, self.g_T = self._init_Theta(self.N, self.K, self.F)
        else:
            print('Setting W, H, g_T!')
            self.W, self.H, self.g_T = Theta
            assert self.W.shape == (
                self.F, self.K), 'W: Wrong shape {}'.format(self.W.shape)
            assert self.H.shape == (
                self.K, self.N), 'H: Wrong shape {}'.format(self.H.shape)
            assert self.g_T.shape == (
                1, self.N), 'g_T: Wrong shape {}'.format(self.g_T.shape)
        self.ones_T = tf.ones((self.F, 1))

        self.eps_sq = eps_sq
        self.num_results = num_results
        self.num_burnin = num_burnin
        self.num_leapfrog_steps = num_leapfrog_steps

        if print_info:
            self.print_info()

    def _init_shapes(self, enc, dX, K):
        N = dX.cardinality().numpy()
        K = K
        L = enc.latent_dim
        for x_sq in dX.take(1):
            F = x_sq.shape[1]
        return N, K, L, F

    def _init_Z(self, enc, dX, N, L):
        Z = tf.Variable(tf.zeros(N, L))
        for n, x_sq in enumerate(dX):
            Z[n].assign(enc(x_sq)[0])
        return Z

    def _init_Theta(self, N, K, F):
        print('Initializing W, H, g_T!')
        W = tf.math.abs(tf.keras.initializers.RandomNormal(
            mean=0., stddev=1.)(shape=(F, K)))
        H = tf.math.abs(tf.keras.initializers.RandomNormal(
            mean=0., stddev=1.)(shape=(K, N)))
        g_T = tf.ones((1, N))
        return W, H, g_T

    def print_info(self):
        print()
        print('++++ MC-EM ++++')
        print(' - W (F,K): {}'.format(self.W.shape))
        print(' - H (K,N): {}'.format(self.H.shape))
        print(' - g (F,K): {}'.format(self.g_T.shape))
        print(' - eps^2: {}'.format(self.eps_sq))
        print(' - Num samples: {}'.format(self.num_results))
        print(' - Num burnin: {}'.format(self.num_burnin))
        print(' - Num leapfrog steps: {}'.format(self.num_leapfrog_steps))

    def single_E_step(self,
                      x_sq,
                      z,
                      num_burning_steps=None,
                      num_results=None):
        """
        - Initialize with encoded noisy signal or current Z-value
        - Return chain of var_out
        """
        def target_log_prob_fn(x_sq, z):
            var_out = tf.math.exp(self.decoder(z)[0])

            loc_pr = tf.zeros([self.L])
            scale_pr = tf.eye(self.L)
            prior = tfd.MultivariateNormalDiag(
                loc=loc_pr,
                scale_identity_multiplier=scale_pr)

            loc_lh = tf.zeros([self.F])
            scale_lh = var_mixture(self.W, self.H, self.g, var_out)
            likelihood = tfd.MultivariateNormalDiag(
                loc=loc_lh,
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
            num_results=self.num_results,
            current_state=z,
            num_burnin_steps=self.num_burnin,
            kernel=hmc)

        return z_chain

    def E_step(self):
        # Axes need to be checked!
        var_out_chain = tf.Variable(
            tf.zeros(self.N, self.F, self.num_results))
        for n, x in enumerate(self.dX):
            z_n_chain = self.single_E_step(x, self.Z[n])
            self.Z[n] = z_n_chain[-1]
            var_out_chain[n].assign(tf.math.exp(self.decoder(z_n_chain)[1]))
        return var_out_chain

    def single_M_step(self, x_sq, var_out_n_R):
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

        V_x_R = var_mixture(self.W, self.H, self.g, var_out_n_R)
        V_sum = tf.reduce_mean(V_x_R, axis=0)
        XV_sq_sum = x_sq * tf.reduce_mean(V_x_R**2, axis=0)
        XVV = x_sq * tf.reduce_mean(var_out_n_R / V_x_R**2, axis=0)
        VV = tf.reduce_mean(var_out_n_R/V_x_R, axis=0)

        H = update_H(XV_sq_sum, V_sum)
        W = update_W(XV_sq_sum, V_sum)
        g = update_g(XVV, VV)

        return W, H, g

    def M_step(self, x_sq, var_out_chain):
        # Needs to be checked:
        # + Summation in H update is missing
        # + Axis ordering var_out_chain
        # for x_sq, var_out_n_chain in zip(x_sq, var_out_chain):
        #     self.W, self.H, self.g_t = self.single_M_step(x_sq, var_out_n_chain)
        pass

    def run_MC_EM(self,
                  max_iter=1000,
                  tol=1e-4,
                  num_burnin=None,
                  num_results=None,
                  num_leapfrog_steps=None,
                  eps_sq=None):
        """
        Run Monte Carlo Expectation Maximization
        """

        if num_burnin == None:
            num_burnin = self.num_burnin
        if num_results == None:
            num_results = self.num_results
        if num_leapfrog_steps == None:
            num_leapfrog_steps = self.num_leapfrog_steps
        if eps_sq == None:
            eps_sq = self.eps_sq

        var_out_chain = self.E_step()
        self.W, self.H, self.g_T = self.M_step(var_out_chain)

        # costs = []

        # for n in range(max_iter):
        #     var_out_r_ds = self.train_ds.map(self.single_E_step)
        #     self.W, self.H, self.g = self.single_M_step(var_out_r_ds)

        #     costs.append(cost_Q(self.W, self.H, self.g, var_out, x_sq))

        #     if n > 0 and abs(costs[n]-costs[n+1]) < tol:
        #         print("MC_EM converged after {} steps!".format(n+1))
        #         break

    def S_reconst(self,
                  c_val_ds,
                  num_results=25,
                  num_burnin=75):
        """ Eq. (18)
        Speech reconstruction
        - Approximate via Metropolis-Hastings using the true posterior distribution.

        Input: Complex STFT X

        Output shape (F, N)
        """
        def s_hat(xn):
            z_samples = self.single_E_step(abs(xn)**2,
                                           num_results=num_results,
                                           num_burnin=num_burnin)
            var_out = tf.math.exp(self.encoder(z_samples))

            V_s = self.g * var_out
            V_x = var_mixture(self.W, self.H, self.g, var_out)

            filter = tf.reduce_sum(V_s/V_x, axis=0)
            filter = tf.complex(filter, tf.zeros_like(filter))

            return tf.reduce_sum(V_s/V_x, axis=0) * xn

        return c_val_ds.map(s_hat)


if __name__ == "__main__":
    pass
