import tensorflow as tf


def target_log_prob_fn():
    """ c.f. Eq. (10)
    Unnormalized log-probability function
    """
    pass


def var_mixture(W, H, g, var_out):
    """ c.f. Eq. (7)
    Variance of the mixture model eq. (6)

    Output shape (R, F, N)
    """
    return g * var_out + W @ H


def s_hat(W, H, g, var_out, X):
    """ Eq. (18)
    Speech reconstruction
    - Approximate via Metropolis-Hastings using the true posterior distribution.

    Output shape (F, N)
    """
    g_blown = tf.repeat(tf.expand_dims(
        tf.repeat(g, var_out.shape[1], axis=0), axis=0), var_out.shape[0], axis=0)

    gvar = g_blown * var_out
    mean = tf.reduce_mean((gvar/(gvar + W @ H)), axis=0)
    c_mean = tf.complex(mean, tf.zeros(mean.shape))

    return c_mean*X


def cost_Q(W, H, g, var_out, X_sq):
    """ Eq. (8)
    Used as cost in MC_EM
    Compute expectation - see section 4. INFERENCE E-step
    """
    V_x = var_mixture(W, H, g, var_out)
    return tf.reduce_sum(tf.reduce_mean(tf.math.log(V_x) + X_sq/V_x, axis=0))


if __name__ == "__main__":
    pass
