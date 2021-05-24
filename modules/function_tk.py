from tf import reduce_mean, matmul
from tf.math import log


def target_log_prob_fn():
    """ c.f. Eq. (10)
    Unnormalized log-probability function
    """
    pass


def var_mixture(W, H, g, var_out):
    """ c.f. Eq. (7)
    Variance of the mixture model eq. (6)
    """
    return g * var_out + matmul(W, H)


def s_hat(W, H, g, var_out, X):
    """ Eq. (18)
    Speech reconstruction
    - Approximate via Metropolis-Hastings using the true posterior distribution.
    """
    gvar = g * var_out
    return reduce_mean((gvar/(gvar + matmul(W, H))), axis=-1) * X


def cost_Q(W, H, g, var_out, X_sq):
    """ Eq. (8)
    Used as cost in MC_EM
    Compute expectation - see section 4. INFERENCE E-step
    """
    V_x = var_mixture(W, H, g, var_out)
    return reduce_mean(log(V_x) + X_sq/V_x, axis=-1)


if __name__ == "__main__":
    pass
