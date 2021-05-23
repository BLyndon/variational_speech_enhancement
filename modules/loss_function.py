import tensorflow as tf


def regularization(mean, log_var):
    return .5 * tf.math.reduce_sum(1 + log_var - tf.math.square(mean) - tf.math.exp(log_var), axis=1)


def itakura_saito_div(S_sq, log_var_out):
    ''' Clean speech Itakura-Saito divergence

    Parameters:
    S_sq (tf.data.Dataset): element-wise norm of complex STFT domain clean speech (real number)
    log_var_out (output VAE): VAE output (real number)
    '''
    var_out = tf.math.exp(log_var_out)
    return tf.math.reduce_sum(S_sq/var_out - tf.math.log(S_sq/var_out) - 1, axis=1)


def itakura_saito_elbo(S_sq, log_var_out, mean, log_var):
    ''' Clean speech loss

    Parameters:
    S_sq (tf.data.Dataset): element-wise norm of complex STFT domain clean speech (real numbers)
    log_var_out (output VAE): VAE output (real numbers)
    mean (output Encoder): Encoder mean (real numbers)
    log_var (output Encoder): Encoder log-variance (real numbers)
    '''
    return tf.math.reduce_mean(regularization(mean, log_var) - itakura_saito_div(S_sq, log_var_out))
