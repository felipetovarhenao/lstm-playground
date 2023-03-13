import tensorflow as tf


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    """ Custom loss function to enforce positive values for note delta and duration """
    mse = (y_true - y_pred) ** 2

    # penalize negative values for y_pred, which get multiplied by 10, significantly increasing the error
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)
