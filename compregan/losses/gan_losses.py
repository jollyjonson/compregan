import tensorflow as tf

from .gan_loss import GANLoss


class LeastSquares(GANLoss):
    @staticmethod
    @tf.function
    def f(z):
        return tf.square(z - 1.0)

    @staticmethod
    @tf.function
    def g(z):
        return tf.square(z)
