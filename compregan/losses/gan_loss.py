from abc import ABCMeta, abstractmethod

import tensorflow as tf


class GANLoss(metaclass=ABCMeta):
    @staticmethod
    @tf.function
    @abstractmethod
    def f(z):
        pass

    @staticmethod
    @tf.function
    @abstractmethod
    def g(z):
        pass
