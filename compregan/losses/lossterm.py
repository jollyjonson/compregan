from abc import ABCMeta, abstractmethod
from typing import Tuple, Union

import tensorflow as tf

from compregan.gan.components import Components


class LossTerm(metaclass=ABCMeta):
    """
    Interface for implementing arbitrary loss terms such as distortion metrics, regularizers and the likes.
    The loss term while training and validating in the CompreGAN class will request the entities it needs to compute
    the loss term via the `needed_components` property. The training process will promptly call the class with the
    requested entities as arguments (in the order in which they were requested). The __call__ method implements the
    actual computation. The value (float or singleton tensor) it returns will subsequently be passed to the `apply`
    method in which the value should be applied to the generator and discriminator loss accordingly. E.g. you would
    apply a distortion term solely to the generator loss and probably use a scalar multiplier for weighting; this
    procedure would be done in the `apply` method like so: `generator_loss += some_multiplier * loss_value`.

    Note: Make sure you use the `tf.function` decorators where applicable when subclassing this interface, to allow for
    correct gradient flow!
    """

    @property
    @abstractmethod
    def needed_components(self) -> Tuple[Components, ...]:
        """
        Components needed to compute the loss term. These could e.g. be the original data and the reconstruction in case
        the loss term is a distortion metric or neural networks in the GAN framework to e.g. implement VGG-style losses.
        """

    @tf.function
    @abstractmethod
    def __call__(self, *components: Union[tf.Tensor, tf.keras.Model]) -> tf.Tensor:
        """
        Compute the actual value of the loss term given the needed components.
        """

    @tf.function
    @abstractmethod
    def apply(self, loss_value: tf.Tensor, generator_loss: tf.Tensor, discr_loss: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Method to apply the actual value of the loss to the corresponding loss functions.

        Parameters
        ----------
        loss_value: tf. Tensor
            The exact same alue computed before by __call__
        generator_loss: tf. Tensor
            The loss which will be backpropagated through the generator (as well as the encoder & quantizer)
        discr_loss: tf. Tensor
            The loss which will be backpropagated through the discriminator

        Returns
        -------
        generator_loss: tf. Tensor
            The loss which will be backpropagated through the generator (as well as the encoder & quantizer) after
            (possible) manipulation e.g. adding the loss term.
            Note that the losses MUST be returned to allow for correct gradient flow!
        discr_loss: tf. Tensor
            The loss which will be backpropagated through the discriminator after
            (possible) manipulation e.g. adding the loss term.
            Note that the losses MUST be returned to allow for correct gradient flow!
        """

    @staticmethod
    @abstractmethod
    def get_key() -> str:
        """
        Returns the name displayed along with the value computed by `__call__` in the history and training progressbars
        """
