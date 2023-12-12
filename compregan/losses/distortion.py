from abc import ABCMeta, abstractmethod
from typing import Tuple

import tensorflow as tf

from ..gan.components import Components
from .lossterm import LossTerm


class Distortion(LossTerm, metaclass=ABCMeta):
    """
    Interface for distortion loss terms.
    Assuming we need the original data as well as the reconstruction to
    compute a metric between the two.
    """

    multiplier: float = 1.0

    @property
    def needed_components(self) -> Tuple[Components, ...]:
        return Components.OriginalCodecData, Components.ReconstructedCodecData

    @tf.function
    @abstractmethod
    def __call__(
        self, data: tf.Tensor, reconstructed_data: tf.Tensor
    ) -> tf.Tensor:
        pass

    @tf.function
    def apply(
        self,
        loss_value: tf.Tensor,
        generator_loss: tf.Tensor,
        discr_loss: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        generator_loss += self.multiplier * loss_value
        return generator_loss, discr_loss
