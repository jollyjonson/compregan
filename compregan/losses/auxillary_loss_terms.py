from typing import List, Tuple

import tensorflow as tf
from typeguard import typechecked

from compregan.gan.components import Components
from compregan.losses.lossterm import LossTerm


class AuxillaryLossTerms(list):
    """
    Wrapper for Auxillary loss terms.

    Parameters
    ----------
    loss_terms: List[List[LossTerm]]
        List of lists of instances of child classes of `LossTerm`. Each list of loss terms will be applied in the
        order that the auxillary output appears at the decoding_generator during `CompressionGAN.train`.
    """

    @typechecked
    def __init__(self, loss_terms: List[List[LossTerm]]):
        super().__init__(loss_terms)


class AuxLoss(LossTerm):
    _class_label_key: str

    def __init__(self, class_label_key: str = 'label'):
        self._class_label_key = class_label_key

    @property
    def needed_components(self) -> Tuple[Components, ...]:
        return Components.AuxillaryOutput, Components.CompleteData


class CategorialCrossEntropy(AuxLoss):

    @tf.function
    def __call__(self, prediction: tf.Tensor, complete_train_data: tf.data.Dataset) -> tf.Tensor:
        ground_truth = complete_train_data[self._class_label_key]
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(ground_truth, prediction))

    @tf.function
    def apply(self, loss_value: tf.Tensor, generator_loss: tf.Tensor, discr_loss: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        generator_loss += loss_value
        return generator_loss, discr_loss

    @staticmethod
    def get_key() -> str:
        return 'categorial_crossentropy'


class CategorialAccuracy(AuxLoss):

    @tf.function
    def __call__(self, prediction: tf.Tensor, complete_train_data: tf.data.Dataset) -> tf.Tensor:
        ground_truth = complete_train_data[self._class_label_key]
        return tf.reduce_mean(tf.keras.metrics.categorical_accuracy(ground_truth, prediction))

    @tf.function
    def apply(self, loss_value: tf.Tensor, generator_loss: tf.Tensor, discr_loss: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        raise ValueError("Accuracy is not differentiable and can only be used as a metric")

    @staticmethod
    def get_key() -> str:
        return 'classification_accuracy'
