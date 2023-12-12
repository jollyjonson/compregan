from typing import Tuple, Union, Callable

import tensorflow.keras as keras
import tensorflow as tf

from compregan.gan.components import Components
from .distortion import Distortion


class SemanticallyWeightedDistortion(Distortion):

    def __init__(self, multiplier: float, distortion: tf.keras.losses.Loss = keras.losses.MSE):
        self.multiplier = multiplier
        self._distortion = distortion

    @property
    def needed_components(self) -> Tuple[Components, ...]:
        return Components.OriginalCodecData, Components.ReconstructedCodecData, Components.Conditional

    @tf.function
    def __call__(self, original_data: tf.Tensor, reconstructed_data: tf.Tensor,
                 conditional: Union[tf.keras.Model, Callable]) -> tf.Tensor:
        semantic_map = conditional(original_data)
        distortion = self._distortion(original_data, reconstructed_data)
        weighted_distortion = distortion * semantic_map
        weighted_distortion_mean = tf.reduce_mean(weighted_distortion)
        return weighted_distortion_mean

    def get_key(self) -> str:
        return f"weighted_{self._distortion.__name__}"
