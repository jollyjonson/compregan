from typing import Tuple, Union

import tensorflow as tf

from compregan.gan.components import Components
from .lossterm import LossTerm


class CombinedDepthEstimationLoss(LossTerm):
    """
    s. Keras Tutorial on the topic: https://keras.io/examples/vision/depth_estimation/
    """
    depth_key: str = 'depth'

    def __init__(self,
                 multiplier: float = 10.,
                 ssim_loss_weight: float = 0.46,
                 l1_loss_weight: float = 0.05,
                 depth_smoothness_loss_weight: float = 0.49,
                 target_interval: Tuple[float, float] = (-1., 1.),
                 ):
        self._multiplier = tf.convert_to_tensor(multiplier, dtype=tf.float32)
        self._ssim_loss_weight = tf.convert_to_tensor(ssim_loss_weight, dtype=tf.float32)
        self._l1_loss_weight = tf.convert_to_tensor(l1_loss_weight, dtype=tf.float32)
        self._depth_smoothness_loss_weight = tf.convert_to_tensor(depth_smoothness_loss_weight, dtype=tf.float32)
        self._target_interval = target_interval
        self._target_range = target_interval[1] - target_interval[0]

    @property
    def needed_components(self) -> Tuple[Components, ...]:
        return Components.AuxillaryOutput, Components.CompleteData

    @tf.function
    def __call__(self, *components: Union[tf.Tensor, tf.keras.Model]) -> tf.Tensor:
        predicted_depth, complete_data = components
        target_depth = complete_data[self.depth_key]

        depth_smoothness_loss_value = self._compute_depth_smoothness_loss(predicted_depth, target_depth)
        l1_loss = self._compute_l1_loss(predicted_depth, target_depth)
        ssim_loss = self._compute_ssim_loss(predicted_depth, target_depth)

        return (self._l1_loss_weight * l1_loss
                + self._ssim_loss_weight * ssim_loss
                + self._depth_smoothness_loss_weight * depth_smoothness_loss_value)

    @tf.function
    def _compute_depth_smoothness_loss(self, predicted_depth: tf.Tensor, target_depth: tf.Tensor) -> tf.Tensor:
        # edges
        dy_true, dx_true = tf.image.image_gradients(target_depth)
        dy_pred, dx_pred = tf.image.image_gradients(predicted_depth)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))
        # depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y
        return tf.reduce_mean(tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y)))

    @tf.function
    def _compute_ssim_loss(self, predicted_depth: tf.Tensor, target_depth: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(
            1 - tf.image.ssim(target_depth, predicted_depth,
                              max_val=self._target_range, filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2))

    @tf.function
    def _compute_l1_loss(self, predicted_depth: tf.Tensor, target_depth: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.abs(predicted_depth - target_depth))

    @tf.function
    def apply(self, loss_value: tf.Tensor, generator_loss: tf.Tensor, discr_loss: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        generator_loss += self._multiplier * loss_value
        return generator_loss, discr_loss

    @staticmethod
    def get_key() -> str:
        return "depth_estimation_loss"


class MAEDepthEstimation(CombinedDepthEstimationLoss):

    def __init__(self,
                 multiplier: float = 10.,
                 target_interval: Tuple[float, float] = (-1., 1.),
                 original_max_depth_in_m: float = 10.,
                 ):
        self._multiplier = multiplier
        self._target_interval = target_interval
        self._target_range = target_interval[1] - target_interval[0]
        self._original_max_depth_in_m = original_max_depth_in_m

    @tf.function
    def __call__(self, *components: Union[tf.Tensor, tf.keras.Model]) -> tf.Tensor:
        predicted_depth, complete_data = components
        target_depth = complete_data[self.depth_key]
        target_depth, predicted_depth = map(self._map_back_to_original_interval, [target_depth, predicted_depth])
        return self._compute_l1_loss(predicted_depth, target_depth)

    @tf.function
    def _map_back_to_original_interval(self, depth_in_target_interval: tf.Tensor) -> tf.Tensor:
        original_depth = (depth_in_target_interval / self._target_range) * self._original_max_depth_in_m
        return original_depth

    @staticmethod
    def get_key() -> str:
        return "depth_estimation_mae"


class RMSEDepthEstimation(MAEDepthEstimation):

    @tf.function
    def __call__(self, *components: Union[tf.Tensor, tf.keras.Model]) -> tf.Tensor:
        predicted_depth, complete_data = components
        target_depth = complete_data[self.depth_key]
        target_depth, predicted_depth = map(self._map_back_to_original_interval, [target_depth, predicted_depth])
        return tf.math.sqrt(tf.reduce_mean(tf.square(target_depth - predicted_depth)))

    @staticmethod
    def get_key() -> str:
        return "depth_estimation_rmse"
