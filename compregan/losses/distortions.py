import tensorflow as tf

from .distortion import Distortion


class MSE(Distortion):

    def __init__(self, mse_multiplier: float):
        self.multiplier = mse_multiplier

    @tf.function
    def __call__(self, data: tf.Tensor, reconstructed_data: tf.Tensor) -> tf.Tensor:
        distortion_term = tf.reduce_mean(tf.keras.losses.MSE(data, reconstructed_data))
        return distortion_term

    def get_key(self) -> str:
        return "mse"


class Lp(Distortion):

    def __init__(self, p: float, multiplier: float, compute_mean: bool = True):
        self._p = tf.cast(p, tf.float32)
        self._display_p = p  # needed for printing: Not having this in a tensor
        self.multiplier = multiplier
        self._compute_mean = compute_mean

    @tf.function
    def __call__(self, data: tf.Tensor, reconstructed_data: tf.Tensor) -> tf.Tensor:
        lp_distortion = tf.pow(tf.reduce_sum(tf.pow(tf.cast(tf.abs(data - reconstructed_data), tf.float32),
                                                    self._p)),
                               1. / self._p)
        if self._compute_mean:
            num_elements = tf.cast(tf.reduce_prod(data.shape), tf.float32)
            lp_distortion /= num_elements
        return lp_distortion

    def get_key(self) -> str:
        return f"l{int(self._display_p) if self._display_p % 1 == 0 else self._display_p}"


class MSSSIM(Distortion):

    def __init__(self, multiplier: float = 1.):
        self.multiplier = multiplier

    @staticmethod
    @tf.function
    def _map_to_0_1(image_in_minus_one_one: tf.Tensor) -> tf.Tensor:
        return (image_in_minus_one_one + 1.) / 2.

    @tf.function
    def __call__(self, data: tf.Tensor, reconstructed_data: tf.Tensor) -> tf.Tensor:
        data, reconstructed_data = self._map_to_0_1(data), self._map_to_0_1(reconstructed_data)
        return tf.image.ssim_multiscale(data, reconstructed_data, max_val=1.)

    def get_key(self) -> str:
        return "msssim"


class PSNR(MSSSIM):

    @tf.function
    def __call__(self, data: tf.Tensor, reconstructed_data: tf.Tensor) -> tf.Tensor:
        data, reconstructed_data = self._map_to_0_1(data), self._map_to_0_1(reconstructed_data)
        return tf.image.psnr(data, reconstructed_data, max_val=1.)

    def get_key(self) -> str:
        return "psnr"
