import unittest

import numpy as np
import numpy.testing
import tensorflow as tf
from typing import Union
from abc import ABCMeta, abstractmethod

from compregan.gan.components import Components
from compregan.losses.distortions import MSE, Distortion, Lp, MSSSIM, PSNR


class DistortionTests(metaclass=ABCMeta):
    multiplier = 99.0
    image_shape = (16, 16, 3)
    image_shape_with_batch_dim = (2, 16, 16, 3)

    @abstractmethod
    def _get_distortion_instance(self) -> Distortion:
        ...

    @abstractmethod
    def _expected_computation(
        self, original: tf.Tensor, reconstruction: tf.Tensor
    ) -> Union[tf.Tensor, np.ndarray]:
        ...

    def setUp(self) -> None:
        self.distortion_instance = self._get_distortion_instance()

    def test_distortion_delivers_correct_results_scalars(self):
        originals, reconstructions = [1, 2, 3], [1, 4, -8]

        for original, reconstruction in zip(originals, reconstructions):
            original_tf = tf.ones(1, dtype=tf.float32) * original
            reconstructed_tf = tf.ones(1, dtype=tf.float32) * reconstruction
            distortion_value = self.distortion_instance(
                original_tf, reconstructed_tf
            )
            numpy.testing.assert_allclose(
                distortion_value,
                self._expected_computation(original, reconstruction),
            )

    def test_distortion_delivers_correct_results_image_shaped_data(self):
        for image_shape in [self.image_shape, self.image_shape_with_batch_dim]:
            original = tf.ones(image_shape, dtype=tf.float32) * 0.5
            reconstruction = tf.ones(image_shape, dtype=tf.float32) * 0.25

            distortion_value = self.distortion_instance(
                original, reconstruction
            )
            print(self.distortion_instance)
            numpy.testing.assert_allclose(
                distortion_value,
                self._expected_computation(original, reconstruction),
                atol=1e-8,
            )

    def test_distortion_delivers_correct_needed_components(self):
        self.assertEqual(
            (Components.OriginalCodecData, Components.ReconstructedCodecData),
            self.distortion_instance.needed_components,
        )

    def test_multiplier_works(self):
        gen_loss = tf.ones(1)
        disc_loss = tf.zeros(1)
        value = tf.ones(1) * 10.9
        (
            returned_gen_loss,
            returned_discr_loss,
        ) = self.distortion_instance.apply(value, gen_loss, disc_loss)
        self.assertEqual(disc_loss, returned_discr_loss)
        self.assertEqual(returned_gen_loss, gen_loss + self.multiplier * value)


class MSETests(DistortionTests, unittest.TestCase):
    def _get_distortion_instance(self):
        return MSE(self.multiplier)

    def _expected_computation(
        self, original: tf.Tensor, reconstruction: tf.Tensor
    ) -> Union[tf.Tensor, np.ndarray]:
        return np.mean((original - reconstruction) ** 2)


class LpTests(DistortionTests, unittest.TestCase):
    p = 5

    def _get_distortion_instance(self):
        return Lp(self.p, self.multiplier, compute_mean=False)

    def _expected_computation(
        self, original: tf.Tensor, reconstruction: tf.Tensor
    ) -> Union[tf.Tensor, np.ndarray]:
        return np.power(
            np.sum(np.power(np.abs((original - reconstruction)), self.p)),
            1.0 / self.p,
        )


class MSSSIMTests(DistortionTests, unittest.TestCase):
    image_shape = (256, 379, 3)
    image_shape_with_batch_dim = (2, 256, 379, 3)

    def test_distortion_delivers_correct_results_scalars(self):
        self.skipTest("computing MSSSIM is not possible for scalars")

    def _get_distortion_instance(self):
        return MSSSIM(multiplier=self.multiplier)

    def _expected_computation(
        self, original: tf.Tensor, reconstruction: tf.Tensor
    ) -> Union[tf.Tensor, np.ndarray]:
        return tf.image.ssim_multiscale(
            MSSSIM._map_to_0_1(original),
            MSSSIM._map_to_0_1(reconstruction),
            max_val=1.0,
        )


class PSNRTests(DistortionTests, unittest.TestCase):
    def test_distortion_delivers_correct_results_scalars(self):
        self.skipTest("computing PSNR is not possible for scalars")

    def _get_distortion_instance(self):
        return PSNR(multiplier=self.multiplier)

    def _expected_computation(
        self, original: tf.Tensor, reconstruction: tf.Tensor
    ) -> Union[tf.Tensor, np.ndarray]:
        return tf.image.psnr(
            MSSSIM._map_to_0_1(original),
            MSSSIM._map_to_0_1(reconstruction),
            max_val=1.0,
        )


if __name__ == "__main__":
    unittest.main()
