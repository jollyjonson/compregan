import unittest
from copy import deepcopy
import keras.layers
import numpy as np
import tensorflow as tf

from compregan.losses.vgg_perceptual_loss import VGGPerceptualLoss


class VGGPerceptualLossTests(unittest.TestCase):
    _input_shape = (10, 10, 3)
    _batch_size = 1
    _multiplier = 10.0
    _layer_names = ("block1_conv2", "block2_conv1")

    def test_vgg_loss_works_as_intended(self):
        vgg_loss = VGGPerceptualLoss(
            self._multiplier, self._input_shape, self._layer_names
        )
        x, x_reconstructed = np.random.randn(
            self._batch_size, *self._input_shape
        ), np.random.randn(self._batch_size, *self._input_shape)
        self.assertGreater(vgg_loss(x, x_reconstructed), 0.0)
        self.assertEqual(vgg_loss(x, x), 0.0)

    def test_vgg_loss_raises_error_if_unkown_layer_requested(self):
        def _illegal_layer_pass():
            VGGPerceptualLoss(
                self._multiplier, self._input_shape, ("unknown_layer",)
            )

        self.assertRaises(AssertionError, _illegal_layer_pass)

    def test_vgg_model_does_not_change_when_backpropagating_losses_though_it(
        self,
    ):
        tf.config.run_functions_eagerly(True)

        model_to_be_optimized = tf.keras.Sequential(
            [
                keras.layers.Conv2D(3, 2, padding="same"),
                keras.layers.Conv2D(3, 2, padding="same"),
            ]
        )
        optimizer = tf.keras.optimizers.Adam()
        vgg_loss = VGGPerceptualLoss(
            10.0, self._input_shape, (self._layer_names[0],)
        )
        vgg_model_ut_idx, layer_ut_idx = 0, 0
        weights_before_backprop_vgg = deepcopy(
            vgg_loss._vgg_models[vgg_model_ut_idx].layers[layer_ut_idx].weights
        )

        input, target = tf.random.normal(
            (1, *self._input_shape)
        ), tf.random.normal((1, *self._input_shape))

        with tf.GradientTape(persistent=True) as tape:
            model_output = model_to_be_optimized(input)
            model_weights_before_backprop = deepcopy(
                model_to_be_optimized.weights
            )
            loss_value = vgg_loss(target, model_output)

        gradient = tape.gradient(
            loss_value, model_to_be_optimized.trainable_variables
        )
        optimizer.apply_gradients(
            zip(gradient, model_to_be_optimized.trainable_variables)
        )

        for weights_before_backprop, weights_after_backprop in zip(
            model_to_be_optimized.weights, model_weights_before_backprop
        ):
            self.assertFalse(
                np.allclose(
                    weights_before_backprop.numpy(),
                    weights_after_backprop.numpy(),
                )
            )

        for weights_actual, weights_desired in zip(
            vgg_loss._vgg_models[vgg_model_ut_idx]
            .layers[layer_ut_idx]
            .weights,
            weights_before_backprop_vgg,
        ):
            np.testing.assert_equal(
                weights_actual.numpy(), weights_desired.numpy()
            )


if __name__ == "__main__":
    unittest.main()
