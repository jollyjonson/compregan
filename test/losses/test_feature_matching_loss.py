import unittest
from typing import Tuple

import numpy as np
import tensorflow as tf

from compregan.losses.feature_matching_loss import FeatureMatchingLoss

INPUT_SHAPE = (10, 10, 3)


def get_mock_discriminator():
    input_ = tf.keras.Input(shape=INPUT_SHAPE)
    x = tf.keras.layers.Conv2D(2, kernel_size=2, strides=2, padding="same")(
        input_
    )
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Conv2D(4, kernel_size=2, strides=2, padding="same")(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inputs=[input_], outputs=[x])


class FeatureMatchingLossTests(unittest.TestCase):
    mock_discriminator = get_mock_discriminator()
    random_input = np.random.randn(1, *INPUT_SHAPE)
    multiplier = 10.0
    relevant_layers = (tf.keras.layers.LeakyReLU,)
    fm_loss = FeatureMatchingLoss(
        mock_discriminator, 10.0, relevant_layer_types=relevant_layers
    )

    def test_feature_matching_loss_nonzero_for_unequal_inputs(self):
        self.assertGreater(
            self.fm_loss(self.random_input, self.random_input + 0.01), 0.0
        )

    def test_feature_matching_loss_zero_for_equal_inputs(self):
        self.assertEqual(
            self.fm_loss(self.random_input, self.random_input), 0.0
        )

    def test_apply_function_works(self):
        gen_loss, discr_loss, term_value = 1.0, 0.0, 99.0
        post_apply_gen_loss, post_apply_disc_loss = self.fm_loss.apply(
            term_value, gen_loss, discr_loss
        )
        self.assertEqual(
            post_apply_gen_loss, gen_loss + self.multiplier * term_value
        )
        self.assertEqual(discr_loss, post_apply_disc_loss)

    def test_decay_works(self):
        decay_factor = 1e-2
        fm_loss = FeatureMatchingLoss(
            self.mock_discriminator,
            10.0,
            relevant_layer_types=self.relevant_layers,
            decay_factor=decay_factor,
        )
        loss_value = tf.ones(1) * 0.5

        for step_idx in range(10):
            gen_loss, _ = fm_loss.apply(
                loss_value, generator_loss=tf.zeros(1), discr_loss=tf.zeros(1)
            )
            if step_idx > 0:
                tf.assert_less(gen_loss, previous_gen_loss)
            previous_gen_loss = gen_loss

        fm_loss._num_steps_loss_has_been_applied = 1e10
        np.testing.assert_allclose(
            fm_loss.apply(tf.ones(1), tf.zeros(1), tf.zeros(1))[0].numpy(), 0.0
        )

    def test_weights_change_in_discr_instance_held_by_fm_loss(self):
        def __alter_weights(
            weights: Tuple[tf.Tensor, tf.Tensor]
        ) -> Tuple[tf.Tensor, tf.Tensor]:
            return tuple([weight_array + 1.0 for weight_array in weights])

        layer_ut_idx = -3
        self.mock_discriminator.layers[layer_ut_idx].set_weights(
            __alter_weights(
                self.mock_discriminator.layers[layer_ut_idx].weights
            )
        )

        discr_instance_from_fm_loss = self.fm_loss._discriminator

        for weights_actual, weights_desired in zip(
            discr_instance_from_fm_loss.layers[layer_ut_idx].weights,
            self.mock_discriminator.layers[layer_ut_idx].weights,
        ):
            np.testing.assert_equal(
                weights_actual.numpy(), weights_desired.numpy()
            )


if __name__ == "__main__":
    unittest.main()
