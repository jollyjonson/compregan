import unittest

import numpy as np
import tensorflow as tf

from compregan.models.layers.selective_dropout import SelectiveDropout
from compregan.util import AllRandomSeedsSetter


class SelectiveDropoutTests(unittest.TestCase):
    def setUp(self) -> None:
        AllRandomSeedsSetter(42)

    def test_replacement_values_work(self):
        num_passes_through_selective_dropout = 50
        input_shape = (10,)
        num_elements = np.prod(input_shape)
        ones_input = tf.ones(input_shape, dtype=tf.float32)
        ratio_of_never_dropped_elements = 0.4
        first_possibly_dropped_idx = int(
            round(ratio_of_never_dropped_elements * num_elements)
        )
        overall_probability_of_dropout = 0.8

        minval_uniform, maxval_uniform = -10.0, -5.0

        for replacement_value_idx, replacement_value in enumerate(
            [
                lambda input_shape: -1.0,
                lambda input_shape: tf.random.uniform(
                    input_shape, minval_uniform, maxval_uniform
                ),
            ]
        ):
            model = tf.keras.Sequential(
                [
                    SelectiveDropout(
                        ratio_of_never_dropped_elements=ratio_of_never_dropped_elements,
                        overall_probability_of_dropout=overall_probability_of_dropout,
                        only_drop_full_channels=False,
                        replacement_value=replacement_value,
                    )
                ]
            )

            for _ in range(num_passes_through_selective_dropout):
                selective_dropout_output = tf.squeeze(
                    model(tf.expand_dims(ones_input, 0), training=True)
                ).numpy()

                for value_idx, value in enumerate(selective_dropout_output):
                    if value_idx < first_possibly_dropped_idx:
                        self.assertEqual(value, 1.0)
                    else:
                        value_dropped = value != 1.0
                        if value_dropped:
                            if replacement_value_idx == 0:
                                self.assertEqual(value, -1.0)
                            elif replacement_value_idx == 1:
                                self.assertGreater(value, minval_uniform)
                                self.assertLess(value, maxval_uniform)

    def test_last_indices_are_dropped_with_increasing_probability_and_first_indices_are_never_dropped_simple_shape(
        self,
    ):
        num_passes_through_selective_dropout = 50
        input_shape = (10,)
        ones_input = tf.ones(input_shape, dtype=tf.float32)
        ratio_of_never_dropped_elements = 0.2
        overall_probability_of_dropout = 0.8
        model = tf.keras.Sequential(
            [
                SelectiveDropout(
                    ratio_of_never_dropped_elements=ratio_of_never_dropped_elements,
                    overall_probability_of_dropout=overall_probability_of_dropout,
                )
            ]
        )

        cumulated_output = tf.zeros_like(ones_input)

        for _ in range(num_passes_through_selective_dropout):
            cumulated_output += tf.squeeze(
                model(tf.expand_dims(ones_input, 0), training=True)
            )

        first_possibly_dropped_idx = int(
            round(ratio_of_never_dropped_elements * input_shape[0])
        )
        last_idx_value = 0
        for idx in range(first_possibly_dropped_idx):
            last_idx_value = cumulated_output.numpy()[idx]
            self.assertEqual(
                last_idx_value, num_passes_through_selective_dropout
            )

        for idx in range(first_possibly_dropped_idx, input_shape[0]):
            this_idx_value = cumulated_output.numpy()[idx]
            self.assertGreater(last_idx_value, this_idx_value)
            last_idx_value = this_idx_value

    def test_last_indices_are_dropped_with_increasing_probability_and_first_indices_are_never_dropped_complex_shape(
        self,
    ):
        num_passes_through_selective_dropout = 50
        num_channels = 3
        input_shape = (3, 3, num_channels)
        num_elements = np.prod(input_shape)
        num_elements_per_channel = np.prod(input_shape[:-1])
        ones_input = tf.ones(input_shape, dtype=tf.float32)
        ratio_of_never_dropped_elements = 0.4
        first_possibly_dropped_idx = int(
            round(ratio_of_never_dropped_elements * num_elements)
        )
        overall_probability_of_dropout = 0.8

        for replacement_value_idx, replacement_value in enumerate(
            [
                lambda input_shape: 0.0,
                lambda input_shape: -1.0,
                lambda input_shape: tf.random.uniform(
                    input_shape, -10.0, -1.0
                ),
            ]
        ):
            for only_drop_full_channels in [True, False]:
                model = tf.keras.Sequential(
                    [
                        SelectiveDropout(
                            ratio_of_never_dropped_elements=ratio_of_never_dropped_elements,
                            overall_probability_of_dropout=overall_probability_of_dropout,
                            only_drop_full_channels=only_drop_full_channels,
                            replacement_value=replacement_value,
                        )
                    ]
                )

                cumulated_output = tf.zeros_like(ones_input)

                for _ in range(num_passes_through_selective_dropout):
                    cumulated_output += tf.squeeze(
                        model(tf.expand_dims(ones_input, 0), training=True)
                    )

                if only_drop_full_channels:
                    num_channels_to_always_keep = int(
                        round(
                            first_possibly_dropped_idx
                            / num_elements_per_channel
                        )
                    )

                    if replacement_value_idx != 2:
                        for channel_idx in range(num_channels):
                            all_values_per_channel_equal = (
                                cumulated_output[:, :, channel_idx].numpy()
                                == cumulated_output[0, :, channel_idx].numpy()
                            ).all()
                            self.assertTrue(all_values_per_channel_equal)

                    for channel_idx in range(num_channels_to_always_keep):
                        channels_to_always_keep_are_always_kept = (
                            cumulated_output[:, :, channel_idx].numpy()
                            == num_passes_through_selective_dropout
                        ).all()
                        self.assertTrue(
                            channels_to_always_keep_are_always_kept
                        )

                    for iter_idx, channel_idx in enumerate(
                        range(num_channels_to_always_keep, num_channels)
                    ):
                        if iter_idx > 0:
                            this_channel_values = cumulated_output[
                                :, :, channel_idx
                            ]
                            this_channels_had_more_dropout_than_last_channel = (
                                last_channel_values.numpy()
                                > this_channel_values.numpy()
                            ).all()
                            self.assertTrue(
                                this_channels_had_more_dropout_than_last_channel
                            )
                        last_channel_values = cumulated_output[
                            :, :, channel_idx
                        ]

                else:
                    last_idx_value = 0
                    for idx in range(first_possibly_dropped_idx):
                        last_idx_value = cumulated_output.numpy().flatten()[
                            idx
                        ]
                        self.assertEqual(
                            last_idx_value,
                            num_passes_through_selective_dropout,
                        )

                    for idx in range(
                        first_possibly_dropped_idx, input_shape[0]
                    ):
                        this_idx_value = cumulated_output.numpy().flatten()[
                            idx
                        ]
                        self.assertGreater(last_idx_value, this_idx_value)
                        last_idx_value = this_idx_value

    def test_input_shape_is_set_correctly_by_calling_the_layer_first_time(
        self,
    ):
        selective_dropout_instance = SelectiveDropout()
        shape = (2, 3, 3, 3)
        random_input = tf.convert_to_tensor(
            np.random.randn(*shape), dtype=tf.float32
        )

        # calling the layers __call__ method should call its build method, setting what we need
        selective_dropout_instance(random_input)

        self.assertEqual(shape, selective_dropout_instance._input_shape)

    def test_never_dropping_values_while_inference_and_dropping_values_while_training_randomly(
        self,
    ):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(4, use_bias=False, input_shape=(4,)),
                SelectiveDropout(overall_probability_of_dropout=0.99),
                tf.keras.layers.Dense(1, use_bias=False),
            ]
        )

        num_batches = 3
        random_data = np.random.randn(3, 4).astype(np.float32)
        model.compile(loss=tf.keras.losses.MSE)
        num_trials_per_item = 5

        for item_idx in range(num_batches):
            item = random_data[item_idx]
            first_result = model(np.expand_dims(item, 0), training=True)
            equals_first_result = list()
            for _ in range(num_trials_per_item):
                equals_first_result.append(
                    (
                        first_result
                        == model(np.expand_dims(item, 0), training=True)
                    ).numpy()
                )
            self.assertFalse(np.array(equals_first_result).all())

        for item_idx in range(num_batches):
            item = random_data[item_idx]
            expected_result = model(np.expand_dims(item, 0)).numpy()
            for _ in range(num_trials_per_item):
                self.assertEqual(
                    expected_result, model(np.expand_dims(item, 0)).numpy()
                )


if __name__ == "__main__":
    unittest.main()
