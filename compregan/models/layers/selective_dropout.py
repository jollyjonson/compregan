from typing import Optional, Callable

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class SelectiveDropout(keras.layers.Layer):
    # TODO [SelectiveDropout]: Docs
    # TODO [SelectiveDropout]: Fade-in functionality

    _pmf: tf.Tensor
    _input_shape: tf.TensorShape

    def __init__(self,
                 ratio_of_never_dropped_elements: float = 0.1,
                 overall_probability_of_dropout: float = 0.75,
                 replacement_value: Callable = lambda input_shape: 0.,
                 only_drop_full_channels: bool = False,
                 ):
        """

        Parameters
        ----------
        ratio_of_never_dropped_elements: float = 0.1
        overall_probability_of_dropout: float = 0.75
        replacement_value: Callable = lambda input_shape: 0.
        only_drop_full_channels: bool = False
        """
        super().__init__(trainable=False, name=type(self).__name__, dtype=tf.float32, dynamic=False)

        assert 0. < ratio_of_never_dropped_elements <= 1.
        self._ratio_of_never_dropped_elements = ratio_of_never_dropped_elements
        assert 0. < overall_probability_of_dropout < 1.
        self._overall_probability_of_dropout = overall_probability_of_dropout

        self._replacement_value = replacement_value
        self._only_drop_full_channels = only_drop_full_channels

    @tf.function
    def call(self, inputs: tf.Tensor, *args, **kwargs) -> tf.Tensor:

        keep_mask = tf.cast(self._get_random_keep_mask_for_input(inputs), inputs.dtype)

        if self._replacement_value is None:
            def __dropped_out_inputs() -> tf.Tensor:
                return tf.multiply(inputs, keep_mask)
        else:
            def __dropped_out_inputs() -> tf.Tensor:
                replace_mask = tf.multiply(tf.ones_like(inputs) * self._replacement_value(tf.shape(inputs)),
                                           1. - keep_mask)
                return tf.add(tf.multiply(inputs, keep_mask),
                              replace_mask)

        return keras.backend.in_train_phase(x=__dropped_out_inputs(), alt=inputs)

    @tf.function
    def _get_random_keep_mask_for_input(self, inputs: tf.Tensor) -> tf.Tensor:
        keep_mask = tf.ones(tf.shape(inputs), dtype=tf.bool)
        num_batches = inputs.shape[0]
        if num_batches is not None:
            for batch_idx in range(num_batches):
                random_choice = tf.random.uniform((1,), dtype=tf.float32)
                per_batch_keep_mask = tf.less(random_choice, self._pmf)
                per_batch_keep_mask_shaped_like_input = tf.expand_dims(tf.reshape(per_batch_keep_mask,
                                                                                  inputs.shape[1:]), 0)
                keep_mask = tf.tensor_scatter_nd_update(keep_mask, [[batch_idx]], per_batch_keep_mask_shaped_like_input)
        return keep_mask

    def build(self, input_shape: tf.TensorShape) -> None:
        num_elements_per_batch = int(np.prod([*input_shape[1:]]))
        num_elements_never_dropped = int(round(self._ratio_of_never_dropped_elements
                                               * float(num_elements_per_batch)))

        if self._only_drop_full_channels:
            num_channels = input_shape[-1]
            assert num_channels > 1
            assert len(input_shape.as_list()) >= 2

            num_elements_per_channel = int(np.prod([*input_shape[1:-1]]))
            assert num_elements_per_channel * num_channels == num_elements_per_batch
            num_channels_to_always_keep = int(round(num_elements_never_dropped / num_elements_per_channel))
            assert 1 <= num_channels_to_always_keep < num_channels

            probability_of_keeping_per_channel = np.linspace(1.,
                                                             1. - self._overall_probability_of_dropout,
                                                             num_channels)
            pmf = np.zeros(input_shape[1:], dtype=np.float32)
            for channel_idx in range(num_channels):
                if channel_idx + 1 <= num_channels_to_always_keep:
                    pmf[:, :, channel_idx] = 1.
                else:
                    pmf[:, :, channel_idx] = probability_of_keeping_per_channel[channel_idx]

        else:
            pmf = np.linspace(1., 1. - self._overall_probability_of_dropout, num_elements_per_batch)
            pmf[:num_elements_never_dropped] = 1.

        self._pmf = tf.convert_to_tensor(pmf, dtype=tf.float32)
        self._input_shape = input_shape
