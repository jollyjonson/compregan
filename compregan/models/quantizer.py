# TODO [quantizers] Double check implementation of quantization, Docs

from typing import Union, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_compression as tfcmp


class Quantizer(keras.Model):

    def __init__(self, input_shape: Union[int, Tuple[int, ...]], num_centers: int = 5, temperature: float = 1.):
        assert num_centers % 2 == 1, "num_centers must be odd to allow for even distribution on ints around 0"
        input_shape = (input_shape,) if type(input_shape) == int else input_shape
        inputs = keras.Input(shape=input_shape)
        centers = tf.cast(tf.range(-num_centers // 2 + 1, num_centers // 2 + 1), tf.float32)
        # Partition W into the Voronoi tesellation over the centers
        w_stack = tf.stack([inputs for _ in range(centers.shape[0])], axis=-1)
        w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)

        smx = tf.nn.softmax(-1.0 / temperature * tf.abs(w_stack - centers), axis=-1)
        # Contract last dimension
        w_soft = tf.tensordot(smx, centers, axes=(-1, 0))

        # Treat quantization as differentiable for optimization
        quantized_latent_representation = tfcmp.SoftRound()(tf.stop_gradient(w_hard - w_soft) + w_soft)
        super().__init__(inputs=inputs, outputs=quantized_latent_representation)


class BinaryQuantizer(keras.Model):

    def __init__(self, input_shape: Union[int, Tuple[int, ...]], temperature: float = 1.):
        input_shape = (input_shape,) if type(input_shape) == int else input_shape
        inputs = keras.Input(shape=input_shape)
        centers = tf.cast([0, 1], tf.float32)
        w_stack = tf.stack([inputs for _ in range(centers.shape[0])], axis=-1)
        w_hard = tf.cast(tf.argmin(tf.abs(w_stack - centers), axis=-1), tf.float32) + tf.reduce_min(centers)
        smx = tf.nn.softmax(-1.0 / temperature * tf.abs(w_stack - centers), axis=-1)
        w_soft = tf.tensordot(smx, centers, axes=(-1, 0))

        quantized_latent_representation = tfcmp.SoftRound()(tf.stop_gradient(w_hard - w_soft) + w_soft)

        super().__init__(inputs=inputs, outputs=quantized_latent_representation)
