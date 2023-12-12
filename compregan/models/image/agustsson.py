from typing import Union, List, Optional, Tuple, Callable

import tensorflow as tf
import tensorflow_addons as tfa


class AgustssonImageModelsBuilder:

    def __init__(self,
                 width: int = 768,
                 height: int = 512,
                 num_channels: int = 3,
                 use_sc_scheme: bool = False,
                 multiply_latent_space_with_binary_heatmap: bool = False,
                 binary_heatmap_channel: Optional[int] = None,
                 num_semantic_classes: int = 1,
                 num_channels_latent_space: Optional[int] = 8,
                 num_noise_channels: int = 0,
                 use_patch_gan: bool = False,
                 initializer: tf.keras.initializers.Initializer = tf.keras.initializers.glorot_normal,
                 normalization: tf.keras.layers.LayerNormalization = tfa.layers.InstanceNormalization,
                 activation: tf.keras.layers.Activation = tf.keras.layers.LeakyReLU,
                 num_filters_per_layer_encoder: Tuple[int, ...] = (60, 120, 240, 480, 960),
                 num_filters_per_layer_decoder: Tuple[int, ...] = (480, 240, 120, 60),
                 num_residual_blocks_decoder: int = 9,
                 num_filters_in_residual_blocks: int = 960,
                 num_filter_per_layer_discriminator: Tuple[int, ...] = (64, 128, 256, 512),
                 decoding_generator_output_activation: Callable = tf.nn.tanh,
                 intermediate_conv_layer_kwargs: Optional[Tuple[dict, ...]] = None
                 ):

        self._width = width
        self._height = height
        self._num_channels = num_channels

        self._use_sc_scheme = use_sc_scheme
        assert not multiply_latent_space_with_binary_heatmap if not self._use_sc_scheme else True
        self._multiply_latent_space_with_binary_heatmap = multiply_latent_space_with_binary_heatmap
        assert binary_heatmap_channel is not None if self._multiply_latent_space_with_binary_heatmap else True

        self._num_semantic_classes = num_semantic_classes

        self._use_patch_gan = use_patch_gan

        self._initializer = initializer
        self._normalization = normalization
        if normalization == tf.keras.layers.BatchNormalization:
            self._normalization_kwargs = {'center': True, 'scale': True, 'training': True, 'fused': True,
                                          'renorm': False}
        elif normalization == tfa.layers.InstanceNormalization:
            self._normalization_kwargs = {'center': True, 'scale': True}
        else:
            raise NotImplementedError(f"No kwargs found for normalization type '{type(normalization)}'")

        self._intermediate_conv_layer_kwargs = intermediate_conv_layer_kwargs

        self._activation = activation

        self._num_channels_latent_space = num_channels_latent_space

        num_filters_per_layer_encoder_default = (60, 120, 240, 480, 960) if not use_sc_scheme else (60, 120, 240, 480)
        self._num_filters_per_layer_encoder = num_filters_per_layer_encoder \
            if num_filters_per_layer_encoder is not None else num_filters_per_layer_encoder_default
        self._num_filters_per_layer_decoder = num_filters_per_layer_decoder
        self._num_filters_per_layer_discriminator = num_filter_per_layer_discriminator
        assert num_residual_blocks_decoder >= 1
        self._num_residual_blocks = num_residual_blocks_decoder
        self._num_filters_in_residual_blocks = num_filters_in_residual_blocks

        self._num_noise_channels = num_noise_channels

        self._decoding_generator_output_activation = decoding_generator_output_activation

        self._encoder = self._build_encoder()
        self._decoding_generator = self._build_decoding_generator()
        self._discriminator = self._build_discriminator()

    @property
    def encoder(self) -> tf.keras.Model:
        return self._encoder

    @property
    def decoding_generator(self) -> tf.keras.Model:
        return self._decoding_generator

    @property
    def discriminator(self) -> tf.keras.Model:
        return self._discriminator

    def _build_encoder(self) -> tf.keras.Model:
        """
        GC: c7s1-60, d120, d240, d480, d960, c3s1-C, q
        SC: c7s1-60, d120, d240, d480, c3s1-C, q, (c3s1-480, d960) - layers in parenthesis are implemented in the decoder
        """

        image_input = tf.keras.Input(shape=(self._width, self._height, self._num_channels))
        image_input_f32 = tf.cast(image_input, tf.float32)

        if self._use_sc_scheme:
            semantic_map_input = tf.keras.Input(shape=(self._width, self._height, self._num_semantic_classes))
            semantic_map_input_f32 = tf.cast(semantic_map_input, tf.float32)
            encoder_input = tf.concat([image_input_f32, semantic_map_input_f32], axis=-1)
            inputs = [image_input, semantic_map_input]
        else:
            encoder_input = image_input_f32
            inputs = [image_input]

        x = tf.pad(encoder_input, [[0, 0], [3, 3], [3, 3], [0, 0]], 'reflect')
        x = self._conv_block(x, num_filters=self._num_filters_per_layer_encoder[0], kernel_size=7, strides=1,
                             padding='valid')

        for num_filters in self._num_filters_per_layer_encoder[1:]:
            x = self._conv_block(x, num_filters=num_filters, kernel_size=3, strides=2)

        # Project channels onto space w/ dimension channels_latent_space
        # Feature maps have dimension ceil(W/16) x ceil(H/16) x channels_latent_space
        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'reflect')

        activation_temp = self._activation
        self._activation = lambda: lambda x: x
        feature_map = self._conv_block(x, num_filters=self._num_channels_latent_space, kernel_size=3, strides=1,
                                       padding='valid')
        self._activation = activation_temp

        return tf.keras.Model(inputs=inputs, outputs=[feature_map])

    def _build_semantic_map_encoder_layers(self, model_input: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        """c7s1-60, d120, d240, d480, d960"""
        x = tf.cast(model_input, tf.float32)
        x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'reflect')
        x = self._conv_block(x, num_filters=60, kernel_size=7, strides=1,
                             padding='valid')

        for num_filters in [120, 240, 480, 960]:
            x = self._conv_block(x, num_filters=num_filters, kernel_size=3, strides=2)
        return x

    def _build_decoding_generator(self) -> tf.keras.Model:
        """
         (c3s1-480, d960)

        Returns
        -------

        """
        output_shape_encoder = self.encoder.output_shape[1:]
        shape_latent_repr_with_noise = (*output_shape_encoder[:-1], output_shape_encoder[-1] + self._num_noise_channels)
        quantized_latent_representation_and_noise = tf.keras.layers.Input(shape=shape_latent_repr_with_noise)
        inputs = [quantized_latent_representation_and_noise]

        if self._use_sc_scheme:
            intermediate_latents = self._conv_block(quantized_latent_representation_and_noise, 480, 3, 1)
            intermediate_latents = self._conv_block(intermediate_latents, 960)

            # TODO [AgustssonModelBuilder]: SC model - add pointwise binary heatmap multiplication

            semantic_label_map_input = tf.keras.layers.Input(shape=(self._width, self._height,
                                                                    self._num_semantic_classes))
            inputs.append(semantic_label_map_input)
            semantic_label_encoder_output = self._build_semantic_map_encoder_layers(semantic_label_map_input)
            generator_input = tf.concat([intermediate_latents, semantic_label_encoder_output],
                                        axis=-1)
        else:
            generator_input = quantized_latent_representation_and_noise
            if self._intermediate_conv_layer_kwargs is not None:
                for conv_layer_kwargs in self._intermediate_conv_layer_kwargs:
                    generator_input = self._conv_block(generator_input, **conv_layer_kwargs)

        w_bar = tf.pad(generator_input, [[0, 0], [1, 1], [1, 1], [0, 0]], 'reflect')

        upsampled = self._conv_block(w_bar, num_filters=self._num_filters_in_residual_blocks, kernel_size=3, strides=1,
                                     padding='valid')

        # Process upsampled feature map with residual blocks
        res = self._residual_block(upsampled, self._num_filters_in_residual_blocks)
        for _ in range(self._num_residual_blocks - 1):
            res = self._residual_block(res, self._num_filters_in_residual_blocks)

        # Upsample to original dimensions - mirror decoder
        ups = res
        for num_filters in self._num_filters_per_layer_decoder:
            ups = self._upsample_block(ups, num_filters, 3, strides=[2, 2], padding='same')

        ups = tf.pad(ups, [[0, 0], [3, 3], [3, 3], [0, 0]], 'reflect')
        ups = tf.keras.layers.Conv2D(3, kernel_size=7, strides=1, padding='valid')(ups)
        out = self._decoding_generator_output_activation(ups)
        return tf.keras.Model(inputs=inputs, outputs=[out])

    def _build_discriminator(self) -> tf.keras.Model:
        """Patch-GAN discriminator based on arXiv 1711.11585"""
        image_input = tf.keras.Input(shape=(self._width, self._height, self._num_channels))
        image_input_f32 = tf.cast(image_input, tf.float32)

        if self._use_sc_scheme:
            semantic_map_input = tf.keras.Input(shape=(self._width, self._height, self._num_semantic_classes))
            semantic_map_input_f32 = tf.cast(semantic_map_input, tf.float32)
            discr_input = tf.concat([image_input_f32, semantic_map_input_f32], axis=-1)
            inputs = [image_input, semantic_map_input]
        else:
            discr_input = image_input_f32
            inputs = [image_input]

        kernel_size = 4
        for layer_idx, num_filters in enumerate(self._num_filters_per_layer_discriminator):
            if layer_idx == 0:
                x = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same')(
                    discr_input)
            else:
                x = tf.keras.layers.Conv2D(num_filters, kernel_size=kernel_size, strides=2, padding='same')(x)
                x = tfa.layers.InstanceNormalization(**self._normalization_kwargs)(x)
            x = self._activation()(x)
        out = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, strides=1, padding='same')(x)

        if not self._use_patch_gan:
            out = tf.keras.layers.Flatten()(out)
            out = tf.keras.layers.Dense(1)(out)

        return tf.keras.Model(inputs=inputs, outputs=[out])

    def _conv_block(self, x, num_filters, kernel_size: Union[int, List[int]] = 3, strides: Union[int, List[int]] = 2,
                    padding: str = 'same'):
        x = tf.keras.layers.Conv2D(num_filters, kernel_size, strides=strides, padding=padding, activation=None,
                                   kernel_initializer=self._initializer, bias_initializer=self._initializer)(x)
        x = self._normalization(**self._normalization_kwargs)(x)
        x = self._activation()(x)
        return x

    def _residual_block(self, x, num_filters: int, kernel_size: Union[int, List[int]] = 3,
                        strides: Union[int, List[int]] = 1):
        identity_map = x

        padding_width = int((kernel_size - 1) / 2)

        def _block_part(input_):
            res = tf.pad(input_, [[0, 0], [padding_width, padding_width], [padding_width, padding_width], [0, 0]],
                         'reflect')
            res = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=kernel_size, strides=strides,
                                         activation=None, padding='valid')(res)
            res = tfa.layers.InstanceNormalization(**self._normalization_kwargs)(res)
            return res

        res = _block_part(x)
        res = self._activation()(res)
        res = _block_part(res)

        assert res.get_shape().as_list() == identity_map.get_shape().as_list(), 'Mismatched shapes between ' \
                                                                                'input/output!'
        out = tf.add(res, identity_map)

        return out

    def _upsample_block(self, x, num_filters: int, kernel_size: Union[int, List[int]] = 3,
                        strides: Union[int, List[int]] = 2,
                        padding: str = 'same'):
        x = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size, strides=strides, padding=padding,
                                            activation=None)(x)
        x = self._normalization(**self._normalization_kwargs)(x)
        x = self._activation()(x)
        return x
