import tensorflow as tf

from .agustsson import AgustssonImageModelsBuilder


class AgustssonImageWithAuxUNetModelsBuilder(AgustssonImageModelsBuilder):
    """
    U-net architecture taken from:
    https://github.com/YichengWu/PhaseCam3D/blob/master/Network.py
    """

    num_channels_task_output: int = 1
    u_net_output_activation: tf.keras.layers.Activation = tf.math.tanh

    @staticmethod
    def _down_block(
        input_: tf.keras.layers.Layer, num_filters: int, max_pool: bool = True
    ) -> tf.keras.layers.Layer:
        if max_pool:
            down_n_0 = tf.nn.max_pool(
                input_,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding="SAME",
            )
        else:
            down_n_0 = input_

        down_n_1 = tf.keras.layers.Conv2D(
            num_filters, 3, strides=(1, 1), activation="relu", padding="same"
        )(down_n_0)
        down_n_1 = tf.keras.layers.BatchNormalization()(down_n_1)
        down_n_2 = tf.keras.layers.Conv2D(
            num_filters, 3, strides=(1, 1), activation="relu", padding="same"
        )(down_n_1)
        down_n_2 = tf.keras.layers.BatchNormalization()(down_n_2)
        return down_n_2

    @staticmethod
    def _up_block(
        input_: tf.keras.layers.Layer,
        num_filters: int,
        layer_to_concat: tf.keras.layers.Layer,
    ) -> tf.keras.layers.Layer:
        up_n_0 = tf.concat(
            [
                tf.keras.layers.Conv2DTranspose(
                    num_filters, 3, strides=(2, 2), padding="same"
                )(input_),
                layer_to_concat,
            ],
            axis=-1,
        )
        up_n_1 = tf.keras.layers.Conv2D(
            num_filters, 3, activation="relu", padding="same"
        )(up_n_0)
        up_n_1 = tf.keras.layers.BatchNormalization()(up_n_1)
        up_n_2 = tf.keras.layers.Conv2D(
            num_filters, 3, activation="relu", padding="same"
        )(up_n_1)
        up_n_2 = tf.keras.layers.BatchNormalization()(up_n_2)
        return up_n_2

    def _build_u_net_task_network(
        self, network_input: tf.keras.layers.Layer
    ) -> tf.keras.layers.Layer:
        num_filters_per_layer = (480, 240, 120, 60, 30)

        down_layers = [network_input]
        for layer_idx, num_filters in enumerate(
            reversed(num_filters_per_layer)
        ):
            down_layers.append(
                self._down_block(
                    down_layers[-1], num_filters, max_pool=layer_idx != 0
                )
            )

        up_layers = [down_layers[-1]]
        for layer_idx, num_filters in enumerate(num_filters_per_layer):
            if layer_idx == 0:
                continue
            up_layers.append(
                self._up_block(
                    up_layers[-1],
                    num_filters,
                    layer_to_concat=down_layers[-(layer_idx + 1)],
                )
            )

        u_net_output = tf.keras.layers.Conv2D(1, 1, padding="same")(
            up_layers[-1]
        )
        u_net_output = tf.keras.layers.BatchNormalization()(u_net_output)
        u_net_output = tf.keras.activations.tanh(u_net_output)
        return u_net_output

    def _build_decoding_generator(self) -> tf.keras.Model:
        decoding_generator = super()._build_decoding_generator()

        task_network_input = decoding_generator.output
        task_network_output = self._build_u_net_task_network(
            task_network_input
        )

        return tf.keras.Model(
            inputs=decoding_generator.inputs,
            outputs=[decoding_generator.output, task_network_output],
        )


class AgustssonImageWithAuxHalfParallelDecoder(AgustssonImageModelsBuilder):
    """
    U-net architecture taken from:
    https://github.com/YichengWu/PhaseCam3D/blob/master/Network.py
    """

    num_channels_task_output: int = 1
    aux_net_output_activation: tf.keras.layers.Activation = tf.math.tanh

    def _build_decoding_generator(self) -> tf.keras.Model:
        decoding_generator = super()._build_decoding_generator()

        last_res_layer_idx = 102 if self._use_sc_scheme else 83
        ups = decoding_generator.layers[last_res_layer_idx].input
        for num_filters in self._num_filters_per_layer_decoder:
            ups = self._upsample_block(
                ups, int(1.4 * num_filters), 3, strides=[2, 2], padding="same"
            )

        ups = tf.pad(ups, [[0, 0], [3, 3], [3, 3], [0, 0]], "reflect")
        ups = tf.keras.layers.Conv2D(
            1, kernel_size=7, strides=1, padding="valid"
        )(ups)
        task_network_output = self._decoding_generator_output_activation(ups)

        return tf.keras.Model(
            inputs=decoding_generator.inputs,
            outputs=[decoding_generator.output, task_network_output],
        )
