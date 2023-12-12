from typing import Optional, Tuple, Type

import tensorflow as tf

from .distortion import Distortion


class VGGPerceptualLoss(Distortion):
    """
    Perceptual loss as proposed by Wang et al. Computing the `metric`
    between features (outputs of layers
    corresponding to `output_layer_names`) computed by a
    pretrained VGG network as shown by Simonyan et al.


    Parameters
    ----------
    multiplier: float
        Weight with which the term is multiplied in the final loss function
    input_shape: Tuple[int, ...]
        Input shape - needs to be known a prori, e.g. (512, 1024, 3)
        for a 1024 x 512 pixel image
    output_layer_names: Optional[Tuple[str, ...]] = None
        Names of the layers of which the distance metric is computed.
        Defaults to None. If None is given the outputs
        of all convolutional layers in the network are used.
        Call `VGGPerceptualLoss.valid_output_layer_names` to display
        the available layer names.
    metric: Type[tf.keras.losses.Loss] = tf.keras.losses.MAE
        Metric computed (and minimized) between the features given
        by the output layer(s).


    References
    ----------
    Wang, T. C., Liu, M. Y., Zhu, J. Y., Tao, A., Kautz, J.,
    & Catanzaro, B. (2018). High-resolution image
    synthesis and semantic manipulation with conditional gans.
    In Proceedings of the IEEE conference on
    computer vision and pattern recognition (pp. 8798-8807).

    Simonyan, K., & Zisserman, A. (2014).
    Very deep convolutional networks for large-scale image recognition.
    arXiv preprint arXiv:1409.1556.
    """

    def __init__(
        self,
        multiplier: float,
        input_shape: Tuple[int, ...],
        output_layer_names: Optional[Tuple[str, ...]] = None,
        metric: Type[tf.keras.losses.Loss] = tf.keras.losses.MAE,
    ):
        self.multiplier = multiplier
        self._input_shape = input_shape
        self._metric = metric

        self._orig_vgg_model = tf.keras.applications.vgg19.VGG19()
        self._orig_vgg_model.trainable = False
        self._first_conv_layer_idx, self._first_fc_layer_idx = 1, -5

        if output_layer_names is not None:
            self._output_layer_names = output_layer_names
        else:
            self._output_layer_names = tuple(
                [
                    layer.name
                    for layer in self._orig_vgg_model.layers[
                        self._first_conv_layer_idx : self._first_fc_layer_idx  # noqa
                    ]
                    if "conv" in layer.name
                ]
            )

        self._valid_layer_names = [
            layer.name
            for layer in self._orig_vgg_model.layers[
                self._first_conv_layer_idx : self._first_fc_layer_idx  # noqa
            ]
        ]
        for output_layer_name in self._output_layer_names:
            assert output_layer_name in self._valid_layer_names, (
                f"Could not find layer name {output_layer_name} "
                f"in valid VGG layers. All layers but the input and "
                f"the fully connected ones at the end of the model "
                f"are valid choices."
            )

        self._vgg_models: Tuple[tf.keras.Model] = tuple(
            [
                self._build_custom_vgg_network(layer_name)
                for layer_name in self._output_layer_names
            ]
        )

    def valid_output_layer_names(self):
        return self._valid_layer_names

    def _build_custom_vgg_network(
        self, output_layer_name: str
    ) -> tf.keras.Model:
        """
        Build a model with the correct input shape, the VGG
        weights and the given output layer name as output
        """
        layers = list()
        for layer_idx, layer in enumerate(
            self._orig_vgg_model.layers[
                self._first_conv_layer_idx : self._first_fc_layer_idx  # noqa
            ]
        ):
            layers.append(layer)
            if layer.name == output_layer_name:
                break

        custom_vgg_network = tf.keras.Sequential(layers)
        custom_vgg_network.trainable = False

        return custom_vgg_network

    @tf.function
    def __call__(
        self, data: tf.Tensor, reconstructed_data: tf.Tensor
    ) -> tf.Tensor:
        vgg_loss = self._compute_loss_for_one_model_instance(
            data, reconstructed_data, self._vgg_models[0]
        )
        for model in self._vgg_models[1:]:  # type: ignore
            vgg_loss += self._compute_loss_for_one_model_instance(
                data, reconstructed_data, model
            )
        return vgg_loss / tf.cast(len(self._vgg_models), tf.float32)

    @tf.function
    def _compute_loss_for_one_model_instance(
        self,
        data: tf.Tensor,
        reconstructed_data: tf.Tensor,
        model: tf.keras.Model,
    ) -> tf.Tensor:
        return tf.reduce_mean(
            self._metric(model(data), model(reconstructed_data))
        )

    @staticmethod
    def get_key() -> str:
        return "vgg_loss"
