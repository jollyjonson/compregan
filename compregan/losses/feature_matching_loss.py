from typing import Tuple, Type, Union

import tensorflow as tf

from .lossterm import LossTerm
from ..gan.components import Components


class FeatureMatchingLoss(LossTerm):
    """
    For details s. Wang et al. Eqs. (4) and (5). Citation from their work:
    "We improve the GAN loss in Eq. (2) by incorporating a feature matching loss based on
    the discriminator. This loss stabilizes the training as the generator has to produce natural statistics at
    multiple scales. Specifically, we extract features from multiple layers of the discriminator and learn to match
    these intermediate representations from the real and the synthesized image."


    Parameters
    ----------
    discriminator: tf.keras.Model
        The same instance of the discriminator model as used in the training process. Extracing the features on the
        fly is not possible due to constraints of `tf.function`
    multiplier: float
        Weight with which the loss term is multiplied before added to the overall loss
    metric: tf.keras.losses.Loss = tf.keras.losses.MAE
        Distance measure for the features
    relevant_layer_types: Tuple[Type[tf.keras.layers.Layer]] = tf.keras.layers.LeakyReLU
        Features produced by all layers of this kind will be used to compute the loss. Most likely these will be
        activation layers of Conv2D layers (as for image processing).
    decay_factor: float = 0.
        Factor determining the decay of the FM loss term. As some literature suggests to only use the fm_loss during
        the first phase of the training, this parameter can be used to exponentially decay this loss term.
        The multiplier, weighting the loss term is computed at every aaplication of the loss term as

        >>>decayed_multiplier = multiplier * tf.exp(-decay_factor * num_steps_loss_has_been_applied)

        The parameter defaults to 0, meaning no decay is applied! For an example schedule check e.g.:

        >>>import numpy as np
        >>>import matplotlib.pyplot as plt
        >>>steps = np.arange(1, 150000)
        >>>multiplier = 10
        >>>decay_factor = 5e-4
        >>>plt.plot(multiplier * np.exp(-decay_factor * steps))
        >>>plt.show()


    References
    ----------
    Wang, T. C., Liu, M. Y., Zhu, J. Y., Tao, A., Kautz, J., & Catanzaro, B. (2018). High-resolution image
    synthesis and semantic manipulation with conditional gans. In Proceedings of the IEEE conference on
    computer vision and pattern recognition (pp. 8798-8807).
    """

    def __init__(self,
                 discriminator: tf.keras.Model,
                 multiplier: float,
                 metric: tf.keras.losses.Loss = tf.keras.losses.MAE,
                 relevant_layer_types: Tuple[Type[tf.keras.layers.Layer], ...] = (tf.keras.layers.LeakyReLU,),
                 decay_factor: float = 0.,
                 use_conditional_discriminator: bool = False,
                 ):
        self._discriminator = discriminator
        self.multiplier = multiplier
        self._metric = metric
        self._relevant_layer_types = relevant_layer_types
        self._use_conditional_discriminator = use_conditional_discriminator

        self._discriminator_layer_output_models = self._assemble_custom_output_models()

        self._num_steps_loss_has_been_applied = 0
        assert decay_factor >= 0., "Negative decay factors are not allowed as they lead to exponential growth of the " \
                                   "loss term!"
        self._decay_factor = decay_factor

    @property
    def needed_components(self) -> Tuple[Components, ...]:
        if self._use_conditional_discriminator:
            return tuple([c for c in Components])
        else:
            return Components.OriginalCodecData, Components.ReconstructedCodecData

    def _assemble_custom_output_models(self):
        custom_output_models = list()
        for layer_idx, layer in enumerate(self._discriminator.layers):
            if layer_idx != 0 and type(layer) in self._relevant_layer_types:
                custom_output_models.append(tf.keras.Model(inputs=[self._discriminator.input], outputs=[layer.output]))
        return tuple(custom_output_models)

    @tf.function
    def __call__(self, *components: Tuple[Union[tf.Tensor, tf.keras.Model], ...]) -> tf.Tensor:
        fm_loss = tf.zeros(1)

        if self._use_conditional_discriminator:
            original_data = components[Components.OriginalCodecData]
            reconstructed_data = components[Components.ReconstructedCodecData]
            conditional = components[Components.Conditional]
            conditional_input = [components[k] for k in conditional.needed_components] \
                if hasattr(conditional, 'needed_components') \
                else [original_data]
            conditioning_information = conditional(*conditional_input)
        else:
            original_data, reconstructed_data = components

        for tmp_model in self._discriminator_layer_output_models:
            if self._use_conditional_discriminator:
                fm_loss += tf.reduce_mean(self._metric(tmp_model([original_data, conditioning_information]),
                                                       tmp_model([reconstructed_data, conditioning_information])))
            else:
                fm_loss += tf.reduce_mean(self._metric(tmp_model(original_data),
                                                       tmp_model(reconstructed_data)))

        return fm_loss / tf.cast(len(self._discriminator_layer_output_models), tf.float32)

    def apply(self, loss_value: tf.Tensor, generator_loss: tf.Tensor, discr_loss: tf.Tensor) \
            -> Tuple[tf.Tensor, tf.Tensor]:
        self._increment_step_counter()
        decayed_multiplier = self.multiplier * tf.exp(-self._decay_factor * self._num_steps_loss_has_been_applied)
        generator_loss += tf.cast(decayed_multiplier, tf.float32) * loss_value
        return generator_loss, discr_loss

    def _increment_step_counter(self):
        self._num_steps_loss_has_been_applied += 1

    def get_key(self) -> str:
        return "fm_loss"
