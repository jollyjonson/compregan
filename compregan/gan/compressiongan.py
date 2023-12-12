import json
import os
import shutil
import subprocess
import warnings
from typing import Callable, Optional, Tuple, Union, Dict
from typeguard import typechecked

import dill as pickle
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar

from compregan.gan.components import Components
from compregan.gan.conditional import Conditional
from compregan.losses.distortions import Distortion, MSE
from compregan.losses.gan_losses import GANLoss, LeastSquares
from compregan.losses.lossterm import LossTerm
from compregan.losses.auxillary_loss_terms import AuxillaryLossTerms


class CompressionGAN:
    """

    Parameters
    ----------
    encoder: tf.keras.Model
    decoding_generator: tf.keras.Model
    discriminator: tf.keras.Model
    loss_terms: Union[LossTerm, Tuple[LossTerm, ...]] = MSE(mse_multiplier=10.)
    quantizer: tf.keras.Model = tf.identity
    gan_loss: GANLoss = LeastSquares()
    noise_prior: Optional[Callable] = None
    noise_dim: Union[int, Tuple[int, ...]] = 0
    conditional: Union[tf.keras.Model, Callable] = None
    use_conditional: bool = True
    encoder_optimizer: tf.keras.optimizers.Optimizer = Adam()
    decoding_generator_optimizer: tf.keras.optimizers.Optimizer = Adam()
    discriminator_optimizer: tf.keras.optimizers.Optimizer = Adam()
    quantizer_optimizer: Optional[tf.keras.optimizers.Optimizer] = None
    metrics: Optional[Union[Distortion, Tuple[Distortion, ...]]] = None
    auxillary_loss_terms: Optional[AuxillaryLossTerms] = None
    auxillary_metrics: Optional[AuxillaryLossTerms] = None
    codec_data_key: Optional[str] = None
    """

    class QuantityNames:
        generator_loss = 'generator_loss'
        discriminator_loss = 'discriminator_loss'
        discriminator_acc = 'discriminator_accuracy'
        validation_prefix = 'val_'

    @typechecked
    def __init__(self,
                 encoder: tf.keras.Model,
                 decoding_generator: tf.keras.Model,
                 discriminator: tf.keras.Model,
                 loss_terms: Union[LossTerm, Tuple[LossTerm, ...]] = MSE(mse_multiplier=10.),
                 quantizer: tf.keras.Model = tf.identity,
                 gan_loss: GANLoss = LeastSquares(),
                 noise_prior: Optional[Callable] = None,
                 noise_dim: Union[int, Tuple[int, ...]] = 0,
                 conditional: Optional[Union[tf.keras.Model, Callable, Conditional]] = None,
                 use_conditional: bool = True,
                 encoder_optimizer: tf.keras.optimizers.Optimizer = Adam(),
                 decoding_generator_optimizer: tf.keras.optimizers.Optimizer = Adam(),
                 discriminator_optimizer: tf.keras.optimizers.Optimizer = Adam(),
                 quantizer_optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                 metrics: Optional[Union[Distortion, Tuple[Distortion, ...]]] = None,
                 auxillary_loss_terms: Optional[AuxillaryLossTerms] = None,
                 auxillary_metrics: Optional[AuxillaryLossTerms] = None,
                 codec_data_key: Optional[str] = None,
                 ):
        self._encoder = encoder
        self._encoder_optimizer = encoder_optimizer
        self._quantizer = quantizer
        self._quantizer_optimizer = quantizer_optimizer
        self._decoding_generator = decoding_generator
        self._decoding_generator_optimizer = decoding_generator_optimizer
        self._loss_terms = (loss_terms,) if issubclass(type(loss_terms),
                                                       LossTerm) else loss_terms
        self._discriminator = discriminator
        self._discriminator_optimizer = discriminator_optimizer
        self._gan_loss = gan_loss
        self._conditional = conditional
        self._conditional_has_needed_components = issubclass(type(self._conditional), Conditional)
        self._using_conditional_gan = self._conditional is not None and use_conditional

        self._noise_prior = noise_prior
        self._noise_dim = (noise_dim,) if type(noise_dim) is int else noise_dim
        self._concatenating_latent_repr_with_noise = self._noise_prior is not None and all(
            n > 0 for n in self._noise_dim)

        self._metrics = tuple() if metrics is None else (metrics,) if type(metrics) is not tuple and \
                                                                      type(metrics) is not list else metrics

        self._auxillary_loss_terms = auxillary_loss_terms if auxillary_loss_terms is not None else list()
        self._auxillary_metrics = auxillary_metrics if auxillary_metrics is not None else list()

        self._codec_data_key = codec_data_key

        self.history = dict()

    @tf.function
    def _get_codec_data(self, data_batch: Union[tf.Tensor, tf.data.Dataset]) -> tf.Tensor:
        if not self._codec_data_key:
            return data_batch
        else:
            return data_batch[self._codec_data_key]

    def train(self,
              training_data: tf.data.Dataset,
              num_epochs: int,
              batch_size: int = 32,
              shuffle: bool = True,
              validation_data: Optional[tf.data.Dataset] = None,
              validation_steps: Optional[int] = None,
              steps_per_epoch: int = None,
              start_epoch_idx: int = 1,
              ) -> Dict[int, Dict[str, float]]:

        self._warn_if_ignoring_data(steps_per_epoch, shuffle)

        for epoch_idx in range(start_epoch_idx, num_epochs + 1):

            batched_dataset = training_data.batch(batch_size)
            print(f"Epoch {epoch_idx}/{num_epochs}:")
            num_train_steps = batched_dataset.cardinality().numpy() if steps_per_epoch is None else steps_per_epoch
            progress_bar = Progbar(num_train_steps)
            for batch_idx, batch in enumerate(batched_dataset):
                monitored_quantities = self._train_step(batch)
                if not batch_idx == num_train_steps - 1 or validation_data is None:  # spare last update if validation
                    progress_bar.update(
                        current=batch_idx + 1,
                        values=[(k, v) for k, v in monitored_quantities.items()])
                if steps_per_epoch == batch_idx + 1:
                    break

            if validation_data is not None:
                monitored_quantities.update(self._validation_step(validation_data, validation_steps, batch_size))
                progress_bar.update(
                    current=batch_idx + 1,
                    values=[(k, v) for k, v in monitored_quantities.items()],
                    finalize=True)

            if shuffle:
                training_data.shuffle(batch_size)

            self.history[epoch_idx] = monitored_quantities

        return self.history

    def _train_step(self, train_data_batch: tf.data.Dataset) -> Dict[str, float]:
        monitored_quantities = self._on_train_step(train_data_batch)
        for key, value in monitored_quantities.items():  # convert tf.Tensors to python scalars
            monitored_quantities[key] = float(value.numpy())
        return monitored_quantities

    @tf.function
    def _on_train_step(self, train_data_batch: tf.data.Dataset) -> Dict[str, tf.Tensor]:

        monitored_quantities = dict()

        codec_data = self._get_codec_data(train_data_batch)

        component_dict = {
            Components.Encoder: self._encoder,
            Components.DecodingGenerator: self._decoding_generator,
            Components.Discriminator: self._discriminator,
            Components.Conditional: self._conditional,
            Components.OriginalCodecData: codec_data,
            Components.CompleteData: train_data_batch,
        }

        with tf.GradientTape(persistent=True) as tape:

            # forward pass
            if self._using_conditional_gan:
                conditional_input = [component_dict[k] for k in self._conditional.needed_components] \
                    if self._conditional_has_needed_components \
                    else [codec_data]
                conditioning_information = self._conditional(*conditional_input)
                encoder_input = [codec_data, conditioning_information]
            else:
                conditioning_information = None
                encoder_input = codec_data

            latent_repr = self._encoder(encoder_input, training=True)
            quantized_latent_repr = self._quantizer(latent_repr, training=True)

            if self._concatenating_latent_repr_with_noise:
                batch_dim = codec_data.shape[0]
                noise = self._noise_prior((batch_dim, *self._noise_dim))
                latents = tf.concat([quantized_latent_repr, noise], axis=-1)
            else:
                latents = quantized_latent_repr

            decoding_generator_inputs = [latents, conditioning_information] if self._using_conditional_gan else latents
            decoding_generator_outputs = self._decoding_generator(decoding_generator_inputs, training=True)

            if type(decoding_generator_outputs) is tf.Tensor:  # single output
                reconstructed_data = decoding_generator_outputs
                auxillary_outputs = list()
            else:  # multiple outputs
                assert type(decoding_generator_outputs) is list
                reconstructed_data = decoding_generator_outputs[0]
                auxillary_outputs = decoding_generator_outputs[1:]

            component_dict[Components.ReconstructedCodecData] = reconstructed_data
            component_dict[Components.AuxillaryOutput] = auxillary_outputs

            discr_input_reconstr = [reconstructed_data, conditioning_information] if self._using_conditional_gan \
                else reconstructed_data
            discr_out_reconstructed_data = self._discriminator(discr_input_reconstr,
                                                               training=True)
            discr_input_real = [codec_data, conditioning_information] if self._using_conditional_gan \
                else codec_data
            discr_out_real_data = self._discriminator(discr_input_real, training=True)

            # compute losses
            discr_loss = (tf.reduce_mean(self._gan_loss.f(discr_out_real_data))
                          + tf.reduce_mean(self._gan_loss.g(discr_out_reconstructed_data)))
            gen_loss = tf.reduce_mean(self._gan_loss.f(discr_out_reconstructed_data))

            # loss terms for compression part
            for loss_term in self._loss_terms:
                needed_components = [component_dict[key] for key in loss_term.needed_components]
                loss_term_value = loss_term(*needed_components)
                monitored_quantities[loss_term.get_key()] = loss_term_value
                gen_loss, discr_loss = loss_term.apply(loss_term_value, gen_loss, discr_loss)

            # loss terms for auxillary output
            for auxillary_output_idx, auxillary_output in enumerate(auxillary_outputs):
                component_dict[Components.AuxillaryOutput] = auxillary_output
                for auxillary_loss_term in self._auxillary_loss_terms[auxillary_output_idx]:
                    needed_components = [component_dict[key] for key in auxillary_loss_term.needed_components]
                    auxillary_loss_term_value = auxillary_loss_term(*needed_components)
                    monitored_quantities[auxillary_loss_term.get_key()] = auxillary_loss_term_value
                    gen_loss, discr_loss = auxillary_loss_term.apply(auxillary_loss_term_value, gen_loss, discr_loss)

        # update encoder, quantizer, generator (decoder)
        gradient_encoder = tape.gradient(gen_loss, self._encoder.trainable_variables)
        self._encoder_optimizer.apply_gradients(zip(gradient_encoder, self._encoder.trainable_variables))

        if self._quantizer_optimizer is not None:
            gradient_quantizer = tape.gradient(gen_loss, self._quantizer.trainable_variables)
            self._quantizer_optimizer.apply_gradients(zip(gradient_quantizer, self._quantizer.trainable_variables))

        gradient_decoding_generator = tape.gradient(gen_loss,
                                                    self._decoding_generator.trainable_variables)
        self._decoding_generator_optimizer.apply_gradients(
            zip(gradient_decoding_generator, self._decoding_generator.trainable_variables))

        # update discriminator
        gradient_discriminator = tape.gradient(discr_loss, self._discriminator.trainable_variables)
        self._discriminator_optimizer.apply_gradients(
            zip(gradient_discriminator, self._discriminator.trainable_variables))

        # store losses
        monitored_quantities[self.QuantityNames.generator_loss] = tf.reduce_mean(gen_loss)
        monitored_quantities[self.QuantityNames.discriminator_loss] = tf.reduce_mean(discr_loss)

        monitored_quantities[self.QuantityNames.discriminator_acc] = self.compute_discriminator_accuracy(
            discr_out_real_data, discr_out_reconstructed_data)

        return monitored_quantities

    def _validation_step(self, validation_data: tf.data.Dataset, validation_steps: Optional[int] = None,
                         batch_size: int = 32) -> Dict[str, float]:
        batched_dataset = validation_data.batch(batch_size)
        for batch_idx, batch in enumerate(batched_dataset):
            if batch_idx == 0:
                monitored_quantities_val = self._on_validation_step(batch)
                for key, value in monitored_quantities_val.items():  # convert tf.Tensors to python scalars
                    monitored_quantities_val[key] = float(value.numpy())
            else:
                monitored_quantities_val_tmp = self._on_validation_step(batch)
                for key, value in monitored_quantities_val.items():  # accumulate losses, metrics
                    monitored_quantities_val[key] += float(monitored_quantities_val_tmp[key].numpy())

            if validation_steps is not None:
                if batch_idx == validation_steps - 1:
                    break

        for key, value in monitored_quantities_val.items():  # compute mean over batches
            monitored_quantities_val[key] /= (batch_idx + 1)

        return monitored_quantities_val

    @tf.function
    def _on_validation_step(self, validation_data_instance: tf.Tensor) -> Dict[str, tf.Tensor]:

        monitored_quantities = dict()

        def __val_name(name: str) -> str:
            return self.QuantityNames.validation_prefix + name

        codec_data = self._get_codec_data(validation_data_instance)

        component_dict = {
            Components.Encoder: self._encoder,
            Components.DecodingGenerator: self._decoding_generator,
            Components.Discriminator: self._discriminator,
            Components.Conditional: self._conditional,
            Components.OriginalCodecData: codec_data,
            Components.CompleteData: validation_data_instance,
        }

        # forward pass
        if self._using_conditional_gan:
            conditional_input = [component_dict[k] for k in self._conditional.needed_components] \
                if self._conditional_has_needed_components \
                else [codec_data]
            conditioning_information = self._conditional(*conditional_input)
            encoder_input = [codec_data, conditioning_information]
        else:
            conditioning_information = None
            encoder_input = codec_data

        latent_repr = self._encoder(encoder_input, training=True)
        quantized_latent_repr = self._quantizer(latent_repr, training=True)

        if self._concatenating_latent_repr_with_noise:
            batch_dim = codec_data.shape[0]
            noise = self._noise_prior((batch_dim, *self._noise_dim))
            latents = tf.concat([quantized_latent_repr, noise], axis=-1)
        else:
            latents = quantized_latent_repr

        decoding_generator_inputs = [latents, conditioning_information] if self._using_conditional_gan else latents
        decoding_generator_outputs = self._decoding_generator(decoding_generator_inputs, training=True)

        if type(decoding_generator_outputs) is tf.Tensor:  # single output
            reconstructed_data = decoding_generator_outputs
            auxillary_outputs = list()
        else:  # multiple outputs
            assert type(decoding_generator_outputs) is list
            reconstructed_data = decoding_generator_outputs[0]
            auxillary_outputs = decoding_generator_outputs[1:]

        discr_input_reconstr = [reconstructed_data, conditioning_information] if self._using_conditional_gan \
            else reconstructed_data
        discr_out_reconstructed_data = self._discriminator(discr_input_reconstr,
                                                           training=True)
        discr_input_real = [codec_data, conditioning_information] if self._using_conditional_gan \
            else codec_data
        discr_out_real_data = self._discriminator(discr_input_real, training=True)

        # compute losses
        monitored_quantities[__val_name(self.QuantityNames.generator_loss)] = (
                tf.reduce_mean(self._gan_loss.f(discr_out_real_data))
                + tf.reduce_mean(self._gan_loss.g(discr_out_reconstructed_data)))
        monitored_quantities[__val_name(self.QuantityNames.discriminator_loss)] = tf.reduce_mean(
            self._gan_loss.f(discr_out_reconstructed_data))

        component_dict[Components.ReconstructedCodecData] = reconstructed_data
        component_dict[Components.AuxillaryOutput] = auxillary_outputs

        for loss_term in self._loss_terms:
            needed_components = [component_dict[k] for k in loss_term.needed_components]
            monitored_quantities[__val_name(loss_term.get_key())] = loss_term(*needed_components)

        for metric in self._metrics:
            monitored_quantities[__val_name(metric.get_key())] = metric(codec_data,
                                                                        reconstructed_data)

        for auxillary_output_idx, auxillary_output in enumerate(auxillary_outputs):
            component_dict[Components.AuxillaryOutput] = auxillary_output
            for auxillary_loss_term_or_metric in [*self._auxillary_loss_terms[auxillary_output_idx],
                                                  *self._auxillary_metrics[auxillary_output_idx]]:
                needed_components = [component_dict[key] for key in auxillary_loss_term_or_metric.needed_components]
                auxillary_loss_term_value = auxillary_loss_term_or_metric(*needed_components)
                monitored_quantities[__val_name(auxillary_loss_term_or_metric.get_key())] = auxillary_loss_term_value

        monitored_quantities[__val_name(self.QuantityNames.discriminator_acc)] = self.compute_discriminator_accuracy(
            discr_out_real_data, discr_out_reconstructed_data)

        return monitored_quantities

    @tf.function
    def compute_discriminator_accuracy(self, discriminator_out_real_data: tf.Tensor,
                                       discriminator_out_reconstructed_data: tf.Tensor) -> tf.Tensor:
        return (tf.reduce_mean(
            keras.metrics.binary_accuracy(tf.ones_like(discriminator_out_real_data),
                                          tf.round(discriminator_out_real_data)))
                + tf.reduce_mean(
                    keras.metrics.binary_accuracy(tf.zeros_like(discriminator_out_reconstructed_data),
                                                  tf.round(discriminator_out_reconstructed_data)))) / 2

    def encode_data(self, data: Union[tf.Tensor, np.ndarray], conditioning_information: Optional[tf.Tensor] = None,
                    return_np_array: bool = True) -> Union[tf.Tensor, np.ndarray]:
        if self._using_conditional_gan and conditioning_information is None:
            conditioning_information = self._conditional(data)
            encoder_input = [data, conditioning_information]
        elif self._using_conditional_gan and conditioning_information is not None:
            encoder_input = [data, conditioning_information]
        else:
            encoder_input = data

        latent_repr = self._encoder(encoder_input)
        quantized_latent_repr = self._quantizer(latent_repr)

        return quantized_latent_repr.numpy() if return_np_array else quantized_latent_repr

    def decode(self, encoded_representation: Union[tf.Tensor, np.ndarray],
               conditioning_information: Optional[tf.Tensor] = None,
               return_np_array: bool = True) -> Union[tf.Tensor, np.ndarray]:
        if self._concatenating_latent_repr_with_noise:
            noise = self._noise_prior((encoded_representation.shape[0], *self._noise_dim))
            z = tf.concat([encoded_representation, noise], axis=-1)
        else:
            z = encoded_representation

        if self._using_conditional_gan:
            decoding_generator_inputs = [z, conditioning_information]
        else:
            decoding_generator_inputs = z

        decoded_data = self._decoding_generator(decoding_generator_inputs)

        if type(decoded_data) is tf.Tensor:
            decoded_data = decoded_data.numpy() if return_np_array else decoded_data
        else:
            decoded_data = list(map(lambda t: t.numpy(), decoded_data)) if return_np_array else decoded_data

        return decoded_data

    def save_history_as_json(self, path: Union[str, os.PathLike]) -> None:
        with open(path, 'w') as json_handle:
            json_handle.write(json.dumps(self.history, indent=2))

    _potential_model_members = ['_encoder', '_decoding_generator', '_discriminator', '_quantizer',
                                '_conditional']
    _saved_instance_filename = 'CompreGANInstance.pkl'

    def save_to_file(self, path: Union[str, os.PathLike]):
        path = os.path.abspath(path)

        tmp_dir = os.path.join(os.path.dirname(path), 'tmpCompressionGAN_save_to_file')
        os.mkdir(tmp_dir)

        try:
            # save all models separately and remove them from this instance for pickling, save them in a dict for later
            models = dict()
            for potential_model_name in self._potential_model_members:
                potential_model = getattr(self, potential_model_name)
                if issubclass(type(potential_model), tf.keras.Model):
                    model_save_path = os.path.join(tmp_dir, potential_model_name)
                    potential_model.save(model_save_path)
                    models[potential_model_name] = potential_model
                    setattr(self, potential_model_name, None)

            # pickle the instance without the models
            instance_save_path = os.path.join(tmp_dir, self._saved_instance_filename)
            with open(instance_save_path, 'wb') as pkl_file_handle:
                pickle.dump(self, pkl_file_handle)

            # reattach the models to this instance
            for name, model in models.items():
                setattr(self, name, model)

            # compress everything into one file
            with subprocess.Popen(
                    ['tar', '-czf', f'{os.path.basename(path)}.tar.gz', '-C', f'{tmp_dir}', '.'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=os.path.dirname(path)) as proc:
                proc.wait()
                assert proc.returncode == 0

        except Exception as exception:
            raise exception

        finally:
            shutil.rmtree(tmp_dir)

    @classmethod
    def load_from_file(cls, path: Union[str, os.PathLike]):

        path = os.path.abspath(path) if path.endswith('.tar.gz') else os.path.abspath(path) + '.tar.gz'

        tmp_dir = os.path.join(os.path.dirname(path), 'tmpCompressionGAN_load_from_file')
        os.mkdir(tmp_dir)

        try:
            # uncompress
            with subprocess.Popen(['tar', '-xf', f'{os.path.basename(path)}',
                                   '-C', f'{os.path.basename(tmp_dir)}'],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  cwd=os.path.dirname(path)) as proc:
                proc.wait()
                assert proc.returncode == 0

            # load instance
            with open(os.path.join(tmp_dir, cls._saved_instance_filename), 'rb') as pkl_file_handle:
                compression_gan_instance = pickle.load(pkl_file_handle)  # type: CompressionGAN

            # any directory corresponds to a model, load the models and again set them as members of the instance
            for item_name in os.listdir(tmp_dir):
                item_path = os.path.join(tmp_dir, item_name)
                is_saved_model = os.path.isdir(item_path)
                if is_saved_model:
                    assert item_name in cls._potential_model_members
                    model = tf.keras.models.load_model(item_path)
                    setattr(compression_gan_instance, item_name, model)

        except Exception as exception:
            raise exception

        finally:
            shutil.rmtree(tmp_dir)

        return compression_gan_instance

    @staticmethod
    def _warn_if_ignoring_data(steps_per_epoch: int, shuffle: bool):
        if steps_per_epoch is not None and not shuffle:
            warnings.warn(f"Performing {steps_per_epoch} steps per epoch "
                          f"without shuffling! This could potentially "
                          f"ignore large parts of the dataset!")
