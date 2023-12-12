import json
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

from compregan.gan.compressiongan import CompressionGAN
from compregan.losses.distortions import MSE, PSNR, MSSSIM
from compregan.losses.feature_matching_loss import FeatureMatchingLoss
from compregan.losses.gan_losses import LeastSquares
from compregan.losses.vgg_perceptual_loss import VGGPerceptualLoss
from compregan.models.image.agustsson import AgustssonImageModelsBuilder
from compregan.models.layers.selective_dropout import SelectiveDropout
from compregan.models.quantizer import Quantizer
from compregan.util import GlobalRandomSeed, save_pip_freeze_to_file, debugger_connected, copy_this_file_to_directory, \
    reduce_dataset_size_to_n_elements


# wait_for_gpu()


class HyperParameters:
    force_retrain_gan = True
    seed = 1999

    experiment_name = os.path.basename(__file__).replace('.py', '')
    result_dir = os.path.join(os.getcwd(), experiment_name)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    compgan_path = os.path.join(result_dir, f'{experiment_name}')

    orig_img_shape = (352, 688, 3)

    learning_rate = 0.0002

    debug = debugger_connected()

    num_filters_per_layer_encoder = (10, 30, 120) if debug else (60, 120, 240, 480, 960)
    num_filters_per_layer_decoder = (60, 60, 30) if debug else (480, 240, 120, 60)

    img_shape = (48, 80, 3) if debug else orig_img_shape

    num_channels_latent_space = 64
    num_noise_channels = 0

    if debug:
        num_epochs = 1
        steps_per_epoch = 3
        batch_size = 1
        num_examples_to_be_saved = 5
    else:
        num_training_steps = 50000
        num_epochs = 50
        batch_size = 1
        num_items_per_epoch = int(num_training_steps / num_epochs)
        steps_per_epoch = int(num_items_per_epoch / batch_size)
        num_examples_to_be_saved = 70

    num_validation_examples = 250
    num_validation_steps = int(num_validation_examples / batch_size)
    num_centers_quantization = 5

    # selective dropout
    use_selective_dropout = True
    overall_probability_of_selective_dropout = 0.6
    ratio_of_never_dropped_channels = 0.05
    replacement_value: Callable = lambda input_shape: -10.


GlobalRandomSeed(HyperParameters.seed)

model_builder = AgustssonImageModelsBuilder(
    *HyperParameters.img_shape,
    use_patch_gan=True,
    num_channels_latent_space=HyperParameters.num_channels_latent_space,
    num_noise_channels=HyperParameters.num_noise_channels,
    num_residual_blocks_decoder=12,
)


def build_dropout_quantizer() -> tf.keras.Model:
    selective_dropout_layer = SelectiveDropout(HyperParameters.ratio_of_never_dropped_channels,
                                               HyperParameters.overall_probability_of_selective_dropout,
                                               replacement_value=HyperParameters.replacement_value,
                                               only_drop_full_channels=True)
    quantizer = Quantizer(model_builder.encoder.output_shape[1:], num_centers=HyperParameters.num_centers_quantization)
    if HyperParameters.use_selective_dropout:
        return tf.keras.Sequential([quantizer, selective_dropout_layer])
    else:
        return quantizer


num_pixels = np.prod(HyperParameters.img_shape)
num_bits_latent_space = np.prod(model_builder.encoder.output_shape[1:]) \
                        * np.log2(HyperParameters.num_centers_quantization)
print(f"Compressing images to {num_bits_latent_space / num_pixels} bpp "
      f"(min {(num_bits_latent_space * HyperParameters.ratio_of_never_dropped_channels) / num_pixels} "
      f"bpp with selective dropout)")

metrics = (PSNR(), MSSSIM())

compression_gan = CompressionGAN(
    encoder=model_builder.encoder,
    decoding_generator=model_builder.decoding_generator,
    discriminator=model_builder.discriminator,
    loss_terms=(MSE(10.),
                VGGPerceptualLoss(10., HyperParameters.img_shape),
                FeatureMatchingLoss(model_builder.discriminator, 10., decay_factor=5e-4)),
    metrics=metrics,
    quantizer=build_dropout_quantizer(),
    gan_loss=LeastSquares(),
    noise_prior=lambda shape: tf.random.normal(shape, mean=0., stddev=1.),
    noise_dim=(*model_builder.encoder.output_shape[1:-1], HyperParameters.num_noise_channels),
    conditional=None,
    encoder_optimizer=tf.keras.optimizers.Adam(HyperParameters.learning_rate),
    decoding_generator_optimizer=tf.keras.optimizers.Adam(HyperParameters.learning_rate),
    discriminator_optimizer=tf.keras.optimizers.Adam(HyperParameters.learning_rate),
    quantizer_optimizer=None,
)


def data_preprocess_func(dataset_item: tf.data.Dataset) -> tf.data.Dataset:
    image = dataset_item['image']
    image = tf.image.resize(image, HyperParameters.img_shape[:-1])
    image = ((tf.cast(image, tf.float32) / 255.) - 0.5) * 2.  # [0, 255] -> [-1, 1]
    return image


flic_dataset_train = tfds.load('flic', split='train').map(data_preprocess_func)
flic_dataset_test = tfds.load('flic', split='test').map(data_preprocess_func)

if HyperParameters.debug:
    flic_dataset_train, flic_dataset_test = list(map(lambda d: reduce_dataset_size_to_n_elements(d, 3),
                                                     [flic_dataset_train, flic_dataset_test]))

compression_gan.train(
    training_data=flic_dataset_train,
    num_epochs=HyperParameters.num_epochs,
    batch_size=HyperParameters.batch_size,
    shuffle=True,
    validation_data=flic_dataset_test,
    validation_steps=HyperParameters.num_validation_steps,
    steps_per_epoch=HyperParameters.steps_per_epoch,
)

compression_gan.save_to_file(HyperParameters.compgan_path)
compression_gan.save_history_as_json(HyperParameters.compgan_path + '_train_history.json')
copy_this_file_to_directory(HyperParameters.result_dir)
save_pip_freeze_to_file(os.path.join(HyperParameters.result_dir, 'dependencies.txt'))

show_plots = HyperParameters.debug
examples_dir = os.path.join(HyperParameters.result_dir, 'examples')
if not os.path.isdir(examples_dir):
    os.mkdir(examples_dir)

metrics_values = {metric.get_key(): dict() for metric in metrics}

for idx, image in tqdm.tqdm(enumerate(flic_dataset_test), desc='Evaluating model'):

    num_channels = model_builder.encoder.output_shape[-1]
    num_channels_to_always_keep = int(round(HyperParameters.ratio_of_never_dropped_channels * num_channels))

    image_with_batch_dim = np.expand_dims(image.numpy(), 0)

    latent_representation = compression_gan.encode_data(image_with_batch_dim)

    if idx < HyperParameters.num_examples_to_be_saved:
        image_dir = os.path.join(examples_dir, str(idx))
        os.mkdir(image_dir)

        np.save(os.path.join(image_dir, str(idx) + f'latent_repr'), latent_representation)

        plt.figure(dpi=300, figsize=(6, 16))
        plt.suptitle("Latent Space")
        for i in range(latent_representation.shape[-1]):
            plt.subplot(int(latent_representation.shape[-1] / 2), 2, i + 1)
            plt.title(f"Channel {i} | std {latent_representation[0, :, :, i].std():.4f}")
            plt.imshow(latent_representation[0, :, :, i])
        plt.savefig(os.path.join(image_dir, str(idx) + f'_latent_space.png'))
        plt.close()

    num_pixels = np.prod(HyperParameters.img_shape)

    if HyperParameters.use_selective_dropout:
        example_channel_configs = np.linspace(num_channels_to_always_keep,
                                              HyperParameters.num_channels_latent_space,
                                              5, dtype=int)
    else:
        example_channel_configs = [HyperParameters.num_channels_latent_space]

    for channel_idx_until_latents_are_kept in example_channel_configs:

        latent_representation_reduced = latent_representation.copy()

        latent_representation_reduced_to_be_replaced_shape = \
            latent_representation_reduced[:, :, :, channel_idx_until_latents_are_kept:].shape
        latent_representation_reduced[:, :, :, channel_idx_until_latents_are_kept:] = \
            HyperParameters.replacement_value(latent_representation_reduced_to_be_replaced_shape)

        num_elements_per_channel = np.prod(latent_representation_reduced.shape[1:-1])
        bpp = ((channel_idx_until_latents_are_kept + 1) * num_elements_per_channel
               * np.log2(HyperParameters.num_centers_quantization)) / num_pixels

        reconstructed_image = compression_gan.decode(latent_representation_reduced)[0]

        for metric in metrics:
            if bpp in metrics_values[metric.get_key()]:
                metrics_values[metric.get_key()][bpp] += float(metric(image, reconstructed_image).numpy())
            else:
                metrics_values[metric.get_key()][bpp] = float(metric(image, reconstructed_image).numpy())

        normalize_for_plot = lambda image: (image + 1.) / 2.
        reconstructed_image, image_plot = normalize_for_plot(reconstructed_image), normalize_for_plot(image)

        if idx < HyperParameters.num_examples_to_be_saved:
            plt.figure(dpi=300)
            plt.suptitle(f"{bpp:.6f} bpp | kept {channel_idx_until_latents_are_kept+1}/{num_channels} channels")
            plt.subplot(211)
            plt.title("Original")
            plt.imshow(np.squeeze(image_plot.numpy()))
            plt.subplot(212)
            plt.title("Reconstruction")
            plt.imshow(np.squeeze(reconstructed_image))

            plt.savefig(os.path.join(image_dir, str(idx) + f'_{str(bpp).replace(".", "_")}.png'))
            if show_plots:
                plt.show()
            plt.close()

for metric_name in metrics_values:
    for bpp, value in metrics_values[metric_name].items():
        metrics_values[metric_name][bpp] /= float(flic_dataset_test.cardinality().numpy())

with open(HyperParameters.compgan_path + '_eval_results.json', 'w') as json_handle:
    json_handle.write(json.dumps(metrics_values, indent=2))
