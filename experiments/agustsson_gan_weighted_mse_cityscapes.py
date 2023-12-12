import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from compregan.data.util import extract_and_preprocess_images
from compregan.gan.compressiongan import CompressionGAN
from compregan.losses.feature_matching_loss import FeatureMatchingLoss
from compregan.losses.gan_losses import LeastSquares
from compregan.losses.semantically_weighted_distortion import (
    SemanticallyWeightedDistortion,
)
from compregan.losses.vgg_perceptual_loss import VGGPerceptualLoss
from compregan.models.image.agustsson import AgustssonImageModelsBuilder
from compregan.models.image.cityscapes_label_getter import (
    CityScapesBooleanSemanticMapGetter,
)
from compregan.models.quantizer import Quantizer
from compregan.util import (
    save_pip_freeze_to_file,
    debugger_connected,
    AllRandomSeedsSetter,
    copy_this_file_to_directory,
)


# wait_for_gpu()


class HyperParameters:
    force_retrain_gan = True
    seed = 1999

    experiment_name = os.path.basename(__file__).replace(".py", "")
    result_dir = os.path.join(os.getcwd(), experiment_name)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    compgan_path = os.path.join(result_dir, f"{experiment_name}.pkl")

    learning_rate = 0.0002

    debug = debugger_connected()

    num_filters_per_layer_encoder = (
        (60, 120, 240, 480, 960) if not debug else (1, 3, 6)
    )
    num_filters_per_layer_decoder = (
        (480, 240, 120, 60) if not debug else (6, 6, 3)
    )

    img_shape = (512, 1024, 3) if not debug else (48, 80, 3)

    num_channels_latent_space = 2 if debug else 4
    num_noise_channels = 0 if debug else 0

    if debug:
        num_epochs = 1
        steps_per_epoch = 3
        batch_size = 1
        num_examples_to_be_saved = 5
    else:
        num_training_steps = 150000
        num_epochs = 60
        batch_size = 1
        num_items_per_epoch = int(num_training_steps / num_epochs)
        steps_per_epoch = int(num_items_per_epoch / batch_size)
        num_examples_to_be_saved = 200

    num_validation_examples = 240
    num_validation_steps = int(num_validation_examples / batch_size)
    num_centers_quantization = 5


AllRandomSeedsSetter(HyperParameters.seed)

model_builder = AgustssonImageModelsBuilder(
    *HyperParameters.img_shape,
    use_patch_gan=True,
    num_channels_latent_space=HyperParameters.num_channels_latent_space,
    num_noise_channels=HyperParameters.num_noise_channels,
)

num_pixels = np.prod(HyperParameters.img_shape)
num_bits_latent_space = np.prod(
    model_builder.encoder.output_shape[1:]
) * np.log2(HyperParameters.num_centers_quantization)
print(f"Compressing images to {num_bits_latent_space / num_pixels} bpp")

dataset_preprocesing_kwargs = {
    "image_key": "image_left",
    "target_size": HyperParameters.img_shape,
    "preprocess_func": lambda image: image / 255.0,
    "image_dataset_dtype": tf.float32,
}

cityscapes_train_orig = tfds.load(
    "cityscapes", split="train"
)  # type: tf.data.Dataset
cityscapes_train_preprocessed = extract_and_preprocess_images(
    cityscapes_train_orig, **dataset_preprocesing_kwargs
)

cityscapes_val_orig = tfds.load("cityscapes", split="validation")
cityscapes_val_preprocessed = extract_and_preprocess_images(
    cityscapes_val_orig, **dataset_preprocesing_kwargs
)

cityscapes_complete_orig = cityscapes_train_orig.concatenate(
    cityscapes_val_orig
)
cityscapes_complete_preprocessed = cityscapes_train_preprocessed.concatenate(
    cityscapes_val_preprocessed
)

label_getter = CityScapesBooleanSemanticMapGetter(
    cityscapes_complete_orig, cityscapes_complete_preprocessed
)

compression_gan = CompressionGAN(
    encoder=model_builder.encoder,
    decoding_generator=model_builder.decoding_generator,
    discriminator=model_builder.discriminator,
    loss_terms=(
        SemanticallyWeightedDistortion(100.0),
        VGGPerceptualLoss(2.0, HyperParameters.img_shape),
        FeatureMatchingLoss(
            model_builder.discriminator, 10.0, decay_factor=5e-4
        ),
    ),
    quantizer=Quantizer(
        model_builder.encoder.output_shape[1:],
        num_centers=HyperParameters.num_centers_quantization,
    ),
    gan_loss=LeastSquares(),
    noise_prior=lambda shape: tf.random.normal(shape, mean=0.0, stddev=1.0),
    noise_dim=(
        *model_builder.encoder.output_shape[1:-1],
        HyperParameters.num_noise_channels,
    ),
    conditional=label_getter,
    use_conditional=False,
    encoder_optimizer=tf.keras.optimizers.Adam(HyperParameters.learning_rate),
    decoding_generator_optimizer=tf.keras.optimizers.Adam(
        HyperParameters.learning_rate
    ),
    discriminator_optimizer=tf.keras.optimizers.Adam(
        HyperParameters.learning_rate
    ),
    quantizer_optimizer=None,
    metrics=None,
)

compression_gan.train(
    training_data=cityscapes_train_preprocessed,
    num_epochs=HyperParameters.num_epochs,
    batch_size=HyperParameters.batch_size,
    shuffle=True,
    validation_data=None
    if HyperParameters.debug
    else cityscapes_val_preprocessed,
    steps_per_epoch=HyperParameters.steps_per_epoch,
)

compression_gan.save_to_file(HyperParameters.compgan_path)
copy_this_file_to_directory(HyperParameters.result_dir)
save_pip_freeze_to_file(
    os.path.join(HyperParameters.result_dir, "dependencies.txt")
)

city_scapes_dataset_test = extract_and_preprocess_images(
    tfds.load("cityscapes", split="test"), **dataset_preprocesing_kwargs
)

show_plots = not HyperParameters.debug
examples_dir = os.path.join(HyperParameters.result_dir, "examples")
if not os.path.isdir(examples_dir):
    os.mkdir(examples_dir)

for idx, image in enumerate(city_scapes_dataset_test):
    image_with_batch_dim = np.expand_dims(image.numpy(), 0)
    latent_representation = compression_gan.encode_data(image_with_batch_dim)
    reconstructed_image = compression_gan.decode(
        latent_representation,
        conditioning_information=None,
        return_np_array=True,
    )

    plt.figure()
    plt.subplot(211)
    plt.title("Original")
    plt.imshow(np.squeeze(image.numpy()))
    plt.subplot(212)
    plt.title("Reconstruction")
    plt.imshow(np.squeeze(reconstructed_image))
    plt.savefig(os.path.join(examples_dir, str(idx) + ".png"))
    if show_plots:
        plt.show()
    else:
        plt.close()

    if idx >= HyperParameters.num_examples_to_be_saved:
        break
