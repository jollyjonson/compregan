"""
Demonstrating the functionality of scalable coding using
SelectiveDropout on the basic MNIST example.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
import tqdm

from compregan.data.util import extract_and_preprocess_images
from compregan.gan.compressiongan import CompressionGAN
from compregan.losses.distortions import MSE, Lp
from compregan.losses.gan_losses import LeastSquares
from compregan.models.quantizer import Quantizer, BinaryQuantizer
from compregan.models.layers.selective_dropout import SelectiveDropout
from compregan.util import GlobalRandomSeed, save_pip_freeze_to_file, copy_this_file_to_directory, debugger_connected
from basic_mnist_example import HyperParameters as OriginalExperimentHyperParameters
from basic_mnist_example import get_trained_classifier, build_encoder, build_decoder, build_discriminator


# tf.config.run_functions_eagerly(True)  # useful when debugging


class HyperParameters(OriginalExperimentHyperParameters):
    use_conditional_gan = False
    force_retrain_gan = True
    experiment_name = f'mnist_selective_dropout_example_{"c" if use_conditional_gan else ""}gan'
    result_dir = os.path.join(os.getcwd(), experiment_name)

    overall_probability_of_selective_dropout = 0.6
    ratio_of_never_dropped_indices = 0.2
    replacement_value = -1
    bottleneck_dim = num_bits = 48


def build_dropout_quantizer() -> tf.keras.Model:
    selective_dropout_layer = SelectiveDropout(HyperParameters.ratio_of_never_dropped_indices,
                                               HyperParameters.overall_probability_of_selective_dropout,
                                               replacement_value=HyperParameters.replacement_value)
    quantizer = BinaryQuantizer(HyperParameters.bottleneck_dim)
    return keras.Sequential([quantizer, selective_dropout_layer])


if __name__ == '__main__':

    GlobalRandomSeed(HyperParameters.seed)

    if HyperParameters.use_conditional_gan:
        possible_trained_classifier_path = os.path.join(HyperParameters.result_dir, 'classifier.h5')
        if not HyperParameters.force_retrain_classifier and os.path.exists(possible_trained_classifier_path):
            classifier = tf.keras.models.load_model(possible_trained_classifier_path)
        else:
            classifier = get_trained_classifier()
            tf.keras.models.save_model(classifier, possible_trained_classifier_path)
        conditional = classifier
    else:
        conditional = None

    compgan_path = os.path.join(HyperParameters.result_dir, f'{HyperParameters.experiment_name}.pkl.tar.gz')

    mnist_test = tfds.load('mnist', split='test')
    mnist_test_images_dataset = extract_and_preprocess_images(
        tf_dataset_with_labels=mnist_test,
        preprocess_func=lambda image: image / 255.,  # [0, 255] -> [0, 1]
        image_dataset_dtype=tf.float32,
    )

    if not HyperParameters.force_retrain_gan and os.path.exists(compgan_path):
        print("Loading Models from file...")
        compression_gan = CompressionGAN.load_from_file(compgan_path)
    else:
        compression_gan = CompressionGAN(
            encoder=build_encoder(HyperParameters.bottleneck_dim, HyperParameters.use_conditional_gan, activation=None),
            decoding_generator=build_decoder(HyperParameters.bottleneck_dim, HyperParameters.use_conditional_gan),
            discriminator=build_discriminator(HyperParameters.use_conditional_gan),
            quantizer=build_dropout_quantizer(),
            noise_prior=lambda shape: tf.random.normal(shape, mean=0., stddev=1.),
            noise_dim=HyperParameters.noise_dim,
            conditional=conditional,
            gan_loss=LeastSquares(),
            loss_terms=(MSE(20.), Lp(1, 20.)),
            encoder_optimizer=keras.optimizers.Adam(),
            decoding_generator_optimizer=keras.optimizers.Adam(),
            discriminator_optimizer=keras.optimizers.Adam(),
        )

        mnist_train_dataset = tfds.load('mnist', split='train')
        mnist_train_images_dataset = extract_and_preprocess_images(
            tf_dataset_with_labels=mnist_train_dataset,
            preprocess_func=lambda image: image / 255.,  # [0, 255] -> [0, 1]
            image_dataset_dtype=tf.float32,
        )

        compression_gan.train(
            mnist_train_images_dataset,
            num_epochs=HyperParameters.num_epochs,
            batch_size=HyperParameters.batch_size,
            validation_data=mnist_test_images_dataset,
        )

    if not os.path.isdir(HyperParameters.result_dir):
        os.mkdir(HyperParameters.result_dir)

    compression_gan.save_to_file(os.path.join(HyperParameters.result_dir, f'{HyperParameters.experiment_name}.pkl'))
    save_pip_freeze_to_file(os.path.join(HyperParameters.result_dir, 'dependencies.txt'))
    copy_this_file_to_directory(HyperParameters.result_dir)

    num_examples_to_be_saved = 100
    show_plots = True if debugger_connected() else False
    examples_dir = os.path.join(HyperParameters.result_dir, 'examples')
    if not os.path.isdir(examples_dir):
        os.mkdir(examples_dir)

    num_never_dropped_bits = int(round(HyperParameters.num_bits * HyperParameters.ratio_of_never_dropped_indices))

    for idx, image in tqdm.tqdm(enumerate(mnist_test_images_dataset)):
        image_with_batch_dim = np.expand_dims(image.numpy(), 0)

        for bit_idx, num_bits in enumerate(np.linspace(HyperParameters.num_bits, num_never_dropped_bits, 6)):

            num_bits = int(np.ceil(num_bits))
            num_pixels = np.prod(HyperParameters.img_shape)
            bpp = (num_bits * 5) / num_pixels

            latent_representation = compression_gan.encode_data(image_with_batch_dim)
            latent_representation[0, num_bits:] = HyperParameters.replacement_value

            if HyperParameters.use_conditional_gan:
                bpp = (num_bits + np.log2(HyperParameters.num_classes)) / num_pixels
            else:
                bpp = num_bits / num_pixels

            print(f"{idx} @ {bpp if bit_idx != 0 else f'{bpp}_full_scale'}: ", latent_representation)

            condition = conditional(image_with_batch_dim) if HyperParameters.use_conditional_gan else None
            reconstructed_image = compression_gan.decode(latent_representation,
                                                         conditioning_information=condition,
                                                         return_np_array=True)

            plt.figure()
            plt.subplot(121)
            plt.title("Original")
            plt.imshow(np.squeeze(image.numpy()))
            plt.subplot(122)
            plt.title("Reconstruction")
            plt.imshow(np.squeeze(reconstructed_image))
            plt.savefig(os.path.join(examples_dir, str(idx) + f'_{str(bpp).replace(".", "_")}.png'))
            if show_plots:
                plt.show()
            else:
                plt.close()

        if idx >= num_examples_to_be_saved:
            break
