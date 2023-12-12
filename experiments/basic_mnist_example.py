"""
Demonstrating the functionality of the CompressionGAN class on a simple MNIST example which
you can easily run on a CPU. Using just plain dense networks for the GAN related networks (E, G, D).
The classifier, i.e. the network delivering the semantic information for conditional GAN
training uses a simple convolutional NN (boilerplate copied right from the tf.keras docs).
Running this script takes about 5 minutes on a ten year old i7...
"""
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

from compregan.gan.compressiongan import CompressionGAN
from compregan.losses.auxillary_loss_terms import (
    AuxillaryLossTerms,
    CategorialAccuracy,
    CategorialCrossEntropy,
)
from compregan.losses.distortions import MSE, Lp
from compregan.models.blocks import dense_block
from compregan.models.quantizer import BinaryQuantizer
from compregan.util import (
    AllRandomSeedsSetter,
    save_pip_freeze_to_file,
    copy_this_file_to_directory,
)


# tf.config.run_functions_eagerly(True)  # useful when debugging


class HyperParameters:
    use_conditional_gan = True
    num_classes = 10
    force_retrain_classifier = False
    force_retrain_gan = True
    seed = 7

    train_decoder_with_auxillary_class_output = True

    experiment_name = (
        f'basic_mnist_dnn_example_{"c" if use_conditional_gan else ""}gan'
    )
    result_dir = os.path.join(os.getcwd(), experiment_name)

    img_shape = (28, 28, 1)

    noise_dim = 12
    bottleneck_dim = num_bits = 12

    num_epochs = 10
    batch_size = 128
    compgan_path = os.path.join(result_dir, f"{experiment_name}.pkl")


# ---------------------------------------------------------------------------
# Function Definitions


def get_trained_classifier():
    print(
        "Training an MNIST classifier for employing a conditional GAN later on!"
    )
    input = tf.keras.Input(shape=HyperParameters.img_shape)
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(
        input
    )
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(
        HyperParameters.num_classes, activation="softmax"
    )(x)
    classifier = keras.Model(inputs=input, outputs=output)

    (ds_train, ds_test), ds_info = tfds.load(
        "mnist",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255.0, label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    classifier.compile(
        tf.keras.optimizers.Adam(),
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    classifier.fit(ds_train, epochs=5, validation_data=ds_test)
    return classifier


def build_encoder(
    bottleneck_dim: int = HyperParameters.bottleneck_dim,
    use_conditional_gan: bool = HyperParameters.use_conditional_gan,
    activation: Optional[tf.keras.layers.Layer] = tf.keras.activations.sigmoid,
):
    input = keras.Input(shape=HyperParameters.img_shape)
    x = keras.layers.Flatten()(input)
    x = dense_block(x, 512)
    if use_conditional_gan:
        class_input = keras.Input(shape=(HyperParameters.num_classes,))
        x1 = dense_block(class_input, 32)
        x1 = dense_block(x1, 64)
        x = keras.layers.Concatenate()([x, x1])
    x = dense_block(x, 256)
    x = dense_block(x, 256)
    x = dense_block(x, 128)
    x = dense_block(x, 64)
    x = dense_block(x, 32)
    output = keras.layers.Dense(bottleneck_dim, activation=activation)(x)
    if use_conditional_gan:
        inputs = [input, class_input]
    else:
        inputs = input
    encoder = keras.Model(inputs=inputs, outputs=output)
    print("Encoder:\n")
    encoder.summary()
    return encoder


def build_decoder(
    bottleneck_dim: int = HyperParameters.bottleneck_dim,
    use_conditional_gan: bool = HyperParameters.use_conditional_gan,
    predict_class_on_aux_output: bool = HyperParameters.train_decoder_with_auxillary_class_output,
):
    input = keras.Input(shape=(bottleneck_dim + HyperParameters.noise_dim,))
    x = dense_block(input, 64)
    x = dense_block(x, 128)
    if use_conditional_gan:
        class_input = keras.Input(shape=(HyperParameters.num_classes,))
        x1 = dense_block(class_input, 32)
        x1 = dense_block(x1, 64)
        x = keras.layers.Concatenate()([x, x1])
    x = dense_block(x, 256)
    x2 = dense_block(x, 256)
    x = dense_block(x2, 512)
    x = keras.layers.Dense(
        HyperParameters.img_shape[0] * HyperParameters.img_shape[1],
        activation=keras.activations.tanh,
    )(x)
    outputs = [keras.layers.Reshape(target_shape=HyperParameters.img_shape)(x)]

    if use_conditional_gan:
        inputs = [input, class_input]
    else:
        inputs = input

    if predict_class_on_aux_output:
        class_pr = dense_block(x2, 64)
        class_pr = dense_block(class_pr, 128)
        class_pr = dense_block(class_pr, 128)
        class_pr = keras.layers.Dense(10, activation="softmax")(class_pr)
        outputs.append(class_pr)

    decoder = keras.Model(inputs=inputs, outputs=outputs)
    print("Decoder/Generator:\n")
    decoder.summary()
    return decoder


def build_discriminator(
    use_conditional_gan: bool = HyperParameters.use_conditional_gan,
):
    input = keras.Input(shape=HyperParameters.img_shape)
    x = keras.layers.Flatten()(input)
    x = dense_block(x, 64)
    if use_conditional_gan:
        class_input = keras.Input(shape=(HyperParameters.num_classes,))
        x = keras.layers.Concatenate()([x, class_input])
    x = dense_block(x, 64)
    x = dense_block(x, 24)
    output = keras.layers.Dense(1, activation=None)(x)
    if use_conditional_gan:
        inputs = [input, class_input]
    else:
        inputs = input
    discriminator = keras.Model(inputs=inputs, outputs=output)
    print("Discriminator:\n")
    discriminator.summary()
    return discriminator


# ---------------------------------------------------------------------------


if __name__ == "__main__":
    AllRandomSeedsSetter(HyperParameters.seed)

    if HyperParameters.use_conditional_gan:
        possible_trained_classifier_path = os.path.join(
            HyperParameters.result_dir, "classifier.h5"
        )
        if not HyperParameters.force_retrain_classifier and os.path.exists(
            possible_trained_classifier_path
        ):
            classifier = tf.keras.models.load_model(
                possible_trained_classifier_path
            )
        else:
            classifier = get_trained_classifier()
            tf.keras.models.save_model(
                classifier, possible_trained_classifier_path
            )
        conditional = classifier
    else:
        conditional = None

    compgan_path = os.path.join(
        HyperParameters.result_dir, f"{HyperParameters.experiment_name}.pkl"
    )

    if not HyperParameters.force_retrain_gan and os.path.exists(compgan_path):
        compression_gan = CompressionGAN.load_from_file(compgan_path)
    else:
        compression_gan = CompressionGAN(
            encoder=build_encoder(),
            decoding_generator=build_decoder(),
            discriminator=build_discriminator(),
            quantizer=BinaryQuantizer(HyperParameters.num_bits),
            noise_prior=lambda shape: tf.random.normal(
                shape, mean=0.0, stddev=1.0
            ),
            noise_dim=HyperParameters.noise_dim,
            conditional=conditional,
            loss_terms=(MSE(20.0), Lp(1, 20.0)),
            codec_data_key="image",
            auxillary_loss_terms=AuxillaryLossTerms(
                [[CategorialCrossEntropy()]]
            ),
            auxillary_metrics=AuxillaryLossTerms([[CategorialAccuracy()]]),
        )

        def preprocess_func(dataset_item):
            image = dataset_item["image"]
            image = tf.cast(image, tf.float32)
            image = ((image / 255.0) - 0.5) * 2.0
            dataset_item["image"] = image
            label = tf.one_hot(dataset_item["label"], 10)
            dataset_item["label"] = label
            return dataset_item

        mnist_train_dataset = tfds.load("mnist", split="train").map(
            preprocess_func
        )
        mnist_test_dataset = tfds.load("mnist", split="test").map(
            preprocess_func
        )

        compression_gan.train(
            mnist_train_dataset,
            num_epochs=HyperParameters.num_epochs,
            batch_size=HyperParameters.batch_size,
            validation_data=mnist_test_dataset,
        )

    if not os.path.isdir(HyperParameters.result_dir):
        os.mkdir(HyperParameters.result_dir)

    compression_gan.save_to_file(
        os.path.join(
            HyperParameters.result_dir,
            f"{HyperParameters.experiment_name}.pkl",
        )
    )
    save_pip_freeze_to_file(
        os.path.join(HyperParameters.result_dir, "dependencies.txt")
    )
    copy_this_file_to_directory(HyperParameters.result_dir)

    num_examples_to_be_saved = 100
    show_plots = True
    examples_dir = os.path.join(HyperParameters.result_dir, "examples")
    if not os.path.isdir(examples_dir):
        os.mkdir(examples_dir)

    for idx, dataset_item in enumerate(mnist_test_dataset):
        image = dataset_item["image"]
        image_with_batch_dim = np.expand_dims(image.numpy(), 0)
        latent_representation = compression_gan.encode_data(
            image_with_batch_dim
        )
        condition = (
            conditional(image_with_batch_dim)
            if HyperParameters.use_conditional_gan
            else None
        )
        decoder_output = compression_gan.decode(
            latent_representation,
            conditioning_information=condition,
            return_np_array=True,
        )

        if HyperParameters.train_decoder_with_auxillary_class_output:
            reconstructed_image, class_probs = decoder_output
        else:
            reconstructed_image = decoder_output
            class_probs = None

        plt.figure()
        subplot_int = 311 if class_probs is not None else 211
        plt.subplot(subplot_int)
        plt.title("Original")
        plt.imshow(np.squeeze(image.numpy()))
        plt.subplot(subplot_int + 1)
        plt.title("Reconstruction")
        plt.imshow(np.squeeze(reconstructed_image))
        if class_probs is not None:
            plt.subplot(subplot_int + 2)
            plt.plot(np.arange(10), np.squeeze(class_probs))
        plt.savefig(os.path.join(examples_dir, str(idx) + ".png"))
        if show_plots:
            plt.show()
        else:
            plt.close()

        if idx >= num_examples_to_be_saved:
            break
