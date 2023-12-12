import os
import shutil
import unittest

import numpy as np
import tensorflow as tf

from compregan.gan.compressiongan import CompressionGAN
from compregan.models.image.agustsson import AgustssonImageModelsBuilder


class CompressionGANTests(unittest.TestCase):
    @staticmethod
    def test_saving_and_loading_works():
        model_builder = AgustssonImageModelsBuilder(
            width=10,
            height=10,
            num_channels=3,
            num_channels_latent_space=1,
            num_filters_per_layer_encoder=(1, 2),
            num_filters_per_layer_decoder=(1, 2),
            num_filters_in_residual_blocks=2,
            num_residual_blocks_decoder=2,
            num_filter_per_layer_discriminator=(2, 2),
        )
        compression_gan = CompressionGAN(
            model_builder.encoder,
            model_builder.decoding_generator,
            model_builder.discriminator,
        )
        tmp_dir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tmpCompressionGANTests_test_saving_and_loading_works",
        )
        os.mkdir(tmp_dir_path)
        try:
            compression_gan_save_path = os.path.join(
                tmp_dir_path, "compressiongan"
            )
            compression_gan.save_to_file(compression_gan_save_path)
            compression_gan_loaded = CompressionGAN.load_from_file(
                compression_gan_save_path
            )  # type: CompressionGAN

            # check if all models work as intended, forward pass without error is enough
            _ = compression_gan_loaded.encode_data(
                np.random.randn(1, *model_builder.encoder.input_shape[1:])
            )
            _ = compression_gan_loaded.decode(
                np.random.randn(1, *model_builder.encoder.output_shape[1:])
            )
            _ = compression_gan_loaded._discriminator(
                np.random.randn(1, *model_builder.encoder.input_shape[1:])
            )

            # check if the reattached models work correctly
            _ = compression_gan.encode_data(
                np.random.randn(1, *model_builder.encoder.input_shape[1:])
            )
            _ = compression_gan.decode(
                np.random.randn(1, *model_builder.encoder.output_shape[1:])
            )
            _ = compression_gan._discriminator(
                np.random.randn(1, *model_builder.encoder.input_shape[1:])
            )

            # check if weights are correct
            for model_expected, model_actual in zip(
                [
                    compression_gan._encoder,
                    compression_gan._decoding_generator,
                    compression_gan._discriminator,
                ],
                [
                    compression_gan_loaded._encoder,
                    compression_gan_loaded._decoding_generator,
                    compression_gan_loaded._discriminator,
                ],
            ):
                for weights_expected, weights_actual in zip(
                    model_expected.weights, model_actual.weights
                ):
                    tf.assert_equal(weights_expected, weights_actual)

        except Exception as e:
            raise e

        finally:
            shutil.rmtree(tmp_dir_path)


if __name__ == "__main__":
    unittest.main()
