import unittest
import numpy as np
from compregan.models.image.agustsson import AgustssonImageModelsBuilder


class AgustssonModelsTest(unittest.TestCase):
    kwargs = {
        "width": 96,
        "height": 32,
        "num_channels": 3,
        "downsampling_factor": 16,
        "num_channels_latent_space": 2,
    }

    image_shape = (kwargs['width'], kwargs['height'], kwargs['num_channels'])

    model_builder = AgustssonImageModelsBuilder(**kwargs)

    encoder = model_builder.encoder
    decoding_generator = model_builder.decoding_generator
    discriminator = model_builder.discriminator

    batch_size = 2
    latent_representation = np.random.randn(batch_size, *encoder.output_shape[1:])

    def test_encoder_correct_input_shape(self):
        self.assertEqual(self.image_shape, self.encoder.input_shape[1:])

    def test_encoder_correct_output_shape(self):
        self.assertEqual(self.encoder.output_shape[1:],
                         (np.ceil(self.kwargs['width'] / self.kwargs['downsampling_factor']),
                          np.ceil(self.kwargs['height'] / self.kwargs['downsampling_factor']),
                          self.kwargs['num_channels_latent_space']))

    def test_encoder_forward_pass(self):
        latent_representation = self.encoder(np.random.randn(self.batch_size, *self.encoder.input_shape[1:]))
        self.assertEqual(self.latent_representation.shape, latent_representation.shape)

    def test_decoding_generator_forward_pass(self):
        reconstructed_image = self.decoding_generator(self.latent_representation)
        self.assertEqual(reconstructed_image.shape, (self.batch_size, *self.image_shape))

    def test_discriminator_forward_pass(self):
        discr_out = self.discriminator(np.random.randn(self.batch_size, *self.encoder.input_shape[1:]))
        self.assertEqual(discr_out.shape, (self.batch_size, 1))


if __name__ == '__main__':
    unittest.main()
