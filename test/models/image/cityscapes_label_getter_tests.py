import unittest

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from compregan.data.util import extract_and_preprocess_images
from compregan.models.image.cityscapes_label_getter import CityScapesBooleanSemanticMapGetter


class CityScapesLabelGetterTests(unittest.TestCase):
    _dataset_preprocesing_kwargs = {'image_key': 'image_left',
                                    'target_size': (48, 80, 3),
                                    'preprocess_func': lambda image: image / 255.,
                                    'image_dataset_dtype': tf.float32,
                                    'verbose': False}

    _testing_dataset_cardinality = 3
    _plot = False

    @staticmethod
    def _reduce_dataset_size_to_n_elements(dataset: tf.data.Dataset, n: int):
        batched_dataset = dataset.batch(n)
        for elements in batched_dataset:
            return tf.data.Dataset.from_tensor_slices(elements)

    def _get_testing_dataset(self) -> tf.data.Dataset:
        return self._reduce_dataset_size_to_n_elements(tfds.load('cityscapes', split='validation'),
                                                       self._testing_dataset_cardinality)

    def test_label_getter_returns_well_defined_labels(self):
        cityscapes_val_orig = self._get_testing_dataset()
        cityscapes_val_preprocessed = extract_and_preprocess_images(cityscapes_val_orig,
                                                                    **self._dataset_preprocesing_kwargs)

        for reduce_classes in [True, False]:

            cityscapes_label_getter = CityScapesBooleanSemanticMapGetter(
                cityscapes_val_orig,
                cityscapes_val_preprocessed,
                reduce_all_classes_to_binary_heatmap=reduce_classes,
                verbose=False)

            for image in cityscapes_val_preprocessed:
                semantic_map = np.squeeze(cityscapes_label_getter(tf.expand_dims(image, 0)).numpy())

                semantic_labels_present = 0. < semantic_map.mean() < 1.
                self.assertTrue(semantic_labels_present)

                for element in np.ravel(semantic_map):
                    is_binary = element == 0. or element == 1.
                    self.assertTrue(is_binary)

                if not reduce_classes and self._plot:
                    import matplotlib.pyplot as plt
                    plt.subplot(121)
                    plt.imshow(image)
                    plt.subplot(122)
                    plt.imshow(semantic_map[:, :, 3:6])
                    plt.show()


if __name__ == '__main__':
    unittest.main()
