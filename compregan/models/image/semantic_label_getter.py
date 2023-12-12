import os.path
from typing import Tuple, Callable, Optional, Union
import atexit

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from compregan.util import get_tmp_file_dir, cleanup_tmp_dir
from compregan.models.image.semantic_labels import SemanticLabels
from compregan.gan.conditional import Components, Conditional


HASH_DTYPE = tf.float64


class FromDatasetSemanticMapGetter(Conditional):

    @property
    def needed_components(self) -> Tuple[Components, ...]:
        return (Components.CompleteData,)

    @tf.function
    def __call__(self, full_data_batch: tf.data.Dataset, **kwargs) -> tf.Tensor:
        return full_data_batch['label']


@tf.function
def get_image_hash(image: tf.Tensor) -> tf.Tensor:
    """Simply using the sum of a row of the image as hash value"""
    for idx, hashed_row_divider in enumerate([2, 3, 4, 5]):
        hashed_row_idx = int(image.shape[0] / 2)
        if idx == 0:
            hashed_row = tf.gather_nd(image, [[hashed_row_idx]])
        else:
            hashed_row += tf.gather_nd(image, [[hashed_row_idx]])
    return tf.ones(1, dtype=HASH_DTYPE) * tf.reduce_sum(tf.cast(hashed_row, HASH_DTYPE))


class HashBasedSemanticMapGetter:
    """
    Mock model for getting perfect semantic maps from images. Works by computing the sum of the pixels
    in the middle row and column of the picture as pseudo-hash-values and mapping these hashes to the sought
    semantic maps.

    Parameters
    ----------
    original_dataset_instance: tf.data.Dataset
        Instance of the original dataset containing images in their original resolution and
        semantic maps, as returned by `tfds.load`.
    preprocessed_dataset_instance: tf.data.Dataset
        Preprocessed instance, i.e. containing only images
    included_classes: Tuple[SemanticLabels, ...]
        Which classes to include in the binary semantic map.
    """

    def __init__(self,
                 original_dataset_instance: tf.data.Dataset,
                 preprocessed_dataset_instance: tf.data.Dataset,
                 included_classes: Optional[Tuple[SemanticLabels, ...]],
                 reduce_all_classes_to_binary_heatmap: bool = True,
                 hashing_function: Callable = get_image_hash,
                 verbose: bool = True,
                 use_file_cache: bool = False,
                 semantic_label_key: str = 'segmentation_label',
                 image_key: str = 'image_left',
                 ):
        self._semantic_label_key = semantic_label_key
        self._image_key = image_key

        self._orig_image_shape = self._get_img_shape(original_dataset_instance)
        self._img_shape = self._get_img_shape(preprocessed_dataset_instance)
        self._included_classes = included_classes
        self._reduce_all_classes_to_binary_heatmap = reduce_all_classes_to_binary_heatmap
        self._shape_semantic_maps = (*self._img_shape.as_list()[:-1],) \
            if self._reduce_all_classes_to_binary_heatmap else \
            (*self._img_shape.as_list()[:-1], len(self._included_classes))
        self._hashing_func = hashing_function
        self._verbose = verbose

        self._epsilon = tf.ones(1, dtype=HASH_DTYPE) * 1e-9

        assert original_dataset_instance.cardinality() == preprocessed_dataset_instance.cardinality()
        self._num_items = int(preprocessed_dataset_instance.cardinality())

        self._use_file_cache = use_file_cache
        semantic_mappings_shape = (self._num_items, *self._shape_semantic_maps)
        if self._use_file_cache:
            self._tmp_file_path = os.path.join(get_tmp_file_dir(),
                                               f'HashBasedSemanticMapGetter{hash(original_dataset_instance)}.npy')
            atexit.register(self._cleanup_file_cache)
            self._semantic_mappings = np.memmap(self._tmp_file_path, mode='w+', dtype=bool,
                                                shape=semantic_mappings_shape)
        else:
            self._semantic_mappings = np.zeros(dtype=bool, shape=semantic_mappings_shape)

        self._hash_to_semantic_map_idx = self._create_hash_to_semantic_map_mapping(original_dataset_instance,
                                                                                   preprocessed_dataset_instance)

        self._assert_hashes_are_unique_with_respect_to_epsilon()

    @tf.function
    def __call__(self, batch_of_codec_data: tf.Tensor) -> tf.Tensor:
        semantic_maps = tf.zeros((batch_of_codec_data.shape[0], *self._shape_semantic_maps), dtype=tf.float32)
        num_imag_in_batch = batch_of_codec_data.shape[0]
        for image_idx_in_batch in range(num_imag_in_batch):
            image = tf.squeeze(tf.gather_nd(batch_of_codec_data, [[image_idx_in_batch]]))
            image_hash_value = self._hashing_func(image)
            semantic_map_idx = tf.where(tf.abs(self._hash_to_semantic_map_idx - image_hash_value) <= self._epsilon)
            semantic_map = tf.gather_nd(self._semantic_mappings, semantic_map_idx)
            semantic_maps = tf.tensor_scatter_nd_update(semantic_maps, [[image_idx_in_batch]],
                                                        tf.cast(semantic_map, tf.float32))
        return semantic_maps

    def _get_img_shape(self, preprocessed_cityscapes_dataset: tf.data.Dataset) -> tf.TensorShape:
        for item in preprocessed_cityscapes_dataset:
            try:
                return item[self._image_key].shape
            except TypeError:
                return item.shape

    def _get_img(self, item_from_preprocessed_dataset: Union[tf.data.Dataset, tf.Tensor]) -> tf.Tensor:
        try:
            return item_from_preprocessed_dataset[self._image_key]
        except TypeError:
            return item_from_preprocessed_dataset

    def _create_hash_to_semantic_map_mapping(self,
                                             original_dataset_instance: tf.data.Dataset,
                                             preprocessed_dataset_instance: tf.data.Dataset):

        hash_map = tf.zeros(original_dataset_instance.cardinality(), dtype=HASH_DTYPE)

        iterable = enumerate(zip(original_dataset_instance, preprocessed_dataset_instance))
        progress_bar_wrapper = tqdm(iterable, desc="Extracting semantic maps and creating hash table",
                                    total=self._num_items) if self._verbose else iterable

        for item_idx, (item_orig, item_preprocessed) in progress_bar_wrapper:
            if self._reduce_all_classes_to_binary_heatmap:
                semantic_map_tmp = np.zeros((*self._orig_image_shape[:-1], 1), dtype=float)  # float needed for resampling
                for class_idx in self._included_classes:
                    semantic_map_tmp[np.where(item_orig[self._semantic_label_key].numpy() == class_idx)] = 255.
                semantic_map_tmp = np.squeeze(tf.image.resize(semantic_map_tmp, self._img_shape[:-1]).numpy())
                semantic_map = np.zeros_like(semantic_map_tmp, dtype=bool)
                semantic_map[np.where(semantic_map_tmp > 128.)] = True
            else:
                semantic_map = np.zeros(self._shape_semantic_maps, dtype=bool)
                for class_idx_sc, class_idx_cityscapes in enumerate(self._included_classes):
                    semantic_map_tmp = np.zeros((*self._orig_image_shape[:-1], 1),
                                                dtype=float)
                    semantic_map_tmp[np.where(item_orig[self._semantic_label_key].numpy() == class_idx_cityscapes)] = 255.
                    semantic_map_tmp = tf.image.resize(semantic_map_tmp, self._shape_semantic_maps[:-1]).numpy()
                    semantic_map_tmp2 = np.zeros_like(np.squeeze(semantic_map_tmp), dtype=bool)
                    semantic_map_tmp2[np.where(np.squeeze(semantic_map_tmp) > 128.)] = True
                    semantic_map[:, :, class_idx_sc] = semantic_map_tmp2

            self._semantic_mappings[item_idx] = semantic_map
            hash_map = tf.tensor_scatter_nd_update(hash_map,
                                                   [[item_idx]],
                                                   self._hashing_func(self._get_img(item_preprocessed)))
            if self._use_file_cache:
                self._semantic_mappings.flush()

        return hash_map

    @staticmethod
    def _cleanup_file_cache():
        cleanup_tmp_dir()

    def _assert_hashes_are_unique_with_respect_to_epsilon(self):
        hashes = self._hash_to_semantic_map_idx.numpy()
        epsilon = self._epsilon.numpy()
        smallest_diff = float('inf')
        for idx_x, value_x in enumerate(hashes):
            for idx_y, value_y in enumerate(hashes):
                if idx_x == idx_y:
                    continue
                diff = np.abs(value_x - value_y)
                assert diff > epsilon
                if diff < smallest_diff:
                    smallest_diff = diff
        if self._verbose:
            print(f'[CityScapesBooleanSemanticMapGetter] smallest diff in hashes: {smallest_diff}')
