import os.path as path
import tempfile
from tempfile import mkdtemp
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf
from tqdm import tqdm


def extract_and_preprocess_images(
        tf_dataset_with_labels: tf.data.Dataset,
        preprocess_func: Optional[Callable] = None,
        image_dataset_dtype: Optional[tf.dtypes.DType] = None,
        verbose: bool = True,
        image_key: str = 'image',
        target_size: Optional[Tuple[int, int, int]] = None) -> tf.data.Dataset:

    for element in tf_dataset_with_labels:
        image = element[image_key]
        image_shape = image.shape.as_list()
        image_dtype = image.dtype
        break

    dtype = image_dataset_dtype if image_dataset_dtype else image_dtype

    if target_size is not None:
        image_shape = target_size

    images_np = np.zeros(shape=(tf_dataset_with_labels.cardinality().numpy(), *image_shape), dtype=dtype.as_numpy_dtype)

    progress_bar_wrapper = tqdm(tf_dataset_with_labels) if verbose else tf_dataset_with_labels
    if verbose:
        progress_bar_wrapper.set_description(f"Extracing and preprocessing images from dataset")

    for idx, element in enumerate(progress_bar_wrapper):
        image = element[image_key]
        if target_size is not None:
            image = tf.image.resize(image, target_size[:-1])
        assert image.shape.as_list() == list(image_shape)
        image_np = image.numpy().astype(dtype.as_numpy_dtype)
        if preprocess_func:
            image_np = preprocess_func(image_np)
        images_np[idx, :] = image_np

    image_dataset = tf.data.Dataset.from_tensor_slices(images_np)
    return image_dataset
