import os
from pathlib import Path
from typing import Optional, Union, Literal
import h5py
import numpy as np
import tensorflow as tf

TFDS_DEFAULT_DATA_DIR = os.path.join(Path.home(), 'tensorflow_datasets')
DEFAULT_DATASET_PATH = os.path.join(TFDS_DEFAULT_DATA_DIR,
                                    'nyu_depth_v2_labeled',
                                    'nyu_depth_v2_labeled.mat')

NUM_VALIDATION_ITEMS, NUM_TEST_ITEMS = 150, 250
SPLIT_SLICES = {
    'train': slice(0, -(NUM_VALIDATION_ITEMS + NUM_TEST_ITEMS)),
    'validation': slice(-(NUM_VALIDATION_ITEMS + NUM_TEST_ITEMS),
                        -NUM_TEST_ITEMS),
    'test': slice(-NUM_TEST_ITEMS, None),
    None: slice(0, None),
}

INCLUDED_ORIGINAL_DATASET_KEYS = ('images', 'labels', 'depths')
TF_DATASET_KEY = {s: s[:-1] for s in INCLUDED_ORIGINAL_DATASET_KEYS}


def load_nyu_depth_dataset_with_semantic_labels(
        split: Optional[Literal['train', 'validation', 'test']] = None,
        dataset_path: Optional[Union[str, os.PathLike]] = DEFAULT_DATASET_PATH) -> tf.data.Dataset:

    assert os.path.exists(dataset_path)
    dataset_as_dict = dict()

    with h5py.File(dataset_path) as h5_file_handle:
        for key in INCLUDED_ORIGINAL_DATASET_KEYS:
            data_as_array = np.array(h5_file_handle[key])[SPLIT_SLICES[split]]
            # transform to W x H x C
            if key == 'images':
                data_as_array = np.swapaxes(data_as_array, -1, -3)
            elif key in ['depths', 'labels']:
                data_as_array = np.swapaxes(data_as_array, -1, -2)

            dataset_as_dict[TF_DATASET_KEY[key]] = data_as_array

    return tf.data.Dataset.from_tensor_slices(dataset_as_dict)
