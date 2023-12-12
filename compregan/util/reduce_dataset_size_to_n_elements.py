import tensorflow as tf


def reduce_dataset_size_to_n_elements(dataset: tf.data.Dataset, n: int) -> tf.data.Dataset:
    batched_dataset = dataset.batch(n)
    for elements in batched_dataset:
        return tf.data.Dataset.from_tensor_slices(elements)
