class AllRandomSeedsSetter:
    """
    Making experiments reproducible by setting all random seeds known
    to Python/NumPy/TensorFlow.
    Instantiate this class before performing any non-deterministic
    operations, if experiments should be reproducible!
    """

    def __init__(self, seed: int):
        import os
        import random

        import numpy as np

        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(seed)
        np.random.seed(seed)
        import tensorflow as tf

        tf.random.set_seed(seed)
