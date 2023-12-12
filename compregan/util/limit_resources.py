import os
import tensorflow as tf


def limit_resources(use_dynamic_gpu_memory_growth: bool = True, num_cpus_utilized: int = 6) -> None:
    os.environ["OMP_NUM_THREADS"] = str(num_cpus_utilized)

    if use_dynamic_gpu_memory_growth:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

            except RuntimeError as e:
                print(e)
