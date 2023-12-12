import os
import subprocess
import warnings

import tensorflow as tf
from time import sleep

import tqdm


def wait_for_gpu(polling_interval_in_s: float = 30, verbose: bool = True) -> None:
    """
    Pause program execution until no more python processes are seen occupying the GPU. Simply return when running on
    a machine without GPU.
    """
    if not tf.test.is_gpu_available():
        warnings.warn("Could not detect GPU, skipping waiting for it!")
        return

    else:
        if verbose:
            progbar = tqdm.tqdm(desc=f"Waiting for GPU with polling interval {polling_interval_in_s}s", unit=" polls")
        while True:

            with subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE) as proc:
                for line in map(lambda byte_str: byte_str.decode('utf-8'), proc.stdout.readlines()):
                    gpu_occupied = 'python' in line and not str(os.getpid()) in line
                    if gpu_occupied:
                        break

            if not gpu_occupied:
                return
            else:
                if verbose:
                    progbar.update(1)
                sleep(polling_interval_in_s)
