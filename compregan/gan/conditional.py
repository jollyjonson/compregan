from abc import ABCMeta, abstractmethod
from typing import Tuple

import tensorflow as tf

from .components import Components


class Conditional(metaclass=ABCMeta):
    @property
    @abstractmethod
    def needed_components(self) -> Tuple[Components, ...]:
        """
        Components needed to compute or assemble the conditioning information.
        """

    @tf.function
    @abstractmethod
    def __call__(self, *components, **kwargs) -> tf.Tensor:
        """
        Compute or assemble the conditioning information
        from the needed components.
        """
