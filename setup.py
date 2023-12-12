import os
from setuptools import setup
from typing import Union


def read(filename: Union[os.PathLike, str]) -> str:
    return open(os.path.join(os.path.dirname(__file__), filename)).read()


setup(
    name="compregan",
    version="0.0.1",
    author="Jonas Hajek-Lellmann",
    author_email="jonas.hajek-lellmann@gmx.de",
    description="T B I",
    keywords="compression gan",
    packages=['compregan', 'test'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
    ],
)
