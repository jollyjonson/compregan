import io
import os

from setuptools import setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely."""
    with io.open(
            os.path.join(os.path.dirname(__file__), *paths),
            encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


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
    install_requires=read_requirements("requirements.txt"),
    extras_require={"test": read_requirements("requirements-test.txt")},
)
