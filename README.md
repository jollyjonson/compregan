# CompreGAN
[![CI](https://github.com/jollyjonson/compregan/actions/workflows/main.yml/badge.svg)](https://github.com/jollyjonson/compregan/actions/workflows/main.yml)

This repository contains some of my ventures into the realm of image coding using Generative Adversarial Networks (GANs).
Most of this code was written in late 2021 and I unfortunately have not had the chance to work on the topic further.
This work was mainly inspired by a paper by Agustsson et al. called [Generative Adversarial Networks for Extreme Learned Image Compression](https://openaccess.thecvf.com/content_ICCV_2019/papers/Agustsson_Generative_Adversarial_Networks_for_Extreme_Learned_Image_Compression_ICCV_2019_paper.pdf).
This Python package contains functionality for training GANs for image compression tasks as well as corresponding experiments.

## Installation
0. Create a conda environment for the package to reside in and activate it
```shell
conda create -n compregan python=3.10
conda activate compregan
conda install -c conda-forge cudnn  # might be needed to utilize the GPU
```

1. To install the python package please check it out, install the dependencies and install the package itself in 
   'develop' mode as follows.
```shell
git clone https://github.com/jollyjonson/compregan
cd compregan
make install
```

2. Make sure everything works as intended, run the unit tests as follows
```shell
pip install -r requirements-test.txt
make test
```
3. You can find examples and experiments in the `experiments` directory! As a basic sandbox try out and explore
```shell
python experiments/basic_mnist_example.py
```
