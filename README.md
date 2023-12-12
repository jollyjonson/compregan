## Toolbox for perceptual data compression using GANs (or so)
[![CI](https://github.com/jollyjonson/compregan/actions/workflows/main.yml/badge.svg)](https://github.com/jollyjonson/compregan/actions/workflows/main.yml)

Still in the very early construction phase...

To use the package please follow the points lined out below

1. Create a conda environment for the package to reside in and activate it [Optional]
```shell
conda create -n compregan python=3.8
conda activate compregan
conda install -c conda-forge cudnn  # might be needed to utilize the GPU e.g. on commit-ws
```

2. To install the python package please check it out, install the dependencies and install thepackage itself in 
   'develop' mode as follows.
```shell
git clone git@git.tu-berlin.de:jonas.hajek-lellmann/compregan CompreGAN
cd CompreGAN
pip install -r requirements.txt
pip install -e .
```

3. Make sure everything works as intended, run the unit tests as follows [Optional]
```shell
pip install -r requirements-test.txt
python test/run_all_and_measure_coverage.py
```

4. You can find examples and experiments in the `bin` directory! As a basic sandbox try out
```shell
python experiments/basic_mnist_example.py
```
