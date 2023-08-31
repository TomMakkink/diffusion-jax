# Diffusion

Implementation of
[Denoising Diffusion Probablistic Models](https://arxiv.org/abs/2006.11239) (DDPM) by Ho
et al. 2020.

This implementation of DDPM is written in Jax using the
[Flax](https://flax.readthedocs.io/en/latest/) framework. The goal was to write simple,
readable, and easy to understand code that illustrates the core mechanics of the paper.

## Setup

Clone the repository:

```shell
git clone git@github.com:TomMakkink/diffusion-jax.git
```

Create a conda environment:

```shell
conda env create --file environment.yaml
```

Activate the conda environment:

```shell
conda activate ddpm-env
```

Install the local package in editable mode:

```shell
pip install --editable .
```

## Training

To train a Diffusion model, launch the following command:

```shell
python diffusion/cli/training.py
```

The training parameters, such as the learning rate, are configured in
`diffusion/ddpm/constants.py`. During training the checkpoints from the latest two
epochs will be saved using the [orbax](https://github.com/google/orbax) library.

## Inference/Sampling

To run inference and generate an image using a saved model, launch the following
command:

```shell
python diffusion/cli/inference.py
```

Note the latest checkpoint saved during training will be used for inference.

## DDPM

The implementation of DDPM is based on the
[original DDPM paper](<(https://arxiv.org/abs/2006.11239)>).

- [The Colab](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=BIc33L9-uK4q)
