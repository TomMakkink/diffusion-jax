# Diffusion

Implementation of
[Denoising Diffusion Probablistic Models](https://arxiv.org/abs/2006.11239) (DDPM) by Ho
et al. 2020.

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
conda activate diffusion-jax-playground
```

Install the local package in editable mode:

```shell
pip install --editable .
```

## Training

To train a Diffusion model, launch the following command:

`python diffusion/cli/training.py`

## DDPM

The implementation of DDPM is based on the
[original DDPM paper](<(https://arxiv.org/abs/2006.11239)>).

- [The Colab](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=BIc33L9-uK4q)
