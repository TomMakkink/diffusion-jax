"""Module for implementing Algorithm 2 in the original DDPM paper."""
from __future__ import annotations

import jax
from jax import numpy as jnp

from diffusion.ddpm.constants import BATCH_SIZE, IMG_SIZE, SEED
from diffusion.ddpm.data_loader import show_tensor_image
from diffusion.ddpm.ddpm import DDPM
from diffusion.ddpm.network import SimpleUnet


def inference() -> None:
    """Implement Algorithm 2 in the paper."""
    T = 100
    ddpm = DDPM(T=T)

    # Define the neural network which predicts noise
    network_fn = SimpleUnet()

    # Set the random key and provide some fake data to initialize the model
    # with the correct shapes.
    rng = jax.random.PRNGKey(SEED)
    init_t = jnp.ones(BATCH_SIZE)
    init_x = jnp.ones((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))

    # Initialize the network
    params, _ = network_fn.init(rngs=rng, t=init_t, x=init_x)

    # Algorithm 2 line 1: sample pure noise at t=T from
    # the standard normal distribution.
    rng, sub_key = jax.random.split(rng)
    x_t = jax.random.normal(key=sub_key, shape=init_x.shape)

    # Algorithm 2 for-loop
    for timestep in range(T, 0):
        t = jnp.array(timestep)

        # Algorithm 2 line 4
        x_t = ddpm.perform_denoising_step(
            x_t=x_t,
            t=t,
            network_fn=network_fn,
            network_params=params,
            rng=rng,
        )

    # Plot the generated/denoised image
    show_tensor_image(x_t)


if __name__ == "__main__":
    inference()
