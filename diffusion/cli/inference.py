"""Module for implementing Algorithm 2 in the original DDPM paper."""
from __future__ import annotations

import jax
import orbax.checkpoint
import torch
from jax import device_get
from jax import numpy as jnp

from diffusion.ddpm.constants import BATCH_SIZE, CHECKPOINT_DIR, IMG_SIZE, SEED
from diffusion.ddpm.data_loader import show_tensor_image
from diffusion.ddpm.ddpm import DDPM


def inference() -> None:
    """Implement Algorithm 2 in the paper."""
    T = 100
    ddpm = DDPM(T=T)

    # Set up the checkpoint manager.
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        CHECKPOINT_DIR, orbax_checkpointer, options
    )

    # Load the latest saved model checkpoint
    step = checkpoint_manager.latest_step()
    checkpoint_dict = checkpoint_manager.restore(step)

    # Get the saved model train state.
    state = checkpoint_dict["state"]

    # Set the random key and provide some fake data to initialize the model
    # with the correct shapes.
    rng = jax.random.PRNGKey(SEED)
    image_shape = (BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3)

    # Algorithm 2 line 1: sample pure noise at t=T from
    # the standard normal distribution.
    rng, sub_key = jax.random.split(rng)
    x_t = jax.random.normal(key=sub_key, shape=image_shape)

    # Algorithm 2 for-loop
    for timestep in range(T, 0):
        t = jnp.array(timestep)

        noise_pred, _ = state.apply_fn(
            {"params": state.params, "batch_stats": state.batch_stats},
            x=x_t,
            t=t,
            train=True,
            mutable=["batch_stats"],
        )

        # Algorithm 2 line 4
        x_t = ddpm.perform_denoising_step(
            x_t=x_t,
            t=t,
            noise_pred=noise_pred,
            rng=rng,
        )

    # Plot the generated/denoised image.
    # Note PyTorch expects (B, C, H, W) format.
    x_t = jnp.transpose(x_t, (0, 3, 1, 2))
    x_t = torch.from_numpy(device_get(x_t))
    show_tensor_image(x_t)


if __name__ == "__main__":
    inference()
