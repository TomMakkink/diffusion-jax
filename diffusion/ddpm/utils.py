"""Utility functions for the DDPM model."""

from jax import numpy as jnp


def get_value_from_index(
    vals: jnp.ndarray, t: jnp.ndarray, x_shape: tuple[int]
) -> jnp.ndarray:
    """Returns the value at index `t` of a passed list of values
    `vals`, all while considering the batch dimension.

    Args:
        vals: the array of values.
        t: the index used to get the desired value from `vals`.
        x_shape: shape of image, usually (B, C, H, W).

    Returns
        Rank-4 array of shape (1, 1, 1, 1), containing the value.
    """
    batch_size = t.shape[0]

    # Get the value of `vals` at index `t`
    out = vals.take(t)

    # Reshape to (1, 1, 1, 1) i.e. same rank as the image.
    non_batch_shape = (1,) * (len(x_shape) - 1)
    out = out.reshape((batch_size, *non_batch_shape))

    return out
