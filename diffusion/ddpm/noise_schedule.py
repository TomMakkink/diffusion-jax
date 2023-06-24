"""Module that implements different noise schedules."""
from jax import numpy as jnp


def linear_beta_schedule(
    num_timesteps: int,
    start: float = 0.0001,
    end: float = 0.02,
) -> jnp.ndarray:
    """Return a tensor of linearly spaced beta values, defining
    the noise schedule (aka variance schedule).

    Args:
        num_timesteps: number of timesteps (T in literature).
        start: initial value of beta from t=0 to t=1.
        end: final value of beta from timestep t=T-1 to t=T.

    Returns:
        1D tensor of shape (T,) representing linearly spaced beta
        values used in the noise schedule of the forward process.
    """
    return jnp.linspace(start=start, stop=end, num=num_timesteps)
