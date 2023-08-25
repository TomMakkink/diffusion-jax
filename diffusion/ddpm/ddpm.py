"""Module for defining the DDPM class, used for implementing the
forward and backward processes.
"""

from __future__ import annotations

from typing import Any

import jax
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional
from flax import linen as nn
from jax import numpy as jnp
from torch.utils.data import DataLoader

from diffusion.ddpm.constants import BATCH_SIZE, SEED
from diffusion.ddpm.data_loader import load_transformed_dataset, show_tensor_image
from diffusion.ddpm.noise_schedule import linear_beta_schedule
from diffusion.ddpm.utils import get_value_from_index


class DDPM:
    """Class for representing the forward (noising) process and backward (denoising)
    process in a denoising diffusion probabilistic model (DDPM). This implementation
    is based on the original DDPM paper: https://arxiv.org/pdf/2006.11239.pdf.

    Forward process, noise/variance schedule, and sampling: we must first create the
    inputs for our network, which are noisy versions of our original images. Rather
    than doing this sequentially, we can use the closed form in Equation 4 of the
    paper to create a noised image x_t given the original image x_0 and a timestep t
    in {1, ..., T}. Importantly,
    - The noise-levels can be pre-computed.
    - There are different types of variance schedules.
    - We can directly sample x_t from x_0 and t (Equation 4 in paper). This follows
      from the reparametrization trick and the fact that the sum of independent
      Gaussian distributions is also Gaussian.
    - No ML model/network is needed in the forward/noising process.
    """

    def __init__(self, T: int = 100) -> None:
        """Initialize the DDPM model.

        Args:
            T: number of timesteps. Defaults to 100.
        """
        self.T = T

        # Create a linear beta schedule which defines the noise schedule
        # and precompute useful quantities.
        self.betas = linear_beta_schedule(num_timesteps=T, start=0.001, end=0.01)
        self.alphas = 1.0 - self.betas
        self.alphabars = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_recip_alphas = jnp.sqrt(1.0 / self.alphas)
        self.sqrt_alphabars = jnp.sqrt(self.alphabars)
        self.sqrt_one_minus_alphabars = jnp.sqrt(1.0 - self.alphabars)
        # Remove the last value and then add 1.0 as the first value
        self.alphabars_prev = jnp.pad(
            self.alphabars[:-1], pad_width=((1, 0)), constant_values=1.0
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphabars_prev) / (1.0 - self.alphabars)
        )

    def create_noised_image(
        self, x_0: jnp.ndarray, t: jnp.ndarray, random_key: jax.random.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Create a noised version of an uncorrupted image. Used in the forward process.

        Specifically, given an uncorrupted image at t=0 and timestep `t`, this method
        returns a noisy version of the image. The level of noise added depends on the
        timestep `t`.

        Note: because the noise added from t-1 to t follows a Gaussian distribution, a
        convenient property follows: one can directly sample the noised image `x_t` at
        any timestep `t` from a distribution q(x_t|x_0) with closed form. To this end,
        this method implements Equation (4) in the original DDPM paper.

        Args:
            x_0: the original image, i.e. the image at t=0. Shape (B, C, H, W).
            t: timestep to sample. Can take values in {1, ..., T}. Shape (B,).
            random_key: jax.random.PRNGKey used for random number generation.

        Returns:
            A tuple containing:
            - `x_t` (a noisy version of the input image). Shape (B, C, H, W).
            - The noise that was added to the input image to get `x_t`.
              Shape (B, C, H, W).
        """
        # Compute factor of x_0 in normal distribution in Equation 4 of DDPM paper
        sqrt_alphabar_t = get_value_from_index(self.sqrt_alphabars, t, x_0.shape)

        # Compute standard deviation of normal distribution in Equation 4 of DDPM paper
        sqrt_one_minus_alphabar_t = get_value_from_index(
            self.sqrt_one_minus_alphabars, t, x_0.shape
        )

        # Sample epsilon from the standard normal distribution (Algorithm 1 line 4)
        noise = jax.random.normal(random_key, x_0.shape)

        # Sample from the normal distribution (Equation 4 in the DDPM paper)
        # using the reparametrization trick: x = μ + sigma * ε
        mean = sqrt_alphabar_t * x_0
        std = sqrt_one_minus_alphabar_t
        noised_image = mean + std * noise

        return noised_image, noise

    def perform_denoising_step(
        self,
        x_t: jnp.ndarray,
        t: jnp.ndarray,
        network_fn: nn.Module,
        network_params: dict[str, Any],
        rng: jax.random.PRNGKey,
    ) -> jnp.ndarray:
        """Denoise a noisy image to get a slightly less noisy image.

        The following steps are done:
        - Call U-net model to predict the total noise in the noised input image `x_t`.
        - Subtract this from `x_t` to give an estimate of the final image `x_0`.
        - If not in the final timestep, we apply noise (but less than previously) which
          results in x_{t-1} which is (hopefully) a slight improvement over `x_t`.

        This is used in Algorithm 2 in the original DDPM paper: it is the successive
        application of this method (starting from pure noise at t=T) that results in a
        final generated/denoised image.

        Args:
            x_t: the noisy image at timestep `t`. Shape (B, C, H, W).
            t: the timestep in question. Shape (B,).
            network_fn: the U-net model used to predict the "total"
                noise in the noisy image. Note this model is expected to already
                be initialized.
            network_params: network parameters to be passed into the apply method
                (e.g. params, state, random key).

        Returns:
            - x_{t-1} if t > 0. This a noised version of the final image. It can be
              thought of as the result of applying a denoising step on `x_t`.
              Shape (B, C, H, W).
            - x_0 if t = 0. This is the final generated image. Shape (B, C, H, W).
        """
        beta_t = get_value_from_index(self.betas, t, x_t.shape)
        sqrt_one_minus_alphabar_t = get_value_from_index(
            self.sqrt_one_minus_alphabars, t, x_t.shape
        )
        sqrt_recip_alpha_t = get_value_from_index(self.sqrt_recip_alphas, t, x_t.shape)

        # Calculate µ_θ in Equation 11 of the DDPM paper, where the noise is predicted
        # using the U-net model (represented as ε_θ(x_t, t) in the paper).
        pred_noise = network_fn.apply(t=t, x=x_t, **network_params)
        model_mean = sqrt_recip_alpha_t * (
            x_t - beta_t * pred_noise / sqrt_one_minus_alphabar_t
        )

        # Compute (sigma_t)^2 in Algorithm 2 line 4
        posterior_variance_t = get_value_from_index(
            self.posterior_variance, t, x_t.shape
        )

        if t == 0:
            x_0 = model_mean
            # As pointed out by Luis Pereira (see YouTube comment)
            # The t's are offset from the t's in the paper
            return x_0

        # Sample z (in Algorithm 2) from a standard normal distribution
        rng, sub_key = jax.random.split(rng)
        noise = jax.random.normal(sub_key, x_t.shape)

        # Return x_{t-1} in Algorithm 2 line 4. This is basically
        # sampling from the distribution q(x_{t-1}|x_t) in Equation 4.
        x_t_1 = model_mean + jnp.sqrt(posterior_variance_t) * noise

        return x_t_1


if __name__ == "__main__":
    # Prepare dataset
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Initialize DDPM model
    T = 100
    ddpm = DDPM(T=T)
    rng = jax.random.PRNGKey(SEED)  # Assume SEED is defined

    # Simulate forward diffusion for an example image
    example_image_tensor = next(iter(dataloader))[0][0]
    # Convert the example image from a PyTorch Tensor to a Jax Array
    example_image = jnp.array(example_image_tensor)

    plt.figure(figsize=(15, 15))
    plt.axis("off")
    num_images = 10
    step_size = int(T / num_images)

    # Plot the sequence of progressively noiser images for the example image
    for idx in range(0, T, step_size):
        t = jnp.array([idx]).astype(jnp.int32)
        rng, sub_rng = jax.random.split(rng)
        noised_image, _ = ddpm.create_noised_image(
            x_0=example_image, t=t, random_key=sub_rng
        )
        plt.subplot(1, num_images + 1, int(idx / step_size) + 1)
        noised_image_tensor = torch.tensor(np.asarray(noised_image))
        show_tensor_image(noised_image_tensor)

    plt.show()
