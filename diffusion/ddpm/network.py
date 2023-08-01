"""Backward process i.e. the denoising process.
    - We use a simple form of a U-Net to predict the noise in each sampled image.
    - The input is a noisy image, the output of the model is the predicted noise
      in the image.
    - Because the parameters are shared across time, we must tell the network in
      which timestep we are: the timestep `t` is positionally encoded.
    - We output one single value (mean), because the variance is fixed.
"""
from typing import Callable

import haiku as hk
import jax
from jax import numpy as jnp
from torch.utils.data import DataLoader

from diffusion.ddpm.constants import BATCH_SIZE
from diffusion.ddpm.data_loader import load_transformed_dataset


class SinusoidalPositionEmbeddings(hk.Module):
    """Positional encoding is needed because the U-Net uses the same network parameters
    regardless of the timestep `t` in question.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, t: jax.Array) -> jax.Array:
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate(
            (jnp.sin(embeddings), jnp.cos(embeddings)), axis=-1
        )
        return embeddings


class Block(hk.Module):
    def __init__(self, in_ch: int, out_ch: int, up: bool = False) -> None:
        """Block used in U-net.

        Args:
            in_ch: number of input channels.
            out_ch: number of output channels.
            up: whether the block is used for upsampling. If False,
                the block is for downsampling.
        """
        super().__init__()
        self.time_mlp = hk.Linear(output_size=out_ch)
        self.up = up
        self.in_ch = in_ch
        self.out_ch = out_ch

    def __call__(
        self, x: jax.Array, t: jax.Array, is_training: bool = True
    ) -> jax.Array:
        """Forward pass of the downsampling/upsampling block.

        Args:
            x: embedding tensor of shape (B, H, W, C).
            t: the timestep of the sample in question.
            is_training: whether in training or inference mode.

        Returns:
            Embedding tensor.
        """
        # Define layers for upsampling vs. downsampling
        if self.up:
            conv1 = hk.Conv2D(
                output_channels=self.out_ch, kernel_shape=3, padding="SAME"
            )
            transform = hk.Conv2DTranspose(
                output_channels=self.out_ch, kernel_shape=4, stride=2, padding="SAME"
            )
        else:
            conv1 = hk.Conv2D(
                output_channels=self.out_ch, kernel_shape=3, padding="SAME"
            )
            transform = hk.Conv2D(
                output_channels=self.out_ch, kernel_shape=4, stride=2, padding="SAME"
            )

        # First convolution, ReLU, and batch normalisation
        h = conv1(x)
        h = jax.nn.relu(h)
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            h, is_training=is_training
        )

        # Time embedding. Shape (1, time_emb_dim)
        time_emb = jax.nn.relu(self.time_mlp(t))

        # time_emb = time_emb.reshape((time_emb.shape[0], time_emb.shape[1], 1, 1))
        time_emb = time_emb[(...,) + (None,) * 2]
        # Reshape the time embedding to (batch, 1, 1, time_emb_dim)
        time_emb = jax.numpy.transpose(time_emb, (0, 2, 3, 1))

        # Add time channel (broadcasting occurs)
        h = h + time_emb

        # Second Conv
        h = hk.Conv2D(output_channels=self.out_ch, kernel_shape=3, padding="SAME")(h)
        h = jax.nn.relu(h)
        h = hk.BatchNorm(create_scale=True, create_offset=True, decay_rate=0.9)(
            h, is_training=is_training
        )

        return transform(h)


class SimpleUnet(hk.Module):
    """A simplified variant of the U-net architecture."""

    def __init__(self) -> None:
        super().__init__()
        self.image_channels = 3
        self.down_channels = (64, 128, 256, 512, 1024)
        self.up_channels = (1024, 512, 256, 128, 64)
        self.out_dim = 3
        self.time_emb_dim = 32

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Do a forward pass of the neural network in the denoising process.

        Args:
            x: during training, this is usually a noisy version of some starting
                image `x_0` sampled at a timestep `t`.
            t: the timestep indicating how much noise has been added to the
                starting image `x_0`. Is in {1, 2, ..., T}.

        Returns:
            A prediction of the noise in the input `x`.
        """
        # Compute the positional encoding of the timestep
        time_mlp = hk.Sequential(
            [
                SinusoidalPositionEmbeddings(self.time_emb_dim),
                hk.Linear(self.time_emb_dim),
                jax.nn.relu,
            ]
        )
        t_encoded = time_mlp(t)

        x = hk.Conv2D(
            output_channels=self.down_channels[0], kernel_shape=3, padding="SAME"
        )(x)

        # U-net: downsampling followed by upsampling.
        residual_inputs = []  # Used as a stack
        for i in range(len(self.down_channels) - 1):
            x = Block(
                in_ch=self.down_channels[i],
                out_ch=self.down_channels[i + 1],
            )(x, t_encoded)
            residual_inputs.append(x)
        for i in range(len(self.up_channels) - 1):
            residual_x = residual_inputs.pop()
            x = jnp.concatenate((x, residual_x), axis=-1)
            x = Block(
                in_ch=self.up_channels[i], out_ch=self.up_channels[i + 1], up=True
            )(x, t_encoded)

        output = hk.Conv2D(self.out_dim, kernel_shape=1)(x)

        return output


def make_simple_unet() -> Callable:
    def forward_fn(x: jax.Array, t: jax.Array) -> jax.Array:
        network = SimpleUnet()
        return network(x=x, t=t)

    return forward_fn


if __name__ == "__main__":
    # Initialise the SimpleUnet network function.
    network_fn = hk.transform_with_state(make_simple_unet())

    # Load example image
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    image_and_label = next(iter(dataloader))
    image = image_and_label[0]
    image = jnp.array(image.numpy())
    # Haiku convolution layers by default expect a NHWC data format,
    # not channels first like PyTorch.
    image = jnp.transpose(image, (0, 2, 3, 1))
    t = jnp.array([3])

    # Set the random key and provide some fake data for shape.
    rng = jax.random.PRNGKey(42)
    fake_t = jnp.array([1.0])
    fake_x = jnp.ones_like(image)

    # Intialize the simple unet
    params, state = network_fn.init(rng=rng, t=fake_t, x=fake_x)

    # Apply the forward function
    rng = jax.random.PRNGKey(7)
    pred_noise = network_fn.apply(params=params, state=state, rng=rng, t=t, x=image)
