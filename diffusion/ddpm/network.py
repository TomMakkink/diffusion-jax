"""Backward process i.e. the denoising process.
    - We use a simple form of a U-Net to predict the noise in each sampled image.
    - The input is a noisy image, the output of the model is the predicted noise
      in the image.
    - Because the parameters are shared across time, we must tell the network in
      which timestep we are: the timestep `t` is positionally encoded.
    - We output one single value (mean), because the variance is fixed.
"""
from collections.abc import Sequence

import jax
from flax import linen as nn
from jax import numpy as jnp
from torch.utils.data import DataLoader

from diffusion.ddpm.constants import BATCH_SIZE
from diffusion.ddpm.data_loader import load_transformed_dataset


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal Positional encoding used to encode the timestep `t`.

    Args:
        dim: embedding dimensions for the positional encoding.
    """

    dim: int  # Emedding dimension

    @nn.compact
    def __call__(self, t: jax.Array) -> jax.Array:
        half_dim = self.dim // 2
        embeddings = jnp.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate(
            (jnp.sin(embeddings), jnp.cos(embeddings)), axis=-1
        )
        return embeddings


class Block(nn.Module):
    """Block used in U-net.

    Args:
        in_ch: number of input channels.
        out_ch: number of output channels.
        up: whether the block is used for upsampling. If False,
            the block is for downsampling.
    """

    in_ch: int
    out_ch: int
    up: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array, train: bool = True) -> jax.Array:
        """Forward pass of the downsampling/upsampling block.

        Args:
            x: embedding tensor of shape (B, H, W, C).
            t: the timestep of the sample in question.
            train: whether in training or inference mode.

        Returns:
            Embedding tensor.
        """
        # Define layers for upsampling vs. downsampling
        if self.up:
            conv1 = nn.Conv(features=self.out_ch, kernel_size=(3, 3), padding="SAME")
            transform = nn.ConvTranspose(
                features=self.out_ch, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
            )
        else:
            conv1 = nn.Conv(features=self.out_ch, kernel_size=(3, 3), padding="SAME")
            transform = nn.Conv(
                features=self.out_ch, kernel_size=(4, 4), strides=(2, 2), padding="SAME"
            )

        # First convolution, ReLU, and batch normalisation
        h = conv1(x)
        h = jax.nn.relu(h)
        h = nn.BatchNorm(momentum=0.9, use_running_average=not train)(h)
        # Time embedding. Shape (1, time_emb_dim)
        time_emb = nn.Dense(features=self.out_ch)(t)
        time_emb = jax.nn.relu(time_emb)

        # time_emb = time_emb.reshape((time_emb.shape[0], time_emb.shape[1], 1, 1))
        time_emb = time_emb[(...,) + (None,) * 2]
        # Reshape the time embedding to (batch, 1, 1, time_emb_dim)
        time_emb = jax.numpy.transpose(time_emb, (0, 2, 3, 1))

        # Add time channel (broadcasting occurs)
        h = h + time_emb

        # Second Conv
        h = nn.Conv(features=self.out_ch, kernel_size=(3, 3), padding="SAME")(h)
        h = jax.nn.relu(h)
        h = nn.BatchNorm(momentum=0.9, use_running_average=not train)(h)

        return transform(h)


class SimpleUnet(nn.Module):
    """A simpliefied variant of the U-net architecture."""

    image_channels: int = 3
    down_channels: Sequence[int] = (64, 128, 256, 512, 1024)
    up_channels: Sequence[int] = (1024, 512, 256, 128, 64)
    out_dim: int = 3
    time_emb_dim: int = 32

    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array, train: bool = True) -> jax.Array:
        """Do a forward pass of the neural network in the denoising process.

        Args:
            x: during training, this is usually a noisy version of some starting
                image `x_0` sampled at a timestep `t`. Shape (batch_size, H, W, C).
            t: the timestep indicating how much noise has been added to the
                starting image `x_0`. Is in {1, 2, ..., T}. Shape (batch_size).
            train: whether in training or inference mode.. Useful for BatchNorm layers.

        Returns:
            A prediction of the noise in the input `x`.
        """
        # Compute the positional encoding of the timestep
        time_mlp = nn.Sequential(
            [
                SinusoidalPositionEmbeddings(dim=self.time_emb_dim),
                nn.Dense(self.time_emb_dim),
                jax.nn.relu,
            ]
        )
        # time encoding with shape (batch_size, time_emb_dim).
        t_encoded = time_mlp(t)

        x = nn.Conv(features=self.down_channels[0], kernel_size=(3, 3), padding="SAME")(
            x
        )

        # U-net: downsampling followed by upsampling.
        residual_inputs = []  # Used as a stack
        for i in range(len(self.down_channels) - 1):
            x = Block(
                in_ch=self.down_channels[i],
                out_ch=self.down_channels[i + 1],
            )(x=x, t=t_encoded, train=train)
            residual_inputs.append(x)
        for i in range(len(self.up_channels) - 1):
            residual_x = residual_inputs.pop()
            x = jnp.concatenate((x, residual_x), axis=-1)
            x = Block(
                in_ch=self.up_channels[i], out_ch=self.up_channels[i + 1], up=True
            )(x=x, t=t_encoded, train=train)
        output = nn.Conv(features=self.out_dim, kernel_size=(1,))(x)

        return output


if __name__ == "__main__":
    # Load example image
    data = load_transformed_dataset()
    dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    image_and_label = next(iter(dataloader))
    image = image_and_label[0]
    image = jnp.array(image.numpy())
    # Flax convolution layers by default expect a NHWC data format,
    # not channels first like PyTorch.
    image = jnp.transpose(image, (0, 2, 3, 1))
    t = jnp.array([3])

    # Set the random key and provide some fake data for shape.
    rng = jax.random.PRNGKey(42)
    fake_t = jnp.array([1.0])
    fake_x = jnp.ones_like(image)

    # Define the network
    model = SimpleUnet()

    # Intialize the simple unet
    params = model.init(rngs=rng, t=fake_t, x=fake_x)

    # Apply the forward function
    rng = jax.random.PRNGKey(7)
    pred_noise = model.apply(params, rngs=rng, t=t, x=image)
