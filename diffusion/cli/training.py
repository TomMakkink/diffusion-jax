"""Module for training. This implements Algorithm 1 in the DDPM paper."""
from __future__ import annotations

import logging

import jax
import omegaconf
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
from torch.utils.data import DataLoader

from diffusion.data_loader import load_transformed_dataset
from diffusion.ddpm.ddpm import DDPM
from diffusion.network import SimpleUnet
from diffusion.utils import create_checkpoint_manager

# Configure logging to show messages at the INFO level or above
logging.basicConfig(level=logging.INFO)


class TrainState(train_state.TrainState):
    batch_stats: dict[str, jax.Array]


def create_train_state(
    model: nn.Module,
    rng: jax.random.PRNGKey,
    learning_rate: float,
    batch_size: int,
    image_size: int,
    image_channels: int,
) -> TrainState:
    """Creates the initial TrainState.

    Args:
        model: flax model.
        rng: random key.
        learning_rate: learning rate for optimizer.
        batch_size: number of examples in each batch.
        image_size: image of size (same for width and height).
        image_channels: number of image channels.

    Returns:
        train_state: training state containing the model apply
          function, model parameters, batch stats and optimizer.
    """
    # Define fake data to initialise the model.
    init_t = jnp.ones(batch_size)
    init_x = jnp.ones((batch_size, image_size, image_size, image_channels))

    # Initialize the network.
    variables = model.init(rngs=rng, t=init_t, x=init_x, train=False)

    # Get the model parameters and the batch norm stats.
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    optimizer = optax.adam(learning_rate=learning_rate)

    state: TrainState = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
    )

    return state


def l1_loss(noise: jax.Array, noise_pred: jax.Array) -> jax.Array:
    """Compute the L1 loss between actual and predicted noise.

    Args:
        noise: original noise added to the image, shape (B, C, H, W).
        noise_pred: predicted noise from the model, shape (B, C, H, W).

    Returns:
        loss: scalar L1 loss.
    """
    loss = jnp.mean(jnp.abs(noise - noise_pred))
    return loss


def grad_fn(
    params: jax.Array,
    x_t: jax.Array,
    t: jax.Array,
    noise: jax.Array,
    state: TrainState,
) -> tuple[dict[str, jax.Array], float, dict[str, jax.Array]]:
    """Compute gradients using a specific loss function.

    Args:
        params: model parameters.
        x_t: the noised image at time t. Shape (B, C, H, W).
        t: timestep to sample. Can take values in {1, ..., T}. Shape (B,).
        noise: noise added to the original image, shape (B, C, H, W).
        state: training state.

    Returns:
        grads: dict containing the gradients of the model's parameters.
        loss: scalar loss.
        updates: dictionary containing training metadata, e.g.
            batch_stats for BatchNorm.
    """

    def wrapped_loss(params: jax.Array) -> tuple[float, dict[str, jax.Array]]:
        noise_pred, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            x=x_t,
            t=t,
            train=True,
            mutable=["batch_stats"],
        )
        loss = l1_loss(noise=noise, noise_pred=noise_pred)
        return loss, updates

    (loss, updates), grads = jax.value_and_grad(wrapped_loss, has_aux=True)(params)

    return grads, loss, updates


@jax.jit
def train_step(
    state: TrainState,
    x_t: jax.Array,
    noise: jax.Array,
    t: jax.Array,
) -> tuple[TrainState, float]:
    """Train for a single step on a batch of data.

    Args:
        state: flax training state object.
        x_t: the noised image at time t. Shape (B, C, H, W).
        noise: noise added to the original image, shape (B, C, H, W).
        t: timestep to sample. Can take values in {1, ..., T}. Shape (B,).

    Returns:
        state: updated training state after updating the network parameters.
        loss: scalar loss.
    """
    # Compute the gradients and loss.
    grads, loss, updates = grad_fn(
        params=state.params,
        x_t=x_t,
        t=t,
        noise=noise,
        state=state,
    )

    # Update training state using the new gradients.
    state = state.apply_gradients(grads=grads)

    # Update the training state with batch_stats (useful for BatchNorm).
    state = state.replace(batch_stats=updates["batch_stats"])

    return state, loss


def training(cfg: omegaconf.OmegaConf) -> None:
    """Implement Training (Algorithm 1) in the DDPM paper."""
    # Load the dataset.
    dataset = load_transformed_dataset(image_size=cfg.image_size)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True
    )

    # Initialise the SimpleUnet network function.
    model = SimpleUnet(image_channels=cfg.image_channels)

    # Create an initial random key.
    init_rng = jax.random.PRNGKey(cfg.seed)
    rng, sub_key = jax.random.split(init_rng)

    state = create_train_state(
        model=model,
        rng=sub_key,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        image_channels=cfg.image_channels,
    )
    checkpoint_manager = create_checkpoint_manager(
        checkpoint_dir=cfg.checkpoint_dir,
        max_ckpts_to_keep=cfg.max_ckpts_to_keep,
    )
    diffuser = DDPM(T=cfg.num_t)

    for epoch in range(cfg.epochs):
        for step, (x_0, _) in enumerate(dataloader):
            # Algorithm 1 line 3: sample `t` from a discrete uniform distribution
            rng, sub_key = jax.random.split(rng)
            t = jax.random.randint(
                sub_key,
                shape=(cfg.batch_size,),
                minval=0,
                maxval=cfg.num_t,
            )

            # Convert the image x0 from a PyTorch tensor to a jax numpy array
            x_0 = jnp.array(x_0.numpy())

            # Flax convolution layers by default expect a NHWC data format,
            # not channels first NCHW like PyTorch.
            x_0 = jnp.transpose(x_0, (0, 2, 3, 1))

            # From the original image x_0, sample a noised image at timestep t.
            rng, sub_key = jax.random.split(rng)
            x_t, noise = diffuser.create_noised_image(x_0=x_0, t=t, random_key=sub_key)

            # Execute a training step
            state, loss = train_step(
                state=state,
                x_t=x_t,
                noise=noise,
                t=t,
            )

            logging.info(f"Epoch {epoch} | step {step:03d} Loss: {loss} ")

        # Save the training state.
        checkpoint_dict = {"state": state}
        checkpoint_manager.save(step=epoch, items=checkpoint_dict)


if __name__ == "__main__":
    # Read in the configuration file
    cfg = omegaconf.OmegaConf.load("diffusion/config/config.yaml")
    training(cfg)
