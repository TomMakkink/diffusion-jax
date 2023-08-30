"""Module for training. This implements Algorithm 1 in the DDPM paper."""
from __future__ import annotations

import logging
from typing import Callable

import jax
import optax
import orbax.checkpoint
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
from torch.utils.data import DataLoader

from diffusion.ddpm.constants import BATCH_SIZE, CHECKPOINT_DIR, IMG_SIZE, SEED
from diffusion.ddpm.data_loader import load_transformed_dataset
from diffusion.ddpm.ddpm import DDPM
from diffusion.ddpm.network import SimpleUnet

# Configure logging to show messages at the INFO level or above
logging.basicConfig(level=logging.INFO)


class TrainState(train_state.TrainState):
    batch_stats: dict[str, jax.Array]


def create_train_state(
    model: nn.Module,
    rng: jax.random.PRNGKey,
    learning_rate: float,
) -> TrainState:
    """Creates the initial TrainState.

    Args:
        model: flax model.
        rng: random key.
        learning_rate: learning rate for optimizer.

    Returns:
        train_state: training state containing the model apply
          function, model parameters, batch stats and optimizer.
    """
    # Define fake data to initialise the model.
    init_t = jnp.ones(BATCH_SIZE)
    init_x = jnp.ones((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))

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
    loss_fn: Callable = l1_loss,
) -> tuple[dict[str, jax.Array], float, dict[str, jax.Array]]:
    """Compute gradients using a specific loss function.

    Args:
        params: model parameters.
        x_t: the original image at time t. Shape (B, C, H, W).
        t: timestep to sample. Can take values in {1, ..., T}. Shape (B,).
        noise: original noise added to the image, shape (B, C, H, W).
        state: training state.
        loss_fn: loss function (e.g. l1 loss).

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
        loss = loss_fn(noise, noise_pred)
        return loss, updates

    (loss, updates), grads = jax.value_and_grad(wrapped_loss, has_aux=True)(params)

    return grads, loss, updates


def train_step(
    state: TrainState,
    x_0: jax.Array,
    t: jax.Array,
    rng: jax.random.PRNGKey,
    ddpm: DDPM,
    loss_fn: Callable = l1_loss,
) -> tuple[TrainState, float]:
    """Train for a single step on a batch of data.

    Args:
        state: flax training state object.
        x_0: the original image, i.e. the image at t=0. Shape (B, C, H, W).
        t: timestep to sample. Can take values in {1, ..., T}. Shape (B,).
        rng: jax.random.PRNGKey used for random number generation.
        ddpm: initialised ddpm class object.
        loss_fn: loss function (e.g. l1 loss).

    Returns:
        state: updated training state after updating the network parameters.
        loss: scalar loss.
    """
    # From the original image x_0, sample a noised image at timestep t.
    x_t, noise = ddpm.create_noised_image(x_0=x_0, t=t, random_key=rng)

    # Compute the gradients and loss.
    grads, loss, updates = grad_fn(
        params=state.params,
        x_t=x_t,
        t=t,
        noise=noise,
        state=state,
        loss_fn=loss_fn,
    )

    # Update training state using the new gradients.
    state = state.apply_gradients(grads=grads)

    # Update the training state with batch_stats (useful for BatchNorm).
    state = state.replace(batch_stats=updates["batch_stats"])

    return state, loss


def training() -> None:
    """Implement Training (Algorithm 1) in the DDPM paper."""
    # Load the dataset.
    dataset = load_transformed_dataset()
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Initialise the SimpleUnet network function.
    model = SimpleUnet()

    # Create an initial random key.
    init_rng = jax.random.PRNGKey(SEED)
    rng, sub_key = jax.random.split(init_rng)

    # Create the training state.
    state = create_train_state(
        model=model,
        rng=sub_key,
        learning_rate=0.001,
    )

    # Set up the DDPM class.
    T = 100
    ddpm = DDPM(T=T)
    epochs = 100

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        CHECKPOINT_DIR, orbax_checkpointer, options
    )

    for epoch in range(epochs):
        for step, (x_0, _) in enumerate(dataloader):
            # Algorithm 1 line 3: sample `t` from a discrete uniform distribution
            rng, sub_key = jax.random.split(rng)
            t = jax.random.randint(sub_key, shape=(BATCH_SIZE,), minval=0, maxval=T)

            # Convert the image x0 from a PyTorch tensor to a jax numpy array
            x_0 = jnp.array(x_0.numpy())

            # Flax convolution layers by default expect a NHWC data format,
            # not channels first like PyTorch.
            x_0 = jnp.transpose(x_0, (0, 2, 3, 1))

            # Execute a training step
            rng, sub_key = jax.random.split(rng)
            state, loss = train_step(
                state=state,
                x_0=x_0,
                t=t,
                rng=sub_key,
                ddpm=ddpm,
            )

            logging.info(f"Epoch {epoch} | step {step:03d} Loss: {loss} ")

        # Save the training state.
        checkpoint_dict = {"state": state}
        checkpoint_manager.save(step=epoch, items=checkpoint_dict)


if __name__ == "__main__":
    training()
