"""Module for training. This implements Algorithm 1 in the DDPM paper."""
import logging

import haiku as hk
import jax
import optax
from jax import numpy as jnp
from torch.utils.data import DataLoader

from diffusion.ddpm.constants import BATCH_SIZE, IMG_SIZE, SEED
from diffusion.ddpm.data_loader import load_transformed_dataset
from diffusion.ddpm.ddpm import DDPM
from diffusion.ddpm.network import SimpleUnet

# Configure logging to show messages at the INFO level or above
logging.basicConfig(level=logging.INFO)


def compute_loss_and_grads(
    params: hk.Params,
    state: hk.State,
    t: jax.Array,
    x_0: jax.Array,
    rng: jax.random.PRNGKey,
    ddpm: DDPM,
    network_fn: hk.Module,
) -> tuple[jax.Array, hk.State]:
    """Computes the loss and state of a denoising diffusion probabilistic model.

    Args:
        params: The parameters of the Haiku network.
        state: The state information for stateful components in the model.
        t: The timestep(s) with shape (batch_size).
        x_0: The original image(s) of shape (batch_size, n_channels, H, W).
        rng: A random number generator key.
        ddpm: An instance of the DDPM class.
        network_fn: The Haiku network function, with the apply
            method for predicting the noise.

    Returns:
        A tuple with the computed loss and the updated network state.
    """
    # From the original image x_0, sample a noised image at timestep t
    x_t, noise = ddpm.create_noised_image(x_0=x_0, t=t, random_key=rng)

    # Predict the noise
    noise_pred, state = network_fn.apply(
        params=params, state=state, rng=rng, t=t, x=x_t
    )

    # Compute L1 loss between actual noise and predicted noise
    loss = jnp.mean(jnp.abs(noise - noise_pred))

    return loss, state


def training() -> None:
    """Implement Algorithm 1 in the DDPM paper."""
    # Load the dataset.
    dataset = load_transformed_dataset()
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Initialise the SimpleUnet network function.
    network_fn = SimpleUnet()

    # Set the random key and provide some fake data to initialize the model
    # with the correct shapes.
    rng = jax.random.PRNGKey(SEED)
    init_t = jnp.ones(BATCH_SIZE)
    init_x = jnp.ones((BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3))

    # Initialize the network
    params, state = network_fn.init(rngs=rng, t=init_t, x=init_x)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate=0.001)

    # Initialize the optimizer with the model parameters.
    opt_state = optimizer.init(params=params)

    # Set up the DDPM class.
    T = 100
    ddpm = DDPM(T=T)
    epochs = 100

    # Define the loss function to compute gradients.
    grad_fn = jax.value_and_grad(compute_loss_and_grads, has_aux=True)

    for epoch in range(epochs):
        for step, (x_0, _) in enumerate(dataloader):
            # Algorithm 1 line 3: sample `t` from a discrete uniform distribution
            rng, sub_key = jax.random.split(rng)
            t = jax.random.randint(sub_key, shape=(BATCH_SIZE,), minval=0, maxval=T)

            # Convert the image x0 from a PyTorch tensor to a jax numpy array.
            x_0 = jnp.array(x_0.numpy())

            # Haiku convolution layers by default expect a NHWC data format,
            # not channels first like PyTorch.
            x_0 = jnp.transpose(x_0, (0, 2, 3, 1))

            # Compute the loss function
            rng, sub_key = jax.random.split(rng)
            (loss, state), grads = grad_fn(
                params,
                state=state,
                t=t,
                x_0=x_0,
                rng=rng,
                ddpm=ddpm,
                network_fn=network_fn,
            )

            # Backpropagation: compute the gradients and update the parameters
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            logging.info(f"Epoch {epoch} | step {step:03d} Loss: {loss} ")


if __name__ == "__main__":
    training()
