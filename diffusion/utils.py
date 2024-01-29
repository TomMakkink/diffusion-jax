"""Module to define utility functions"""
from __future__ import annotations

import orbax.checkpoint


def create_checkpoint_manager(
    checkpoint_dir: str, max_ckpts_to_keep: int
) -> orbax.checkpoint.CheckpointManager:
    """Create a orbax checkpoint manager.

    Args:
        checkpoint_dir: path to directory where checkpoints are saved.
        max_ckpts_to_keep: maximum number of checkpoints to save.

    Returns:
        checkpoint_manager: orbax checkpoint manager.
    """
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(
        max_to_keep=max_ckpts_to_keep,
        create=True,
    )
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        checkpoint_dir, orbax_checkpointer, options
    )
    return checkpoint_manager
