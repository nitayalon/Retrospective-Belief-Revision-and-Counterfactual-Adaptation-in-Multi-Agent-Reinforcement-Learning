"""Quick test script for train_state."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"

import sys
print("Starting imports...")

import jax
print("✓ JAX imported")

from him_her.training.train_state import HIMHERTrainState, create_train_state
print("✓ train_state imported")

import numpy as np
import jax.numpy as jnp

print("\nTesting create_train_state with minimal network...")
rng = jax.random.PRNGKey(0)
log_priors = np.log(np.array([0.5, 0.5]))

print("Creating train state with small hidden sizes...")
train_state, _ = create_train_state(
    rng=rng,
    obs_dim=2,
    goal_dim=2,
    action_dim=1,
    model_embed_dim=2,
    log_priors=log_priors,
    hidden_sizes=(4,),  # Single small layer
)

print(f"\n✓✓✓ SUCCESS! Train state created!")
print(f"  - current_model_id: {train_state.current_model_id} (type: {type(train_state.current_model_id).__name__})")
print(f"  - step: {train_state.step}")
print(f"  - log_belief shape: {train_state.log_belief.shape}")
print(f"  - Actor params keys: {list(train_state.actor_state.params.keys())}")
print(f"  - Critic params keys: {list(train_state.critic_state.params.keys())}")

