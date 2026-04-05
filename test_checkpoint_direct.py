"""Direct test for checkpoint roundtrip without pytest."""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import tempfile
import shutil
from pathlib import Path
import orbax.checkpoint as ocp

from him_her.training.train_state import HIMHERTrainState, create_train_state

print("Creating train state...")
rng = jax.random.PRNGKey(0)
log_priors = np.log(np.array([0.5, 0.5]))

train_state, _ = create_train_state(
    rng=rng,
    obs_dim=4,
    goal_dim=2,
    action_dim=2,
    model_embed_dim=2,
    log_priors=log_priors,
    hidden_sizes=(8, 8),
)
print("✓ Train state created")

# Update some fields to make it more interesting
train_state = train_state.replace(
    current_model_id=1,
    step=42,
    log_belief=jnp.array([0.3, 0.7])
)
print("✓ Train state updated")

# Create temporary directory for checkpoint
tmpdir = tempfile.mkdtemp()
try:
    ckpt_path = Path(tmpdir) / "checkpoint"
    
    # Save checkpoint
    print(f"Saving checkpoint to {ckpt_path}...")
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(ckpt_path, train_state)
    print("✓ Checkpoint saved")
    
    # Load checkpoint
    print("Loading checkpoint...")
    restored_state = checkpointer.restore(ckpt_path, item=train_state)
    print("✓ Checkpoint loaded")
    
    # Verify all fields are preserved
    print("\nVerifying restored state:")
    assert restored_state.current_model_id == 1
    print(f"  ✓ current_model_id: {restored_state.current_model_id}")
    
    assert restored_state.step == 42
    print(f"  ✓ step: {restored_state.step}")
    
    assert jnp.allclose(restored_state.log_belief, jnp.array([0.3, 0.7]))
    print(f"  ✓ log_belief: {restored_state.log_belief}")
    
    # Verify params are preserved
    def check_params_equal(p1, p2):
        leaves1 = jax.tree_util.tree_leaves(p1)
        leaves2 = jax.tree_util.tree_leaves(p2)
        assert len(leaves1) == len(leaves2)
        for l1, l2 in zip(leaves1, leaves2):
            assert jnp.allclose(l1, l2)
    
    check_params_equal(restored_state.actor_state.params, train_state.actor_state.params)
    print("  ✓ actor_state.params")
    
    check_params_equal(restored_state.critic_state.params, train_state.critic_state.params)
    print("  ✓ critic_state.params")
    
    check_params_equal(restored_state.target_critic_params, train_state.target_critic_params)
    print("  ✓ target_critic_params")
    
    print("\n✓✓✓ SUCCESS! Checkpoint roundtrip test passed!")
    
finally:
    # Clean up temporary directory
    shutil.rmtree(tmpdir)
    print(f"\nCleaned up temporary directory: {tmpdir}")
