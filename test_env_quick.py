"""Quick test script for predator-prey environment."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ["JAX_PLATFORMS"] = "cpu"

import sys
sys.path.insert(0, '.')

import numpy as np
import jax
import jax.numpy as jnp

from him_her.envs.predator_prey import PredatorPreyEnv, predator_prey_model_forward
from him_her.models.base_model import AgentModel

print("Testing Predator-Prey Environment...")

# Test 1: Reset returns correct shapes
print("\n1. Testing reset()...")
env = PredatorPreyEnv(max_episode_length=50, grid_size=16, seed=42)
obs, info = env.reset(seed=0)
assert obs.shape == (4,), f"Expected obs shape (4,), got {obs.shape}"
assert "switch_point" in info
assert "initial_policy" in info
print(f"✓ Reset successful. Obs shape: {obs.shape}, switch_point: {info['switch_point']}")

# Test 2: Switch point in range
print("\n2. Testing switch point range...")
env = PredatorPreyEnv(max_episode_length=60, grid_size=16, seed=42)
lower_bound = 60 // 3
upper_bound = 2 * 60 // 3
switch_points = []

for episode in range(10):
    obs, info = env.reset(seed=episode)
    switch_point = info["switch_point"]
    switch_points.append(switch_point)
    assert lower_bound <= switch_point <= upper_bound, (
        f"switch_point {switch_point} not in [{lower_bound}, {upper_bound}]"
    )

print(f"✓ All 10 switch points in range [{lower_bound}, {upper_bound}]")
print(f"  Switch points: {switch_points}")

# Test 3: Info keys present
print("\n3. Testing info keys...")
env = PredatorPreyEnv(max_episode_length=50, grid_size=16, seed=42)
obs, info = env.reset(seed=0)

for step in range(10):
    action = np.random.randint(0, 5)
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    assert "other_action" in info
    assert "achieved_goal" in info
    assert "desired_goal" in info
    
    if terminated or truncated:
        break

print(f"✓ All required info keys present in {step+1} steps")

# Test 4: Reward functions match
print("\n4. Testing compute_reward_jax matches compute_reward...")
env = PredatorPreyEnv(max_episode_length=50, grid_size=16, seed=42)
dummy_model = AgentModel(
    model_id=0,
    name="test_model",
    reward_weights=np.ones(5),
    policy_params=np.ones(5),
    prior=1.0,
)

rng = np.random.RandomState(123)
mismatches = 0

for i in range(20):
    state = rng.uniform(0, 16, size=4).astype(np.float32)
    ego_action = rng.randint(0, 5)
    goal = state[2:4]
    reward_weights = rng.uniform(0, 1, size=5).astype(np.float32)
    
    reward_np = env.compute_reward(state, ego_action, goal, dummy_model)
    reward_jax = env.compute_reward_jax(
        jnp.array(state),
        jnp.array(ego_action),
        jnp.array(goal),
        jnp.array(reward_weights),
    )
    reward_jax = float(reward_jax)
    
    diff = np.abs(reward_np - reward_jax)
    if diff >= 1e-5:
        mismatches += 1
        print(f"  Mismatch {i}: NumPy={reward_np}, JAX={reward_jax}, diff={diff}")

assert mismatches == 0, f"Found {mismatches} mismatches"
print(f"✓ All 20 reward computations match (max diff < 1e-5)")

# Test 5: Model forward is pure JAX
print("\n5. Testing model_forward JIT compilation...")
policy_params = jnp.ones(5)
state = jnp.array([1.0, 2.0, 3.0, 4.0])

jitted_model_forward = jax.jit(predator_prey_model_forward)
logits = jitted_model_forward(policy_params, state)
assert logits.shape == (5,)
assert jnp.all(jnp.isfinite(logits))
print(f"✓ model_forward compiled successfully. Logits shape: {logits.shape}")

# Test 6: Policy switch occurs
print("\n6. Testing policy switch...")
env = PredatorPreyEnv(max_episode_length=60, grid_size=16, seed=42)
obs, info = env.reset(seed=0)
switch_point = info["switch_point"]
initial_policy = info["initial_policy"]

for step in range(switch_point + 2):
    action = 0
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    if step == switch_point:
        # Should have switched
        assert info["current_policy"] != initial_policy, (
            f"Policy did not switch at step {switch_point}"
        )
        print(f"✓ Policy switched from '{initial_policy}' to '{info['current_policy']}' at step {switch_point}")
        break
    
    if terminated or truncated:
        break

print("\n" + "="*50)
print("ALL TESTS PASSED!")
print("="*50)
