"""Test suite for environment wrappers.

Tests verify correct implementation of BaseMultiAgentEnv interface and
environment-specific behavior (e.g., model switching in Predator-Prey).
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from him_her.envs.predator_prey import PredatorPreyEnv, predator_prey_model_forward
from him_her.envs.cooperative_nav import (
    CooperativeNavEnv,
    cooperative_nav_model_forward,
)
from him_her.envs.intersection import (
    IntersectionEnv,
    intersection_model_forward,
)
from him_her.envs.hide_and_seek import (
    HideAndSeekEnv,
    hide_and_seek_model_forward,
)
from him_her.models.base_model import AgentModel


@pytest.fixture
def predator_prey_env():
    """Create a Predator-Prey environment for testing."""
    return PredatorPreyEnv(max_episode_length=50, grid_size=16, seed=42)


def test_reset_returns_correct_shapes(predator_prey_env):
    """Test that reset() returns observation and info with correct structure."""
    obs, info = predator_prey_env.reset(seed=0)
    
    # Check observation shape: [predator_x, predator_y, prey_x, prey_y]
    assert obs.shape == (4,), f"Expected obs shape (4,), got {obs.shape}"
    assert obs.dtype == np.float32
    
    # Check info keys
    assert "switch_point" in info
    assert "initial_policy" in info
    
    # Check switch_point is set
    assert info["switch_point"] is not None
    assert isinstance(info["switch_point"], (int, np.integer))


def test_switch_point_in_range():
    """Test that switch point is always in [T/3, 2T/3] across multiple episodes."""
    env = PredatorPreyEnv(max_episode_length=60, grid_size=16, seed=42)
    
    lower_bound = 60 // 3  # 20
    upper_bound = 2 * 60 // 3  # 40
    
    switch_points = []
    
    for episode in range(10):
        obs, info = env.reset(seed=episode)
        switch_point = info["switch_point"]
        switch_points.append(switch_point)
        
        # Assert switch_point is in valid range
        assert lower_bound <= switch_point <= upper_bound, (
            f"Episode {episode}: switch_point {switch_point} not in [{lower_bound}, {upper_bound}]"
        )
    
    # Verify we got 10 switch points
    assert len(switch_points) == 10
    
    # Sanity check: not all switch points should be the same (randomness)
    assert len(set(switch_points)) > 1, "All switch points are identical (not random)"


def test_info_keys_present(predator_prey_env):
    """Test that step() returns required info keys after every step."""
    obs, info = predator_prey_env.reset(seed=0)
    
    # Run 20 steps
    for step in range(20):
        action = predator_prey_env.rng.randint(0, 5)  # Random action
        next_obs, reward, terminated, truncated, info = predator_prey_env.step(action)
        
        # Check required info keys
        assert "other_action" in info, f"Step {step}: missing 'other_action' in info"
        assert "achieved_goal" in info, f"Step {step}: missing 'achieved_goal' in info"
        assert "desired_goal" in info, f"Step {step}: missing 'desired_goal' in info"
        
        # Check types
        assert isinstance(info["other_action"], (int, np.integer, np.ndarray))
        assert isinstance(info["achieved_goal"], np.ndarray)
        assert isinstance(info["desired_goal"], np.ndarray)
        
        if terminated or truncated:
            break


def test_compute_reward_jax_matches_numpy():
    """Test that compute_reward_jax and compute_reward agree."""
    env = PredatorPreyEnv(max_episode_length=50, grid_size=16, seed=42)
    
    # Create a dummy model
    dummy_model = AgentModel(
        model_id=0,
        name="test_model",
        reward_weights=np.ones(5),
        policy_params=np.ones(5),
        prior=1.0,
    )
    
    # Test 20 random (state, action, goal, reward_weights) tuples
    rng = np.random.RandomState(123)
    
    for i in range(20):
        # Generate random state
        state = rng.uniform(0, 16, size=4).astype(np.float32)
        
        # Generate random ego_action (not used in reward but required by interface)
        ego_action = rng.randint(0, 5)
        
        # Goal is just prey position for this simple env
        goal = state[2:4]
        
        # Reward weights (not used in this simple reward)
        reward_weights = rng.uniform(0, 1, size=5).astype(np.float32)
        
        # Compute reward with NumPy version
        reward_np = env.compute_reward(state, ego_action, goal, dummy_model)
        
        # Compute reward with JAX version
        reward_jax = env.compute_reward_jax(
            jnp.array(state),
            jnp.array(ego_action),
            jnp.array(goal),
            jnp.array(reward_weights),
        )
        reward_jax = float(reward_jax)
        
        # Check they match
        assert np.abs(reward_np - reward_jax) < 1e-5, (
            f"Test {i}: NumPy reward {reward_np} != JAX reward {reward_jax} "
            f"(diff: {np.abs(reward_np - reward_jax)})"
        )


def test_model_forward_pure_jax():
    """Test that predator_prey_model_forward compiles without error in JAX JIT."""
    
    # Create dummy inputs
    policy_params = jnp.ones(5)
    state = jnp.array([1.0, 2.0, 3.0, 4.0])
    
    # JIT compile the function
    jitted_model_forward = jax.jit(predator_prey_model_forward)
    
    # Call it — should not raise an error
    try:
        logits = jitted_model_forward(policy_params, state)
        assert logits.shape == (5,), f"Expected logits shape (5,), got {logits.shape}"
    except Exception as e:
        pytest.fail(f"predator_prey_model_forward failed to JIT compile: {e}")
    
    # Verify output is finite
    assert jnp.all(jnp.isfinite(logits)), "model_forward produced non-finite logits"


def test_policy_switch_occurs():
    """Test that the policy actually switches at the switch point."""
    env = PredatorPreyEnv(max_episode_length=60, grid_size=16, seed=42)
    
    obs, info = env.reset(seed=0)
    switch_point = info["switch_point"]
    initial_policy = info["initial_policy"]
    
    # Step until just before switch point
    for step in range(switch_point):
        action = 0  # Stay
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        if step < switch_point - 1:
            # Should still be initial policy
            assert info["current_policy"] == initial_policy
        
        if terminated or truncated:
            break
    
    # Step once more — should trigger switch
    if not (terminated or truncated):
        next_obs, reward, terminated, truncated, info = env.step(0)
        
        # Policy should have switched
        assert info["current_policy"] != initial_policy, (
            f"Policy did not switch at step {switch_point}. "
            f"Initial: {initial_policy}, Current: {info['current_policy']}"
        )


def test_reward_is_binary():
    """Test that reward is always 0.0 or 1.0."""
    env = PredatorPreyEnv(max_episode_length=50, grid_size=16, seed=42)
    
    obs, info = env.reset(seed=0)
    
    rewards = []
    for step in range(30):
        action = env.rng.randint(0, 5)
        next_obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        # Check reward is binary
        assert reward in [0.0, 1.0], f"Reward {reward} is not binary"
        
        if terminated or truncated:
            break
    
    assert len(rewards) > 0, "No rewards collected"


# ---------------------------------------------------------------------------
# Cooperative Navigation tests
# ---------------------------------------------------------------------------

@pytest.fixture
def coop_nav_env():
    return CooperativeNavEnv(max_episode_length=50, seed=42)


def test_coop_nav_reset_shapes(coop_nav_env):
    """CooperativeNavEnv reset() returns correct obs shape and info keys."""
    obs, info = coop_nav_env.reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.ndim == 1
    assert "switch_point" in info
    assert "initial_policy" in info
    assert "achieved_goal" in info
    assert "desired_goal" in info


def test_coop_nav_info_keys(coop_nav_env):
    """CooperativeNavEnv step() returns required info keys for 20 steps."""
    coop_nav_env.reset(seed=0)
    for _ in range(20):
        action = int(np.random.randint(0, 5))
        obs, reward, terminated, truncated, info = coop_nav_env.step(action)
        assert "other_action" in info
        assert "achieved_goal" in info
        assert "desired_goal" in info
        if terminated or truncated:
            break


def test_coop_nav_reward_jax_matches_numpy(coop_nav_env):
    """CooperativeNavEnv compute_reward_jax agrees with compute_reward."""
    dummy_model = AgentModel(
        model_id=0,
        name="uniform",
        reward_weights=np.ones(5),
        policy_params=np.array([0.0, 1.0]),
        prior=0.5,
    )
    rng = np.random.RandomState(7)
    for _ in range(10):
        state = rng.randn(4).astype(np.float32)
        goal = rng.randn(2).astype(np.float32)
        ego_action = rng.randint(0, 5)
        r_np = coop_nav_env.compute_reward(state, ego_action, goal, dummy_model)
        r_jax = float(
            coop_nav_env.compute_reward_jax(
                jnp.array(state), jnp.array(ego_action), jnp.array(goal)
            )
        )
        assert abs(r_np - r_jax) < 1e-5, f"NumPy {r_np} != JAX {r_jax}"


def test_coop_nav_model_forward_pure_jax():
    """cooperative_nav_model_forward compiles under jax.jit."""
    jitted = jax.jit(cooperative_nav_model_forward)
    params = jnp.array([2.0, 4.0])
    state = jnp.array([0.1, 0.2, 1.0, 1.0])
    logits = jitted(params, state)
    assert logits.shape == (5,)
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# Intersection tests
# ---------------------------------------------------------------------------

@pytest.fixture
def intersection_env():
    return IntersectionEnv(max_episode_length=30, seed=42)


def test_intersection_reset_shapes(intersection_env):
    """IntersectionEnv reset() returns obs as 1-D float32 array."""
    obs, info = intersection_env.reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.ndim == 1
    assert "switch_point" in info
    assert "initial_policy" in info
    assert "achieved_goal" in info


def test_intersection_info_keys(intersection_env):
    """IntersectionEnv step() returns required info keys for 10 steps."""
    intersection_env.reset(seed=0)
    for _ in range(10):
        obs, reward, terminated, truncated, info = intersection_env.step(2)
        assert "other_action" in info
        assert "achieved_goal" in info
        assert "desired_goal" in info
        if terminated or truncated:
            break


def test_intersection_reward_jax_matches_numpy(intersection_env):
    """IntersectionEnv compute_reward_jax agrees with compute_reward."""
    dummy_model = AgentModel(
        model_id=0,
        name="aggressive",
        reward_weights=np.ones(5),
        policy_params=np.array([2.0, 0.0, 4.0]),
        prior=0.333,
    )
    rng = np.random.RandomState(99)
    obs, _ = intersection_env.reset(seed=0)
    for _ in range(5):
        state = rng.randn(4).astype(np.float32)
        goal = rng.randn(2).astype(np.float32)
        ego_action = rng.randint(0, 5)
        r_np = intersection_env.compute_reward(state, ego_action, goal, dummy_model)
        r_jax = float(
            intersection_env.compute_reward_jax(
                jnp.array(state), jnp.array(ego_action), jnp.array(goal)
            )
        )
        assert abs(r_np - r_jax) < 1e-5, f"NumPy {r_np} != JAX {r_jax}"


def test_intersection_model_forward_pure_jax():
    """intersection_model_forward compiles under jax.jit."""
    jitted = jax.jit(intersection_model_forward)
    params = jnp.array([2.0, 0.0, 4.0])
    state = jnp.array([0.1, 0.2, 0.5, 0.5])
    logits = jitted(params, state)
    assert logits.shape == (5,)
    assert jnp.all(jnp.isfinite(logits))


# ---------------------------------------------------------------------------
# Hide-and-Seek tests
# ---------------------------------------------------------------------------

@pytest.fixture
def hide_and_seek_env():
    return HideAndSeekEnv(max_episode_length=30, seed=42)


def test_hide_and_seek_reset_shapes(hide_and_seek_env):
    """HideAndSeekEnv reset() returns 1-D float32 obs and required info keys."""
    obs, info = hide_and_seek_env.reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.ndim == 1
    assert "switch_point" in info
    assert "initial_policy" in info
    assert "achieved_goal" in info


def test_hide_and_seek_info_keys(hide_and_seek_env):
    """HideAndSeekEnv step() returns required info keys for 10 steps."""
    hide_and_seek_env.reset(seed=0)
    for _ in range(10):
        obs, reward, terminated, truncated, info = hide_and_seek_env.step(0)
        assert "other_action" in info
        assert "achieved_goal" in info
        assert "desired_goal" in info
        if terminated or truncated:
            break


def test_hide_and_seek_reward_jax_matches_numpy(hide_and_seek_env):
    """HideAndSeekEnv compute_reward_jax agrees with compute_reward."""
    dummy_model = AgentModel(
        model_id=0,
        name="direct",
        reward_weights=np.ones(5),
        policy_params=np.array([0.0, 2.0, 4.0]),
        prior=0.333,
    )
    rng = np.random.RandomState(55)
    for _ in range(10):
        state = rng.randn(4).astype(np.float32)
        goal = rng.randn(2).astype(np.float32)
        ego_action = rng.randint(0, 5)
        r_np = hide_and_seek_env.compute_reward(state, ego_action, goal, dummy_model)
        r_jax = float(
            hide_and_seek_env.compute_reward_jax(
                jnp.array(state), jnp.array(ego_action), jnp.array(goal)
            )
        )
        assert abs(r_np - r_jax) < 1e-5, f"NumPy {r_np} != JAX {r_jax}"


def test_hide_and_seek_model_forward_pure_jax():
    """hide_and_seek_model_forward compiles under jax.jit."""
    jitted = jax.jit(hide_and_seek_model_forward)
    params = jnp.array([0.0, 2.0, 4.0])
    state = jnp.array([0.1, 0.2, 1.0, 1.0])
    logits = jitted(params, state)
    assert logits.shape == (5,)
    assert jnp.all(jnp.isfinite(logits))
