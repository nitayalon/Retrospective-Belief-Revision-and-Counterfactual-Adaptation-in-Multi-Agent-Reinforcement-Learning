"""Tests for HER (Hindsight Experience Replay) module.

Tests cover goal sampling strategies, reward relabeling, and batch augmentation.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from him_her.models.base_model import Transition, Episode
from him_her.her.goal_sampler import GoalSampler
from him_her.her.reward_relabeler import make_relabeler
from him_her.her.her_buffer import apply_her


@pytest.fixture
def test_episode():
    """Create a simple test episode with 10 transitions."""
    transitions = []
    for t in range(10):
        transition = Transition(
            state=np.array([float(t), float(t) * 0.5, 0.0]),
            ego_action=np.array([0.1 * t]),
            other_action=np.array([0.2 * t]),
            next_state=np.array([float(t + 1), float(t + 1) * 0.5, 0.0]),
            goal=np.array([10.0, 5.0, 0.0]),  # Same goal for all
            model_id=0,
            reward=-1.0,
            done=(t == 9),
        )
        transitions.append(transition)
    
    episode = Episode(
        transitions=transitions,
        total_reward=-10.0,
        model_used=0,
        him_triggered=False,
        revised_model=None,
    )
    
    return episode


@pytest.fixture
def goal_sampler():
    """Create a GoalSampler instance."""
    return GoalSampler()


@pytest.fixture
def simple_reward_fn():
    """Simple distance-based reward function for testing."""
    def reward_fn(state, ego_action, goal, reward_weights):
        # Negative L2 distance to goal (weighted)
        distance = jnp.linalg.norm(state[:2] - goal[:2])
        return -distance * reward_weights[0]
    return reward_fn


@pytest.fixture
def relabeler(simple_reward_fn):
    """Create a JIT-compiled relabeler using the factory pattern."""
    return make_relabeler(simple_reward_fn)


class TestGoalSampler:
    """Tests for GoalSampler strategies."""
    
    def test_future_goals_are_future(self, goal_sampler, test_episode):
        """Test that future strategy only returns goals from future timesteps."""
        rng = np.random.default_rng(42)
        transition_idx = 3
        k = 4
        
        # Sample future goals
        goals = goal_sampler.sample_goals(
            episode=test_episode,
            transition_idx=transition_idx,
            strategy="future",
            k=k,
            rng=rng,
        )
        
        # Each goal should come from a state at index > transition_idx
        # Since we're sampling next_states, they should correspond to indices 4-9
        # The next_states at these indices are [4.0, 2.0, 0.0], [5.0, 2.5, 0.0], etc.
        
        # Verify that all goals have first component > transition_idx
        # (since next_state at index i has first component i+1)
        for goal in goals:
            # Goal should be a next_state from a future transition
            # next_state[0] = t + 1 where t > transition_idx
            # So goal[0] should be > transition_idx + 1
            assert goal[0] > float(transition_idx + 1) - 0.1  # Allow small numerical error
    
    def test_k_goals_returned(self, goal_sampler, test_episode):
        """Test that each strategy returns exactly k goals."""
        rng = np.random.default_rng(42)
        k = 4
        
        for strategy in ["future", "episode", "final"]:
            goals = goal_sampler.sample_goals(
                episode=test_episode,
                transition_idx=3,
                strategy=strategy,
                k=k,
                rng=rng,
            )
            
            assert len(goals) == k, f"Strategy '{strategy}' should return {k} goals"
            
            # All goals should be NumPy arrays
            for goal in goals:
                assert isinstance(goal, np.ndarray)
    
    def test_final_strategy_constant(self, goal_sampler, test_episode):
        """Test that final strategy returns identical goals."""
        k = 5
        
        goals = goal_sampler.sample_goals(
            episode=test_episode,
            transition_idx=3,
            strategy="final",
            k=k,
            rng=None,  # final doesn't need rng
        )
        
        # All goals should be identical (the final state)
        final_state = test_episode.transitions[-1].next_state
        
        for goal in goals:
            np.testing.assert_array_equal(goal, final_state)
    
    def test_episode_strategy_samples_from_episode(self, goal_sampler, test_episode):
        """Test that episode strategy samples from the entire episode."""
        rng = np.random.default_rng(42)
        k = 20  # More than episode length to force replacement
        
        goals = goal_sampler.sample_goals(
            episode=test_episode,
            transition_idx=3,
            strategy="episode",
            k=k,
            rng=rng,
        )
        
        assert len(goals) == k
        
        # All goals should be valid next_states from the episode
        episode_next_states = [t.next_state for t in test_episode.transitions]
        
        for goal in goals:
            # Check that this goal matches at least one next_state in the episode
            matches = [np.allclose(goal, state) for state in episode_next_states]
            assert any(matches), "Goal should match a state from the episode"
    
    def test_random_strategy_not_implemented(self, goal_sampler, test_episode):
        """Test that random strategy raises NotImplementedError."""
        rng = np.random.default_rng(42)
        
        with pytest.raises(NotImplementedError):
            goal_sampler.sample_goals(
                episode=test_episode,
                transition_idx=3,
                strategy="random",
                k=4,
                rng=rng,
            )
    
    def test_invalid_strategy_raises_error(self, goal_sampler, test_episode):
        """Test that invalid strategy raises ValueError."""
        rng = np.random.default_rng(42)
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            goal_sampler.sample_goals(
                episode=test_episode,
                transition_idx=3,
                strategy="invalid_strategy",
                k=4,
                rng=rng,
            )


class TestRewardRelabeler:
    """Tests for reward relabeling with model-conditioned rewards."""
    
    def test_relabeler_deterministic(self, relabeler):
        """Test that relabeler returns the same result on repeated calls."""
        state = jnp.array([1.0, 2.0, 3.0])
        ego_action = jnp.array([0.5])
        goal = jnp.array([5.0, 6.0, 7.0])
        reward_weights = jnp.array([1.0])
        
        # Call relabeler twice with identical inputs
        reward_1 = relabeler(state, ego_action, goal, reward_weights)
        reward_2 = relabeler(state, ego_action, goal, reward_weights)
        
        # Results should be identical
        assert jnp.allclose(reward_1, reward_2)
        
        # Should be a scalar
        assert reward_1.shape == ()
        assert reward_2.shape == ()
    
    def test_relabeler_different_goals_different_rewards(self, relabeler):
        """Test that different goals produce different rewards."""
        state = jnp.array([1.0, 2.0, 3.0])
        ego_action = jnp.array([0.5])
        reward_weights = jnp.array([1.0])
        
        goal_1 = jnp.array([1.0, 2.0, 0.0])  # Close to state
        goal_2 = jnp.array([10.0, 20.0, 0.0])  # Far from state
        
        reward_1 = relabeler(state, ego_action, goal_1, reward_weights)
        reward_2 = relabeler(state, ego_action, goal_2, reward_weights)
        
        # Rewards should be different
        assert not jnp.allclose(reward_1, reward_2)
        
        # Reward for closer goal should be higher (less negative)
        assert reward_1 > reward_2
    
    def test_relabeler_accepts_jax_arrays(self, relabeler):
        """Test that relabeler accepts JAX arrays."""
        state = jnp.array([1.0, 2.0, 3.0])
        ego_action = jnp.array([0.5])
        goal = jnp.array([5.0, 6.0, 7.0])
        reward_weights = jnp.array([1.0])
        
        # Should not raise an error
        reward = relabeler(state, ego_action, goal, reward_weights)
        
        # Result should be a JAX array
        assert isinstance(reward, jnp.ndarray)


class TestHERBuffer:
    """Tests for HER batch augmentation."""
    
    def test_her_buffer_augmentation(self, test_episode, goal_sampler, relabeler):
        """Test that apply_her augments batch to (1 + k) * batch_size."""
        rng = np.random.default_rng(42)
        batch_size = 8
        k = 4
        
        # Create a simple batch
        batch = {
            'states': np.random.randn(batch_size, 3).astype(np.float32),
            'ego_actions': np.random.randn(batch_size, 1).astype(np.float32),
            'other_actions': np.random.randn(batch_size, 1).astype(np.float32),
            'next_states': np.random.randn(batch_size, 3).astype(np.float32),
            'goals': np.random.randn(batch_size, 3).astype(np.float32),
            'model_ids': np.zeros(batch_size, dtype=np.int32),
            'rewards': np.random.randn(batch_size).astype(np.float32),
            'dones': np.zeros(batch_size, dtype=bool),
        }
        
        # Apply HER
        augmented = apply_her(
            batch=batch,
            episode=test_episode,
            goal_sampler=goal_sampler,
            relabeler=relabeler,
            k=k,
            rng=rng,
            strategy="final",  # Use final strategy for determinism
        )
        
        # Check that augmented batch has (1 + k) * batch_size transitions
        expected_size = (1 + k) * batch_size
        assert augmented['states'].shape[0] == expected_size
        assert augmented['ego_actions'].shape[0] == expected_size
        assert augmented['other_actions'].shape[0] == expected_size
        assert augmented['next_states'].shape[0] == expected_size
        assert augmented['goals'].shape[0] == expected_size
        assert augmented['model_ids'].shape[0] == expected_size
        assert augmented['rewards'].shape[0] == expected_size
        assert augmented['dones'].shape[0] == expected_size
    
    def test_no_jax_arrays_in_her_buffer(self, test_episode, goal_sampler, relabeler):
        """Test that all arrays in augmented batch are NumPy, not JAX."""
        rng = np.random.default_rng(42)
        batch_size = 4
        k = 2
        
        # Create a simple batch
        batch = {
            'states': np.random.randn(batch_size, 3).astype(np.float32),
            'ego_actions': np.random.randn(batch_size, 1).astype(np.float32),
            'other_actions': np.random.randn(batch_size, 1).astype(np.float32),
            'next_states': np.random.randn(batch_size, 3).astype(np.float32),
            'goals': np.random.randn(batch_size, 3).astype(np.float32),
            'model_ids': np.zeros(batch_size, dtype=np.int32),
            'rewards': np.random.randn(batch_size).astype(np.float32),
            'dones': np.zeros(batch_size, dtype=bool),
        }
        
        # Apply HER
        augmented = apply_her(
            batch=batch,
            episode=test_episode,
            goal_sampler=goal_sampler,
            relabeler=relabeler,
            k=k,
            rng=rng,
            strategy="final",
        )
        
        # All arrays should be NumPy, not JAX
        for key, value in augmented.items():
            assert isinstance(value, np.ndarray), f"{key} should be np.ndarray"
            assert not isinstance(value, jnp.ndarray), f"{key} should not be jnp.ndarray"
            # Note: jnp.ndarray is a subclass of np.ndarray, so we need both checks
            assert type(value).__module__ == 'numpy', f"{key} should be from numpy module"
    
    def test_her_preserves_original_transitions(self, test_episode, goal_sampler, relabeler):
        """Test that apply_her preserves original transitions in the augmented batch."""
        rng = np.random.default_rng(42)
        batch_size = 4
        k = 2
        
        # Create a simple batch with distinctive values
        batch = {
            'states': np.arange(batch_size * 3).reshape(batch_size, 3).astype(np.float32),
            'ego_actions': np.arange(batch_size).reshape(batch_size, 1).astype(np.float32),
            'other_actions': np.arange(batch_size).reshape(batch_size, 1).astype(np.float32),
            'next_states': np.arange(batch_size * 3).reshape(batch_size, 3).astype(np.float32) + 100,
            'goals': np.arange(batch_size * 3).reshape(batch_size, 3).astype(np.float32) + 200,
            'model_ids': np.arange(batch_size, dtype=np.int32),
            'rewards': np.arange(batch_size, dtype=np.float32) - 10,
            'dones': np.array([False, False, True, False], dtype=bool),
        }
        
        # Apply HER
        augmented = apply_her(
            batch=batch,
            episode=test_episode,
            goal_sampler=goal_sampler,
            relabeler=relabeler,
            k=k,
            rng=rng,
            strategy="final",
        )
        
        # First batch_size transitions should be identical to original
        np.testing.assert_array_equal(augmented['states'][:batch_size], batch['states'])
        np.testing.assert_array_equal(augmented['ego_actions'][:batch_size], batch['ego_actions'])
        np.testing.assert_array_equal(augmented['other_actions'][:batch_size], batch['other_actions'])
        np.testing.assert_array_equal(augmented['next_states'][:batch_size], batch['next_states'])
        np.testing.assert_array_equal(augmented['goals'][:batch_size], batch['goals'])
        np.testing.assert_array_equal(augmented['model_ids'][:batch_size], batch['model_ids'])
        np.testing.assert_array_equal(augmented['rewards'][:batch_size], batch['rewards'])
        np.testing.assert_array_equal(augmented['dones'][:batch_size], batch['dones'])
