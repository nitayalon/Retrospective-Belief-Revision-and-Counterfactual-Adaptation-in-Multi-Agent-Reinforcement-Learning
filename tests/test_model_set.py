"""Tests for core data structures (Transition, AgentModel, Episode).

These tests verify that the dataclasses instantiate correctly and use NumPy arrays
(not JAX arrays or callables) to maintain compatibility with the JAX/NumPy boundary.
"""

import numpy as np
import pytest

from him_her.models.base_model import Transition, AgentModel, Episode


class TestTransition:
    """Tests for Transition dataclass."""
    
    def test_transition_instantiation(self):
        """Test that Transition can be instantiated with valid numpy arrays."""
        state = np.array([1.0, 2.0, 3.0])
        ego_action = np.array([0.5])
        other_action = np.array([0.3])
        next_state = np.array([1.1, 2.1, 3.1])
        goal = np.array([5.0, 5.0])
        model_id = 0
        reward = 1.0
        done = False
        
        transition = Transition(
            state=state,
            ego_action=ego_action,
            other_action=other_action,
            next_state=next_state,
            goal=goal,
            model_id=model_id,
            reward=reward,
            done=done,
        )
        
        assert np.array_equal(transition.state, state)
        assert np.array_equal(transition.ego_action, ego_action)
        assert np.array_equal(transition.other_action, other_action)
        assert np.array_equal(transition.next_state, next_state)
        assert np.array_equal(transition.goal, goal)
        assert transition.model_id == model_id
        assert transition.reward == reward
        assert transition.done == done
    
    def test_transition_arrays_are_numpy(self):
        """Verify that Transition uses numpy arrays, not JAX arrays."""
        state = np.array([1.0, 2.0])
        ego_action = np.array([0.5])
        other_action = np.array([0.3])
        next_state = np.array([1.1, 2.1])
        goal = np.array([5.0])
        
        transition = Transition(
            state=state,
            ego_action=ego_action,
            other_action=other_action,
            next_state=next_state,
            goal=goal,
            model_id=0,
            reward=1.0,
            done=False,
        )
        
        # Verify all array fields are numpy arrays
        assert isinstance(transition.state, np.ndarray)
        assert isinstance(transition.ego_action, np.ndarray)
        assert isinstance(transition.other_action, np.ndarray)
        assert isinstance(transition.next_state, np.ndarray)
        assert isinstance(transition.goal, np.ndarray)


class TestAgentModel:
    """Tests for AgentModel dataclass."""
    
    def test_agent_model_instantiation(self):
        """Test that AgentModel can be instantiated with policy_params as array."""
        model_id = 0
        name = "test_model"
        reward_weights = np.array([1.0, 0.5, 0.3])
        policy_params = np.array([0.1, 0.2, 0.3, 0.4])  # Parameters, not a callable
        prior = 0.5
        
        model = AgentModel(
            model_id=model_id,
            name=name,
            reward_weights=reward_weights,
            policy_params=policy_params,
            prior=prior,
        )
        
        assert model.model_id == model_id
        assert model.name == name
        assert np.array_equal(model.reward_weights, reward_weights)
        assert np.array_equal(model.policy_params, policy_params)
        assert model.prior == prior
    
    def test_agent_model_policy_params_is_array(self):
        """Verify that policy_params is an array (not a callable).
        
        This is critical for JAX vmap compatibility. The policy function lives
        separately as a pure JAX function in him/inconsistency.py.
        """
        policy_params = np.array([0.1, 0.2, 0.3])
        
        model = AgentModel(
            model_id=0,
            name="test",
            reward_weights=np.array([1.0]),
            policy_params=policy_params,
            prior=1.0,
        )
        
        # Verify policy_params is an array, not a function
        assert isinstance(model.policy_params, np.ndarray)
        assert not callable(model.policy_params)
    
    def test_agent_model_arrays_are_numpy(self):
        """Verify that AgentModel uses numpy arrays, not JAX arrays."""
        reward_weights = np.array([1.0, 0.5])
        policy_params = np.array([0.1, 0.2])
        
        model = AgentModel(
            model_id=0,
            name="test",
            reward_weights=reward_weights,
            policy_params=policy_params,
            prior=0.5,
        )
        
        assert isinstance(model.reward_weights, np.ndarray)
        assert isinstance(model.policy_params, np.ndarray)


class TestEpisode:
    """Tests for Episode dataclass."""
    
    def test_episode_instantiation(self):
        """Test that Episode can be instantiated with a list of Transitions."""
        # Create a few transitions
        transitions = []
        for i in range(3):
            transition = Transition(
                state=np.array([float(i), float(i)]),
                ego_action=np.array([0.5]),
                other_action=np.array([0.3]),
                next_state=np.array([float(i+1), float(i+1)]),
                goal=np.array([5.0]),
                model_id=0,
                reward=1.0,
                done=(i == 2),
            )
            transitions.append(transition)
        
        total_reward = 3.0
        model_used = 0
        him_triggered = False
        revised_model = None
        
        episode = Episode(
            transitions=transitions,
            total_reward=total_reward,
            model_used=model_used,
            him_triggered=him_triggered,
            revised_model=revised_model,
        )
        
        assert len(episode.transitions) == 3
        assert episode.total_reward == total_reward
        assert episode.model_used == model_used
        assert episode.him_triggered == him_triggered
        assert episode.revised_model is None
    
    def test_episode_with_him_triggered(self):
        """Test Episode when HIM is triggered and model is revised."""
        transition = Transition(
            state=np.array([1.0]),
            ego_action=np.array([0.5]),
            other_action=np.array([0.3]),
            next_state=np.array([1.1]),
            goal=np.array([5.0]),
            model_id=0,
            reward=1.0,
            done=True,
        )
        
        episode = Episode(
            transitions=[transition],
            total_reward=1.0,
            model_used=0,
            him_triggered=True,
            revised_model=1,  # Model was revised from 0 to 1
        )
        
        assert episode.him_triggered is True
        assert episode.revised_model == 1
    
    def test_episode_empty_transitions(self):
        """Test that Episode can be created with an empty transition list."""
        episode = Episode(
            transitions=[],
            total_reward=0.0,
            model_used=0,
            him_triggered=False,
            revised_model=None,
        )
        
        assert len(episode.transitions) == 0
        assert episode.total_reward == 0.0
