"""Simple replay buffer for storing and sampling transitions.

This is a NumPy-backed circular buffer. All storage is NumPy arrays (not JAX).
Conversion to JAX happens once per gradient step when batches are sampled.
"""

import numpy as np
from typing import Dict, Tuple
from him_her.models.base_model import Transition


class ReplayBuffer:
    """Circular buffer for storing (s, a, s', r, done, goal, model_id) transitions.
    
    All arrays are NumPy (not JAX). This keeps the buffer outside the JIT boundary.
    
    Attributes:
        capacity: Maximum number of transitions to store
        obs_dim: Observation dimensionality
        action_dim: Action dimensionality
        goal_dim: Goal dimensionality
        size: Current number of transitions stored
        ptr: Write pointer for circular buffer
    """
    
    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        goal_dim: int,
    ):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum buffer size
            obs_dim: Observation dimensionality
            action_dim: Action dimensionality
            goal_dim: Goal dimensionality
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # Pre-allocate numpy arrays
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.goals = np.zeros((capacity, goal_dim), dtype=np.float32)
        self.model_ids = np.zeros((capacity,), dtype=np.int32)
        
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        goal: np.ndarray,
        model_id: int,
    ):
        """Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            goal: Goal for this transition
            model_id: Model ID used for this transition
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.goals[self.ptr] = goal
        self.model_ids[self.ptr] = model_id
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Dictionary with keys: states, actions, rewards, next_states, dones, goals, model_ids
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
            "goals": self.goals[indices],
            "model_ids": self.model_ids[indices],
        }

    def relabel_episode(self, episode_start_idx: int, episode_length: int, new_model_id: int) -> None:
        """Relabel the model_id for a contiguous episode segment in-place."""
        if episode_length <= 0 or self.size == 0:
            return

        for offset in range(episode_length):
            idx = (episode_start_idx + offset) % self.capacity
            self.model_ids[idx] = new_model_id
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size
