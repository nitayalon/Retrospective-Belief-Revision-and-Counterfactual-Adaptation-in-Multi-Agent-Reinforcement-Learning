"""Scripted other-agent policies for controlled experiments.

These policies are used in environments where the other agent follows deterministic
rules. Each policy exposes action_log_probs(state) for likelihood computation.
"""

import numpy as np
from typing import Dict, Any


class EvasivePolicy:
    """Evasive policy: maximizes distance from ego agent.
    
    In predator-prey scenarios, this represents prey that tries to escape.
    
    Attributes:
        temperature: Boltzmann temperature for softmax over action preferences.
                    Higher temperature = more random, lower = more deterministic.
    """
    
    def __init__(self, temperature: float = 0.1):
        """Initialize evasive policy.
        
        Args:
            temperature: Boltzmann temperature for action selection
        """
        self.temperature = temperature
        self.name = "evasive"
    
    def get_action_preferences(self, state: np.ndarray) -> np.ndarray:
        """Compute unnormalized preferences (scores) for each action.
        
        For predator-prey: actions that increase distance from predator get higher scores.
        
        Args:
            state: Current state observation containing positions
        
        Returns:
            Array of shape (n_actions,) with unnormalized preference scores
        
        Note:
            This is task-specific. For predator-prey, we assume state contains:
            - prey position (first 2 dims)
            - predator position (next 2 dims)
            And actions are discrete directions: [stay, up, down, left, right]
        """
        # Extract positions (assuming 2D grid)
        # State format: [prey_x, prey_y, predator_x, predator_y, ...]
        prey_pos = state[:2]
        predator_pos = state[2:4]
        
        # Compute direction away from predator
        direction_away = prey_pos - predator_pos
        
        # Action preferences: 5 actions [stay, up, down, left, right]
        # Map actions to direction vectors
        action_directions = np.array([
            [0, 0],   # stay
            [0, 1],   # up
            [0, -1],  # down
            [-1, 0],  # left
            [1, 0],   # right
        ])
        
        # Preference = dot product with direction away from predator
        # Higher dot product = better alignment with escape direction
        preferences = np.array([
            np.dot(action_dir, direction_away)
            for action_dir in action_directions
        ])
        
        return preferences
    
    def action_log_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute log-probabilities for all actions under Boltzmann distribution.
        
        Args:
            state: Current state observation
        
        Returns:
            Array of shape (n_actions,) with log-probabilities summing to 1
        """
        preferences = self.get_action_preferences(state)
        logits = preferences / self.temperature
        # Log-softmax for numerical stability
        log_probs = logits - np.log(np.sum(np.exp(logits)))
        return log_probs
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select an action given the current state.
        
        Args:
            state: Current state observation
            deterministic: If True, return argmax; if False, sample from distribution
        
        Returns:
            Action index (integer)
        """
        log_probs = self.action_log_probs(state)
        
        if deterministic:
            return int(np.argmax(log_probs))
        else:
            probs = np.exp(log_probs)
            return int(np.random.choice(len(probs), p=probs))


class TerritorialPolicy:
    """Territorial policy: stays within a fixed home region.
    
    In predator-prey scenarios, this represents prey that defends territory
    rather than fleeing.
    
    Attributes:
        home_center: Center coordinates of the home territory
        temperature: Boltzmann temperature for action selection
    """
    
    def __init__(self, home_center: np.ndarray = None, temperature: float = 0.1):
        """Initialize territorial policy.
        
        Args:
            home_center: Center of home territory. If None, defaults to map center.
            temperature: Boltzmann temperature for action selection
        """
        self.home_center = home_center if home_center is not None else np.array([0.5, 0.5])
        self.temperature = temperature
        self.name = "territorial"
    
    def get_action_preferences(self, state: np.ndarray) -> np.ndarray:
        """Compute action preferences based on staying near home.
        
        Actions that move toward or keep agent near home get higher scores.
        
        Args:
            state: Current state observation
        
        Returns:
            Array of shape (n_actions,) with unnormalized preference scores
        """
        # Extract prey position
        prey_pos = state[:2]
        
        # Compute direction toward home
        direction_to_home = self.home_center - prey_pos
        
        # Action directions
        action_directions = np.array([
            [0, 0],   # stay
            [0, 1],   # up
            [0, -1],  # down
            [-1, 0],  # left
            [1, 0],   # right
        ])
        
        # Preference = dot product with direction to home
        # Also penalize being far from home
        distance_from_home = np.linalg.norm(direction_to_home)
        
        preferences = np.array([
            np.dot(action_dir, direction_to_home) - 0.5 * distance_from_home
            for action_dir in action_directions
        ])
        
        return preferences
    
    def action_log_probs(self, state: np.ndarray) -> np.ndarray:
        """Compute log-probabilities for all actions under Boltzmann distribution.
        
        Args:
            state: Current state observation
        
        Returns:
            Array of shape (n_actions,) with log-probabilities
        """
        preferences = self.get_action_preferences(state)
        logits = preferences / self.temperature
        log_probs = logits - np.log(np.sum(np.exp(logits)))
        return log_probs
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select an action given the current state.
        
        Args:
            state: Current state observation
            deterministic: If True, return argmax; if False, sample
        
        Returns:
            Action index (integer)
        """
        log_probs = self.action_log_probs(state)
        
        if deterministic:
            return int(np.argmax(log_probs))
        else:
            probs = np.exp(log_probs)
            return int(np.random.choice(len(probs), p=probs))
