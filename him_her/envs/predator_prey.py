"""Predator-Prey environment wrapper using PettingZoo.

Wraps pettingzoo.magent2.adversarial_pursuit_v4 for HIM+HER experiments.
The other agent switches between evasive and territorial policies at a random point.
"""

import numpy as np
import jax.numpy as jnp
import jax
from typing import Tuple, Dict
from pathlib import Path

from him_her.envs.base_env import BaseMultiAgentEnv
from him_her.models.base_model import AgentModel
from him_her.other_agents.type_agents import EvasivePolicy, TerritorialPolicy


def predator_prey_model_forward(
    policy_params: jnp.ndarray, state: jnp.ndarray
) -> jnp.ndarray:
    """Task-specific model_forward for predator-prey environment.
    
    This is the function passed to make_likelihood_fns(). It computes logits
    over the action space as a linear function of policy_params and state features.
    
    Args:
        policy_params: Parameters defining the policy, shape (param_dim,)
                      For predator-prey: shape (n_actions,) representing preference weights
        state: Current state observation, shape (obs_dim,)
    
    Returns:
        Logits over action space, shape (n_actions,)
    
    Note:
        This is a pure JAX function with no Python control flow.
        The logits are computed as a simple linear model for simplicity.
    """
    # Extract relevant features from state
    # State format: [prey_x, prey_y, predator_x, predator_y, ...]
    prey_pos = state[:2]
    predator_pos = state[2:4]
    
    # Compute basic features
    direction_away = prey_pos - predator_pos
    
    # Action directions (must match those in type_agents.py)
    action_directions = jnp.array([
        [0, 0],   # stay
        [0, 1],   # up
        [0, -1],  # down
        [-1, 0],  # left
        [1, 0],   # right
    ])
    
    # Compute preferences as dot products with direction away from predator
    raw_preferences = jnp.array([
        jnp.dot(action_dir, direction_away)
        for action_dir in action_directions
    ])
    
    # Combine with policy_params (weights for each action)
    # policy_params acts as a weighting over base preferences
    logits = policy_params * raw_preferences
    
    return logits


class PredatorPreyEnv(BaseMultiAgentEnv):
    """Predator-Prey environment with model switching.
    
    Wraps a simple 2-agent predator-prey scenario where:
    - Ego agent is the predator (trying to catch prey)
    - Other agent is the prey (uses evasive or territorial policy)
    - The prey policy switches at a random timestep in [T/3, 2T/3]
    
    This environment is crucial for HIM validation because the ground-truth
    model switch is known, allowing direct verification of HIM's detection.
    
    Attributes:
        max_episode_length: Maximum timesteps per episode
        current_step: Current timestep within episode
        switch_point: Timestep when policy switches (sampled per episode)
        current_policy: Currently active policy for the other agent
        evasive_policy: Evasive policy instance
        territorial_policy: Territorial policy instance
    """
    
    def __init__(
        self,
        max_episode_length: int = 50,
        grid_size: int = 16,
        seed: int = None,
    ):
        """Initialize the Predator-Prey environment.
        
        Args:
            max_episode_length: Maximum episode length
            grid_size: Size of the grid world
            seed: Random seed for reproducibility
        """
        self.max_episode_length = max_episode_length
        self.grid_size = grid_size
        self.current_step = 0
        self.switch_point = None
        
        # Initialize policies
        self.evasive_policy = EvasivePolicy(temperature=0.1)
        self.territorial_policy = TerritorialPolicy(
            home_center=np.array([grid_size / 2, grid_size / 2]),
            temperature=0.1
        )
        self.current_policy = self.evasive_policy
        
        # State tracking
        self.predator_pos = None
        self.prey_pos = None
        self.rng = np.random.RandomState(seed)
        
        # Capture distance threshold
        self.capture_distance = 1.0
    
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for this episode
        
        Returns:
            obs: Initial observation for ego agent (predator)
            info: Dictionary with metadata
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        
        # Sample switch point uniformly in [T/3, 2T/3]
        lower = self.max_episode_length // 3
        upper = 2 * self.max_episode_length // 3
        self.switch_point = self.rng.randint(lower, upper + 1)
        
        # Reset step counter
        self.current_step = 0
        
        # Initialize positions randomly
        self.predator_pos = self.rng.uniform(0, self.grid_size, size=2)
        self.prey_pos = self.rng.uniform(0, self.grid_size, size=2)
        
        # Start with evasive policy
        self.current_policy = self.evasive_policy

        # Construct observation
        obs = self._get_observation()

        info = {
            "switch_point": self.switch_point,
            "initial_policy": self.current_policy.name,
            "achieved_goal": self.prey_pos.copy(),
            "desired_goal": self.prey_pos.copy(),
        }

        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation for ego agent (predator).
        
        Returns:
            Observation array containing predator and prey positions
        """
        # Simple observation: [predator_x, predator_y, prey_x, prey_y]
        obs = np.concatenate([self.predator_pos, self.prey_pos])
        return obs.astype(np.float32)
    
    def _get_state(self) -> np.ndarray:
        """Construct state for other agent (prey).
        
        The prey sees its own position first, then predator position.
        
        Returns:
            State array: [prey_x, prey_y, predator_x, predator_y]
        """
        state = np.concatenate([self.prey_pos, self.predator_pos])
        return state.astype(np.float32)
    
    def step(
        self, ego_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step.
        
        Args:
            ego_action: Predator's action (integer index)
        
        Returns:
            next_obs: Next observation
            reward: Reward for predator
            terminated: Whether episode ended (capture or escape)
            truncated: Whether episode was truncated (time limit)
            info: Dictionary with other_action, achieved_goal, desired_goal
        """
        # Check if we need to switch policy
        if self.current_step == self.switch_point:
            # Silent switch: toggle between evasive and territorial
            if self.current_policy.name == "evasive":
                self.current_policy = self.territorial_policy
            else:
                self.current_policy = self.evasive_policy
        
        # Get other agent's state and action
        other_state = self._get_state()
        other_action = self.current_policy.select_action(other_state, deterministic=False)
        
        # Apply actions (move agents)
        self._apply_action(ego_action, is_predator=True)
        self._apply_action(other_action, is_predator=False)
        
        # Increment step
        self.current_step += 1
        
        # Compute distance
        distance = np.linalg.norm(self.predator_pos - self.prey_pos)
        
        # Check if captured
        captured = distance <= self.capture_distance
        
        # Compute reward
        reward = 1.0 if captured else 0.0
        
        # Check termination
        terminated = captured
        truncated = self.current_step >= self.max_episode_length
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Construct info
        info = {
            "other_action": np.array(other_action, dtype=np.int32),
            "achieved_goal": self.prey_pos.copy(),
            "desired_goal": self.prey_pos.copy(),  # In this simple version, goal = prey position
            "distance": distance,
            "current_policy": self.current_policy.name,
        }
        
        return next_obs, reward, terminated, truncated, info
    
    def _apply_action(self, action: int, is_predator: bool):
        """Apply action to move agent.
        
        Args:
            action: Action index (0=stay, 1=up, 2=down, 3=left, 4=right)
            is_predator: Whether this is the predator (True) or prey (False)
        """
        # Action to direction mapping
        action_to_direction = {
            0: np.array([0, 0]),    # stay
            1: np.array([0, 1]),    # up
            2: np.array([0, -1]),   # down
            3: np.array([-1, 0]),   # left
            4: np.array([1, 0]),    # right
        }
        
        direction = action_to_direction[int(action)]
        
        # Update position
        if is_predator:
            self.predator_pos = self.predator_pos + direction
            # Clip to grid bounds
            self.predator_pos = np.clip(self.predator_pos, 0, self.grid_size)
        else:
            self.prey_pos = self.prey_pos + direction
            self.prey_pos = np.clip(self.prey_pos, 0, self.grid_size)
    
    def compute_reward(
        self,
        state: np.ndarray,
        ego_action: np.ndarray,
        goal: np.ndarray,
        model: AgentModel,
    ) -> float:
        """Compute reward: +1 if within capture distance, 0 otherwise.
        
        Args:
            state: Current state observation
            ego_action: Predator's action
            goal: Goal (prey position to reach)
            model: Other agent's model (not used in this simple reward)
        
        Returns:
            Reward value
        """
        # Extract positions from state
        predator_pos = state[:2]
        prey_pos = state[2:4]
        
        # Compute distance
        distance = np.linalg.norm(predator_pos - prey_pos)
        
        # Reward is 1 if captured, 0 otherwise
        reward = 1.0 if distance <= self.capture_distance else 0.0
        
        return float(reward)
    
    def compute_reward_jax(
        self,
        state: jnp.ndarray,
        ego_action: jnp.ndarray,
        goal: jnp.ndarray,
        reward_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """JAX version of compute_reward for use inside update_step.
        
        Args:
            state: Current state observation
            ego_action: Predator's action
            goal: Goal (prey position)
            reward_weights: Model's reward weights (not used here)
        
        Returns:
            Reward value (scalar JAX array)
        """
        # Extract positions
        predator_pos = state[:2]
        prey_pos = state[2:4]
        
        # Compute distance
        distance = jnp.linalg.norm(predator_pos - prey_pos)
        
        # Use jnp.where for no Python control flow
        reward = jnp.where(distance <= self.capture_distance, 1.0, 0.0)
        
        return reward
    
    def get_other_action_log_probability(
        self, policy_params: np.ndarray, state: np.ndarray, action: np.ndarray
    ) -> float:
        """Compute log pi_o^m(a^o | s) for the other agent.
        
        Uses Boltzmann distribution over scripted policy's action preferences.
        
        Args:
            policy_params: Parameters defining the policy (preference weights)
            state: Current state for the other agent
            action: Action taken by other agent
        
        Returns:
            Log-probability of the action
        """
        # Extract features from state
        prey_pos = state[:2]
        predator_pos = state[2:4]
        direction_away = prey_pos - predator_pos
        
        # Compute action preferences
        action_directions = np.array([
            [0, 0],   # stay
            [0, 1],   # up
            [0, -1],  # down
            [-1, 0],  # left
            [1, 0],   # right
        ])
        
        raw_preferences = np.array([
            np.dot(action_dir, direction_away)
            for action_dir in action_directions
        ])
        
        # Weight by policy_params
        logits = policy_params * raw_preferences
        
        # Compute log-probabilities via log-softmax
        log_probs = logits - np.log(np.sum(np.exp(logits)))
        
        # Return log-prob for the given action
        action_idx = int(action)
        return float(log_probs[action_idx])
