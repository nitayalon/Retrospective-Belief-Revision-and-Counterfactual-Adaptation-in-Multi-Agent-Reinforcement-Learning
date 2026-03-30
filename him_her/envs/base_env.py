"""Abstract base class for multi-agent environment wrappers.

Environment stepping is ALWAYS outside the JAX boundary — never JIT-compile env.step().
All inputs and outputs are NumPy arrays, not JAX arrays.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict
import numpy as np

from him_her.models.base_model import AgentModel


class BaseMultiAgentEnv(ABC):
    """Abstract interface for multi-agent environments.
    
    All concrete environment implementations must inherit from this class and implement
    the abstract methods. This ensures a consistent interface across all environments
    (Predator-Prey, Cooperative Navigation, Intersection, Hide-and-Seek).
    
    Note:
        Environment stepping happens outside the JAX JIT boundary. All arrays are
        NumPy arrays (np.ndarray), never JAX arrays (jnp.ndarray). This separation
        is critical for the JAX/NumPy boundary architecture (see architecture.md §5).
    """
    
    @abstractmethod
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to an initial state.
        
        Returns:
            obs: NumPy array containing the ego agent's observation
            info: Dictionary with auxiliary information
        
        Note:
            Returns NumPy arrays, not JAX arrays.
        """
        pass
    
    @abstractmethod
    def step(
        self, ego_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment.
        
        Args:
            ego_action: NumPy array containing the ego agent's action
        
        Returns:
            next_obs: NumPy array of next observation
            reward: Scalar reward for the ego agent
            terminated: Whether the episode has terminated (goal reached/failed)
            truncated: Whether the episode was truncated (time limit)
            info: Dictionary with required keys:
                - "other_action": np.ndarray — observed other agent action
                - "achieved_goal": np.ndarray — achieved goal for HER relabeling
                - "desired_goal": np.ndarray — original desired goal
        
        Note:
            All arrays are NumPy arrays. This method is never JIT-compiled.
        """
        pass
    
    @abstractmethod
    def compute_reward(
        self,
        state: np.ndarray,
        ego_action: np.ndarray,
        goal: np.ndarray,
        model: AgentModel,
    ) -> float:
        """Task-specific reward function r_e(s, a, g, m).
        
        This NumPy version is used for debugging and unit tests. Task wrappers should
        also expose a compute_reward_jax method (pure JAX) for use inside update_step.
        
        Args:
            state: Current state observation (NumPy array)
            ego_action: Ego agent's action (NumPy array)
            goal: Goal specification (NumPy array)
            model: Other agent's model (contains reward_weights)
        
        Returns:
            Scalar reward value
        
        Note:
            Must be recomputable for HER goal relabeling. Should not rely on hidden
            environment state that isn't captured in the (state, action, goal, model) tuple.
        """
        pass
    
    @abstractmethod
    def get_other_action_log_probability(
        self, policy_params: np.ndarray, state: np.ndarray, action: np.ndarray
    ) -> float:
        """Compute log pi_o^m(a^o | s) for the other agent.
        
        This NumPy version is used for debugging and unit tests only. The actual
        likelihood computation happens in JAX (see him/inconsistency.py) and is
        vmapped over all models for efficiency.
        
        Args:
            policy_params: Parameters defining the other agent's policy (NumPy array)
            state: Current state observation (NumPy array)
            action: Other agent's action (NumPy array)
        
        Returns:
            Log-probability of the action under the policy
        
        Note:
            The JAX version (for HIM likelihood computation) lives in him/inconsistency.py
            and operates on JAX arrays. This method is for testing/validation only.
        """
        pass
