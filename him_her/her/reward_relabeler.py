"""Reward relabeling for HER with model-conditioned rewards.

Uses the factory pattern to bake in the reward function, enabling JIT compilation.
The reward function must be a pure JAX callable injected per environment.
"""

import jax
import jax.numpy as jnp
from typing import Callable


def make_relabeler(reward_fn: Callable) -> Callable:
    """Factory function that creates a JIT-compiled reward relabeler.
    
    This function bakes in the task-specific reward function as a closure,
    avoiding the need to pass callables as runtime arguments to JIT-compiled code.
    
    Args:
        reward_fn: Task-specific pure JAX function with signature:
                   (state, ego_action, goal, reward_weights) -> scalar reward
                   Must be a pure function compatible with JAX transformations.
    
    Returns:
        A JIT-compiled function with signature:
        (state, ego_action, new_goal, new_model_reward_weights) -> scalar reward
        
    Example:
        >>> def my_reward_fn(state, ego_action, goal, reward_weights):
        ...     # Task-specific reward computation
        ...     return -jnp.linalg.norm(state[:2] - goal[:2])
        >>> relabeler = make_relabeler(my_reward_fn)
        >>> reward = relabeler(state, action, new_goal, new_weights)
    
    Note:
        The returned function is JIT-compiled and can be called inside the
        JIT-compiled update_step. All inputs must be JAX arrays.
    """
    
    @jax.jit
    def relabel_reward(
        state: jnp.ndarray,
        ego_action: jnp.ndarray,
        new_goal: jnp.ndarray,
        new_model_reward_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """Recompute reward r_e(s_t, a^e_t, g_tilde, m_tilde).
        
        Called inside the JIT-compiled update_step — all inputs must be JAX arrays.
        
        Args:
            state: Current state observation (JAX array)
            ego_action: Ego agent's action (JAX array)
            new_goal: Relabeled goal (JAX array)
            new_model_reward_weights: Reward weights from the revised model (JAX array)
        
        Returns:
            Scalar reward (JAX array)
        
        Note:
            The reward_fn is closed over (baked in) from the factory, so it doesn't
            need to be passed as an argument.
        """
        return reward_fn(state, ego_action, new_goal, new_model_reward_weights)
    
    return relabel_reward
