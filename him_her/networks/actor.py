"""Actor network for goal- and model-conditioned policy.

Flax Linen implementation of pi_e(a | s, g, m_embed).
"""

import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Tuple


class Actor(nn.Module):
    """Goal- and model-conditioned policy network.
    
    Outputs a Gaussian action distribution: pi_e(a | s, g, m_embed).
    
    Attributes:
        hidden_sizes: Sequence of hidden layer dimensions (e.g., [256, 256])
        action_dim: Dimensionality of the action space
    
    Note:
        Flax modules are stateless. Parameters are initialized separately and
        passed explicitly via apply(). This module does not store parameters.
    
    Example:
        >>> actor = Actor(hidden_sizes=[256, 256], action_dim=2)
        >>> params = actor.init(rng, obs, goal, model_embed)
        >>> mean, log_std = actor.apply(params, obs, goal, model_embed)
    """
    hidden_sizes: Sequence[int]
    action_dim: int
    
    @nn.compact
    def __call__(
        self, 
        obs: jnp.ndarray, 
        goal: jnp.ndarray, 
        model_embed: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through the actor network.
        
        Args:
            obs: Observation/state, shape (batch, obs_dim)
            goal: Goal, shape (batch, goal_dim)
            model_embed: Model embedding, shape (batch, model_embed_dim)
        
        Returns:
            Tuple of (mean, log_std) where:
            - mean: Action mean, shape (batch, action_dim)
            - log_std: Log standard deviation, shape (batch, action_dim), clipped to [-5, 2]
        
        Note:
            The log_std is clipped to [-5, 2] to prevent numerical issues.
            This corresponds to std in [exp(-5), exp(2)] ≈ [0.0067, 7.39].
        """
        # Concatenate all inputs along the last axis
        x = jnp.concatenate([obs, goal, model_embed], axis=-1)
        
        # Hidden layers with ReLU activation
        for hidden_size in self.hidden_sizes:
            x = nn.Dense(hidden_size)(x)
            x = nn.relu(x)
        
        # Two separate output heads for mean and log_std
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        
        # Clip log_std to prevent numerical issues
        log_std = jnp.clip(log_std, -5.0, 2.0)
        
        return mean, log_std
