"""Critic network with double-Q to reduce overestimation bias.

Flax Linen implementation of Q(s, a, g, m_embed) with two independent Q-functions.
"""

import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence, Tuple


class Critic(nn.Module):
    """Double-Q critic network for goal- and model-conditioned value estimation.
    
    Implements two independent Q-functions Q1 and Q2 to reduce overestimation bias.
    The two networks do NOT share weights — they are completely independent.
    
    Attributes:
        hidden_sizes: Sequence of hidden layer dimensions (e.g., [256, 256])
    
    Note:
        Flax modules are stateless. Parameters are initialized separately and
        passed explicitly via apply(). This module does not store parameters.
    
    Example:
        >>> critic = Critic(hidden_sizes=[256, 256])
        >>> params = critic.init(rng, obs, action, goal, model_embed)
        >>> q1, q2 = critic.apply(params, obs, action, goal, model_embed)
    """
    hidden_sizes: Sequence[int]
    
    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        goal: jnp.ndarray,
        model_embed: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass through both Q-networks.
        
        Args:
            obs: Observation/state, shape (batch, obs_dim)
            action: Action, shape (batch, action_dim)
            goal: Goal, shape (batch, goal_dim)
            model_embed: Model embedding, shape (batch, model_embed_dim)
        
        Returns:
            Tuple of (q1, q2) where:
            - q1: Q-value from first network, shape (batch, 1)
            - q2: Q-value from second network, shape (batch, 1)
        
        Note:
            The two Q-networks are completely independent — they do not share weights.
            This is achieved by using separate Dense layers for each path.
        """
        # Concatenate all inputs along the last axis
        x = jnp.concatenate([obs, action, goal, model_embed], axis=-1)
        
        # Initialize two independent paths with the same input
        q1 = x
        q2 = x
        
        # Process through independent hidden layers
        for i, hidden_size in enumerate(self.hidden_sizes):
            # Q1 path - independent Dense layers
            q1 = nn.Dense(hidden_size, name=f'q1_dense_{i}')(q1)
            q1 = nn.relu(q1)
            
            # Q2 path - independent Dense layers (different parameters)
            q2 = nn.Dense(hidden_size, name=f'q2_dense_{i}')(q2)
            q2 = nn.relu(q2)
        
        # Output layers - each produces a scalar Q-value
        q1 = nn.Dense(1, name='q1_out')(q1)
        q2 = nn.Dense(1, name='q2_out')(q2)
        
        return q1, q2
