"""Model encoding utilities for JAX neural networks.

Converts discrete model IDs to continuous embeddings for neural network input.
"""

import jax
import jax.numpy as jnp
import chex

from him_her.models.base_model import ModelSet


def encode_model(model_id: int, model_set: ModelSet, mode: str = "onehot") -> jnp.ndarray:
    """Encode a model ID as a 1D embedding array.
    
    This function converts the discrete model_id (a Python int) to a continuous
    embedding that can be fed to neural networks. The encoding happens **before**
    calling the JIT-compiled update_step to prevent JAX from tracing model_id as
    a dynamic value (which would cause retracing on every HIM trigger).
    
    Args:
        model_id: Index of the model in model_set.models (Python int, not JAX array)
        model_set: ModelSet containing the hypothesis set
        mode: Encoding strategy, one of:
              - "onehot": One-hot vector of length |M|
              - "reward_weights": Model's reward weight vector
    
    Returns:
        1D JAX array encoding the model
    
    Raises:
        AssertionError: If model_id is not a Python int (prevents tracing issues)
        ValueError: If mode is not recognized
    
    Note:
        The model_id MUST be a plain Python int, never a traced JAX value. This is
        enforced by the isinstance check. This is the fix for the retracing risk
        discussed in Section 15, item 5 of the architecture document.
    
    Example:
        >>> embed = encode_model(0, model_set, mode="onehot")
        >>> assert embed.ndim == 1  # Always 1D
    """
    # CRITICAL: model_id must be a Python int, not a JAX traced value
    # This prevents JAX from retracing update_step on every HIM trigger
    assert isinstance(model_id, int), (
        f"model_id must be a Python int, got {type(model_id)}. "
        "This is required to prevent JAX retracing on model changes."
    )
    
    if mode == "onehot":
        # One-hot encoding: [0, 0, ..., 1, ..., 0] with 1 at position model_id
        num_models = len(model_set.models)
        result = jax.nn.one_hot(model_id, num_classes=num_models)
        
    elif mode == "reward_weights":
        # Use the model's reward weights directly as the embedding
        result = jnp.array(model_set.models[model_id].reward_weights)
        
    else:
        raise ValueError(
            f"Unknown encoding mode '{mode}'. Must be 'onehot' or 'reward_weights'."
        )
    
    # Ensure the result is always a 1D array
    chex.assert_rank(result, 1)
    
    return result
