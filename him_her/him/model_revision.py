"""Model revision via MAP selection and buffer relabeling.

MAP selection runs in JAX for efficiency. Buffer relabeling is a NumPy mutation that
happens outside the JIT boundary.
"""

import jax
import jax.numpy as jnp
from typing import Callable


@jax.jit(static_argnames=['all_model_log_likelihoods_fn'])
def select_map_model(
    stacked_policy_params: jnp.ndarray,
    log_priors: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    all_model_log_likelihoods_fn: Callable,
) -> jnp.ndarray:
    """Select the MAP (maximum a posteriori) model given trajectory.
    
    Computes: m_tilde = argmax_{m in M} [log p(m) + log L(m | tau)]
    
    Args:
        stacked_policy_params: Shape (|M|, param_dim) — all model parameters
        log_priors: Shape (|M|,) — log p(m) for each model
        states: Trajectory of states, shape (T, obs_dim)
        actions: Trajectory of actions, shape (T,)
        all_model_log_likelihoods_fn: JIT-compiled likelihood function from 
                                      make_likelihood_fns(). This is a pre-compiled
                                      closure with signature (stacked_params, states,
                                      actions) and no callable arguments. Marked as
                                      static so JAX treats it as a constant during
                                      compilation.
    
    Returns:
        Scalar integer index into ModelSet indicating the MAP model
    
    Note:
        This function is JIT-compiled with static_argnames for the likelihood function.
        This means the function will be compiled once per unique likelihood function,
        and subsequent calls with the same likelihood function will reuse the cached
        compilation. The returned value must be converted to a Python int before 
        passing to buffer relabeling: int(jax_scalar).
    
    Usage:
        >>> from him_her.him.inconsistency import make_likelihood_fns
        >>> _, all_models_fn = make_likelihood_fns(my_model_forward)
        >>> map_id = select_map_model(params, priors, states, actions, all_models_fn)
    """
    log_likelihoods = all_model_log_likelihoods_fn(
        stacked_policy_params, states, actions
    )
    log_posteriors = log_likelihoods + log_priors
    return jnp.argmax(log_posteriors)


def relabel_trajectory_in_buffer(
    buffer,
    episode_start_idx: int,
    episode_length: int,
    new_model_id: int,
) -> None:
    """Relabel all transitions in an episode with a new model_id.
    
    This is a NumPy in-place mutation that happens OUTSIDE the JAX JIT boundary.
    Rewards are NOT recomputed here — that happens during HER gradient updates where
    the reward function can be JIT-compiled.
    
    Args:
        buffer: ReplayBuffer instance (NumPy-backed)
        episode_start_idx: Index in buffer where the episode begins
        episode_length: Number of transitions in the episode
        new_model_id: The revised model_id (must be a Python int, not JAX array)
    
    Returns:
        None (modifies buffer in-place)
    
    Note:
        This function is NOT JIT-compiled. It operates on NumPy arrays and mutates
        buffer state. Call this AFTER converting the JAX scalar from select_map_model
        to a Python int.
    """
    buffer.relabel_episode(episode_start_idx, episode_length, new_model_id)
