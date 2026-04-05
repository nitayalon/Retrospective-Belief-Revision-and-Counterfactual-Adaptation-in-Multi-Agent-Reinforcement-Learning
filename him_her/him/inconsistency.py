"""Likelihood computation and inconsistency detection using JAX.

All functions are pure JAX — JIT-compiled and vmapped over the model dimension.
The key advantage: likelihoods under all |M| models are computed in a single vmap call
rather than a Python loop.

IMPORTANT: To avoid passing functions as runtime arguments to JIT-compiled code, use the
factory pattern via make_likelihood_fns(). This returns closures with model_forward baked in.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple


def make_likelihood_fns(model_forward: Callable) -> Tuple[Callable, Callable]:
    """Factory function that creates JIT-compiled likelihood functions.
    
    This is the RECOMMENDED way to create likelihood functions. It avoids passing
    model_forward as a runtime argument to JIT-compiled code, which JAX cannot trace.
    
    Args:
        model_forward: Task-specific pure JAX function that computes logits from 
                      (params, state). Must be a pure function with signature:
                      (policy_params: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray
    
    Returns:
        A tuple of two JIT-compiled functions:
        - trajectory_log_likelihood_fn(policy_params, states, actions) -> scalar
        - all_model_log_likelihoods_fn(stacked_policy_params, states, actions) -> array
    
    Example:
        >>> def my_model_forward(params, state):
        ...     return jnp.dot(params, state)  # Linear model
        >>> traj_fn, all_models_fn = make_likelihood_fns(my_model_forward)
        >>> log_lik = traj_fn(params, states, actions)
    
    Note:
        The returned functions have model_forward closed over (baked in), so they only
        take array arguments and can be safely JIT-compiled and vmapped.
    """
    
    def single_step_log_prob(
        policy_params: jnp.ndarray,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log pi_o^m(a | s) for one timestep.
        
        Args:
            policy_params: Parameters defining the other agent's policy
            state: Current state observation
            action: Action taken (scalar index for discrete actions)
        
        Returns:
            Log-probability of the action under the policy
        """
        logits = model_forward(policy_params, state)
        log_probs = jax.nn.log_softmax(logits)
        # For discrete actions: index into log_probs
        # action is assumed to be an integer index
        return log_probs[jnp.int32(action)]
    
    @jax.jit
    def trajectory_log_likelihood_fn(
        policy_params: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log L(m | tau) = sum_t log pi_o^m(a^o_t | s_t).
        
        Uses vmap to vectorize over the timestep dimension.
        
        Args:
            policy_params: Parameters defining the other agent's policy
            states: Trajectory of states, shape (T, obs_dim)
            actions: Trajectory of actions, shape (T,)
        
        Returns:
            Log-likelihood of the trajectory under the model (scalar)
        """
        # vmap over timesteps: policy_params is constant, states and actions are batched
        log_probs = jax.vmap(
            lambda s, a: single_step_log_prob(policy_params, s, a),
            in_axes=(0, 0)
        )(states, actions)
        return jnp.sum(log_probs)
    
    @jax.jit
    def all_model_log_likelihoods_fn(
        stacked_policy_params: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute log L(m | tau) for ALL models simultaneously via vmap.
        
        This is the key function that enables efficient evaluation of all models in the
        hypothesis set without a Python loop.
        
        Args:
            stacked_policy_params: Shape (|M|, param_dim) — all model parameters stacked
            states: Trajectory of states, shape (T, obs_dim)
            actions: Trajectory of actions, shape (T,)
        
        Returns:
            Log-likelihoods for all models, shape (|M|,)
        """
        # vmap over models: states and actions are constant, policy_params is batched
        return jax.vmap(
            lambda params: trajectory_log_likelihood_fn(params, states, actions),
            in_axes=(0,)
        )(stacked_policy_params)
    
    return trajectory_log_likelihood_fn, all_model_log_likelihoods_fn


@jax.jit
def is_inconsistent_ratio(
    current_model_id: int,
    all_log_likelihoods: jnp.ndarray,
    log_priors: jnp.ndarray,
    ratio_delta: float,
) -> jnp.ndarray:
    """Trigger HIM if the best alternative model's log-posterior exceeds current model's.
    
    Uses likelihood ratio test rather than absolute threshold. This is preferred because
    it avoids per-task tuning of an absolute epsilon threshold.
    
    Args:
        current_model_id: Index of the currently assumed model
        all_log_likelihoods: Log-likelihoods for all models, shape (|M|,)
        log_priors: Log-priors p(m) for all models, shape (|M|,)
        ratio_delta: Threshold for log-posterior gap to trigger HIM
    
    Returns:
        Boolean (as JAX array) indicating whether HIM should be triggered
    
    Note:
        Returns True if:
            max_{m != current} [log p(m) + log L(m | tau)] - 
            [log p(current) + log L(current | tau)] > ratio_delta
    """
    log_posteriors = all_log_likelihoods + log_priors
    
    # Find the best alternative model (excluding current model)
    # Set current model's posterior to -inf so it's not selected
    masked_posteriors = jnp.where(
        jnp.arange(len(log_priors)) == current_model_id,
        -jnp.inf,
        log_posteriors
    )
    best_alternative = jnp.max(masked_posteriors)
    
    # Trigger HIM if the gap exceeds ratio_delta
    return best_alternative - log_posteriors[current_model_id] > ratio_delta


@jax.jit
def is_inconsistent_absolute(
    current_model_id: int,
    all_log_likelihoods: jnp.ndarray,
    threshold: float,
) -> jnp.ndarray:
    """Trigger HIM if current model's log-likelihood falls below threshold.
    
    This is an alternative to ratio mode. Less recommended because it requires
    per-task tuning of the threshold.
    
    Args:
        current_model_id: Index of the currently assumed model
        all_log_likelihoods: Log-likelihoods for all models, shape (|M|,)
        threshold: Absolute log-likelihood threshold (e.g., -10.0)
    
    Returns:
        Boolean (as JAX array) indicating whether HIM should be triggered
    """
    return all_log_likelihoods[current_model_id] < threshold


# Example model_forward function for testing
def example_linear_model_forward(policy_params: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
    """Example linear model: logits = policy_params @ state.
    
    This is a simple placeholder for testing. Real tasks should provide their own
    model_forward function that matches their policy architecture.
    
    Args:
        policy_params: Shape (n_actions, obs_dim)
        state: Shape (obs_dim,)
    
    Returns:
        Logits for each action, shape (n_actions,)
    """
    return jnp.dot(policy_params, state)
