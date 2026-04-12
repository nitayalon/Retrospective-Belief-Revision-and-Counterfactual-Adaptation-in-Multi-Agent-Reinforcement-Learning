"""Likelihood computation and inconsistency detection using JAX.

All functions are pure JAX — JIT-compiled and vmapped over the model dimension.
The key advantage: likelihoods under all |M| models are computed in a single vmap call
rather than a Python loop.

IMPORTANT: To avoid passing functions as runtime arguments to JIT-compiled code, use the
factory pattern via make_likelihood_fns(). This returns closures with model_forward baked in.
"""

import jax
import jax.numpy as jnp
from functools import partial
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

    def _trajectory_log_prob_sum(
        policy_params: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        log_probs = jax.vmap(
            lambda s, a: single_step_log_prob(policy_params, s, a),
            in_axes=(0, 0)
        )(states, actions)
        return jnp.sum(log_probs)

    def _window_slice(
        states: jnp.ndarray,
        actions: jnp.ndarray,
        window_fraction: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not 0.0 < window_fraction <= 1.0:
            raise ValueError(f"window_fraction must be in (0, 1], got {window_fraction}")

        trajectory_length = states.shape[0]
        start = int(trajectory_length * (1.0 - window_fraction))
        start = max(0, min(start, trajectory_length - 1))
        return states[start:], actions[start:]
    
    @partial(jax.jit, static_argnames=['window_fraction'])
    def current_model_log_likelihood(
        policy_params: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        window_fraction: float = 0.5,
    ) -> jnp.ndarray:
        """Compute the current model log-likelihood over a recent trajectory window.
        
        This is the cheap monitoring path used every episode before any full
        model-set comparison is needed.
        
        Args:
            policy_params: Parameters defining the other agent's policy
            states: Trajectory of states, shape (T, obs_dim)
            actions: Trajectory of actions, shape (T,)
            window_fraction: Fraction of final trajectory steps to evaluate
        
        Returns:
            Mean log-likelihood per step under the model (scalar)
        """
        states_window, actions_window = _window_slice(states, actions, window_fraction)

        len_window = states_window.shape[0]
        log_prob_sum = _trajectory_log_prob_sum(policy_params, states_window, actions_window)
        return log_prob_sum / jnp.maximum(len_window, 1)
    
    @partial(jax.jit, static_argnames=['window_fraction'])
    def all_model_log_likelihoods_windowed(
        stacked_policy_params: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        window_fraction: float = 0.5,
    ) -> jnp.ndarray:
        """Compute log-likelihoods for all models over a recent trajectory window.
        
        This is the expensive model-selection path, intended to run only after an
        inconsistency signal indicates that the current assumed model may be wrong.
        
        Args:
            stacked_policy_params: Shape (|M|, param_dim) — all model parameters stacked
            states: Trajectory of states, shape (T, obs_dim)
            actions: Trajectory of actions, shape (T,)
            window_fraction: Fraction of final trajectory steps to evaluate
        
        Returns:
            Log-likelihoods for all models, shape (|M|,)
        """
        states_window, actions_window = _window_slice(states, actions, window_fraction)

        # vmap over models: states and actions are constant, policy_params is batched
        return jax.vmap(
            lambda params: _trajectory_log_prob_sum(params, states_window, actions_window),
            in_axes=(0,)
        )(stacked_policy_params)
    
    return current_model_log_likelihood, all_model_log_likelihoods_windowed


def compute_him_likelihood(
    stacked_policy_params: jnp.ndarray,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    window_fraction: float = 0.5,
    *,
    all_model_log_likelihoods_fn: Callable,
) -> jnp.ndarray:
    """Compute model likelihoods using only the final fraction of a trajectory.

    This focuses HIM on recent behavior in mixed-policy episodes, where the start
    of the trajectory may reflect an outdated model.
    """
    return all_model_log_likelihoods_fn(
        stacked_policy_params,
        states,
        actions,
        window_fraction=window_fraction,
    )


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
