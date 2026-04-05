"""Bayesian belief updating over model hypothesis set.

Used by the Bayesian baseline and soft-HIM variant. Belief is maintained in NumPy
log-space for numerical stability. JAX is called only for likelihood computation.
"""

import numpy as np
import jax.numpy as jnp
from scipy.special import logsumexp
from typing import Callable

from him_her.models.base_model import ModelSet


class BeliefUpdater:
    """Maintains a Bayesian belief distribution over the model hypothesis set.
    
    The belief is represented in log-space for numerical stability and updated via
    Bayesian inference: b'(m) ∝ p(m) * L(m | tau).
    
    Attributes:
        log_belief: NumPy array of shape (|M|,) containing log b(m) for each model
        all_model_log_likelihoods_fn: JIT-compiled function for computing likelihoods
    """
    
    def __init__(self, model_set: ModelSet, all_model_log_likelihoods_fn: Callable):
        """Initialize belief with prior distribution.
        
        Args:
            model_set: ModelSet containing models and their priors
            all_model_log_likelihoods_fn: JIT-compiled likelihood function from
                                          make_likelihood_fns(). Takes (stacked_params,
                                          states, actions) -> log_likelihoods
        
        Usage:
            >>> from him_her.him.inconsistency import make_likelihood_fns
            >>> _, all_models_fn = make_likelihood_fns(my_model_forward)
            >>> updater = BeliefUpdater(model_set, all_models_fn)
        """
        # Initialize belief to prior (in log-space)
        self.log_belief = model_set.log_priors.copy()  # NumPy array
        self._num_models = len(model_set.models)
        self._all_model_log_likelihoods_fn = all_model_log_likelihoods_fn
    
    def update(
        self,
        stacked_policy_params: jnp.ndarray,
        states: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> np.ndarray:
        """Perform Bayesian update given observed trajectory.
        
        Update rule: log b'(m) = log b(m) + log L(m | tau), then normalize.
        
        Args:
            stacked_policy_params: Shape (|M|, param_dim) — JAX array
            states: Trajectory of states, shape (T, obs_dim) — JAX array
            actions: Trajectory of actions, shape (T,) — JAX array
        
        Returns:
            Normalized posterior distribution as NumPy array, shape (|M|,)
            This is b(m), not log b(m) — exponentiated for convenience
        
        Note:
            The belief state (self.log_belief) remains in log-space internally.
        """
        # Compute log-likelihoods using JAX (JIT-compiled)
        log_likelihoods = self._all_model_log_likelihoods_fn(
            stacked_policy_params, states, actions
        )
        
        # Convert to NumPy for belief update
        log_likelihoods_np = np.array(log_likelihoods)
        
        # Bayesian update in log-space
        log_unnorm = self.log_belief + log_likelihoods_np
        
        # Normalize using logsumexp for numerical stability
        self.log_belief = log_unnorm - logsumexp(log_unnorm)
        
        # Return normalized posterior (exponentiated)
        return np.exp(self.log_belief)
    
    def get_belief(self) -> np.ndarray:
        """Get current belief distribution.
        
        Returns:
            Normalized belief distribution as NumPy array, shape (|M|,)
        """
        return np.exp(self.log_belief)
    
    def get_log_belief(self) -> np.ndarray:
        """Get current belief distribution in log-space.
        
        Returns:
            Log-belief distribution as NumPy array, shape (|M|,)
        """
        return self.log_belief.copy()
    
    def map_model_id(self) -> int:
        """Get the MAP (maximum a posteriori) model.
        
        Returns:
            Index of the model with highest posterior probability
        """
        return int(np.argmax(self.log_belief))
    
    def sample_model_id(self, rng: np.random.Generator) -> int:
        """Sample a model from the posterior distribution (Thompson sampling).
        
        Args:
            rng: NumPy random generator
        
        Returns:
            Sampled model index
        
        Note:
            This enables Thompson sampling for exploration. The Bayesian baseline
            can use this to sample policies from the posterior rather than always
            using the MAP estimate.
        """
        belief = np.exp(self.log_belief)
        return int(rng.choice(self._num_models, p=belief))
    
    def reset_to_prior(self, log_priors: np.ndarray) -> None:
        """Reset belief to prior distribution.
        
        Args:
            log_priors: Log-prior distribution, shape (|M|,)
        
        Note:
            Useful for episodic resets in some experimental settings.
        """
        self.log_belief = log_priors.copy()
