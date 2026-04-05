"""Core data structures for HIM+HER.

All dataclasses use np.ndarray (not jnp.ndarray) because they live in the NumPy replay
buffer. Conversion to JAX arrays happens once per gradient step.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class Transition:
    """The fundamental unit stored in the replay buffer.
    
    Attributes:
        state: s_t — state at time t
        ego_action: a^e_t — ego agent's action at time t
        other_action: a^o_t — other agent's action at time t
        next_state: s_{t+1} — next state
        goal: g — original goal
        model_id: index into ModelSet — the assumed model m at time t
        reward: r_e(s_t, a^e_t, g, m) — can be recomputed
        done: episode termination flag
    """
    state: np.ndarray
    ego_action: np.ndarray
    other_action: np.ndarray
    next_state: np.ndarray
    goal: np.ndarray
    model_id: int
    reward: float
    done: bool


@dataclass
class AgentModel:
    """An element of the hypothesis set M.
    
    Attributes:
        model_id: unique identifier for this model
        name: human-readable name (e.g., "obedient", "food_motivated")
        reward_weights: parameterizes r_o^m if using reward model
        policy_params: parameters of pi_o^m — passed to JAX likelihood function
        prior: p(m) — prior probability, sums to 1 over all models in M
    
    Note:
        policy_params is a plain array (not a Callable) to enable JAX vmap over models.
        The policy function lives separately as a pure JAX function in him/inconsistency.py.
    """
    model_id: int
    name: str
    reward_weights: np.ndarray
    policy_params: np.ndarray
    prior: float


@dataclass
class Episode:
    """A complete episode trajectory with metadata.
    
    Attributes:
        transitions: list of Transition objects comprising the episode
        total_reward: undiscounted sum of rewards
        model_used: model_id used throughout this episode
        him_triggered: whether HIM was applied at end of episode
        revised_model: model_id after HIM revision, if triggered (None otherwise)
    """
    transitions: List[Transition]
    total_reward: float
    model_used: int
    him_triggered: bool
    revised_model: Optional[int]


@dataclass
class ModelSet:
    """A set of agent models with pre-stacked parameters for efficient vmap.
    
    Attributes:
        models: list of AgentModel instances
        stacked_policy_params: shape (|M|, param_dim) — for vmap over models
        log_priors: shape (|M|,) — log p(m) for each model
    
    Note:
        stacked_policy_params is pre-computed at init so all_model_log_likelihoods
        can vmap over the model dimension without Python-level iteration.
    """
    models: List[AgentModel]
    stacked_policy_params: np.ndarray
    log_priors: np.ndarray
    
    @classmethod
    def from_config(cls, config):
        """Create ModelSet from configuration (OmegaConf or dict).
        
        Args:
            config: Config with 'models' list, each having 'name', 'policy_params', and 'prior'
        
        Returns:
            ModelSet instance
        """
        models = []
        
        # Handle both dict and OmegaConf
        models_list = config.models if hasattr(config, 'models') else config['models']
        
        for i, model_cfg in enumerate(models_list):
            # Extract fields from either dict or OmegaConf
            if isinstance(model_cfg, dict):
                name = model_cfg['name']
                policy_params = np.array(model_cfg['policy_params'])
                prior = model_cfg['prior']
            else:
                name = model_cfg.name
                policy_params = np.array(model_cfg.policy_params)
                prior = model_cfg.prior
            
            model = AgentModel(
                model_id=i,
                name=name,
                reward_weights=np.ones(5),  # Not used for this simple env
                policy_params=policy_params,
                prior=prior,
            )
            models.append(model)
        
        # Stack policy params for vmap
        stacked_policy_params = np.stack([m.policy_params for m in models])
        log_priors = np.log(np.array([m.prior for m in models]))
        
        return cls(
            models=models,
            stacked_policy_params=stacked_policy_params,
            log_priors=log_priors,
        )
