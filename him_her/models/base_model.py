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
