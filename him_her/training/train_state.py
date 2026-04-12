"""Training state for HIM+HER agent.

All mutable training state lives in a single immutable pytree (HIMHERTrainState).
Updating any field returns a new state via .replace() — standard Flax pattern.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from typing import Any, Dict

from him_her.networks.actor import Actor
from him_her.networks.critic import Critic


@struct.dataclass
class HIMHERTrainState:
    """Immutable pytree containing all training state for HIM+HER agent.
    
    All fields are immutable. Updates return a new HIMHERTrainState via .replace().
    
    Attributes:
        actor_state: Flax TrainState for the actor (params + optimizer state)
        critic_state: Flax TrainState for the critic (params + optimizer state)
        target_critic_params: Soft-updated target network parameters
        log_belief: Log posterior distribution over model set, shape (|M|,)
        current_model_id: Index of the MAP model (Python int, not traced)
        step: Global gradient step counter
    
    Note:
        current_model_id is marked as pytree_node=False, which prevents JAX from
        tracing it as a dynamic value. This is critical to avoid retracing the
        update_step function on every HIM trigger (see Section 15, item 5).
    """
    model_policies: Dict[int, TrainState]
    critic_state: TrainState
    target_critic_params: Any
    log_belief: jnp.ndarray
    current_model_id: int = struct.field(pytree_node=False)
    step: int

    @property
    def actor_state(self) -> TrainState:
        """Compatibility view of the currently selected actor state."""
        return self.model_policies[self.current_model_id]


def create_train_state(
    rng: jax.random.PRNGKey,
    obs_dim: int,
    goal_dim: int,
    action_dim: int,
    model_embed_dim: int,
    log_priors: np.ndarray,
    hidden_sizes: tuple = (256, 256),
    lr_actor: float = 3e-4,
    lr_critic: float = 3e-4,
) -> tuple[HIMHERTrainState, jax.random.PRNGKey]:
    """Initialize HIMHERTrainState with randomly initialized networks.
    
    Args:
        rng: JAX PRNG key for random initialization
        obs_dim: Observation/state dimensionality
        goal_dim: Goal dimensionality
        action_dim: Action dimensionality
        model_embed_dim: Dimensionality of model embedding
        log_priors: Log prior distribution over models, shape (|M|,)
        hidden_sizes: Hidden layer sizes for actor and critic networks
        lr_actor: Learning rate for actor optimizer
        lr_critic: Learning rate for critic optimizer
    
    Returns:
        Tuple of (train_state, rng) where:
        - train_state: Initialized HIMHERTrainState
        - rng: Updated PRNG key for downstream use
    
    Note:
        The PRNG key is split three ways: one for actor init, one for critic init,
        and one returned for downstream use. This ensures proper key hygiene.
        
        Dummy inputs have shape (1, dim) with a batch dimension, as Flax expects
        batched inputs for parameter initialization.
    
    Example:
        >>> rng = jax.random.PRNGKey(0)
        >>> train_state, rng = create_train_state(
        ...     rng, obs_dim=10, goal_dim=2, action_dim=2,
        ...     model_embed_dim=3, log_priors=np.log([0.5, 0.5])
        ... )
        >>> assert train_state.current_model_id == 0
        >>> assert train_state.step == 0
    """
    # Create network instances
    actor = Actor(hidden_sizes=hidden_sizes, action_dim=action_dim)
    critic = Critic(hidden_sizes=hidden_sizes)

    num_models = int(len(log_priors))

    # Split RNG key for one actor per model, the critic, and downstream use.
    split_keys = jax.random.split(rng, num_models + 2)
    actor_keys = split_keys[:num_models]
    critic_key = split_keys[-2]
    rng = split_keys[-1]
    
    # Create dummy inputs for parameter initialization
    # Note: Shape is (1, dim) with batch dimension, not just (dim,)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_goal = jnp.zeros((1, goal_dim))
    dummy_actor_embed = jnp.zeros((1, 0))
    dummy_critic_embed = jnp.zeros((1, model_embed_dim))
    dummy_action = jnp.zeros((1, action_dim))

    # Initialize one actor policy per model. The actor is conditioned only on
    # observation and goal; model identity is represented by policy selection.
    model_policies = {}
    actor_tx = optax.adam(lr_actor)
    for model_id, actor_key in enumerate(actor_keys):
        actor_params = actor.init(actor_key, dummy_obs, dummy_goal, dummy_actor_embed)
        model_policies[model_id] = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=actor_tx,
        )

    critic_params = critic.init(critic_key, dummy_obs, dummy_action, dummy_goal, dummy_critic_embed)
    
    # Create optimizers
    critic_tx = optax.adam(lr_critic)

    # Create TrainState instance for the shared critic
    critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=critic_params,
        tx=critic_tx,
    )
    
    # Initialize HIMHERTrainState
    train_state = HIMHERTrainState(
        model_policies=model_policies,
        critic_state=critic_state,
        target_critic_params=critic_params,  # Initialize target with same params
        log_belief=jnp.array(log_priors),
        current_model_id=0,  # Start with first model
        step=0,
    )
    
    return train_state, rng
