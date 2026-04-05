"""HER-aware replay buffer operations.

Augments sampled batches with goal-relabeled transitions using HER strategies.
All operations run in NumPy before JAX conversion in the update step.
"""

import numpy as np
from typing import Callable

from him_her.models.base_model import Episode
from him_her.her.goal_sampler import GoalSampler


def apply_her(
    batch: dict,
    episode: Episode,
    goal_sampler: GoalSampler,
    relabeler: Callable,
    k: int = 4,
    rng: np.random.Generator = None,
    strategy: str = "future",
) -> dict:
    """Augment a batch with HER-relabeled transitions.
    
    For each transition in the batch, samples k alternative goals and recomputes
    rewards using the relabeler. Returns an augmented batch containing both original
    and relabeled transitions.
    
    Args:
        batch: NumPy batch dict sampled from replay buffer with keys:
               'states', 'ego_actions', 'other_actions', 'next_states', 
               'goals', 'model_ids', 'rewards', 'dones'
        episode: Episode object for goal sampling context
        goal_sampler: GoalSampler instance for sampling alternative goals
        relabeler: JIT-compiled reward relabeling function from make_relabeler()
        k: Number of alternative goals to sample per transition
        rng: NumPy random generator (required for sampling strategies)
        strategy: Goal sampling strategy ("future", "episode", "final")
    
    Returns:
        Augmented batch dict (NumPy) containing (1 + k) * batch_size transitions.
        All arrays are np.ndarray, not jnp.ndarray.
    
    Note:
        This function runs entirely in NumPy. The relabeler is called here with
        NumPy arrays converted to JAX inside the relabeler itself. The batch is
        converted to JAX arrays later in the update_step via buffer.to_jax().
    
    Example:
        >>> batch_size = 32
        >>> batch = buffer.sample(batch_size)  # NumPy dict
        >>> augmented = apply_her(batch, episode, sampler, relabeler, k=4)
        >>> assert augmented['states'].shape[0] == (1 + 4) * 32
    """
    # TODO: vectorize relabeler across batch
    if rng is None and strategy in ["future", "episode"]:
        raise ValueError(f"rng is required for strategy '{strategy}'")
    
    batch_size = batch['states'].shape[0]
    
    # Initialize lists to accumulate original + relabeled transitions
    all_states = [batch['states']]
    all_ego_actions = [batch['ego_actions']]
    all_other_actions = [batch['other_actions']]
    all_next_states = [batch['next_states']]
    all_goals = [batch['goals']]
    all_model_ids = [batch['model_ids']]
    all_rewards = [batch['rewards']]
    all_dones = [batch['dones']]
    
    # For each transition in the batch, generate k HER relabeled versions
    for i in range(batch_size):
        # Sample k alternative goals for this transition
        # Note: transition_idx is within the episode, not the batch
        # For simplicity, we'll use index i as the transition index within episode
        # In a real implementation, you'd need to track which episode each transition came from
        # For now, we'll sample goals from the entire episode
        alternative_goals = goal_sampler.sample_goals(
            episode=episode,
            transition_idx=i % len(episode.transitions),  # Wrap around for demo
            strategy=strategy,
            k=k,
            rng=rng,
        )
        
        # For each alternative goal, create a relabeled transition
        for new_goal in alternative_goals:
            # Extract original transition data
            state = batch['states'][i]
            ego_action = batch['ego_actions'][i]
            other_action = batch['other_actions'][i]
            next_state = batch['next_states'][i]
            model_id = batch['model_ids'][i]
            done = batch['dones'][i]
            
            # The relabeler expects JAX arrays and returns a JAX scalar
            # Convert to JAX, call relabeler, then convert back to NumPy
            import jax.numpy as jnp
            
            # Get model reward weights (placeholder - in real implementation, 
            # this would come from the model associated with model_id)
            # For now, use a dummy weight vector
            model_reward_weights = np.ones(1, dtype=np.float32)
            
            # Relabel the reward with the new goal
            # Convert inputs to JAX arrays for the relabeler
            new_reward = relabeler(
                jnp.array(state),
                jnp.array(ego_action),
                jnp.array(new_goal),
                jnp.array(model_reward_weights),
            )
            # Convert back to NumPy
            new_reward = np.array(new_reward, dtype=np.float32)
            
            # Append the relabeled transition
            all_states.append(state.reshape(1, -1))
            all_ego_actions.append(ego_action.reshape(1, -1))
            all_other_actions.append(other_action.reshape(1, -1))
            all_next_states.append(next_state.reshape(1, -1))
            all_goals.append(new_goal.reshape(1, -1))
            all_model_ids.append(np.array([model_id], dtype=np.int32))
            all_rewards.append(np.array([new_reward], dtype=np.float32))
            all_dones.append(np.array([done], dtype=bool))
    
    # Concatenate all transitions (original + k relabeled per original)
    augmented_batch = {
        'states': np.concatenate(all_states, axis=0),
        'ego_actions': np.concatenate(all_ego_actions, axis=0),
        'other_actions': np.concatenate(all_other_actions, axis=0),
        'next_states': np.concatenate(all_next_states, axis=0),
        'goals': np.concatenate(all_goals, axis=0),
        'model_ids': np.concatenate(all_model_ids, axis=0),
        'rewards': np.concatenate(all_rewards, axis=0),
        'dones': np.concatenate(all_dones, axis=0),
    }
    
    return augmented_batch
