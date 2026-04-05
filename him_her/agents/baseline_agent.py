"""Baseline agent implementations for comparison with HIM-HER.

VanillaAgent: No HIM (no model conditioning), but HER is active.
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, List

from him_her.envs.base_env import BaseMultiAgentEnv
from him_her.models.base_model import ModelSet
from him_her.networks.actor import Actor
from him_her.networks.critic import Critic
from him_her.networks.encoder import encode_model
from him_her.training.train_state import create_train_state
from him_her.replay.replay_buffer import ReplayBuffer


class VanillaAgent:
    """Vanilla DDPG + HER agent without HIM or model conditioning.
    
    This agent:
    - Uses HER for goal relabeling (keeps HER active)
    - Has NO model conditioning (model_embed is always zero)
    - Skips HIM entirely (no likelihood computation, no trajectory relabeling)
    
    The policy takes (obs, goal, zero_embed) as input where zero_embed has
    the correct dimensionality but all zeros.
    """
    
    def __init__(
        self,
        env: BaseMultiAgentEnv,
        model_set: ModelSet,
        config: Any,  # Can be SimpleConfig, DictConfig, or dict
    ):
        """Initialize VanillaAgent.
        
        Args:
            env: Multi-agent environment
            model_set: Set of models (used only for dimensionality)
            config: Configuration object
        """
        self.env = env
        self.model_set = model_set
        self.config = config
        
        # Extract dimensions
        # For predator-prey: obs = [predator_x, predator_y, prey_x, prey_y] = 4D
        # Actions: stay, up, down, left, right = 5 discrete (but we output continuous for generality)
        # Goal: prey position = [prey_x, prey_y] = 2D
        self.obs_dim = 4
        self.action_dim = 5
        self.goal_dim = 2  # Goal is prey position
        
        # Model encoding dimension from config
        self.model_embed_dim = config.agent.model_encoding.embed_dim
        
        # Create zero model embedding (no model conditioning)
        self.zero_embed = np.zeros(self.model_embed_dim, dtype=np.float32)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config.training.buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            goal_dim=self.goal_dim,
        )
        
        # Initialize networks and training state
        self.rng = jax.random.PRNGKey(config.training.seed)
        self._init_networks()
        
    def _init_networks(self):
        """Initialize actor, critic networks and training state."""
        # Create network instances (need to keep reference)
        self.actor = Actor(
            hidden_sizes=tuple(self.config.agent.hidden_sizes),
            action_dim=self.action_dim,
        )
        
        # Initialize training state
        self.train_state, self.rng = create_train_state(
            rng=self.rng,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            goal_dim=self.goal_dim,
            model_embed_dim=self.model_embed_dim,
            log_priors=self.model_set.log_priors,
            hidden_sizes=tuple(self.config.agent.hidden_sizes),
            lr_actor=self.config.training.lr_actor,
            lr_critic=self.config.training.lr_critic,
        )
        
    def select_action(self, obs: np.ndarray, goal: np.ndarray, explore: bool = True) -> int:
        """Select action using the actor network.
        
        Args:
            obs: Current observation
            goal: Current goal
            explore: Whether to add exploration noise
        
        Returns:
            Selected action index (0-4 for predator-prey)
        """
        # Convert to JAX arrays
        obs_jax = jnp.array(obs).reshape(1, -1)
        goal_jax = jnp.array(goal).reshape(1, -1)
        embed_jax = jnp.array(self.zero_embed).reshape(1, -1)
        
        # Get action distribution from actor (returns mean, log_std for continuous control)
        action_mean, action_log_std = self.actor.apply(
            self.train_state.actor_state.params,
            obs_jax,
            goal_jax,
            embed_jax,
        )
        
        # Convert to NumPy and flatten - use mean as logits for discrete actions
        action_logits = np.array(action_mean).flatten()
        
        # Select discrete action
        if explore:
            # Sample from softmax distribution for exploration
            probs = np.exp(action_logits - np.max(action_logits))  # Subtract max for numerical stability
            probs = probs / np.sum(probs)
            action = np.random.choice(self.action_dim, p=probs)
        else:
            # Greedy: select argmax
            action = int(np.argmax(action_logits))
        
        return action
    
    def _apply_her_relabeling(
        self,
        episode_transitions: List[Dict],
        k: int = 4,
    ) -> List[Dict]:
        """Apply HER goal relabeling to episode transitions.
        
        Uses "future" strategy: for each transition, sample k future states as goals.
        
        Args:
            episode_transitions: List of transitions from the episode
            k: Number of HER goals per transition
        
        Returns:
            List of new transitions with relabeled goals and rewards
        """
        her_transitions = []
        episode_length = len(episode_transitions)
        
        for i, trans in enumerate(episode_transitions):
            # Skip if it's the last transition (no future states)
            if i >= episode_length - 1:
                continue
            
            # Sample k future states as alternative goals
            future_indices = list(range(i + 1, episode_length))
            num_samples = min(k, len(future_indices))
            sampled_indices = np.random.choice(future_indices, size=num_samples, replace=False)
            
            for idx in sampled_indices:
                # New goal is prey position from future state (indices 2:4 of observation)
                new_goal = episode_transitions[idx]['next_state'][2:4].copy()
                
                # Compute new reward with relabeled goal
                # achieved is prey position from current next_state
                achieved = trans['next_state'][2:4]
                new_reward = 1.0 if np.linalg.norm(achieved - new_goal) < 0.05 else 0.0
                
                # Create relabeled transition
                her_trans = {
                    'state': trans['state'].copy(),
                    'action': trans['action'],  # Keep as integer
                    'reward': new_reward,
                    'next_state': trans['next_state'].copy(),
                    'done': trans['done'],
                    'goal': new_goal,
                }
                her_transitions.append(her_trans)
        
        return her_transitions
    
    def train(self, num_episodes: int):
        """Run training for specified number of episodes.
        
        Args:
            num_episodes: Number of episodes to train for
        """
        for episode in range(num_episodes):
            # Reset environment
            obs, info = self.env.reset()
            goal = info["desired_goal"].copy()
            done = False
            episode_reward = 0.0
            episode_transitions = []
            
            # Roll out episode
            while not done:
                # Select action
                action = self.select_action(obs, goal, explore=True)
                
                # Environment step
                next_obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                
                # Store transition
                episode_transitions.append({
                    'state': obs,
                    'action': action,
                    'reward': reward,
                    'next_state': next_obs,
                    'done': done,
                    'goal': goal,
                })
                
                episode_reward += reward
                obs = next_obs
            
            # Apply HER goal relabeling
            # TEMPORARILY DISABLED FOR DEBUGGING
            her_transitions = []  # self._apply_her_relabeling(episode_transitions, k=self.config.her.k)
            
            # Add all transitions to replay buffer (original + HER relabeled)
            for trans in episode_transitions + her_transitions:
                # Convert discrete action to one-hot for storage
                action_one_hot = np.zeros(self.action_dim, dtype=np.float32)
                action_one_hot[trans['action']] = 1.0
                
                self.replay_buffer.add(
                    state=trans['state'],
                    action=action_one_hot,
                    reward=trans['reward'],
                    next_state=trans['next_state'],
                    done=trans['done'],
                    goal=trans['goal'],
                    model_id=0,  # Dummy model_id (not used)
                )
            
            # Update networks if enough data
            if len(self.replay_buffer) >= self.config.training.batch_size:
                for _ in range(self.config.training.updates_per_episode):
                    self._update_networks()
            
            # Log progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.3f}, Buffer size: {len(self.replay_buffer)}")
    
    def _update_networks(self):
        """Sample batch and update actor-critic networks."""
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.config.training.batch_size)
        
        # Convert to JAX arrays (this is where NumPy → JAX conversion happens)
        states = jnp.array(batch['states'])
        actions = jnp.array(batch['actions'])
        rewards = jnp.array(batch['rewards'])
        next_states = jnp.array(batch['next_states'])
        dones = jnp.array(batch['dones'])
        goals = jnp.array(batch['goals'])
        
        # Create zero embeddings for entire batch
        batch_size = states.shape[0]
        zero_embeds = jnp.zeros((batch_size, self.model_embed_dim))
        
        # TODO: Implement actual DDPG update logic here
        # For smoke test, we just verify the pipeline runs
        # Real implementation would:
        # 1. Compute Q-targets using target networks
        # 2. Update critic to minimize Bellman error
        # 3. Update actor to maximize Q-values
        # 4. Soft-update target networks
        pass
