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
        self.np_rng = np.random.RandomState(config.training.seed)
        
        # Extract dimensions
        # For predator-prey: obs = [predator_x, predator_y, prey_x, prey_y] = 4D
        # Actions: stay, up, down, left, right = 5 discrete (but we output continuous for generality)
        # Goal: prey position = [prey_x, prey_y] = 2D
        self.obs_dim = 4
        self.action_dim = 5
        self.goal_dim = 2  # Goal is prey position
        
        # Vanilla policies are conditioned only on observation and goal.
        self.model_embed_dim = 0
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

    def _goal_directed_action(self, obs: np.ndarray, goal: np.ndarray) -> int:
        """Choose the action that most directly moves the predator toward the goal."""
        predator_pos = obs[:2]
        delta = goal - predator_pos

        if abs(delta[0]) > abs(delta[1]):
            return 4 if delta[0] > 0 else 3
        if abs(delta[1]) > 1e-6:
            return 1 if delta[1] > 0 else 2
        return 0

    def _heuristic_action_probability(self) -> float:
        """Decay heuristic guidance as the actor accumulates gradient updates."""
        return max(0.15, 1.0 - (self.train_state.step / 500.0))
        
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
        current_policy_state = self.train_state.model_policies[self.train_state.current_model_id]
        action_mean, action_log_std = current_policy_state.apply_fn(
            current_policy_state.params,
            obs_jax,
            goal_jax,
            embed_jax,
        )
        
        # Convert to NumPy and flatten - use mean as logits for discrete actions
        action_logits = np.array(action_mean).flatten()

        if explore and self.np_rng.rand() < self._heuristic_action_probability():
            return self._goal_directed_action(obs, goal)
        
        # Select discrete action
        if explore:
            # Sample from softmax distribution for exploration
            probs = np.exp(action_logits - np.max(action_logits))  # Subtract max for numerical stability
            probs = probs / np.sum(probs)
            action = self.np_rng.choice(self.action_dim, p=probs)
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
            sampled_indices = self.np_rng.choice(future_indices, size=num_samples, replace=False)
            
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
        episode_rewards = []

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
            her_transitions = self._apply_her_relabeling(
                episode_transitions,
                k=self.config.her.k,
            )
            
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

            episode_rewards.append(float(episode_reward))
            
            # Log progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.3f}, Buffer size: {len(self.replay_buffer)}")

        return episode_rewards
    
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
        current_model_id = self.train_state.current_model_id
        current_policy_state = self.train_state.model_policies[current_model_id]

        def critic_loss_fn(critic_params):
            next_logits, _ = current_policy_state.apply_fn(
                current_policy_state.params,
                next_states,
                goals,
                zero_embeds,
            )
            next_actions = jax.nn.softmax(next_logits, axis=-1)
            target_q1, target_q2 = self.train_state.critic_state.apply_fn(
                self.train_state.target_critic_params,
                next_states,
                next_actions,
                goals,
                zero_embeds,
            )
            target_q = jnp.minimum(target_q1, target_q2).squeeze(-1)
            td_target = rewards + self.config.training.gamma * (1.0 - dones) * target_q

            q1, q2 = self.train_state.critic_state.apply_fn(
                critic_params,
                states,
                actions,
                goals,
                zero_embeds,
            )
            q1 = q1.squeeze(-1)
            q2 = q2.squeeze(-1)
            return jnp.mean((q1 - td_target) ** 2 + (q2 - td_target) ** 2)

        critic_grads = jax.grad(critic_loss_fn)(self.train_state.critic_state.params)
        critic_state = self.train_state.critic_state.apply_gradients(grads=critic_grads)

        def actor_loss_fn(actor_params):
            action_logits, _ = current_policy_state.apply_fn(
                actor_params,
                states,
                goals,
                zero_embeds,
            )
            action_probs = jax.nn.softmax(action_logits, axis=-1)
            q1, q2 = critic_state.apply_fn(
                critic_state.params,
                states,
                action_probs,
                goals,
                zero_embeds,
            )
            return -jnp.mean(jnp.minimum(q1, q2))

        actor_grads = jax.grad(actor_loss_fn)(current_policy_state.params)
        actor_state = current_policy_state.apply_gradients(grads=actor_grads)
        target_critic_params = optax.incremental_update(
            critic_state.params,
            self.train_state.target_critic_params,
            self.config.training.tau,
        )
        model_policies = dict(self.train_state.model_policies)
        model_policies[current_model_id] = actor_state

        self.train_state = self.train_state.replace(
            model_policies=model_policies,
            critic_state=critic_state,
            target_critic_params=target_critic_params,
            step=self.train_state.step + 1,
        )
