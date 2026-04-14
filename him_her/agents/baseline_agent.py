"""Baseline agent implementations for comparison with HIM-HER.

VanillaAgent: No HIM, but HER is active.
StaticModelAgent: Uses the initial pretrained model policy throughout.
BayesianAgent: Updates a posterior after each episode and switches to the MAP model.
"""

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from him_her.envs.base_env import BaseMultiAgentEnv
from him_her.envs.predator_prey import predator_prey_model_forward
from him_her.him.belief_updater import BeliefUpdater
from him_her.him.inconsistency import make_likelihood_fns
from him_her.models.base_model import ModelSet
from him_her.networks.actor import Actor
from him_her.replay.replay_buffer import ReplayBuffer
from him_her.training.train_state import create_train_state
from him_her.utils.logging import EpisodeMetricsLogger
from him_her.utils.trajectory_logger import EpisodeRecord, StepRecord, TrajectoryLogger


class VanillaAgent:
    """Vanilla DDPG + HER agent without HIM or model revision."""

    def __init__(
        self,
        env: BaseMultiAgentEnv,
        model_set: ModelSet,
        config: Any,
    ):
        self.env = env
        self.model_set = model_set
        self.config = config
        self.np_rng = np.random.RandomState(config.training.seed)

        self.obs_dim = 4
        self.action_dim = 5
        self.goal_dim = 2

        self.model_embed_dim = 0
        self.zero_embed = np.zeros(self.model_embed_dim, dtype=np.float32)

        self.replay_buffer = ReplayBuffer(
            capacity=config.training.buffer_capacity,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            goal_dim=self.goal_dim,
        )

        self.rng = jax.random.PRNGKey(config.training.seed)
        self.metrics_logger: Optional[EpisodeMetricsLogger] = None
        self.traj_logger: Optional[TrajectoryLogger] = None
        self._last_episode_metrics: Optional[Dict[str, Any]] = None
        self._recent_rewards: deque = deque(maxlen=100)
        self._cumulative_reward: float = 0.0
        self._init_networks()

    def _metrics_agent_type(self) -> str:
        return 'vanilla'

    def _ensure_metrics_logger(self) -> EpisodeMetricsLogger:
        if self.metrics_logger is None:
            self.metrics_logger = EpisodeMetricsLogger(
                agent_type=self._metrics_agent_type(),
                seed=self.config.training.seed,
                config=self.config,
            )
        return self.metrics_logger

    def _log_episode_metrics(self, metrics: Optional[Dict[str, Any]]) -> None:
        if metrics is None:
            return
        self._ensure_metrics_logger().log(metrics)

    def _current_policy_name(self) -> str:
        if hasattr(self.env, 'current_policy_name'):
            return str(self.env.current_policy_name)
        current_policy = getattr(self.env, 'current_policy', None)
        return str(getattr(current_policy, 'name', 'unknown'))

    def _true_model_id(self) -> int:
        true_policy_name = self._current_policy_name()
        for model_id, model in enumerate(self.model_set.models):
            if model.name == true_policy_name:
                return model_id
        return -1

    def _window_fraction(self) -> float:
        him_config = getattr(self.config, 'him', None)
        return float(getattr(him_config, 'window_fraction', 1.0))

    def _switch_point_value(self) -> int:
        switch_point = getattr(self.env, 'switch_point', None)
        return int(switch_point) if switch_point is not None else -1

    def _log_episode_to_trajectory(
        self,
        episode: int,
        episode_transitions: List[Dict[str, Any]],
        episode_model_id: int,
        episode_reward: float,
        cumulative_reward: float,
        her_count: int = 0,
        *,
        him_triggered: bool = False,
        him_trigger_step: Optional[int] = None,
        old_model_id: Optional[int] = None,
        new_model_id: Optional[int] = None,
        log_lik_at_trigger: Optional[float] = None,
        log_lik_ratio_at_trigger: Optional[float] = None,
    ) -> None:
        """Log per-step records and an episode summary to TrajectoryLogger.

        All values are plain Python — never stores or passes JAX arrays (§5).
        """
        if self.traj_logger is None:
            return

        sp = self._switch_point_value()
        n_models = len(self.model_set.models)
        run_id = self.traj_logger.run_id
        env_name = self.traj_logger.env_name
        total = len(episode_transitions)
        save_steps = getattr(getattr(self.config, 'logging', None), 'save_steps', True)

        # --- Per-step records ---
        correct_steps = 0
        cumulative_ep_reward = 0.0
        for t, trans in enumerate(episode_transitions):
            ego_pos = [float(trans['state'][0]), float(trans['state'][1])]
            other_pos = [float(trans['state'][2]), float(trans['state'][3])]
            true_mid_t = 1 if (sp >= 0 and t >= sp) else 0
            true_mname_t = (
                self.model_set.models[true_mid_t].name
                if true_mid_t < n_models else "unknown"
            )
            dist = float(np.linalg.norm(np.array(ego_pos) - np.array(other_pos)))
            cumulative_ep_reward += float(trans['reward'])
            steps_to_switch = (sp - t) if sp >= 0 else 0
            if int(episode_model_id) == true_mid_t:
                correct_steps += 1

            if save_steps:
                self.traj_logger.log_step(StepRecord(
                    run_id=run_id,
                    episode=episode,
                    step=t,
                    ego_pos=ego_pos,
                    other_pos=other_pos,
                    ego_action=int(trans['action']),
                    other_action=int(trans['other_action']),
                    reward=float(trans['reward']),
                    cumulative_episode_reward=cumulative_ep_reward,
                    current_model_id=int(episode_model_id),
                    current_model_name=self.model_set.models[episode_model_id].name,
                    true_model_id=true_mid_t,
                    true_model_name=true_mname_t,
                    log_lik_per_step=0.0,
                    log_lik_all_models=[0.0] * n_models,
                    belief_state=None,
                    him_triggered_this_episode=him_triggered,
                    steps_since_last_trigger=0,
                    steps_to_switch=steps_to_switch,
                    distance=dist,
                ))

        # --- Episode summary ---
        model_correct_fraction = correct_steps / total if total > 0 else 0.0
        true_model_at_end = 1 if (sp >= 0 and total > sp) else 0
        true_model_name_at_end = (
            self.model_set.models[true_model_at_end].name
            if true_model_at_end < n_models else "unknown"
        )
        detection_lag: Optional[int] = None
        if him_triggered and sp >= 0:
            detection_lag = max(0, total - sp)

        total_trans = total + her_count
        her_fraction = her_count / total_trans if total_trans > 0 else 0.0

        rewards_list = list(self._recent_rewards)
        r10 = (
            float(np.mean(rewards_list[-10:])) if len(rewards_list) >= 10
            else (float(np.mean(rewards_list)) if rewards_list else 0.0)
        )
        r50 = (
            float(np.mean(rewards_list[-50:])) if len(rewards_list) >= 50
            else (float(np.mean(rewards_list)) if rewards_list else 0.0)
        )
        r100 = float(np.mean(rewards_list)) if rewards_list else 0.0

        # Read bayesian/HIM metrics from _last_episode_metrics when available
        bayes_switched = False
        bayes_belief: Optional[List[float]] = None
        if self._last_episode_metrics is not None:
            metrics = self._last_episode_metrics
            b_evasive = metrics.get('belief_evasive')
            b_territorial = metrics.get('belief_territorial')
            if b_evasive is not None and b_territorial is not None:
                bayes_belief = [float(b_evasive), float(b_territorial)]
            bayes_switched = bool(metrics.get('switched', False))

        him_frac: Optional[float] = None
        if him_trigger_step is not None and total > 0:
            him_frac = float(him_trigger_step) / float(total)

        gs = self.traj_logger.get_episode_gradient_stats()

        ep_record = EpisodeRecord(
            run_id=run_id,
            env_name=env_name,
            agent_type=self._metrics_agent_type(),
            seed=int(self.config.training.seed),
            episode=episode,
            switch_point=sp,
            total_steps=total,
            episode_reward=float(episode_reward),
            cumulative_reward=float(cumulative_reward),
            reward_10ep_mean=r10,
            reward_50ep_mean=r50,
            reward_100ep_mean=r100,
            final_model_id=int(self.train_state.current_model_id),
            final_model_name=self.model_set.models[int(self.train_state.current_model_id)].name,
            true_model_id_at_end=true_model_at_end,
            true_model_name_at_end=true_model_name_at_end,
            model_correct_fraction=model_correct_fraction,
            detection_lag=detection_lag,
            him_triggered=him_triggered,
            him_trigger_step=him_trigger_step,
            him_trigger_episode_step_fraction=him_frac,
            old_model_id=old_model_id,
            new_model_id=new_model_id,
            log_lik_at_trigger=log_lik_at_trigger,
            log_lik_ratio_at_trigger=log_lik_ratio_at_trigger,
            bayesian_switched_this_episode=bayes_switched,
            bayesian_belief_at_end=bayes_belief,
            critic_loss_mean=gs['critic_loss_mean'],
            actor_loss_mean=gs['actor_loss_mean'],
            her_fraction_mean=her_fraction,
            buffer_size=len(self.replay_buffer),
            gradient_steps=gs['gradient_steps'],
            wall_clock_time=gs['wall_clock_time'],
        )
        self.traj_logger.end_episode(ep_record)

    def _goal_directed_action(self, obs: np.ndarray, goal: np.ndarray) -> int:
        predator_pos = obs[:2]
        delta = goal - predator_pos

        if abs(delta[0]) > abs(delta[1]):
            return 4 if delta[0] > 0 else 3
        if abs(delta[1]) > 1e-6:
            return 1 if delta[1] > 0 else 2
        return 0

    def _heuristic_action_probability(self) -> float:
        return max(0.15, 1.0 - (self.train_state.step / 500.0))

    def _init_networks(self) -> None:
        self.actor = Actor(
            hidden_sizes=tuple(self.config.agent.hidden_sizes),
            action_dim=self.action_dim,
        )

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

    def _zero_embed_batch(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.model_embed_dim), dtype=jnp.float32)

    def _episode_model_id(self) -> int:
        return int(self.train_state.current_model_id)

    def _post_episode_update(
        self,
        episode_transitions: List[Dict[str, Any]],
        episode_model_id: int,
        episode_reward: float = 0.0,
        episode_index: int = 0,
    ) -> None:
        del episode_transitions, episode_model_id, episode_reward, episode_index

    def select_action(self, obs: np.ndarray, goal: np.ndarray, explore: bool = True) -> int:
        obs_jax = jnp.array(obs).reshape(1, -1)
        goal_jax = jnp.array(goal).reshape(1, -1)
        embed_jax = jnp.array(self.zero_embed).reshape(1, -1)

        current_policy_state = self.train_state.model_policies[self.train_state.current_model_id]
        action_mean, _ = current_policy_state.apply_fn(
            current_policy_state.params,
            obs_jax,
            goal_jax,
            embed_jax,
        )
        action_logits = np.array(action_mean).flatten()

        if explore and self.np_rng.rand() < self._heuristic_action_probability():
            return self._goal_directed_action(obs, goal)

        if explore:
            probs = np.exp(action_logits - np.max(action_logits))
            probs = probs / np.sum(probs)
            action = self.np_rng.choice(self.action_dim, p=probs)
        else:
            action = int(np.argmax(action_logits))

        return int(action)

    def _apply_her_relabeling(
        self,
        episode_transitions: List[Dict[str, Any]],
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        her_transitions = []
        episode_length = len(episode_transitions)

        for i, trans in enumerate(episode_transitions):
            if i >= episode_length - 1:
                continue

            future_indices = list(range(i + 1, episode_length))
            num_samples = min(k, len(future_indices))
            sampled_indices = self.np_rng.choice(future_indices, size=num_samples, replace=False)

            for idx in sampled_indices:
                new_goal = episode_transitions[idx]['next_state'][2:4].copy()
                achieved = trans['next_state'][2:4]
                new_reward = 1.0 if np.linalg.norm(achieved - new_goal) < 0.05 else 0.0

                her_transitions.append({
                    'state': trans['state'].copy(),
                    'action': trans['action'],
                    'reward': new_reward,
                    'next_state': trans['next_state'].copy(),
                    'done': trans['done'],
                    'goal': new_goal,
                })

        return her_transitions

    def _collect_episode(self) -> tuple[List[Dict[str, Any]], float]:
        obs, info = self.env.reset()
        goal = info['desired_goal'].copy()
        done = False
        episode_reward = 0.0
        episode_transitions: List[Dict[str, Any]] = []

        while not done:
            other_state = self.env._get_state().copy()
            action = self.select_action(obs, goal, explore=True)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            episode_transitions.append({
                'state': obs.copy(),
                'other_state': other_state,
                'action': action,
                'other_action': int(info['other_action']),
                'reward': reward,
                'next_state': next_obs.copy(),
                'done': done,
                'goal': goal.copy(),
            })

            episode_reward += reward
            obs = next_obs

        return episode_transitions, float(episode_reward)

    def _store_transitions(
        self,
        transitions: List[Dict[str, Any]],
        model_id: int,
    ) -> None:
        for trans in transitions:
            action_one_hot = np.zeros(self.action_dim, dtype=np.float32)
            action_one_hot[trans['action']] = 1.0

            self.replay_buffer.add(
                state=trans['state'],
                action=action_one_hot,
                reward=trans['reward'],
                next_state=trans['next_state'],
                done=trans['done'],
                goal=trans['goal'],
                model_id=model_id,
            )

    def train(self, num_episodes: int):
        episode_rewards = []

        if self.traj_logger is not None:
            self.traj_logger.save_metadata(self.config)

        try:
            for episode in range(num_episodes):
                episode_model_id = self._episode_model_id()
                self.train_state = self.train_state.replace(current_model_id=episode_model_id)

                episode_transitions, episode_reward = self._collect_episode()
                self._store_transitions(episode_transitions, episode_model_id)

                her_transitions = self._apply_her_relabeling(
                    episode_transitions,
                    k=self.config.her.k,
                )
                self._store_transitions(her_transitions, episode_model_id)
                her_count = len(her_transitions)

                self._last_episode_metrics = None
                self._post_episode_update(
                    episode_transitions,
                    episode_model_id,
                    episode_reward=episode_reward,
                    episode_index=episode,
                )
                self._log_episode_metrics(self._last_episode_metrics)

                if len(self.replay_buffer) >= self.config.training.batch_size:
                    for _ in range(self.config.training.updates_per_episode):
                        self._update_networks()

                episode_rewards.append(float(episode_reward))
                self._recent_rewards.append(float(episode_reward))
                self._cumulative_reward += float(episode_reward)
                rolling_mean = float(np.mean(self._recent_rewards))

                if self._last_episode_metrics is not None:
                    self._last_episode_metrics['reward_100ep_mean'] = rolling_mean
                else:
                    self._last_episode_metrics = {
                        'episode': int(episode),
                        'agent_type': self._metrics_agent_type(),
                        'episode_reward': float(episode_reward),
                        'reward_100ep_mean': rolling_mean,
                    }
                self._log_episode_metrics(self._last_episode_metrics)

                # Per-step trajectory logging (after metrics are finalised).
                self._log_episode_to_trajectory(
                    episode, episode_transitions, episode_model_id, episode_reward,
                    self._cumulative_reward, her_count,
                )

                if (episode + 1) % 10 == 0:
                    print(
                        f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.3f}, "
                        f"Buffer size: {len(self.replay_buffer)}"
                    )
        finally:
            if self.traj_logger is not None:
                self.traj_logger.close()

        return episode_rewards

    def _update_networks(self) -> None:
        batch = self.replay_buffer.sample(self.config.training.batch_size)

        states = jnp.array(batch['states'])
        actions = jnp.array(batch['actions'])
        rewards = jnp.array(batch['rewards'])
        next_states = jnp.array(batch['next_states'])
        dones = jnp.array(batch['dones'])
        goals = jnp.array(batch['goals'])

        zero_embeds = self._zero_embed_batch(states.shape[0])
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

        critic_loss_val, critic_grads = jax.value_and_grad(critic_loss_fn)(self.train_state.critic_state.params)
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

        actor_loss_val, actor_grads = jax.value_and_grad(actor_loss_fn)(current_policy_state.params)
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

        # Log gradient step OUTSIDE JAX boundary (§5 of ARCHITECTURE.md).
        if self.traj_logger is not None:
            self.traj_logger.log_gradient_step(
                float(critic_loss_val),
                float(actor_loss_val),
            )


class _PretrainedPolicyAgent(VanillaAgent):
    """Baseline variant that loads per-model pretrained policies when available."""

    def __init__(self, env: BaseMultiAgentEnv, model_set: ModelSet, config: Any):
        super().__init__(env, model_set, config)
        pretrain = getattr(self.config.training, 'pretrain', True)
        if pretrain and not getattr(self.config.training, 'concurrent_training', False):
            self._load_pretrained_policies()

    def _load_pretrained_policies(self) -> None:
        checkpointer = ocp.PyTreeCheckpointer()
        model_policies = dict(self.train_state.model_policies)

        for model_id, policy_state in model_policies.items():
            ckpt_dir = (Path('checkpoints') / f'policy_m{model_id}').resolve()
            if not ckpt_dir.exists():
                continue
            model_policies[model_id] = checkpointer.restore(ckpt_dir, item=policy_state)

        self.train_state = self.train_state.replace(model_policies=model_policies)


class StaticModelAgent(_PretrainedPolicyAgent):
    """Baseline that keeps the initial model policy active for every episode."""

    def __init__(self, env: BaseMultiAgentEnv, model_set: ModelSet, config: Any):
        super().__init__(env, model_set, config)
        self.static_model_id = 0
        self.train_state = self.train_state.replace(current_model_id=self.static_model_id)

    def _metrics_agent_type(self) -> str:
        return 'static'

    def _episode_model_id(self) -> int:
        return self.static_model_id


class BayesianAgent(_PretrainedPolicyAgent):
    """Baseline that updates a posterior over models after each episode."""

    def __init__(self, env: BaseMultiAgentEnv, model_set: ModelSet, config: Any):
        super().__init__(env, model_set, config)
        (
            self.current_model_log_likelihood_fn,
            all_model_log_likelihoods_fn,
        ) = make_likelihood_fns(predator_prey_model_forward)
        self.belief_updater = BeliefUpdater(model_set, all_model_log_likelihoods_fn)
        self.train_state = self.train_state.replace(
            log_belief=jnp.array(self.belief_updater.get_log_belief()),
            current_model_id=self.belief_updater.map_model_id(),
        )

    def _metrics_agent_type(self) -> str:
        return 'bayesian'

    def _collect_episode(self):
        # Reset belief to prior at episode start — opponent may have changed type
        self.belief_updater.reset_to_prior()
        return super()._collect_episode()

    def _post_episode_update(
        self,
        episode_transitions: List[Dict[str, Any]],
        episode_model_id: int,
        episode_reward: float = 0.0,
        episode_index: int = 0,
    ) -> None:
        if not episode_transitions:
            return

        old_model_id = int(episode_model_id)
        states = jnp.array(np.stack([trans['other_state'] for trans in episode_transitions]))
        actions = jnp.array(
            np.array([trans['other_action'] for trans in episode_transitions], dtype=np.int32)
        )
        stacked_params = jnp.array(self.model_set.stacked_policy_params)
        current_policy_params = stacked_params[old_model_id]
        log_lik_per_step = float(
            self.current_model_log_likelihood_fn(
                current_policy_params,
                states,
                actions,
                window_fraction=self._window_fraction(),
            )
        )
        belief_probs = self.belief_updater.update(stacked_params, states, actions)
        map_model_id = self.belief_updater.map_model_id()
        true_policy_name = self._current_policy_name()
        true_model_id = self._true_model_id()

        print(
            f"[BAYES] ep={episode_index} | "
            f"prior_model={old_model_id} ({self.model_set.models[old_model_id].name}) | "
            f"posterior={[f'{v:.3f}' for v in belief_probs.tolist()]} | "
            f"map_model={map_model_id} ({self.model_set.models[map_model_id].name}) | "
            f"switched={'YES' if map_model_id != old_model_id else 'NO'} | "
            f"true_end_policy={true_policy_name} | "
            f"switch_point={self._switch_point_value()}"
        )

        self._last_episode_metrics = {
            'episode': int(episode_index),
            'agent_type': 'bayesian',
            'belief_evasive': float(belief_probs[0]),
            'belief_territorial': float(belief_probs[1]),
            'map_model_id': int(map_model_id),
            'map_model_name': self.model_set.models[map_model_id].name,
            'prior_model_id': int(old_model_id),
            'prior_model_name': self.model_set.models[old_model_id].name,
            'switched': int(map_model_id != old_model_id),
            'true_end_policy': true_policy_name,
            'switch_point': self._switch_point_value(),
            'model_correct': int(map_model_id == true_model_id),
            'episode_reward': float(episode_reward),
            'log_lik_per_step': float(log_lik_per_step),
        }

        self.train_state = self.train_state.replace(
            log_belief=jnp.array(self.belief_updater.get_log_belief()),
            current_model_id=map_model_id,
        )
