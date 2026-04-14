"""Minimal HIM+HER agent used by the smoke-test training entry point."""

from pathlib import Path
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from him_her.agents.baseline_agent import VanillaAgent
from him_her.envs.predator_prey import predator_prey_model_forward
from him_her.him.inconsistency import make_likelihood_fns
from him_her.him.model_revision import relabel_trajectory_in_buffer
from him_her.utils.trajectory_logger import StepRecord


class HIMHERAgent(VanillaAgent):
    """Smoke-test implementation of a HIM+HER agent."""

    def __init__(self, env, model_set, config: Any):
        super().__init__(env, model_set, config)
        self.current_model_log_likelihood_fn, self.all_model_log_likelihoods_fn = make_likelihood_fns(
            predator_prey_model_forward
        )
        pretrain = getattr(self.config.training, 'pretrain', True)
        if pretrain and not getattr(self.config.training, "concurrent_training", False):
            self._load_pretrained_policies()

    def _init_networks(self):
        self.model_embed_dim = 0
        self.zero_embed = np.zeros(self.model_embed_dim, dtype=np.float32)
        super()._init_networks()

    def _metrics_agent_type(self) -> str:
        return 'him_her'

    def _zero_embed_batch(self, batch_size: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, self.model_embed_dim), dtype=jnp.float32)

    def _load_pretrained_policies(self) -> None:
        checkpointer = ocp.PyTreeCheckpointer()
        model_policies = dict(self.train_state.model_policies)

        for model_id, policy_state in model_policies.items():
            ckpt_dir = (Path("checkpoints") / f"policy_m{model_id}").resolve()
            if not ckpt_dir.exists():
                continue
            model_policies[model_id] = checkpointer.restore(ckpt_dir, item=policy_state)

        self.train_state = self.train_state.replace(model_policies=model_policies)

    def select_action(self, obs: np.ndarray, goal: np.ndarray, explore: bool = True) -> int:
        obs_jax = jnp.array(obs).reshape(1, -1)
        goal_jax = jnp.array(goal).reshape(1, -1)
        model_embed = jnp.zeros((1, self.model_embed_dim), dtype=jnp.float32)
        current_policy = self.train_state.model_policies[self.train_state.current_model_id]

        action_mean, _ = current_policy.apply_fn(
            current_policy.params,
            obs_jax,
            goal_jax,
            model_embed,
        )
        action_logits = np.array(action_mean).flatten()

        if explore and self.np_rng.rand() < self._heuristic_action_probability():
            return self._goal_directed_action(obs, goal)

        if explore:
            probs = np.exp(action_logits - np.max(action_logits))
            probs = probs / np.sum(probs)
            return int(self.np_rng.choice(self.action_dim, p=probs))

        return int(np.argmax(action_logits))

    def _update_networks(self):
        batch = self.replay_buffer.sample(self.config.training.batch_size)
        states = jnp.array(batch['states'])
        actions = jnp.array(batch['actions'])
        rewards = jnp.array(batch['rewards'])
        next_states = jnp.array(batch['next_states'])
        dones = jnp.array(batch['dones'])
        goals = jnp.array(batch['goals'])
        model_embeds = self._zero_embed_batch(states.shape[0])
        current_model_id = self.train_state.current_model_id
        current_policy_state = self.train_state.model_policies[current_model_id]

        def critic_loss_fn(critic_params):
            next_logits, _ = current_policy_state.apply_fn(
                current_policy_state.params,
                next_states,
                goals,
                model_embeds,
            )
            next_actions = jax.nn.softmax(next_logits, axis=-1)
            target_q1, target_q2 = self.train_state.critic_state.apply_fn(
                self.train_state.target_critic_params,
                next_states,
                next_actions,
                goals,
                model_embeds,
            )
            target_q = jnp.minimum(target_q1, target_q2).squeeze(-1)
            td_target = rewards + self.config.training.gamma * (1.0 - dones) * target_q

            q1, q2 = self.train_state.critic_state.apply_fn(
                critic_params,
                states,
                actions,
                goals,
                model_embeds,
            )
            return jnp.mean((q1.squeeze(-1) - td_target) ** 2 + (q2.squeeze(-1) - td_target) ** 2)

        critic_loss_val, critic_grads = jax.value_and_grad(critic_loss_fn)(self.train_state.critic_state.params)
        critic_state = self.train_state.critic_state.apply_gradients(grads=critic_grads)

        def actor_loss_fn(actor_params):
            action_logits, _ = current_policy_state.apply_fn(
                actor_params,
                states,
                goals,
                model_embeds,
            )
            action_probs = jax.nn.softmax(action_logits, axis=-1)
            q1, q2 = critic_state.apply_fn(
                critic_state.params,
                states,
                action_probs,
                goals,
                model_embeds,
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

    def _apply_him(self, ep: int, episode_transitions: List[Dict[str, Any]], episode_start_idx: int) -> Dict[str, float]:
        if not episode_transitions:
            return {
                'him_triggered': 0.0,
                'log_lik_per_step': float('nan'),
            }

        states = jnp.array(np.stack([trans['other_state'] for trans in episode_transitions]))
        actions = jnp.array(np.array([trans['other_action'] for trans in episode_transitions], dtype=np.int32))
        stacked_params = jnp.array(self.model_set.stacked_policy_params)
        log_priors = jnp.array(self.model_set.log_priors)
        old_model_id = self.train_state.current_model_id
        current_policy_params = stacked_params[old_model_id]

        if ep == 0:
            test_state = jnp.zeros(states.shape[1], dtype=states.dtype)
            for model_id in range(len(self.model_set.models)):
                params = stacked_params[model_id]
                logits = predator_prey_model_forward(params, test_state)
                probs = jax.nn.softmax(logits)
                print(f"[MODEL] model={self.model_set.models[model_id].name} | "
                      f"logits={logits.tolist()} | "
                      f"probs={[f'{p:.3f}' for p in probs.tolist()]}")

        current_log_lik = self.current_model_log_likelihood_fn(
            current_policy_params,
            states,
            actions,
            window_fraction=self.config.him.window_fraction,
        )
        window_len = max(int(states.shape[0] * self.config.him.window_fraction), 1)
        log_lik_per_step = current_log_lik
        threshold = getattr(self.config.him, 'threshold', -2.0)
        warmup_episodes = getattr(self.config.him, 'warmup_episodes', 20)
        him_active = ep >= warmup_episodes
        triggered = him_active and float(log_lik_per_step) < threshold

        if ep < 10:
            print(f"[DIAG] ep={ep} | "
                  f"window_steps={window_len} | "
                  f"log_lik_per_step={float(log_lik_per_step):.4f} | "
                  f"threshold={threshold} | "
                  f"him_active={him_active} | "
                  f"would_trigger={'YES' if him_active and float(log_lik_per_step) < threshold else 'NO'}")

        if triggered:
            all_log_liks = self.all_model_log_likelihoods_fn(
                stacked_params,
                states,
                actions,
                window_fraction=self.config.him.window_fraction,
            )
            new_model_id = int(jnp.argmax(all_log_liks + log_priors))
            if new_model_id != old_model_id:
                print(f"[HIM] Episode {ep} | "
                      f"old={old_model_id} ({self.model_set.models[old_model_id].name}) | "
                      f"new={int(new_model_id)} ({self.model_set.models[int(new_model_id)].name}) | "
                      f"log_lik={float(log_lik_per_step):.3f} | "
                      f"threshold={float(threshold):.3f} | "
                      f"switch_point={self.env.switch_point}")
                relabel_trajectory_in_buffer(
                    self.replay_buffer,
                    episode_start_idx,
                    len(episode_transitions),
                    int(new_model_id),
                )
                self.train_state = self.train_state.replace(current_model_id=int(new_model_id))

        return {
            'him_triggered': float(triggered),
            'log_lik_per_step': float(log_lik_per_step),
        }

    def train(self, num_episodes: int):
        episode_rewards = []

        if self.traj_logger is not None:
            self.traj_logger.save_metadata(self.config)

        try:
            for episode in range(num_episodes):
                obs, info = self.env.reset()
                goal = info['desired_goal'].copy()
                done = False
                episode_reward = 0.0
                episode_transitions = []
                episode_start_idx = self.replay_buffer.ptr
                current_model_id = self.train_state.current_model_id

                sp = getattr(self.env, 'switch_point', None)
                switch_point = int(sp) if sp is not None else (self.env.max_episode_length + 1)

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

                for trans in episode_transitions:
                    action_one_hot = np.zeros(self.action_dim, dtype=np.float32)
                    action_one_hot[trans['action']] = 1.0
                    self.replay_buffer.add(
                        state=trans['state'],
                        action=action_one_hot,
                        reward=trans['reward'],
                        next_state=trans['next_state'],
                        done=trans['done'],
                        goal=trans['goal'],
                        model_id=current_model_id,
                    )

                old_model_id_before_him = int(self.train_state.current_model_id)
                him_metrics = self._apply_him(episode, episode_transitions, episode_start_idx)
                new_model_id_after_him = int(self.train_state.current_model_id)
                him_triggered = bool(him_metrics['him_triggered'])
                him_switched = him_triggered and new_model_id_after_him != old_model_id_before_him

                her_transitions = self._apply_her_relabeling(episode_transitions, k=self.config.her.k)
                her_count = len(her_transitions)

                verbose = getattr(getattr(self.config, 'logging', None), 'verbose', False)
                if verbose:
                    orig_count = len(episode_transitions)
                    total_count = orig_count + her_count
                    her_fraction = her_count / total_count if total_count > 0 else 0.0
                    mean_relabeled_reward = float(np.mean([t['reward'] for t in her_transitions])) if her_transitions else 0.0
                    mean_original_reward = float(np.mean([t['reward'] for t in episode_transitions])) if episode_transitions else 0.0
                    print(f"[HER] step={self.train_state.step} | "
                          f"her_fraction={her_fraction:.3f} | "
                          f"mean_relabeled_reward={mean_relabeled_reward:.3f} | "
                          f"mean_original_reward={mean_original_reward:.3f}")

                model_id_for_storage = self.train_state.current_model_id
                for trans in her_transitions:
                    action_one_hot = np.zeros(self.action_dim, dtype=np.float32)
                    action_one_hot[trans['action']] = 1.0
                    self.replay_buffer.add(
                        state=trans['state'],
                        action=action_one_hot,
                        reward=trans['reward'],
                        next_state=trans['next_state'],
                        done=trans['done'],
                        goal=trans['goal'],
                        model_id=model_id_for_storage,
                    )

                if getattr(self.config.training, 'concurrent_training', False) and len(self.replay_buffer) >= self.config.training.batch_size:
                    for _ in range(self.config.training.updates_per_episode):
                        self._update_networks()

                true_policy_name = self._current_policy_name()
                true_model_id = self._true_model_id()
                current_model_id = int(self.train_state.current_model_id)
                self._recent_rewards.append(float(episode_reward))
                self._cumulative_reward += float(episode_reward)
                rolling_mean = float(np.mean(self._recent_rewards))
                self._log_episode_metrics({
                    'episode': int(episode),
                    'agent_type': 'him_her',
                    'current_model_id': current_model_id,
                    'current_model_name': self.model_set.models[current_model_id].name,
                    'true_end_policy': true_policy_name,
                    'switch_point': self._switch_point_value(),
                    'model_correct': int(current_model_id == true_model_id),
                    'him_triggered': int(him_metrics['him_triggered']),
                    'log_lik_per_step': float(him_metrics['log_lik_per_step']),
                    'episode_reward': float(episode_reward),
                    'reward_100ep_mean': rolling_mean,
                })

                # Trajectory logging using shared method (batch-style, after HIM).
                self._log_episode_to_trajectory(
                    episode, episode_transitions,
                    int(self.train_state.current_model_id),
                    episode_reward,
                    self._cumulative_reward,
                    her_count,
                    him_triggered=him_triggered,
                    him_trigger_step=len(episode_transitions) if him_triggered else None,
                    old_model_id=old_model_id_before_him if him_switched else None,
                    new_model_id=new_model_id_after_him if him_switched else None,
                    log_lik_at_trigger=float(him_metrics['log_lik_per_step']) if him_triggered else None,
                )

                episode_rewards.append(float(episode_reward))

                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.3f}, Buffer size: {len(self.replay_buffer)}")
        finally:
            if self.traj_logger is not None:
                self.traj_logger.close()

        return episode_rewards


class HIMOnlyAgent(HIMHERAgent):
    """HIM revision with standard experience replay — HER goal relabeling is disabled.

    Identical to HIMHERAgent except _apply_her_relabeling() returns an empty list so
    no goal-relabeled transitions are ever added to the replay buffer. The HIM
    inconsistency detection and model switching run exactly as in HIMHERAgent.
    """

    def _metrics_agent_type(self) -> str:
        return 'him_only'

    def _apply_her_relabeling(
        self,
        episode_transitions: List[Dict[str, Any]],
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """HER disabled — return empty list so no relabeled transitions are stored."""
        return []