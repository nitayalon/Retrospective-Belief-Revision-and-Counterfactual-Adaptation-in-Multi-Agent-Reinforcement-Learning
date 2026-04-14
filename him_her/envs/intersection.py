"""Intersection Negotiation environment wrapping highway_env.envs.IntersectionEnv.

Three model types for the other vehicle:
  - aggressive: always advances (action FASTER)
  - cautious: always yields (action SLOWER / IDLE)
  - reciprocal: mirrors ego's last discrete action (non-stationary as ego policy evolves)

NOTE on reciprocal: because it mirrors the ego's *learned* action, the effective
model is non-stationary — the observed behavior distribution changes as the ego
policy improves over training. This is intentional and tests HIM's ability to
track an evolving opponent.

Observation (ego): flattened highway_env kinematics vector, shape (obs_dim,)
Other-agent state for HIM: [rel_x, rel_y, other_vx, other_vy] — shape (4,)
Action space: Discrete(5) — highway_env DiscreteMetaAction
"""

from typing import Dict, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401 — registers envs
import jax
import jax.numpy as jnp
import numpy as np

from him_her.envs.base_env import BaseMultiAgentEnv
from him_her.models.base_model import AgentModel


# ---------------------------------------------------------------------------
# Pure JAX model_forward
# ---------------------------------------------------------------------------

def intersection_model_forward(
    policy_params: jnp.ndarray, state: jnp.ndarray
) -> jnp.ndarray:
    """Pure JAX model_forward for Intersection environment.

    policy_params layout (shape 3):
        [advance_preference, yield_preference, temperature]

    state layout (shape 4):
        [rel_x, rel_y, other_vx, other_vy]

    Returns logits over 5 discrete actions.
    """
    advance_pref = policy_params[0]
    yield_pref = policy_params[1]
    temperature = policy_params[2]

    # High speed / FASTER actions: indices 3, 4.  Slow / IDLE: 0, 1, 2.
    action_advance = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
    action_yield = jnp.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)

    preferences = advance_pref * action_advance + yield_pref * action_yield
    return temperature * preferences


# ---------------------------------------------------------------------------
# Scripted policies
# ---------------------------------------------------------------------------

class AggressivePolicy:
    name = "aggressive"

    def select_action(self, state: np.ndarray, ego_last_action: int) -> int:
        return 3  # FASTER

    def get_params(self) -> np.ndarray:
        return np.array([2.0, 0.0, 4.0], dtype=np.float32)


class CautiousPolicy:
    name = "cautious"

    def select_action(self, state: np.ndarray, ego_last_action: int) -> int:
        return 1  # SLOWER / IDLE

    def get_params(self) -> np.ndarray:
        return np.array([0.0, 2.0, 4.0], dtype=np.float32)


class ReciprocalPolicy:
    """Mirrors the ego agent's last discrete action.

    Non-stationary because the ego's action distribution changes during training.
    """
    name = "reciprocal"

    def select_action(self, state: np.ndarray, ego_last_action: int) -> int:
        return int(ego_last_action)

    def get_params(self) -> np.ndarray:
        # Neutral — neither advance nor yield dominated
        return np.array([1.0, 1.0, 2.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class IntersectionEnv(BaseMultiAgentEnv):
    """Intersection Negotiation environment.

    Wraps highway-env's intersection-v1 with a discrete action space.
    The 'other vehicle' policy is one of {aggressive, cautious, reciprocal}.

    Observation (ego): flattened kinematics matrix, shape obs_dim.
    Other-agent state (for HIM): first 4 elements of the other vehicle row.
    """

    action_dim: int = 5

    def __init__(
        self,
        max_episode_length: int = 50,
        seed: Optional[int] = None,
        fixed_policy_name: Optional[str] = None,
    ) -> None:
        self.max_episode_length = max_episode_length
        self.fixed_policy_name = fixed_policy_name
        self.rng = np.random.RandomState(seed)

        self._env = gym.make(
            "intersection-v1",
            config={"action": {"type": "DiscreteMetaAction"}},
        )

        # Determine obs_dim from a sample reset
        obs, _ = self._env.reset()
        flat_obs = obs.flatten().astype(np.float32)
        self.obs_dim: int = flat_obs.size
        self.goal_dim: int = 2

        self.current_step = 0
        self.switch_point: Optional[int] = None
        self._ego_last_action: int = 2  # IDLE

        self._policies = {
            "aggressive": AggressivePolicy(),
            "cautious": CautiousPolicy(),
            "reciprocal": ReciprocalPolicy(),
        }
        self.current_policy = self._policies["aggressive"]
        self._policy_names = list(self._policies.keys())

        # Cache last raw obs for other-agent state extraction
        self._last_raw_obs: np.ndarray = obs

    @property
    def current_policy_name(self) -> str:
        return self.current_policy.name

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._env.reset(seed=seed)
        raw_obs, _ = self._env.reset()
        self._last_raw_obs = raw_obs

        self.current_step = 0
        self._ego_last_action = 2

        if self.fixed_policy_name is None:
            lower = self.max_episode_length // 3
            upper = 2 * self.max_episode_length // 3
            self.switch_point = int(self.rng.randint(lower, upper + 1))
            # Start with a random policy
            start_name = self._policy_names[self.rng.randint(len(self._policy_names))]
            self.current_policy = self._policies[start_name]
        else:
            self.switch_point = self.max_episode_length + 1
            self.current_policy = self._policies[self.fixed_policy_name]

        obs = raw_obs.flatten().astype(np.float32)
        # Goal: goal position is ego start position (dummy — intersection doesn't have explicit goal)
        goal = obs[:2].copy()
        info = {
            "switch_point": self.switch_point,
            "initial_policy": self.current_policy.name,
            "achieved_goal": goal.copy(),
            "desired_goal": goal.copy(),
        }
        return obs, info

    def _get_state(self) -> np.ndarray:
        """Other-agent state for HIM: first 4 values of vehicle row 1."""
        if self._last_raw_obs is not None and self._last_raw_obs.ndim == 2 and self._last_raw_obs.shape[0] > 1:
            return self._last_raw_obs[1, :4].astype(np.float32)
        return np.zeros(4, dtype=np.float32)

    def step(self, ego_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.fixed_policy_name is None and self.current_step == self.switch_point:
            current_idx = self._policy_names.index(self.current_policy.name)
            next_idx = (current_idx + 1) % len(self._policy_names)
            self.current_policy = self._policies[self._policy_names[next_idx]]

        other_state = self._get_state()
        other_action = self.current_policy.select_action(other_state, self._ego_last_action)
        self._ego_last_action = int(ego_action)

        raw_obs, hw_reward, terminated, truncated, hw_info = self._env.step(int(ego_action))
        self._last_raw_obs = raw_obs
        self.current_step += 1

        # Map highway reward to our scheme: +1 cross, -1 crash, 0 otherwise
        crashed = bool(hw_info.get("crashed", False))
        if crashed:
            reward = -1.0
            terminated = True
        elif terminated:
            reward = 1.0
        else:
            reward = float(hw_reward) * 0.1  # small shaped signal

        truncated = truncated or (self.current_step >= self.max_episode_length)

        obs = raw_obs.flatten().astype(np.float32)
        goal = obs[:2].copy()
        info = {
            "other_action": np.array(other_action, dtype=np.int32),
            "achieved_goal": goal.copy(),
            "desired_goal": goal.copy(),
            "crashed": crashed,
        }
        return obs, reward, terminated, truncated, info

    def compute_reward(
        self,
        state: np.ndarray,
        ego_action: np.ndarray,
        goal: np.ndarray,
        model: AgentModel,
    ) -> float:
        """NumPy: +1 if close to goal, 0 otherwise (simplified for relabeling)."""
        pos = state[:2]
        return 1.0 if float(np.linalg.norm(pos - goal)) < 0.5 else 0.0

    def compute_reward_jax(
        self,
        state: jnp.ndarray,
        ego_action: jnp.ndarray,
        goal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Pure JAX reward."""
        pos = state[:2]
        return jnp.where(jnp.linalg.norm(pos - goal) < 0.5, 1.0, 0.0)

    def get_other_action_log_probability(
        self,
        policy_params: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        params_jax = jnp.array(policy_params)
        state_jax = jnp.array(state)
        logits = intersection_model_forward(params_jax, state_jax)
        log_probs = jax.nn.log_softmax(logits)
        return float(log_probs[int(action)])
