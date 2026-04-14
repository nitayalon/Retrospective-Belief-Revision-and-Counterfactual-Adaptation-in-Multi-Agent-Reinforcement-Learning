"""Cooperative Navigation environment wrapper using mpe2.simple_spread_v3.

Ego is agent_0; agent_1 is the other agent whose hidden type we want to infer.
agent_1 is sampled from one of two model types each episode:
  - uniform: no landmark preference (uniform random)
  - landmark_biased: strong additive preference toward landmark 0

The switch point mechanic from predator-prey is reused here: agent_1 silently
changes its type at a random step in [T/3, 2T/3].
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mpe2 import simple_spread_v3

from him_her.envs.base_env import BaseMultiAgentEnv
from him_her.models.base_model import AgentModel


# ---------------------------------------------------------------------------
# Pure JAX model_forward — same structure as predator_prey_model_forward
# ---------------------------------------------------------------------------

def cooperative_nav_model_forward(
    policy_params: jnp.ndarray, state: jnp.ndarray
) -> jnp.ndarray:
    """Pure JAX model_forward for Cooperative Navigation.

    policy_params layout (shape 3):
        [landmark_bias_weight, temperature_scale, landmark0_x, landmark0_y]
        → actually shape (4,) for two-param sets; see configs/cooperative_nav.yaml

    state layout (subset of observation relevant to other agent):
        [agent1_x, agent1_y, landmark0_x, landmark0_y] — shape (4,)

    Returns:
        Logits over the 5-action discrete space (stay, up, down, left, right).
    """
    agent_pos = state[:2]
    landmark0_pos = state[2:4]
    direction_to_landmark = landmark0_pos - agent_pos

    action_directions = jnp.array([
        [0.0, 0.0],   # stay
        [0.0, 1.0],   # up
        [0.0, -1.0],  # down
        [-1.0, 0.0],  # left
        [1.0, 0.0],   # right
    ], dtype=jnp.float32)

    coordinate_scale = jnp.maximum(jnp.max(jnp.abs(state[:4])), 1.0)
    feature_scale = 4.0 * coordinate_scale

    landmark_scores = (action_directions @ direction_to_landmark) / feature_scale

    bias_weight = policy_params[0]
    temperature = policy_params[1]

    preferences = bias_weight * landmark_scores
    return temperature * preferences


# ---------------------------------------------------------------------------
# Scripted policies for the other agent
# ---------------------------------------------------------------------------

class UniformPolicy:
    name = "uniform"

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.RandomState(seed)

    def select_action(self, state: np.ndarray) -> int:
        return int(self._rng.randint(0, 5))


class LandmarkBiasedPolicy:
    name = "landmark_biased"

    def __init__(self, landmark0_pos: np.ndarray, temperature: float = 1.0) -> None:
        self.landmark0_pos = landmark0_pos.copy()
        self.temperature = temperature
        self._rng = np.random.default_rng()

    def select_action(self, state: np.ndarray) -> int:
        agent_pos = state[:2]
        direction = self.landmark0_pos - agent_pos
        # Directional preference scores for [stay, up, down, left, right]
        action_directions = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ])
        scores = action_directions @ direction
        logits = self.temperature * scores
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return int(self._rng.choice(5, p=probs))


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class CooperativeNavEnv(BaseMultiAgentEnv):
    """Cooperative Navigation: ego (agent_0) and one other agent (agent_1).

    The other agent silently switches policy type at a random step in [T/3, 2T/3].
    The ego observes joint positional state but not which policy agent_1 is using.

    Observation (ego, shape 8):
        [agent0_x, agent0_y, agent1_x, agent1_y,
         landmark0_x, landmark0_y, landmark1_x, landmark1_y]

    Other-agent state visible to HIM (shape 4):
        [agent1_x, agent1_y, landmark0_x, landmark0_y]

    Action space: Discrete(5) — 0=stay, 1=up, 2=down, 3=left, 4=right.
    """

    obs_dim: int = 8
    action_dim: int = 5
    goal_dim: int = 2

    def __init__(
        self,
        max_episode_length: int = 50,
        n_agents: int = 2,
        n_landmarks: int = 2,
        world_size: float = 1.0,
        landmark_bias_weight: float = 2.0,
        landmark_bias_temperature: float = 4.0,
        seed: Optional[int] = None,
        fixed_policy_name: Optional[str] = None,
    ) -> None:
        self.max_episode_length = max_episode_length
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.world_size = world_size
        self.landmark_bias_weight = landmark_bias_weight
        self.landmark_bias_temperature = landmark_bias_temperature
        self.fixed_policy_name = fixed_policy_name

        self.current_step = 0
        self.switch_point: Optional[int] = None
        self.rng = np.random.RandomState(seed)

        # These are set during reset()
        self._agent0_pos = np.zeros(2, dtype=np.float32)
        self._agent1_pos = np.zeros(2, dtype=np.float32)
        self._landmark_positions: np.ndarray = np.zeros((n_landmarks, 2), dtype=np.float32)

        self._uniform_policy = UniformPolicy(seed=seed if seed is not None else 0)
        self._biased_policy: Optional[LandmarkBiasedPolicy] = None  # created after reset
        self.current_policy: UniformPolicy | LandmarkBiasedPolicy = self._uniform_policy

    @property
    def current_policy_name(self) -> str:
        return self.current_policy.name

    def _policy_from_name(self, name: str):
        if name == "uniform":
            return self._uniform_policy
        if name == "landmark_biased":
            assert self._biased_policy is not None
            return self._biased_policy
        raise ValueError(f"Unknown policy name: {name}")

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.current_step = 0

        if self.fixed_policy_name is None:
            lower = self.max_episode_length // 3
            upper = 2 * self.max_episode_length // 3
            self.switch_point = int(self.rng.randint(lower, upper + 1))
        else:
            self.switch_point = self.max_episode_length + 1

        # Random positions in [-world_size, world_size]
        self._agent0_pos = self.rng.uniform(-self.world_size, self.world_size, 2).astype(np.float32)
        self._agent1_pos = self.rng.uniform(-self.world_size, self.world_size, 2).astype(np.float32)
        self._landmark_positions = self.rng.uniform(
            -self.world_size, self.world_size, (self.n_landmarks, 2)
        ).astype(np.float32)

        # Update biased policy with new landmark 0 position
        self._biased_policy = LandmarkBiasedPolicy(
            self._landmark_positions[0],
            temperature=self.landmark_bias_temperature,
        )

        if self.fixed_policy_name is None:
            self.current_policy = self._uniform_policy
        else:
            self.current_policy = self._policy_from_name(self.fixed_policy_name)

        obs = self._get_observation()
        info = {
            "switch_point": self.switch_point,
            "initial_policy": self.current_policy.name,
            "achieved_goal": self._landmark_positions[0].copy(),
            "desired_goal": self._landmark_positions[0].copy(),
        }
        return obs, info

    def _get_observation(self) -> np.ndarray:
        """Ego observation: [agent0_pos, agent1_pos, landmark0_pos, landmark1_pos]."""
        obs = np.concatenate([
            self._agent0_pos,
            self._agent1_pos,
            self._landmark_positions[0],
            self._landmark_positions[1] if self.n_landmarks > 1 else self._landmark_positions[0],
        ]).astype(np.float32)
        return obs

    def _get_state(self) -> np.ndarray:
        """State visible to HIM for agent_1: [agent1_pos, landmark0_pos]."""
        return np.concatenate([
            self._agent1_pos,
            self._landmark_positions[0],
        ]).astype(np.float32)

    def _apply_action(self, pos: np.ndarray, action: int) -> np.ndarray:
        step_size = 0.1
        deltas = np.array([
            [0.0, 0.0],
            [0.0, step_size],
            [0.0, -step_size],
            [-step_size, 0.0],
            [step_size, 0.0],
        ], dtype=np.float32)
        new_pos = pos + deltas[action]
        return np.clip(new_pos, -self.world_size, self.world_size)

    def step(self, ego_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Policy switch at switch_point
        if self.fixed_policy_name is None and self.current_step == self.switch_point:
            if self.current_policy.name == "uniform":
                self.current_policy = self._biased_policy
            else:
                self.current_policy = self._uniform_policy

        # Other agent acts
        other_state = self._get_state()
        other_action = int(self.current_policy.select_action(other_state))

        # Move agents
        self._agent0_pos = self._apply_action(self._agent0_pos, ego_action)
        self._agent1_pos = self._apply_action(self._agent1_pos, other_action)

        self.current_step += 1

        reward = self._compute_shaped_reward()
        terminated = False
        truncated = self.current_step >= self.max_episode_length

        next_obs = self._get_observation()
        info = {
            "other_action": np.array(other_action, dtype=np.int32),
            "achieved_goal": self._agent0_pos.copy(),
            "desired_goal": self._landmark_positions[0].copy(),
        }
        return next_obs, reward, terminated, truncated, info

    def _compute_shaped_reward(self) -> float:
        """Shaped reward: negative sum of min agent-to-landmark distances."""
        agent_positions = np.stack([self._agent0_pos, self._agent1_pos])
        dists = np.linalg.norm(
            agent_positions[:, None, :] - self._landmark_positions[None, :, :], axis=-1
        )  # (n_agents, n_landmarks)
        min_dists = dists.min(axis=0)  # min over agents for each landmark
        return float(-np.sum(min_dists))

    def compute_reward(
        self,
        state: np.ndarray,
        ego_action: np.ndarray,
        goal: np.ndarray,
        model: AgentModel,
    ) -> float:
        """NumPy reward: negative distance from agent0 to goal landmark."""
        agent0_pos = state[:2]
        return float(-np.linalg.norm(agent0_pos - goal))

    def compute_reward_jax(
        self,
        state: jnp.ndarray,
        ego_action: jnp.ndarray,
        goal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Pure JAX reward: negative distance from agent0 to goal landmark."""
        agent0_pos = state[:2]
        return -jnp.linalg.norm(agent0_pos - goal)

    def get_other_action_log_probability(
        self,
        policy_params: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        """NumPy: log probability of other_action under policy_params."""
        params_jax = jnp.array(policy_params)
        state_jax = jnp.array(state)
        logits = cooperative_nav_model_forward(params_jax, state_jax)
        log_probs = jax.nn.log_softmax(logits)
        return float(log_probs[int(action)])
