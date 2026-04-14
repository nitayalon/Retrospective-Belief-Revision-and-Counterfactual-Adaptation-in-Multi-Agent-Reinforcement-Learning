"""Hide-and-Seek environment wrapper using mpe2.simple_tag_v3.

Ego is agent_0 (the prey); agents adversary_0 and adversary_1 are the chasers
whose hidden cooperative strategy we want to infer.

Three adversary strategy models:
  - flanking:  adversary_0 circles left, adversary_1 circles right (coordinate)
  - direct:    both adversaries move straight toward the prey (no coordination)
  - ambush:    adversary_0 intercepts predicted path; adversary_1 drives directly

Only adversary_0's behaviour is modelled (the visible "leader" adversary). The
switch point mechanic from predator-prey is reused here.

NOTE: HER relabeling is disabled for this environment (no_her=True) because the
goal is fixed (survival) and counterfactual goal substitution is not meaningful.

Observation (ego/prey): flattened observation from simple_tag_v3, shape obs_dim.
Other-agent state for HIM: [adv0_x, adv0_y, adv0_vx, adv0_vy] — shape (4,).
Action space: Discrete(5) — 0=stay, 1=up, 2=down, 3=left, 4=right.
Reward: +1 per step alive (survival), -1 on capture.
"""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from mpe2 import simple_tag_v3

from him_her.envs.base_env import BaseMultiAgentEnv
from him_her.models.base_model import AgentModel


# ---------------------------------------------------------------------------
# Pure JAX model_forward
# ---------------------------------------------------------------------------

def hide_and_seek_model_forward(
    policy_params: jnp.ndarray, state: jnp.ndarray
) -> jnp.ndarray:
    """Pure JAX model_forward for Hide-and-Seek (adversary_0's policy).

    policy_params layout (shape 3):
        [intercept_weight, direct_weight, temperature]

    state layout (shape 4):
        [adv0_x, adv0_y, prey_x, prey_y]

    Returns logits over 5 discrete actions.
    """
    adv_pos = state[:2]
    prey_pos = state[2:4]
    direction_to_prey = prey_pos - adv_pos

    # Action directions: stay, up, down, left, right
    action_directions = jnp.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [-1.0, 0.0],
        [1.0, 0.0],
    ], dtype=jnp.float32)

    scale = jnp.maximum(jnp.linalg.norm(direction_to_prey), 1e-6)
    norm_dir = direction_to_prey / scale

    direct_scores = action_directions @ norm_dir
    intercept_scores = action_directions @ jnp.array([-norm_dir[1], norm_dir[0]])

    direct_weight = policy_params[1]
    intercept_weight = policy_params[0]
    temperature = policy_params[2]

    preferences = intercept_weight * intercept_scores + direct_weight * direct_scores
    return temperature * preferences


# ---------------------------------------------------------------------------
# Scripted adversary policies
# ---------------------------------------------------------------------------

class DirectPolicy:
    """Both adversaries move directly toward the prey."""
    name = "direct"

    def select_action(self, adv_pos: np.ndarray, prey_pos: np.ndarray) -> int:
        direction = prey_pos - adv_pos
        # Pick the action closest to the direction
        action_directions = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ])
        scores = action_directions @ direction
        return int(np.argmax(scores))

    def get_params(self) -> np.ndarray:
        return np.array([0.0, 2.0, 4.0], dtype=np.float32)


class FlankingPolicy:
    """Adversary_0 circles left (perpendicular cut); adversary_1 drives direct."""
    name = "flanking"

    def select_action(self, adv_pos: np.ndarray, prey_pos: np.ndarray) -> int:
        direction = prey_pos - adv_pos
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return 0
        nd = direction / norm
        # Perpendicular: rotate 90° counterclockwise
        perp = np.array([-nd[1], nd[0]])
        flanking_dir = nd + perp
        action_directions = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ])
        scores = action_directions @ flanking_dir
        return int(np.argmax(scores))

    def get_params(self) -> np.ndarray:
        return np.array([1.5, 0.5, 4.0], dtype=np.float32)


class AmbushPolicy:
    """Adversary_0 intercepts predicted path (ahead of prey's movement)."""
    name = "ambush"

    def __init__(self) -> None:
        self._prev_prey_pos: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_prey_pos = None

    def select_action(self, adv_pos: np.ndarray, prey_pos: np.ndarray) -> int:
        if self._prev_prey_pos is None:
            prey_velocity = np.zeros(2)
        else:
            prey_velocity = prey_pos - self._prev_prey_pos
        self._prev_prey_pos = prey_pos.copy()

        # Intercept point: 2 steps ahead
        intercept = prey_pos + 2.0 * prey_velocity
        direction = intercept - adv_pos
        action_directions = np.array([
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [1.0, 0.0],
        ])
        scores = action_directions @ direction
        return int(np.argmax(scores))

    def get_params(self) -> np.ndarray:
        return np.array([2.0, 1.0, 3.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment wrapper
# ---------------------------------------------------------------------------

class HideAndSeekEnv(BaseMultiAgentEnv):
    """Hide-and-Seek environment.

    Wraps mpe2.simple_tag_v3. Ego is agent_0 (prey). adversary_0 uses a scripted
    policy drawn from {direct, flanking, ambush}; adversary_1 always uses DirectPolicy.

    HER is disabled via the `no_her` flag.
    """

    action_dim: int = 5
    no_her: bool = True  # Trainer checks env.no_her before calling HER

    def __init__(
        self,
        max_episode_length: int = 50,
        seed: Optional[int] = None,
        fixed_policy_name: Optional[str] = None,
    ) -> None:
        self.max_episode_length = max_episode_length
        self.fixed_policy_name = fixed_policy_name
        self.rng = np.random.RandomState(seed)

        self._env = simple_tag_v3.parallel_env(
            num_good=1,
            num_adversaries=2,
            num_obstacles=2,
            max_cycles=max_episode_length,
            continuous_actions=False,
        )

        # Determine obs_dim by running a sample reset
        obs_dict, _ = self._env.reset(seed=seed if seed is not None else 0)
        # Ego is "agent_0"
        ego_obs = obs_dict.get("agent_0", next(iter(obs_dict.values())))
        self.obs_dim: int = int(np.array(ego_obs).flatten().size)
        self.goal_dim: int = 2

        self._policies = {
            "direct": DirectPolicy(),
            "flanking": FlankingPolicy(),
            "ambush": AmbushPolicy(),
        }
        self._policy_names = list(self._policies.keys())
        self.current_policy = self._policies["direct"]
        self.current_step = 0
        self.switch_point: Optional[int] = None

        # Latest raw observations dict (for state extraction)
        self._last_obs_dict: Dict = obs_dict

    @property
    def current_policy_name(self) -> str:
        return self.current_policy.name

    def _extract_agent_pos(self, agent_name: str) -> np.ndarray:
        """Extract (x, y) position from agent's observation (first two elements)."""
        obs = self._last_obs_dict.get(agent_name)
        if obs is None:
            return np.zeros(2, dtype=np.float32)
        return np.array(obs[:2], dtype=np.float32)

    def _get_state(self) -> np.ndarray:
        """Other-agent state for HIM: [adv0_x, adv0_y, prey_x, prey_y]."""
        adv0_pos = self._extract_agent_pos("adversary_0")
        prey_pos = self._extract_agent_pos("agent_0")
        return np.concatenate([adv0_pos, prey_pos]).astype(np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        rng_seed = seed if seed is not None else int(self.rng.randint(2**31))
        obs_dict, info_dict = self._env.reset(seed=rng_seed)
        self._last_obs_dict = obs_dict

        self.current_step = 0

        # Reset ambush policy's velocity estimate
        ambush = self._policies["ambush"]
        if hasattr(ambush, "reset"):
            ambush.reset()

        if self.fixed_policy_name is None:
            lower = self.max_episode_length // 3
            upper = 2 * self.max_episode_length // 3
            self.switch_point = int(self.rng.randint(lower, upper + 1))
            start_name = self._policy_names[self.rng.randint(len(self._policy_names))]
            self.current_policy = self._policies[start_name]
        else:
            self.switch_point = self.max_episode_length + 1
            self.current_policy = self._policies[self.fixed_policy_name]

        ego_obs = obs_dict.get("agent_0", next(iter(obs_dict.values())))
        obs = np.array(ego_obs, dtype=np.float32).flatten()
        goal = obs[:2].copy()
        info = {
            "switch_point": self.switch_point,
            "initial_policy": self.current_policy.name,
            "achieved_goal": goal.copy(),
            "desired_goal": goal.copy(),
        }
        return obs, info

    def step(self, ego_action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.fixed_policy_name is None and self.current_step == self.switch_point:
            current_idx = self._policy_names.index(self.current_policy.name)
            next_idx = (current_idx + 1) % len(self._policy_names)
            self.current_policy = self._policies[self._policy_names[next_idx]]

        # Compute adversary actions using scripted policies
        adv0_pos = self._extract_agent_pos("adversary_0")
        prey_pos = self._extract_agent_pos("agent_0")
        adv1_pos = self._extract_agent_pos("adversary_1")

        adv0_action = self.current_policy.select_action(adv0_pos, prey_pos)
        adv1_action = DirectPolicy().select_action(adv1_pos, prey_pos)

        action_dict = {
            "agent_0": int(ego_action),
            "adversary_0": int(adv0_action),
            "adversary_1": int(adv1_action),
        }

        obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict = self._env.step(action_dict)
        self._last_obs_dict = obs_dict
        self.current_step += 1

        ego_obs = obs_dict.get("agent_0", np.zeros(self.obs_dim, dtype=np.float32))
        obs = np.array(ego_obs, dtype=np.float32).flatten()

        ego_reward = float(reward_dict.get("agent_0", 0.0))
        terminated = bool(terminated_dict.get("agent_0", False))
        truncated = bool(truncated_dict.get("agent_0", False)) or (
            self.current_step >= self.max_episode_length
        )

        # Shaped reward: +1 alive, -1 captured (MPE signals capture by high negative reward)
        if ego_reward < -0.5:
            shaped_reward = -1.0
            terminated = True
        else:
            shaped_reward = 1.0

        goal = obs[:2].copy()
        info = {
            "other_action": np.array(adv0_action, dtype=np.int32),
            "achieved_goal": goal.copy(),
            "desired_goal": goal.copy(),
        }
        return obs, shaped_reward, terminated, truncated, info

    def compute_reward(
        self,
        state: np.ndarray,
        ego_action: np.ndarray,
        goal: np.ndarray,
        model: AgentModel,
    ) -> float:
        """NumPy: +1 if prey is far from adversary, else 0."""
        adv0_pos = state[:2]
        prey_pos = state[2:4]
        dist = float(np.linalg.norm(prey_pos - adv0_pos))
        return 1.0 if dist > 0.5 else 0.0

    def compute_reward_jax(
        self,
        state: jnp.ndarray,
        ego_action: jnp.ndarray,
        goal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Pure JAX reward."""
        adv0_pos = state[:2]
        prey_pos = state[2:4]
        dist = jnp.linalg.norm(prey_pos - adv0_pos)
        return jnp.where(dist > 0.5, 1.0, 0.0)

    def get_other_action_log_probability(
        self,
        policy_params: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
    ) -> float:
        params_jax = jnp.array(policy_params)
        state_jax = jnp.array(state)
        logits = hide_and_seek_model_forward(params_jax, state_jax)
        log_probs = jax.nn.log_softmax(logits)
        return float(log_probs[int(action)])
