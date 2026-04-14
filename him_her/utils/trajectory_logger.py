"""Per-step trajectory logger for spatial analysis and paper figures.

Directory structure per run:
    logs/{env_name}/{run_id}/
        episodes.csv     — one flat row per episode  (pandas-ready)
        steps.csv        — one flat row per timestep (pandas-ready, can be large)
        him_triggers.csv — subset of episodes.csv where HIM triggered
        metadata.json    — config snapshot + run provenance

Hard constraints (§5 of ARCHITECTURE.md):
  - Never stores JAX arrays; all values converted to plain Python before logging.
  - File handles opened once in __init__ and closed by close().  Never re-opened
    inside the hot training loop.
  - log_gradient_step() called OUTSIDE the JAX boundary (after update_step returns).
"""

import csv
import json
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One timestep of data within an episode (20 flat fields)."""

    # --- Identifiers ---
    run_id: str
    episode: int
    step: int
    # --- Spatial ---
    ego_pos: List[float]                  # [x, y] predator position
    other_pos: List[float]                # [x, y] prey position
    # --- Actions & reward ---
    ego_action: int
    other_action: int
    reward: float
    cumulative_episode_reward: float      # running sum within this episode
    # --- Model identity ---
    current_model_id: int
    current_model_name: str
    true_model_id: int
    true_model_name: str
    # --- Likelihoods ---
    log_lik_per_step: float               # current model log-likelihood
    log_lik_all_models: List[float]       # one per model (JSON string in CSV)
    belief_state: Optional[List[float]]   # posterior over models (JSON string or "")
    # --- HIM state ---
    him_triggered_this_episode: bool
    steps_since_last_trigger: int
    steps_to_switch: int                  # switch_point - step (can be ≤0 after switch)
    # --- Distance ---
    distance: float                       # Euclidean distance ego–other


@dataclass
class EpisodeRecord:
    """Summary of one episode (34 fields)."""

    # --- Identifiers ---
    run_id: str
    env_name: str
    agent_type: str
    seed: int
    episode: int
    # --- Environment ---
    switch_point: int
    total_steps: int
    # --- Reward ---
    episode_reward: float
    cumulative_reward: float
    reward_10ep_mean: float
    reward_50ep_mean: float
    reward_100ep_mean: float
    # --- Model identity ---
    final_model_id: int
    final_model_name: str
    true_model_id_at_end: int
    true_model_name_at_end: str
    model_correct_fraction: float
    detection_lag: Optional[int]          # steps after switch until agent corrects
    # --- HIM ---
    him_triggered: bool
    him_trigger_step: Optional[int]       # episode step at which HIM ran (= total_steps)
    him_trigger_episode_step_fraction: Optional[float]
    old_model_id: Optional[int]
    new_model_id: Optional[int]
    log_lik_at_trigger: Optional[float]
    log_lik_ratio_at_trigger: Optional[float]
    # --- Bayesian ---
    bayesian_switched_this_episode: bool
    bayesian_belief_at_end: Optional[List[float]]  # JSON string in CSV
    # --- Training ---
    critic_loss_mean: float
    actor_loss_mean: float
    her_fraction_mean: float
    buffer_size: int
    gradient_steps: int
    wall_clock_time: float                # seconds for this episode


# ---------------------------------------------------------------------------
# CSV headers
# ---------------------------------------------------------------------------

_STEPS_HEADER = [
    "run_id", "episode", "step",
    "ego_pos_x", "ego_pos_y", "other_pos_x", "other_pos_y",
    "ego_action", "other_action", "reward", "cumulative_episode_reward",
    "current_model_id", "current_model_name", "true_model_id", "true_model_name",
    "log_lik_per_step", "log_lik_all_models", "belief_state",
    "him_triggered_this_episode", "steps_since_last_trigger", "steps_to_switch",
    "distance",
]

_EPISODES_HEADER = [
    "run_id", "env_name", "agent_type", "seed", "episode",
    "switch_point", "total_steps",
    "episode_reward", "cumulative_reward",
    "reward_10ep_mean", "reward_50ep_mean", "reward_100ep_mean",
    "final_model_id", "final_model_name",
    "true_model_id_at_end", "true_model_name_at_end",
    "model_correct_fraction", "detection_lag",
    "him_triggered", "him_trigger_step", "him_trigger_episode_step_fraction",
    "old_model_id", "new_model_id",
    "log_lik_at_trigger", "log_lik_ratio_at_trigger",
    "bayesian_switched_this_episode", "bayesian_belief_at_end",
    "critic_loss_mean", "actor_loss_mean", "her_fraction_mean",
    "buffer_size", "gradient_steps", "wall_clock_time",
]

_TRIGGERS_HEADER = [
    "run_id", "episode",
    "him_trigger_step", "him_trigger_episode_step_fraction",
    "old_model_id", "new_model_id",
    "log_lik_at_trigger", "log_lik_ratio_at_trigger",
    "switch_point", "episode_reward",
]


# ---------------------------------------------------------------------------
# TrajectoryLogger
# ---------------------------------------------------------------------------

class TrajectoryLogger:
    """Incremental per-step trajectory logger with persistent file handles.

    Usage::

        logger = TrajectoryLogger(env_name="predatorprey", run_id="him_her_seed42")
        logger.save_metadata(config)
        try:
            for episode in range(N):
                for step_record in episode:
                    logger.log_step(step_record)          # immediate write
                logger.log_gradient_step(cl, al)          # after each update
                logger.end_episode(episode_record)        # immediate write
        finally:
            logger.close()
    """

    def __init__(
        self,
        env_name: str,
        run_id: str,
        log_dir: str = "logs",
        save_steps: bool = True,
    ) -> None:
        self.env_name = env_name
        self.run_id = run_id
        self.save_steps = save_steps

        self.log_dir = Path(log_dir) / env_name / run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._episodes_path = self.log_dir / "episodes.csv"
        self._steps_path = self.log_dir / "steps.csv"
        self._triggers_path = self.log_dir / "him_triggers.csv"
        self._metadata_path = self.log_dir / "metadata.json"

        # Open persistent file handles (append mode preserves prior runs).
        self._ep_fh = open(self._episodes_path, "a", newline="")
        self._ep_writer = csv.writer(self._ep_fh)
        self._trig_fh = open(self._triggers_path, "a", newline="")
        self._trig_writer = csv.writer(self._trig_fh)

        # steps.csv is optional (can be large).
        self._step_fh = None
        self._step_writer = None
        if self.save_steps:
            self._step_fh = open(self._steps_path, "a", newline="")
            self._step_writer = csv.writer(self._step_fh)

        # Write headers on first creation only.
        if self._episodes_path.stat().st_size == 0:
            self._ep_writer.writerow(_EPISODES_HEADER)
        if self._triggers_path.stat().st_size == 0:
            self._trig_writer.writerow(_TRIGGERS_HEADER)
        if self.save_steps and self._steps_path.stat().st_size == 0:
            self._step_writer.writerow(_STEPS_HEADER)

        # Per-episode gradient accumulators.
        self._ep_critic_losses: List[float] = []
        self._ep_actor_losses: List[float] = []
        self._ep_gradient_steps: int = 0
        self._ep_start_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # Step logging
    # ------------------------------------------------------------------

    def log_step(self, s: StepRecord) -> None:
        """Write one step row to steps.csv immediately.

        Never stores JAX arrays — ``s`` must contain only plain Python types.
        """
        if self._step_writer is None:
            return
        self._step_writer.writerow([
            s.run_id, s.episode, s.step,
            s.ego_pos[0], s.ego_pos[1],
            s.other_pos[0], s.other_pos[1],
            s.ego_action, s.other_action,
            s.reward, s.cumulative_episode_reward,
            s.current_model_id, s.current_model_name,
            s.true_model_id, s.true_model_name,
            s.log_lik_per_step,
            json.dumps(s.log_lik_all_models),
            json.dumps(s.belief_state) if s.belief_state is not None else "",
            int(s.him_triggered_this_episode),
            s.steps_since_last_trigger,
            s.steps_to_switch,
            s.distance,
        ])

    # ------------------------------------------------------------------
    # Gradient-step logging (called OUTSIDE JAX boundary)
    # ------------------------------------------------------------------

    def log_gradient_step(self, critic_loss: float, actor_loss: float) -> None:
        """Accumulate one gradient-step loss for the current episode summary."""
        self._ep_critic_losses.append(float(critic_loss))
        self._ep_actor_losses.append(float(actor_loss))
        self._ep_gradient_steps += 1

    def get_episode_gradient_stats(self) -> Dict[str, Any]:
        """Return accumulated gradient stats and reset for the next episode."""
        stats = {
            "critic_loss_mean": (
                float(np.mean(self._ep_critic_losses))
                if self._ep_critic_losses else 0.0
            ),
            "actor_loss_mean": (
                float(np.mean(self._ep_actor_losses))
                if self._ep_actor_losses else 0.0
            ),
            "gradient_steps": self._ep_gradient_steps,
            "wall_clock_time": time.monotonic() - self._ep_start_time,
        }
        self._ep_critic_losses = []
        self._ep_actor_losses = []
        self._ep_gradient_steps = 0
        self._ep_start_time = time.monotonic()
        return stats

    # ------------------------------------------------------------------
    # Episode logging
    # ------------------------------------------------------------------

    def end_episode(self, ep: EpisodeRecord) -> None:
        """Write one episode row to episodes.csv (and him_triggers.csv if triggered)."""
        self._ep_writer.writerow([
            ep.run_id, ep.env_name, ep.agent_type, ep.seed, ep.episode,
            ep.switch_point, ep.total_steps,
            ep.episode_reward, ep.cumulative_reward,
            ep.reward_10ep_mean, ep.reward_50ep_mean, ep.reward_100ep_mean,
            ep.final_model_id, ep.final_model_name,
            ep.true_model_id_at_end, ep.true_model_name_at_end,
            ep.model_correct_fraction, ep.detection_lag,
            int(ep.him_triggered), ep.him_trigger_step,
            ep.him_trigger_episode_step_fraction,
            ep.old_model_id, ep.new_model_id,
            ep.log_lik_at_trigger, ep.log_lik_ratio_at_trigger,
            int(ep.bayesian_switched_this_episode),
            json.dumps(ep.bayesian_belief_at_end)
            if ep.bayesian_belief_at_end is not None else "",
            ep.critic_loss_mean, ep.actor_loss_mean,
            ep.her_fraction_mean, ep.buffer_size,
            ep.gradient_steps, ep.wall_clock_time,
        ])
        self._ep_fh.flush()

        if ep.him_triggered:
            self._trig_writer.writerow([
                ep.run_id, ep.episode,
                ep.him_trigger_step, ep.him_trigger_episode_step_fraction,
                ep.old_model_id, ep.new_model_id,
                ep.log_lik_at_trigger, ep.log_lik_ratio_at_trigger,
                ep.switch_point, ep.episode_reward,
            ])
            self._trig_fh.flush()

        if self._step_fh is not None:
            self._step_fh.flush()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def save_metadata(self, config: Any) -> None:
        """Write a JSON metadata file with config and run provenance."""
        def _to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dict__"):
                return {k: _to_dict(v) for k, v in vars(obj).items()}
            if isinstance(obj, (list, tuple)):
                return [_to_dict(x) for x in obj]
            return obj

        metadata = {
            "run_id": self.run_id,
            "env_name": self.env_name,
            "log_dir": str(self.log_dir),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config": _to_dict(config),
        }
        with open(self._metadata_path, "w") as fh:
            json.dump(metadata, fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close all open file handles."""
        for fh in (self._ep_fh, self._trig_fh, self._step_fh):
            if fh is not None and not fh.closed:
                fh.flush()
                fh.close()

