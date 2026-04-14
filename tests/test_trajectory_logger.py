"""Tests for TrajectoryLogger — him_her/utils/trajectory_logger.py.

Covers:
  test_step_record_fields       — StepRecord has all 20 expected fields with correct types
  test_episode_record_fields    — EpisodeRecord has all 34 expected fields after direct construction
  test_csv_files_created        — episodes.csv / steps.csv / him_triggers.csv created; row counts correct
  test_incremental_checkpoint   — each log_step / end_episode write is immediate (no buffering)
  test_close_flushes            — close() flushes and files are readable after close
"""

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from him_her.utils.trajectory_logger import (
    EpisodeRecord,
    StepRecord,
    TrajectoryLogger,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_step(
    episode: int = 0,
    step: int = 0,
    run_id: str = "test_run",
    him_triggered_this_episode: bool = False,
) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        episode=episode,
        step=step,
        ego_pos=[1.5, 2.5],
        other_pos=[8.0, 9.0],
        ego_action=2,
        other_action=4,
        reward=0.0,
        cumulative_episode_reward=0.0,
        current_model_id=0,
        current_model_name="evasive",
        true_model_id=0,
        true_model_name="evasive",
        log_lik_per_step=-1.23,
        log_lik_all_models=[-1.23, -2.50],
        belief_state=[0.8, 0.2],
        him_triggered_this_episode=him_triggered_this_episode,
        steps_since_last_trigger=0,
        steps_to_switch=20 - step,
        distance=float(np.linalg.norm(np.array([1.5, 2.5]) - np.array([8.0, 9.0]))),
    )


def _make_episode(
    episode: int = 0,
    run_id: str = "test_run",
    him_triggered: bool = False,
) -> EpisodeRecord:
    return EpisodeRecord(
        run_id=run_id,
        env_name="test_env",
        agent_type="test_agent",
        seed=0,
        episode=episode,
        switch_point=20,
        total_steps=10,
        episode_reward=1.0,
        cumulative_reward=float(episode + 1),
        reward_10ep_mean=1.0,
        reward_50ep_mean=1.0,
        reward_100ep_mean=1.0,
        final_model_id=0,
        final_model_name="evasive",
        true_model_id_at_end=0,
        true_model_name_at_end="evasive",
        model_correct_fraction=1.0,
        detection_lag=None,
        him_triggered=him_triggered,
        him_trigger_step=10 if him_triggered else None,
        him_trigger_episode_step_fraction=1.0 if him_triggered else None,
        old_model_id=0 if him_triggered else None,
        new_model_id=1 if him_triggered else None,
        log_lik_at_trigger=-2.5 if him_triggered else None,
        log_lik_ratio_at_trigger=1.3 if him_triggered else None,
        bayesian_switched_this_episode=False,
        bayesian_belief_at_end=None,
        critic_loss_mean=0.05,
        actor_loss_mean=0.02,
        her_fraction_mean=0.4,
        buffer_size=512,
        gradient_steps=5,
        wall_clock_time=0.12,
    )


def _make_logger(tmp_path: Path, run_id: str = "test_run") -> TrajectoryLogger:
    return TrajectoryLogger(
        env_name="test_env",
        run_id=run_id,
        log_dir=str(tmp_path),
        save_steps=True,
    )


# ---------------------------------------------------------------------------
# test_step_record_fields
# ---------------------------------------------------------------------------

def test_step_record_fields():
    """StepRecord must have all 20 fields with the expected Python types."""
    s = _make_step(episode=3, step=7, run_id="myrun")

    # Identifiers
    assert isinstance(s.run_id, str) and s.run_id == "myrun"
    assert isinstance(s.episode, int) and s.episode == 3
    assert isinstance(s.step, int) and s.step == 7

    # Spatial
    assert isinstance(s.ego_pos, list) and len(s.ego_pos) == 2
    assert all(isinstance(v, float) for v in s.ego_pos)
    assert isinstance(s.other_pos, list) and len(s.other_pos) == 2

    # Actions & reward
    assert isinstance(s.ego_action, int)
    assert isinstance(s.other_action, int)
    assert isinstance(s.reward, float)
    assert isinstance(s.cumulative_episode_reward, float)

    # Model identity
    assert isinstance(s.current_model_id, int)
    assert isinstance(s.current_model_name, str)
    assert isinstance(s.true_model_id, int)
    assert isinstance(s.true_model_name, str)

    # Likelihoods
    assert isinstance(s.log_lik_per_step, float)
    assert isinstance(s.log_lik_all_models, list)
    assert all(isinstance(v, float) for v in s.log_lik_all_models)
    assert s.belief_state is None or isinstance(s.belief_state, list)

    # HIM state
    assert isinstance(s.him_triggered_this_episode, bool)
    assert isinstance(s.steps_since_last_trigger, int)
    assert isinstance(s.steps_to_switch, int)

    # Distance — no JAX arrays
    assert isinstance(s.distance, float)
    assert abs(s.distance - float(np.linalg.norm([6.5, 6.5]))) < 1e-6


# ---------------------------------------------------------------------------
# test_episode_record_fields
# ---------------------------------------------------------------------------

def test_episode_record_fields():
    """EpisodeRecord must have all 34 expected fields with correct types."""
    ep = _make_episode(episode=5, him_triggered=True)

    # Identifiers
    assert isinstance(ep.run_id, str)
    assert isinstance(ep.env_name, str)
    assert isinstance(ep.agent_type, str)
    assert isinstance(ep.seed, int)
    assert ep.episode == 5

    # Environment
    assert ep.switch_point == 20
    assert ep.total_steps == 10

    # Reward
    assert isinstance(ep.episode_reward, float)
    assert isinstance(ep.cumulative_reward, float)
    assert isinstance(ep.reward_10ep_mean, float)
    assert isinstance(ep.reward_50ep_mean, float)
    assert isinstance(ep.reward_100ep_mean, float)

    # Model identity
    assert isinstance(ep.final_model_id, int)
    assert isinstance(ep.final_model_name, str)
    assert isinstance(ep.true_model_id_at_end, int)
    assert isinstance(ep.true_model_name_at_end, str)
    assert isinstance(ep.model_correct_fraction, float)
    assert ep.detection_lag is None or isinstance(ep.detection_lag, int)

    # HIM
    assert ep.him_triggered is True
    assert ep.him_trigger_step == 10
    assert abs(ep.him_trigger_episode_step_fraction - 1.0) < 1e-9
    assert ep.old_model_id == 0
    assert ep.new_model_id == 1
    assert isinstance(ep.log_lik_at_trigger, float)
    assert isinstance(ep.log_lik_ratio_at_trigger, float)

    # Bayesian
    assert isinstance(ep.bayesian_switched_this_episode, bool)
    assert ep.bayesian_belief_at_end is None or isinstance(ep.bayesian_belief_at_end, list)

    # Training
    assert isinstance(ep.critic_loss_mean, float)
    assert isinstance(ep.actor_loss_mean, float)
    assert isinstance(ep.her_fraction_mean, float)
    assert isinstance(ep.buffer_size, int)
    assert isinstance(ep.gradient_steps, int)
    assert isinstance(ep.wall_clock_time, float)


# ---------------------------------------------------------------------------
# test_csv_files_created
# ---------------------------------------------------------------------------

def test_csv_files_created(tmp_path):
    """After logging, episodes.csv / steps.csv / him_triggers.csv must exist with correct row counts."""
    logger = _make_logger(tmp_path)

    # Episode 0 — no HIM trigger
    for t in range(5):
        logger.log_step(_make_step(episode=0, step=t))
    logger.end_episode(_make_episode(episode=0, him_triggered=False))

    # Episode 1 — HIM triggered
    for t in range(8):
        logger.log_step(_make_step(episode=1, step=t, him_triggered_this_episode=True))
    logger.end_episode(_make_episode(episode=1, him_triggered=True))

    logger.close()

    log_path = tmp_path / "test_env" / "test_run"
    episodes_csv = log_path / "episodes.csv"
    steps_csv = log_path / "steps.csv"
    triggers_csv = log_path / "him_triggers.csv"

    assert episodes_csv.exists(), "episodes.csv not created"
    assert steps_csv.exists(), "steps.csv not created"
    assert triggers_csv.exists(), "him_triggers.csv not created"

    # episodes.csv: 1 header + 2 rows
    with open(episodes_csv) as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 3, f"Expected 3 rows in episodes.csv, got {len(rows)}"

    # steps.csv: 1 header + 5 + 8 = 14 rows
    with open(steps_csv) as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 14, f"Expected 14 rows in steps.csv, got {len(rows)}"

    # him_triggers.csv: 1 header + 1 trigger row
    with open(triggers_csv) as fh:
        trows = list(csv.reader(fh))
    assert len(trows) == 2, f"Expected 2 rows in him_triggers.csv, got {len(trows)}"
    assert trows[1][1] == "1", "Trigger row should be for episode 1"

    # Verify JSON-serialized fields round-trip
    with open(steps_csv) as fh:
        step_rows = list(csv.DictReader(fh))
    assert json.loads(step_rows[0]["log_lik_all_models"]) == [-1.23, -2.5]
    assert json.loads(step_rows[0]["belief_state"]) == [0.8, 0.2]


# ---------------------------------------------------------------------------
# test_incremental_checkpoint
# ---------------------------------------------------------------------------

def test_incremental_checkpoint(tmp_path):
    """Each log_step / end_episode write must be immediately visible on disk."""
    logger = _make_logger(tmp_path)
    log_path = tmp_path / "test_env" / "test_run"

    def step_row_count() -> int:
        p = log_path / "steps.csv"
        if not p.exists():
            return 0
        with open(p) as fh:
            return sum(1 for line in fh if line.strip()) - 1  # exclude header

    def episode_row_count() -> int:
        p = log_path / "episodes.csv"
        if not p.exists():
            return 0
        with open(p) as fh:
            return sum(1 for line in fh if line.strip()) - 1  # exclude header

    # After 3 steps, flush and verify
    for t in range(3):
        logger.log_step(_make_step(episode=0, step=t))
    logger._step_fh.flush()
    assert step_row_count() == 3

    # End episode 0 -> episodes.csv should have 1 row
    logger.end_episode(_make_episode(episode=0))
    assert episode_row_count() == 1

    # After episode 1
    for t in range(4):
        logger.log_step(_make_step(episode=1, step=t))
    logger._step_fh.flush()
    assert step_row_count() == 7

    logger.end_episode(_make_episode(episode=1))
    assert episode_row_count() == 2

    logger.close()


# ---------------------------------------------------------------------------
# test_close_flushes
# ---------------------------------------------------------------------------

def test_close_flushes(tmp_path):
    """After close(), all CSV data is readable and file handles are closed."""
    logger = _make_logger(tmp_path)

    for t in range(6):
        logger.log_step(_make_step(episode=0, step=t))
    logger.log_gradient_step(critic_loss=0.1, actor_loss=0.05)
    logger.end_episode(_make_episode(episode=0))
    logger.close()

    log_path = tmp_path / "test_env" / "test_run"

    # File handles must be closed
    assert logger._ep_fh.closed
    assert logger._trig_fh.closed
    assert logger._step_fh is None or logger._step_fh.closed

    # Data must be fully readable
    with open(log_path / "steps.csv") as fh:
        step_rows = list(csv.reader(fh))
    assert len(step_rows) == 7, f"Expected header + 6 rows, got {len(step_rows)}"

    with open(log_path / "episodes.csv") as fh:
        ep_rows = list(csv.reader(fh))
    assert len(ep_rows) == 2, f"Expected header + 1 episode row, got {len(ep_rows)}"

    # Append mode: re-opening should add to existing data
    logger2 = TrajectoryLogger(
        env_name="test_env",
        run_id="test_run",
        log_dir=str(tmp_path),
        save_steps=True,
    )
    logger2.log_step(_make_step(episode=1, step=0))
    logger2.end_episode(_make_episode(episode=1))
    logger2.close()

    with open(log_path / "episodes.csv") as fh:
        ep_rows = list(csv.reader(fh))
    assert len(ep_rows) == 3, f"Expected 3 rows after re-open append, got {len(ep_rows)}"
