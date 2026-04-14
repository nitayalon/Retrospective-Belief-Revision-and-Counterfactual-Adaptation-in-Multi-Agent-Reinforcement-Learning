"""Spatial analysis and paper-figure generation for HIM+HER experiments.

Loads per-step trajectory CSV files produced by TrajectoryLogger and generates:
  - 2D spatial heatmaps of other-agent positions (detection events)
  - HIM vs Bayesian trigger comparison panels
  - Individual episode behaviour-change trajectories
  - Multi-agent learning curves with mean ± std bands

Run::

    python him_her/utils/spatial_analysis.py

Figures are written to figures/ in the project root.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_steps(agent_type: str, seed: int, log_dir: str = "logs/trajectories") -> pd.DataFrame:
    """Load the flat steps CSV for (agent_type, seed) into a DataFrame."""
    path = Path(log_dir) / agent_type / f"seed{seed}" / "steps.csv"
    if not path.exists():
        raise FileNotFoundError(f"Steps file not found: {path}")
    return pd.read_csv(path)


def load_him_triggers(agent_type: str, seed: int, log_dir: str = "logs/trajectories") -> pd.DataFrame:
    """Load the HIM triggers CSV for (agent_type, seed)."""
    path = Path(log_dir) / agent_type / f"seed{seed}" / "triggers.csv"
    if not path.exists():
        raise FileNotFoundError(f"Triggers file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df


def load_episodes_jsonl(
    agent_type: str, seed: int, log_dir: str = "logs/trajectories"
) -> List[dict]:
    """Load full episode records from JSONL."""
    path = Path(log_dir) / agent_type / f"seed{seed}" / "episodes.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Episodes JSONL not found: {path}")
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_reward_csv(agent_type: str, seed: int, log_dir: str = "logs") -> Optional[pd.DataFrame]:
    """Load the episode-level reward CSV from the standard logs directory."""
    path = Path(log_dir) / f"{agent_type}_seed{seed}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _ensure_figures_dir() -> Path:
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    return fig_dir


# ---------------------------------------------------------------------------
# Plot: 2D spatial heatmap
# ---------------------------------------------------------------------------

def plot_spatial_heatmap(
    steps_df: pd.DataFrame,
    title: str,
    output_path: str,
    grid_size: int = 20,
) -> None:
    """2D heatmap of other-agent positions coloured by detection event type.

    Colours:
      Red   — steps in an episode where HIM triggered
      Blue  — steps where the assumed model was wrong (true != current)
      Green — steps where the assumed model was correct
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Bin positions
    bins = np.linspace(0, grid_size, grid_size + 1)

    for label, colour, mask_fn in [
        ("Model correct", "green", lambda df: df["true_model_id"] == df["current_model_id"]),
        ("Model wrong",   "blue",  lambda df: df["true_model_id"] != df["current_model_id"]),
    ]:
        subset = steps_df[mask_fn(steps_df)]
        if subset.empty:
            continue
        h, xe, ye = np.histogram2d(subset["other_pos_x"], subset["other_pos_y"], bins=bins)
        if h.max() > 0:
            h_norm = h / h.max()
            rgba = plt.get_cmap("Greens" if colour == "green" else "Blues")(h_norm)
            rgba[..., 3] = np.where(h > 0, 0.6, 0.0)
            ax.imshow(
                rgba.transpose(1, 0, 2),
                extent=[0, grid_size, 0, grid_size],
                origin="lower",
                aspect="auto",
            )

    # HIM-triggered episodes overlay
    him_eps = steps_df[steps_df.get("him_triggered", pd.Series([False] * len(steps_df))).astype(bool)] \
        if "him_triggered" in steps_df.columns else pd.DataFrame()
    if not him_eps.empty:
        ax.scatter(
            him_eps["other_pos_x"], him_eps["other_pos_y"],
            c="red", s=4, alpha=0.4, label="HIM trigger episode",
        )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xlabel("Other-agent x")
    ax.set_ylabel("Other-agent y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: HIM vs Bayesian trigger comparison
# ---------------------------------------------------------------------------

def plot_him_vs_bayesian_triggers(
    him_df: pd.DataFrame,
    bayes_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Side-by-side comparison of HIM trigger episodes vs Bayesian belief state.

    Left panel  — episodes and steps where HIM triggered (reward over time).
    Right panel — same episodes showing Bayesian belief in evasive model.
    Vertical lines mark the true switch point per episode.
    """
    him_triggered_eps = him_df[him_df["him_triggered"] == 1]["episode"].unique() \
        if "him_triggered" in him_df.columns else np.array([])

    if len(him_triggered_eps) == 0:
        print("No HIM-triggered episodes to plot; skipping him_vs_bayesian figure.")
        return

    sample_eps = him_triggered_eps[:min(5, len(him_triggered_eps))]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: HIM reward for triggered episodes
    ax_him = axes[0]
    for ep in sample_eps:
        ep_steps = him_df[him_df["episode"] == ep]
        cum_reward = ep_steps["reward"].cumsum().values
        ax_him.plot(cum_reward, alpha=0.7, label=f"ep {ep}")
        sp = ep_steps["switch_point"].iloc[0] if "switch_point" in ep_steps.columns else None
        if sp is not None:
            ax_him.axvline(x=sp, color="red", linestyle="--", alpha=0.5)
    ax_him.set_xlabel("Step")
    ax_him.set_ylabel("Cumulative reward")
    ax_him.set_title("HIM — triggered episodes")
    ax_him.legend(fontsize=7)

    # Right: Bayesian belief for same episodes
    ax_bayes = axes[1]
    for ep in sample_eps:
        if "belief_evasive" not in bayes_df.columns:
            break
        ep_bayes = bayes_df[bayes_df["episode"] == ep]
        if ep_bayes.empty:
            continue
        ax_bayes.plot(
            ep_bayes["step"].values if "step" in ep_bayes.columns else range(len(ep_bayes)),
            ep_bayes["belief_evasive"].values,
            alpha=0.7, label=f"ep {ep}",
        )
        sp_col = "switch_point" if "switch_point" in ep_bayes.columns else None
        if sp_col:
            sp = ep_bayes[sp_col].iloc[0]
            ax_bayes.axvline(x=sp, color="red", linestyle="--", alpha=0.5)
    ax_bayes.set_xlabel("Step")
    ax_bayes.set_ylabel("Belief P(evasive)")
    ax_bayes.set_title("Bayesian — belief in evasive model")
    ax_bayes.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: Behaviour change for a specific episode
# ---------------------------------------------------------------------------

def plot_behavior_change(
    him_df: pd.DataFrame,
    trigger_episode: int,
    output_path: str,
) -> None:
    """Two-panel figure for one episode where HIM triggered.

    Top:    Spatial trajectory of ego and other on the grid.
            Blue = pre-trigger steps, red = post-trigger steps, star = trigger location.
    Bottom: Log-likelihood per step with a vertical line at the trigger step.
    """
    ep_data = him_df[him_df["episode"] == trigger_episode]
    if ep_data.empty:
        print(f"Episode {trigger_episode} not found in DataFrame; skipping.")
        return

    trigger_step = None
    if "him_trigger_step" in ep_data.columns:
        ts_vals = ep_data["him_trigger_step"].dropna()
        if not ts_vals.empty:
            trigger_step = int(ts_vals.iloc[0])

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    ax_traj = axes[0]
    ax_lik  = axes[1]

    steps = ep_data["step"].values
    ego_x  = ep_data["ego_pos_x"].values
    ego_y  = ep_data["ego_pos_y"].values
    oth_x  = ep_data["other_pos_x"].values
    oth_y  = ep_data["other_pos_y"].values
    log_lik = ep_data["log_lik_per_step"].values

    split = trigger_step if trigger_step is not None else len(steps)

    # Pre-trigger
    ax_traj.plot(ego_x[:split],  ego_y[:split],  "b-o",  markersize=3, alpha=0.7, label="Ego  (pre)")
    ax_traj.plot(oth_x[:split],  oth_y[:split],  "b--s", markersize=3, alpha=0.5, label="Other (pre)")
    # Post-trigger
    if split < len(steps):
        ax_traj.plot(ego_x[split:], ego_y[split:], "r-o",  markersize=3, alpha=0.7, label="Ego  (post)")
        ax_traj.plot(oth_x[split:], oth_y[split:], "r--s", markersize=3, alpha=0.5, label="Other (post)")
        ax_traj.scatter([ego_x[split]], [ego_y[split]], marker="*", s=200, color="gold", zorder=5, label="Trigger")

    ax_traj.legend(fontsize=8)
    ax_traj.set_xlabel("x")
    ax_traj.set_ylabel("y")
    ax_traj.set_title(f"Episode {trigger_episode} — trajectory (trigger step={trigger_step})")

    # Log-likelihood
    ax_lik.plot(steps, log_lik, "k-", linewidth=1.5)
    if trigger_step is not None:
        ax_lik.axvline(x=trigger_step, color="red", linestyle="--", label=f"Trigger @ step {trigger_step}")
    ax_lik.set_xlabel("Step")
    ax_lik.set_ylabel("Log-likelihood per step")
    ax_lik.set_title("Log-likelihood (current model vs observed other-agent actions)")
    ax_lik.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot: Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curves(
    agents: List[str],
    seeds: List[int],
    output_path: str,
    window: int = 50,
) -> None:
    """Mean ± std reward curves across seeds for all agents."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colours = {"vanilla": "gray", "bayesian": "blue", "him_only": "orange", "him_her": "green"}

    for agent in agents:
        all_rewards = []
        for seed in seeds:
            df = load_reward_csv(agent, seed)
            if df is None or "episode_reward" not in df.columns:
                continue
            r = df["episode_reward"].values.astype(float)
            # Rolling mean
            rolled = pd.Series(r).rolling(window=window, min_periods=1).mean().values
            all_rewards.append(rolled)

        if not all_rewards:
            continue

        min_len = min(len(r) for r in all_rewards)
        arr = np.array([r[:min_len] for r in all_rewards])
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        xs   = np.arange(min_len)

        colour = colours.get(agent, "black")
        ax.plot(xs, mean, color=colour, label=agent)
        ax.fill_between(xs, mean - std, mean + std, color=colour, alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Reward (rolling mean, window={window})")
    ax.set_title("Learning curves — mean ± std across seeds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main: generate all paper figures
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agents = ["vanilla", "bayesian", "him_only", "him_her"]
    seeds = [0, 42]
    fig_dir = _ensure_figures_dir()

    # Per-agent spatial heatmaps
    for agent in agents:
        for seed in seeds:
            try:
                df = load_steps(agent, seed)
            except FileNotFoundError as exc:
                print(f"[SKIP] {exc}")
                continue
            out = str(fig_dir / f"heatmap_{agent}_seed{seed}.png")
            plot_spatial_heatmap(df, title=f"{agent} seed={seed} — spatial heatmap", output_path=out)

    # HIM vs Bayesian comparison
    try:
        him_steps  = load_steps("him_her", 0)
        bayes_steps = load_steps("bayesian", 0)
        out = str(fig_dir / "him_vs_bayesian_seed0.png")
        plot_him_vs_bayesian_triggers(him_steps, bayes_steps, output_path=out)
    except FileNotFoundError as exc:
        print(f"[SKIP] him_vs_bayesian: {exc}")

    # Behaviour change for a triggered episode (first available)
    try:
        him_steps = load_steps("him_her", 0)
        triggers  = load_him_triggers("him_her", 0)
        if not triggers.empty:
            ep = int(triggers["episode"].iloc[0])
            out = str(fig_dir / f"behavior_change_ep{ep}.png")
            plot_behavior_change(him_steps, trigger_episode=ep, output_path=out)
        else:
            print("[SKIP] No HIM triggers found for him_her seed=0")
    except FileNotFoundError as exc:
        print(f"[SKIP] behavior_change: {exc}")

    # Learning curves
    out = str(fig_dir / "learning_curves.png")
    plot_learning_curves(agents, seeds, output_path=out)

    print("All figures generated.")
