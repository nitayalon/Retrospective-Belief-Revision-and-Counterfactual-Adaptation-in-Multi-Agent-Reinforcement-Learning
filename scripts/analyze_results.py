#!/usr/bin/env python3
"""Comprehensive analysis script for HIM-HER experiment results.

Reads the episode CSVs produced by TrajectoryLogger and generates four
publication-quality figures plus a statistical summary table.

Usage
-----
    # Analyse all runs in logs/ and save figures to figures/
    python scripts/analyze_results.py

    # Analyse a merged CSV (from slurm/collect_results.sh)
    python scripts/analyze_results.py --input logs/all_results.csv

    # Only print stats, skip figures
    python scripts/analyze_results.py --no-figures

Figures produced
----------------
    figures/fig1_learning_curves.pdf    — episode reward vs episode (rolling mean)
    figures/fig2_him_detection.pdf      — HIM trigger rate + detection lag CDF
    figures/fig3_model_accuracy.pdf     — model-correctness fraction over training
    figures/fig4_her_impact.pdf         — HER fraction vs reward correlation
"""

import argparse
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------
# Optional imports (graceful degradation)
# -----------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("[warn] pandas not installed — some statistics will be skipped.")

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[warn] matplotlib not installed — figures will not be generated.")

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[warn] scipy not installed — t-tests and Cohen's d will be skipped.")


# -----------------------------------------------------------------------
# Colour palette (colour-blind friendly, 5 agents)
# -----------------------------------------------------------------------
AGENT_COLOURS = {
    "him_her":  "#2166AC",
    "him_only": "#4DAC26",
    "bayesian": "#D01C8B",
    "vanilla":  "#F1A340",
    "static":   "#999999",
}
AGENT_ORDER = ["him_her", "him_only", "bayesian", "vanilla", "static"]


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def _load_single_csv(path: str) -> List[dict]:
    """Return list of row dicts from one episodes.csv file."""
    rows = []
    with open(path) as fh:
        import csv
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def load_all_episodes(log_root: str = "logs") -> List[dict]:
    """Recursively discover and load all episodes.csv files under log_root."""
    root = Path(log_root)
    all_rows: List[dict] = []
    for csv_path in sorted(root.rglob("episodes.csv")):
        all_rows.extend(_load_single_csv(str(csv_path)))
    print(f"Loaded {len(all_rows)} episode rows from {log_root}/")
    return all_rows


def rows_to_df(rows: List[dict]):
    """Convert list of dicts to a pandas DataFrame with typed columns.

    Falls back to a plain list if pandas is not available.
    """
    if not HAS_PANDAS:
        return rows

    df = pd.DataFrame(rows)

    # Cast numeric columns
    numeric_cols = [
        "seed", "episode", "switch_point", "total_steps",
        "episode_reward", "cumulative_reward",
        "reward_10ep_mean", "reward_50ep_mean", "reward_100ep_mean",
        "final_model_id", "true_model_id_at_end",
        "model_correct_fraction", "detection_lag",
        "him_trigger_step", "old_model_id", "new_model_id",
        "log_lik_at_trigger", "log_lik_ratio_at_trigger",
        "critic_loss_mean", "actor_loss_mean", "her_fraction_mean",
        "buffer_size", "gradient_steps", "wall_clock_time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = ["him_triggered", "bayesian_switched_this_episode"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map({"1": True, "0": False, "True": True, "False": False})

    return df


# -----------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------

def rolling_mean(values: List[float], window: int = 50) -> List[float]:
    """Compute a centred rolling mean (edges use available data)."""
    result = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        result.append(float(np.mean(values[lo:hi])))
    return result


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Effect size (Cohen's d) between two independent samples."""
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0.0:
        return float("nan")
    return (float(np.mean(group_a)) - float(np.mean(group_b))) / pooled_std


def two_sample_t_test(
    group_a: List[float], group_b: List[float]
) -> Tuple[float, float]:
    """Two-sided Welch t-test. Returns (t_stat, p_value)."""
    if not HAS_SCIPY or len(group_a) < 2 or len(group_b) < 2:
        return float("nan"), float("nan")
    result = scipy_stats.ttest_ind(group_a, group_b, equal_var=False)
    return float(result.statistic), float(result.pvalue)


# -----------------------------------------------------------------------
# Statistics table
# -----------------------------------------------------------------------

def print_statistics_table(df, last_n: int = 100) -> None:
    """Print mean ± std of episode_reward over the last N episodes per seed."""
    print("\n" + "=" * 72)
    print(f"  Reward statistics — last {last_n} episodes per seed")
    print("=" * 72)
    print(f"  {'Agent':<18} {'Seeds':>6} {'Mean±Std reward':>22}  {'vs him_her d':>12}")
    print("-" * 72)

    if HAS_PANDAS and hasattr(df, "groupby"):
        seed_means = (
            df.groupby(["agent_type", "seed"])
            .apply(lambda g: g.nlargest(last_n, "episode")["episode_reward"].mean())
            .reset_index(name="seed_mean")
        )
        agent_stats: Dict[str, List[float]] = defaultdict(list)
        for _, row in seed_means.iterrows():
            agent_stats[row["agent_type"]].append(float(row["seed_mean"]))

        him_vals = agent_stats.get("him_her", [])
        for agent in AGENT_ORDER:
            vals = agent_stats.get(agent, [])
            if not vals:
                continue
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            d = cohens_d(him_vals, vals) if agent != "him_her" else float("nan")
            t, p = two_sample_t_test(him_vals, vals)
            d_str = f"{d:+.2f}" if not math.isnan(d) else "   —"
            p_str = f"p={p:.3f}" if not math.isnan(p) else ""
            print(f"  {agent:<18} {len(vals):>6} {mu:>10.3f} ± {sd:>6.3f}  {d_str:>8}  {p_str}")
    else:
        print("  (pandas required for full statistics table)")

    print("=" * 72 + "\n")


# -----------------------------------------------------------------------
# Figure 1 — Learning curves
# -----------------------------------------------------------------------

def fig_learning_curves(df, out_path: str, window: int = 50) -> None:
    """Rolling-mean episode reward vs episode, one curve per agent type."""
    if not HAS_MPL or not HAS_PANDAS:
        print("[skip] fig_learning_curves requires matplotlib + pandas.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for agent in AGENT_ORDER:
        sub = df[df["agent_type"] == agent].copy()
        if sub.empty:
            continue
        colour = AGENT_COLOURS.get(agent, "black")
        # Average over seeds, align by episode index
        ep_rewards = sub.groupby("episode")["episode_reward"].agg(["mean", "std"])
        eps = ep_rewards.index.to_numpy()
        mu = ep_rewards["mean"].to_numpy()
        sd = ep_rewards["std"].fillna(0).to_numpy()
        mu_smooth = rolling_mean(mu.tolist(), window=window)
        ax.plot(eps, mu_smooth, label=agent, color=colour, linewidth=1.8)
        ax.fill_between(
            eps,
            np.array(mu_smooth) - sd,
            np.array(mu_smooth) + sd,
            alpha=0.12, color=colour,
        )

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode reward (rolling mean)", fontsize=12)
    ax.set_title("Learning curves — HIM-HER vs baselines", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------
# Figure 2 — HIM detection
# -----------------------------------------------------------------------

def fig_him_detection(df, out_path: str) -> None:
    """HIM trigger rate per episode + CDF of detection lag."""
    if not HAS_MPL or not HAS_PANDAS:
        print("[skip] fig_him_detection requires matplotlib + pandas.")
        return

    him_agents = ["him_her", "him_only"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ---- Left: trigger rate rolling mean ----
    ax_left = axes[0]
    for agent in him_agents:
        sub = df[df["agent_type"] == agent].copy()
        if sub.empty:
            continue
        colour = AGENT_COLOURS.get(agent, "black")
        rate = sub.groupby("episode")["him_triggered"].mean()
        smooth = rolling_mean(rate.tolist(), window=20)
        ax_left.plot(rate.index, smooth, label=agent, color=colour, linewidth=1.8)

    ax_left.set_xlabel("Episode", fontsize=11)
    ax_left.set_ylabel("HIM trigger rate", fontsize=11)
    ax_left.set_title("HIM trigger rate over training", fontsize=12)
    ax_left.legend(fontsize=9)
    ax_left.grid(alpha=0.3)

    # ---- Right: detection-lag CDF ----
    ax_right = axes[1]
    for agent in him_agents:
        sub = df[(df["agent_type"] == agent) & df["him_triggered"]].copy()
        if sub.empty or "detection_lag" not in sub.columns:
            continue
        lags = sub["detection_lag"].dropna().to_numpy()
        if len(lags) == 0:
            continue
        colour = AGENT_COLOURS.get(agent, "black")
        sorted_lags = np.sort(lags)
        cdf = np.arange(1, len(sorted_lags) + 1) / len(sorted_lags)
        ax_right.step(sorted_lags, cdf, label=agent, color=colour, linewidth=1.8)

    ax_right.set_xlabel("Detection lag (steps after switch)", fontsize=11)
    ax_right.set_ylabel("Cumulative fraction", fontsize=11)
    ax_right.set_title("Detection-lag CDF", fontsize=12)
    ax_right.legend(fontsize=9)
    ax_right.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------
# Figure 3 — Model accuracy
# -----------------------------------------------------------------------

def fig_model_accuracy(df, out_path: str, window: int = 50) -> None:
    """Model-correctness fraction vs episode, per agent type."""
    if not HAS_MPL or not HAS_PANDAS:
        print("[skip] fig_model_accuracy requires matplotlib + pandas.")
        return
    if "model_correct_fraction" not in df.columns:
        print("[skip] fig_model_accuracy: model_correct_fraction column missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for agent in AGENT_ORDER:
        sub = df[df["agent_type"] == agent].copy()
        if sub.empty:
            continue
        colour = AGENT_COLOURS.get(agent, "black")
        acc = sub.groupby("episode")["model_correct_fraction"].mean()
        smooth = rolling_mean(acc.tolist(), window=window)
        ax.plot(acc.index, smooth, label=agent, color=colour, linewidth=1.8)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Model-correctness fraction", fontsize=12)
    ax.set_title("Per-step model correctness over training", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.8, alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------
# Figure 4 — HER impact
# -----------------------------------------------------------------------

def fig_her_impact(df, out_path: str) -> None:
    """Scatter: HER fraction vs episode reward for HIM-HER + vanilla agents."""
    if not HAS_MPL or not HAS_PANDAS:
        print("[skip] fig_her_impact requires matplotlib + pandas.")
        return
    if "her_fraction_mean" not in df.columns:
        print("[skip] fig_her_impact: her_fraction_mean column missing.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    agents_to_plot = ["him_her", "vanilla"]
    for i, agent in enumerate(agents_to_plot):
        ax = axes[i]
        sub = df[df["agent_type"] == agent].copy()
        if sub.empty:
            continue
        colour = AGENT_COLOURS.get(agent, "grey")
        x = sub["her_fraction_mean"].to_numpy()
        y = sub["episode_reward"].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], alpha=0.3, s=8, color=colour)

        if mask.sum() >= 5:
            m, b = np.polyfit(x[mask], y[mask], 1)
            xline = np.linspace(x[mask].min(), x[mask].max(), 100)
            ax.plot(xline, m * xline + b, color="black", linewidth=1.5,
                    linestyle="--", label=f"slope={m:.3f}")
            ax.legend(fontsize=9)

        ax.set_xlabel("HER fraction", fontsize=11)
        ax.set_ylabel("Episode reward", fontsize=11)
        ax.set_title(f"HER impact — {agent}", fontsize=12)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse HIM-HER experiment results.")
    parser.add_argument(
        "--input", default=None,
        help="Path to a merged episodes CSV (from slurm/collect_results.sh). "
             "If omitted, all episodes.csv files under --log-root are loaded.",
    )
    parser.add_argument(
        "--log-root", default="logs",
        help="Root directory containing run subdirectories (default: logs/).",
    )
    parser.add_argument(
        "--figures-dir", default="figures",
        help="Output directory for figure PDFs (default: figures/).",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="Skip figure generation, only print statistics.",
    )
    parser.add_argument(
        "--last-n", type=int, default=100,
        help="Number of final episodes per seed to use for statistics (default: 100).",
    )
    args = parser.parse_args()

    # ---- load data ----
    if args.input:
        rows = _load_single_csv(args.input)
        print(f"Loaded {len(rows)} episode rows from {args.input}")
    else:
        rows = load_all_episodes(args.log_root)

    if not rows:
        print("No episode data found. Run training first.")
        sys.exit(0)

    df = rows_to_df(rows)

    # ---- statistics ----
    if HAS_PANDAS and hasattr(df, "groupby"):
        print_statistics_table(df, last_n=args.last_n)
    else:
        print(f"Loaded {len(rows)} episode rows (pandas unavailable for full stats).")

    # ---- figures ----
    if args.no_figures or not HAS_MPL:
        print("Skipping figure generation.")
        return

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating figures → {figures_dir}/")

    fig_learning_curves(df, str(figures_dir / "fig1_learning_curves.pdf"))
    fig_him_detection(df, str(figures_dir / "fig2_him_detection.pdf"))
    fig_model_accuracy(df, str(figures_dir / "fig3_model_accuracy.pdf"))
    fig_her_impact(df, str(figures_dir / "fig4_her_impact.pdf"))

    print("\nDone.")


if __name__ == "__main__":
    main()
