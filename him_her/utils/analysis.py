"""Helpers for loading and comparing per-episode training logs."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_run(agent_type: str, seed: int) -> pd.DataFrame:
    """Load a training log CSV into a DataFrame."""
    path = Path('logs') / f'{agent_type}_seed{seed}.csv'
    return pd.read_csv(path)


def belief_accuracy(df: pd.DataFrame) -> float:
    """Fraction of episodes where the logged model matches the true end policy."""
    if 'model_correct' in df.columns:
        return float(df['model_correct'].mean())

    if 'map_model_name' in df.columns:
        return float((df['map_model_name'] == df['true_end_policy']).mean())

    if 'current_model_name' in df.columns:
        return float((df['current_model_name'] == df['true_end_policy']).mean())

    raise KeyError('DataFrame is missing model correctness columns.')


def detection_lag_distribution(df: pd.DataFrame) -> np.ndarray:
    """For HIM+HER: array of logged detection lags per trigger."""
    if 'him_triggered' not in df.columns:
        raise KeyError('DataFrame does not contain him_triggered.')

    triggered = df[df['him_triggered'] == 1]
    if triggered.empty:
        return np.array([], dtype=np.int32)

    return (triggered['episode'].to_numpy(dtype=np.int32) - triggered['switch_point'].to_numpy(dtype=np.int32))


def belief_collapse_episodes(df: pd.DataFrame, threshold: float = 0.95) -> List[int]:
    """For Bayesian: episodes where belief exceeds the collapse threshold."""
    belief_cols = [col for col in df.columns if col.startswith('belief_')]
    if not belief_cols:
        raise KeyError('DataFrame does not contain belief_* columns.')

    collapsed = df[belief_cols].max(axis=1) > threshold
    return df.loc[collapsed, 'episode'].astype(int).tolist()


def compare_agents(him_df: pd.DataFrame, bayes_df: pd.DataFrame) -> dict:
    """Side-by-side comparison of belief accuracy, detection timing, and reward."""
    bayes_switches = bayes_df.loc[bayes_df.get('switched', 0) == 1, 'episode'].astype(int).tolist()
    him_triggers = him_df.loc[him_df.get('him_triggered', 0) == 1, 'episode'].astype(int).tolist()

    return {
        'him_model_accuracy': belief_accuracy(him_df),
        'bayesian_model_accuracy': belief_accuracy(bayes_df),
        'him_mean_reward': float(him_df['episode_reward'].mean()),
        'bayesian_mean_reward': float(bayes_df['episode_reward'].mean()),
        'him_trigger_episodes': him_triggers,
        'bayesian_switch_episodes': bayes_switches,
        'shared_detection_episodes': sorted(set(him_triggers).intersection(bayes_switches)),
        'him_detection_lag': detection_lag_distribution(him_df).tolist(),
        'bayesian_collapse_episodes': belief_collapse_episodes(bayes_df),
    }