"""Training entry point for smoke test (simplified without Hydra).

Usage:
    python scripts/train.py
    python scripts/train.py --config-name cooperative_nav agent.type=vanilla training.total_episodes=50
"""

from typing import Dict, Any
import os
import numpy as np
import sys

from him_her.envs.predator_prey import PredatorPreyEnv
from him_her.envs.cooperative_nav import CooperativeNavEnv
from him_her.envs.intersection import IntersectionEnv
from him_her.envs.hide_and_seek import HideAndSeekEnv
from him_her.models.base_model import ModelSet, AgentModel
from him_her.agents.baseline_agent import BayesianAgent, StaticModelAgent, VanillaAgent
from him_her.agents.him_her_agent import HIMHERAgent, HIMOnlyAgent
from him_her.utils.device import setup_device
from him_her.utils.trajectory_logger import TrajectoryLogger


def load_yaml_config(config_name: str) -> Dict[str, Any]:
    """Load a YAML config file from the configs/ directory."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for --config-name support: pip install pyyaml")
    # Locate configs/ relative to this script or the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    yaml_path = os.path.join(project_root, "configs", f"{config_name}.yaml")
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)
    # Remove Hydra defaults key if present
    raw.pop("defaults", None)
    return raw


def create_config() -> Dict[str, Any]:
    """Create hard-coded configuration for smoke test."""
    return {
        'env': {
            'name': 'PredatorPrey',
            'max_episode_length': 50,
        },
        'model_set': {
            'models': [
                {
                    'name': 'evasive',
                    'policy_params': [1.0, 0.0, 8.0, 0.0, 0.0],
                    'prior': 0.5,
                },
                {
                    'name': 'territorial',
                    'policy_params': [0.0, 1.0, 14.0, 8.0, 8.0],
                    'prior': 0.5,
                },
            ],
        },
        'her': {
            'strategy': 'future',
            'k': 4,
        },
        'him': {
            'ratio_delta': 2.0,
            'threshold': -2.0,
            'log_likelihoods': False,
            'window_fraction': 0.5,
            'warmup_episodes': 20,
        },
        'training': {
            'total_episodes': 50,
            'pretrain_episodes': 0,
            'pretrain_mode': 'fixed',
            'concurrent_training': False,
            'batch_size': 128,
            'buffer_capacity': 100000,
            'updates_per_episode': 5,
            'lr_actor': 0.0003,
            'lr_critic': 0.001,
            'gamma': 0.95,
            'tau': 0.005,
            'alpha': 0.2,
            'seed': 42,
            'pretrain': False,
        },
        'agent': {
            'type': 'vanilla',
            'hidden_sizes': [256, 256],
            'model_encoding_mode': 'onehot',
            'model_encoding': {
                'embed_dim': 64,
                'num_layers': 2,
                'use_layer_norm': True,
            },
        },
        'logging': {
            'use_wandb': False,
            'verbose': False,
            'save_trajectories': False,
            'save_steps': True,
        },
    }


def _parse_override_value(raw: str):
    lowered = raw.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    try:
        if raw.startswith('0') and raw != '0' and not raw.startswith('0.'):
            raise ValueError
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def apply_overrides(config: Dict[str, Any], args: list[str]) -> Dict[str, Any]:
    for arg in args:
        if '=' not in arg:
            continue
        key, raw_value = arg.split('=', 1)
        value = _parse_override_value(raw_value)
        cursor = config
        parts = key.split('.')
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return config


class SimpleConfig:
    """Simple namespace for configuration access."""
    def __init__(self, d: Dict[str, Any]):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, SimpleConfig(v))
            else:
                setattr(self, k, v)


def main():
    """Main training loop.
    
    Args:
        config: Configuration dictionary
    """
    # Parse --config-name <name> from argv (strip it before override processing)
    remaining_args = sys.argv[1:]
    config_name = None
    filtered_args = []
    i = 0
    while i < len(remaining_args):
        if remaining_args[i] == "--config-name" and i + 1 < len(remaining_args):
            config_name = remaining_args[i + 1]
            i += 2
        else:
            filtered_args.append(remaining_args[i])
            i += 1

    if config_name is not None:
        base_config = load_yaml_config(config_name)
    else:
        base_config = create_config()

    config_dict = apply_overrides(base_config, filtered_args)
    config = SimpleConfig(config_dict)

    # Configure JAX backend before any JAX operations.
    setup_device(config)

    print("=" * 60)
    print("HIM-HER Training (Smoke Test)")
    print("=" * 60)
    print(f"Agent type: {config.agent.type}")
    print(f"Environment: {config.env.name}")
    print(f"Total episodes: {config.training.total_episodes}")
    print(f"Seed: {config.training.seed}")
    print("=" * 60)
    
    env_registry = {
        'PredatorPrey': PredatorPreyEnv,
        'CooperativeNav': CooperativeNavEnv,
        'Intersection': IntersectionEnv,
        'HideAndSeek': HideAndSeekEnv,
    }
    env_name = getattr(config.env, 'name', 'PredatorPrey')
    if env_name not in env_registry:
        raise ValueError(f"Unknown env.name: {env_name}")
    # Create environment
    env = env_registry[env_name](max_episode_length=config.env.max_episode_length)
    
    # Create model set from config
    model_set = ModelSet.from_config(config.model_set)
    print(f"Loaded {len(model_set.models)} models: {[m.name for m in model_set.models]}")
    
    agent_registry = {
        'vanilla': VanillaAgent,
        'him_her': HIMHERAgent,
        'him_only': HIMOnlyAgent,
        'static': StaticModelAgent,
        'bayesian': BayesianAgent,
    }
    if config.agent.type not in agent_registry:
        raise ValueError(f"Unknown agent.type: {config.agent.type}")
    agent = agent_registry[config.agent.type](env, model_set, config)

    # Attach trajectory logger if requested.
    save_traj = getattr(getattr(config, 'logging', None), 'save_trajectories', False)
    if save_traj:
        env_name = getattr(config.env, 'name', 'PredatorPrey').lower()
        run_id = f"{config.agent.type}_seed{config.training.seed}"
        save_steps = getattr(getattr(config, 'logging', None), 'save_steps', True)
        agent.traj_logger = TrajectoryLogger(
            env_name=env_name,
            run_id=run_id,
            save_steps=save_steps,
        )
        print(f"Trajectory logging \u2192 logs/{env_name}/{run_id}/")

    # Run training
    print("Starting training...")
    episode_rewards = agent.train(num_episodes=config.training.total_episodes)

    positive_reward_episodes = sum(reward > 0.0 for reward in episode_rewards)
    print(f"Positive-reward episodes: {positive_reward_episodes}/{len(episode_rewards)}")
    print(f"Max episode reward: {max(episode_rewards, default=0.0):.3f}")

    if positive_reward_episodes == 0:
        raise RuntimeError("Smoke test failed: no non-zero rewards observed.")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
