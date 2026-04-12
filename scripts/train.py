"""Training entry point for smoke test (simplified without Hydra).

Usage:
    python scripts/train.py
"""

from typing import Dict, Any
import numpy as np
import sys

from him_her.envs.predator_prey import PredatorPreyEnv
from him_her.models.base_model import ModelSet, AgentModel
from him_her.agents.baseline_agent import VanillaAgent
from him_her.agents.him_her_agent import HIMHERAgent


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
            'window_fraction': 0.2,
        },
        'training': {
            'total_episodes': 50,
            'pretrain_episodes': 200,
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
    config_dict = apply_overrides(create_config(), sys.argv[1:])
    config = SimpleConfig(config_dict)
    
    print("=" * 60)
    print("HIM-HER Training (Smoke Test)")
    print("=" * 60)
    print(f"Agent type: {config.agent.type}")
    print(f"Environment: {config.env.name}")
    print(f"Total episodes: {config.training.total_episodes}")
    print(f"Seed: {config.training.seed}")
    print("=" * 60)
    
    # Create environment
    env = PredatorPreyEnv(max_episode_length=config.env.max_episode_length)
    
    # Create model set from config
    model_set = ModelSet.from_config(config.model_set)
    print(f"Loaded {len(model_set.models)} models: {[m.name for m in model_set.models]}")
    
    # Instantiate agent
    if config.agent.type == 'him_her':
        agent = HIMHERAgent(env, model_set, config)
    else:
        agent = VanillaAgent(env, model_set, config)
    
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
