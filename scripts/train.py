"""Training entry point for smoke test (simplified without Hydra).

Usage:
    python scripts/train.py
"""

from typing import Dict, Any
import numpy as np

from him_her.envs.predator_prey import PredatorPreyEnv
from him_her.models.base_model import ModelSet, AgentModel
from him_her.agents.baseline_agent import VanillaAgent


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
                    'policy_params': [0.1, 1.0, 1.0, 1.0, 1.0],
                    'prior': 0.5,
                },
                {
                    'name': 'territorial',
                    'policy_params': [0.5, 0.8, 0.8, 0.8, 0.8],
                    'prior': 0.5,
                },
            ],
        },
        'her': {
            'strategy': 'future',
            'k': 4,
        },
        'training': {
            'total_episodes': 50,
            'batch_size': 128,
            'buffer_capacity': 100000,
            'updates_per_episode': 40,
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
            'model_encoding': {
                'embed_dim': 64,
                'num_layers': 2,
                'use_layer_norm': True,
            },
        },
    }


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
    config_dict = create_config()
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
    agent = VanillaAgent(env, model_set, config)
    
    # Run training
    print("Starting training...")
    agent.train(num_episodes=config.training.total_episodes)
    
    print("Training complete!")


if __name__ == "__main__":
    main()
