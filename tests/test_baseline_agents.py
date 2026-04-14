import importlib.util
from pathlib import Path

import numpy as np

from him_her.agents.baseline_agent import BayesianAgent, StaticModelAgent
from him_her.envs.predator_prey import PredatorPreyEnv
from him_her.models.base_model import ModelSet


class SimpleConfig:
    def __init__(self, values):
        for key, value in values.items():
            if isinstance(value, dict):
                setattr(self, key, SimpleConfig(value))
            else:
                setattr(self, key, value)


def make_config(seed: int = 0, concurrent_training: bool = True) -> SimpleConfig:
    return SimpleConfig(
        {
            'training': {
                'seed': seed,
                'buffer_capacity': 256,
                'batch_size': 512,
                'updates_per_episode': 1,
                'lr_actor': 3e-4,
                'lr_critic': 1e-3,
                'gamma': 0.95,
                'tau': 0.005,
                'concurrent_training': concurrent_training,
            },
            'agent': {
                'hidden_sizes': [32, 32],
            },
            'her': {
                'k': 2,
            },
        }
    )


def make_model_set() -> ModelSet:
    return ModelSet.from_config(
        SimpleConfig(
            {
                'models': [
                    {'name': 'evasive', 'policy_params': [1.0, 0.0, 8.0, 0.0, 0.0], 'prior': 0.5},
                    {'name': 'territorial', 'policy_params': [0.0, 1.0, 14.0, 8.0, 8.0], 'prior': 0.5},
                ]
            }
        )
    )


def test_static_agent_keeps_initial_model():
    env = PredatorPreyEnv(max_episode_length=5, seed=0)
    agent = StaticModelAgent(env, make_model_set(), make_config(seed=0))

    rewards = agent.train(num_episodes=2)

    assert len(rewards) == 2
    assert agent.train_state.current_model_id == 0
    assert np.all(agent.replay_buffer.model_ids[: len(agent.replay_buffer)] == 0)


def test_bayesian_agent_uses_map_model_for_next_episode():
    env = PredatorPreyEnv(max_episode_length=5, seed=1)
    agent = BayesianAgent(env, make_model_set(), make_config(seed=1))

    agent.belief_updater.log_belief = np.log(np.array([0.1, 0.9]))
    agent._post_episode_update([], 0)
    assert agent.train_state.current_model_id == 0

    transitions = [
        {
            'other_state': np.zeros(4, dtype=np.float32),
            'other_action': 1,
        }
    ]
    agent.belief_updater.update = lambda stacked_params, states, actions: np.array([0.1, 0.9])
    agent.belief_updater.map_model_id = lambda: 1

    agent._post_episode_update(transitions, 0)

    assert agent.train_state.current_model_id == 1
    np.testing.assert_allclose(np.exp(np.array(agent.train_state.log_belief)), np.array([0.1, 0.9]))


def test_train_script_agent_registry_supports_new_types():
    train_path = Path(__file__).resolve().parents[1] / 'scripts' / 'train.py'
    spec = importlib.util.spec_from_file_location('train_script', train_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    static_config = module.apply_overrides(module.create_config(), ['agent.type=static'])
    bayesian_config = module.apply_overrides(module.create_config(), ['agent.type=bayesian'])

    assert static_config['agent']['type'] == 'static'
    assert bayesian_config['agent']['type'] == 'bayesian'