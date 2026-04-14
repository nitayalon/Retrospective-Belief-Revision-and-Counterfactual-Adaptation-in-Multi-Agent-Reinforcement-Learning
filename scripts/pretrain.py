"""Pretrain one ego policy per scripted other-agent model."""

from pathlib import Path
import shutil
import sys

import orbax.checkpoint as ocp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from him_her.agents.baseline_agent import VanillaAgent
from him_her.envs.predator_prey import PredatorPreyEnv
from him_her.models.base_model import ModelSet
from scripts.train import SimpleConfig, apply_overrides, create_config


def main() -> None:
    config_dict = apply_overrides(create_config(), sys.argv[1:])
    config = SimpleConfig(config_dict)
    model_set = ModelSet.from_config(config.model_set)
    checkpoints_root = (PROJECT_ROOT / "checkpoints").resolve()
    checkpointer = ocp.PyTreeCheckpointer()

    print("="    * 60)
    print("HIM-HER Policy Pretraining")
    print("=" * 60)
    print(f"Pretrain episodes: {config.training.pretrain_episodes}")
    print(f"Pretrain mode: {config.training.pretrain_mode}")
    print("=" * 60)

    for model in model_set.models:
        env = PredatorPreyEnv(
            max_episode_length=config.env.max_episode_length,
            seed=config.training.seed + model.model_id,
            fixed_policy_name=model.name,
        )
        agent = VanillaAgent(env, model_set, config)
        agent.train_state = agent.train_state.replace(current_model_id=model.model_id)

        print(f"Pretraining policy for model {model.model_id}: {model.name}")
        rewards = agent.train(config.training.pretrain_episodes)

        ckpt_dir = checkpoints_root / f"policy_m{model.model_id}"
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
        checkpointer.save(ckpt_dir, agent.train_state.model_policies[model.model_id])

        print(
            f"Saved {ckpt_dir} | max_reward={max(rewards, default=0.0):.3f} | "
            f"positive_episodes={sum(r > 0.0 for r in rewards)}/{len(rewards)}"
        )


if __name__ == "__main__":
    main()