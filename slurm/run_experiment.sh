#!/bin/bash
# SLURM job script — single seed of one experiment configuration.
# Submitted indirectly via submit_single.sh / submit_all.sh.
#
# Required environment variables (exported by the submitter):
#   AGENT_TYPE   — one of: him_her, him_only, vanilla, bayesian, static
#   SEED         — integer seed (0-9 for full job arrays, or arbitrary)
#   TOTAL_EPS    — total number of training episodes
#   ENV_NAME     — config name matching configs/{ENV_NAME}.yaml (optional)
#
# SBATCH directives are minimal — tune for your cluster.

#SBATCH --job-name=himher_${AGENT_TYPE}_s${SEED}
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1              # Request 1 GPU; remove if CPU-only cluster

# ---- environment setup ----
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [[ -f "$PROJECT_DIR/venv/bin/activate" ]]; then
    source "$PROJECT_DIR/venv/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    : # Already in a venv
else
    echo "[warn] No venv found; using system Python."
fi

echo "============================================"
echo "HOST         : $(hostname)"
echo "AGENT_TYPE   : ${AGENT_TYPE}"
echo "SEED         : ${SEED}"
echo "TOTAL_EPS    : ${TOTAL_EPS:-500}"
echo "ENV_NAME     : ${ENV_NAME:-PredatorPrey}"
echo "SLURM_JOB_ID : ${SLURM_JOB_ID:-local}"
echo "============================================"

# ---- build override args ----
OVERRIDES=(
    "agent.type=${AGENT_TYPE}"
    "training.seed=${SEED}"
    "training.total_episodes=${TOTAL_EPS:-500}"
    "logging.save_trajectories=true"
    "logging.save_steps=true"
)

if [[ -n "${ENV_NAME:-}" && "${ENV_NAME}" != "PredatorPrey" ]]; then
    OVERRIDES+=("--config-name" "${ENV_NAME,,}")  # lowercase config name
fi

# ---- run ----
python scripts/train.py "${OVERRIDES[@]}" 2>&1

echo "Done: agent=${AGENT_TYPE} seed=${SEED}"
