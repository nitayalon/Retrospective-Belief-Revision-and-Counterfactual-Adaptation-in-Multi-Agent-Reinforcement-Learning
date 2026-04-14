#!/bin/bash -l
#SBATCH -o ./slurm_logs/himher_%x_%A_%a.out
#SBATCH -e ./slurm_logs/himher_%x_%A_%a.err
#SBATCH -D ./
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nitay.alon@tuebingen.mpg.de
#SBATCH --time=3-00:00:00
#SBATCH --job-name=himher

# ============================================================================
# HIM-HER EXPERIMENT — single (agent_type, seed) run
# ============================================================================
# Required environment variables (exported by the submitter):
#   AGENT_TYPE   — one of: him_her, him_only, vanilla, bayesian, static
#   TOTAL_EPS    — total number of training episodes (default: 500)
#   ENV_NAME     — config name matching configs/{ENV_NAME}.yaml (optional)
#
# SEED is taken from SLURM_ARRAY_TASK_ID when submitted as an array job,
# or from the SEED env var when submitted via submit_single.sh.
# ============================================================================

module purge
module load singularity

export SINGULARITY_BIND="/run,/ptmp,/scratch,/tmp,/opt/ohpc,${HOME}"
export CONTAINER_PATH=/ptmp/containers/pytorch_1.10.0-cuda.11.3_latest-2021-12-02-ec95d31ea677.sif
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

mkdir -p slurm_logs
mkdir -p logs

# Resolve seed: array task ID takes precedence, then explicit SEED var
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"

START_TIME=$(date +"%Y-%m-%d %H:%M:%S")

echo "============================================================================"
echo "HIM-HER EXPERIMENT - Task ${SLURM_ARRAY_TASK_ID:-single}"
echo "============================================================================"
echo "Job ID:       ${SLURM_JOB_ID:-local}"
echo "Array Task:   ${SLURM_ARRAY_TASK_ID:-n/a}"
echo "Agent type:   ${AGENT_TYPE}"
echo "Seed:         ${SEED}"
echo "Total eps:    ${TOTAL_EPS:-500}"
echo "Env:          ${ENV_NAME:-PredatorPrey}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "Start Time:   ${START_TIME}"
echo "Container:    ${CONTAINER_PATH}"
echo "============================================================================"

# ---- build Hydra override args ----
OVERRIDES=(
    "agent.type=${AGENT_TYPE}"
    "training.seed=${SEED}"
    "training.total_episodes=${TOTAL_EPS:-500}"
    "logging.save_trajectories=true"
    "logging.save_steps=true"
)

if [[ -n "${ENV_NAME:-}" && "${ENV_NAME}" != "PredatorPrey" ]]; then
    OVERRIDES+=("--config-name" "${ENV_NAME,,}")
fi

# ---- run inside container (--nv passes through GPU) ----
singularity exec --nv ${CONTAINER_PATH} \
    python scripts/train.py "${OVERRIDES[@]}"

EXIT_CODE=$?

echo "Task ${SLURM_ARRAY_TASK_ID:-single} completed with exit code: ${EXIT_CODE}"

# Verify episodes CSV was created
RUN_ID="${AGENT_TYPE}_seed${SEED}"
ENV_LOWER=$(echo "${ENV_NAME:-predatorprey}" | tr '[:upper:]' '[:lower:]' | tr -d '_')
EPISODES_CSV="logs/${ENV_LOWER}/${RUN_ID}/episodes.csv"
if [[ -f "${EPISODES_CSV}" ]]; then
    N=$(($(wc -l < "${EPISODES_CSV}") - 1))
    echo "✓ episodes.csv verified: ${N} episodes logged"
else
    echo "✗ WARNING: ${EPISODES_CSV} not found!"
    exit 1
fi

exit ${EXIT_CODE}
