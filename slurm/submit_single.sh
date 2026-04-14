#!/bin/bash
# Submit a single (agent_type, seed) job.
# Usage:  bash slurm/submit_single.sh <agent_type> <seed> [total_episodes]
#
# Examples:
#   bash slurm/submit_single.sh him_her 42 500
#   bash slurm/submit_single.sh vanilla 0

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AGENT_TYPE="${1:?Usage: submit_single.sh <agent_type> <seed> [total_episodes]}"
SEED="${2:?Missing seed argument}"
TOTAL_EPS="${3:-500}"

mkdir -p "$PROJECT_DIR/slurm_logs"

echo "Submitting: agent=${AGENT_TYPE} seed=${SEED} episodes=${TOTAL_EPS}"

JOB_ID=$(
    sbatch \
        --job-name="himher_${AGENT_TYPE}_s${SEED}" \
        --export="ALL,AGENT_TYPE=${AGENT_TYPE},SEED=${SEED},TOTAL_EPS=${TOTAL_EPS}" \
        "$PROJECT_DIR/slurm/run_experiment.sh" \
    | awk '{print $NF}'
)

echo "Job submitted: ${JOB_ID}"
echo "Output: slurm_logs/himher_himher_${AGENT_TYPE}_s${SEED}_${JOB_ID}_*.out"
