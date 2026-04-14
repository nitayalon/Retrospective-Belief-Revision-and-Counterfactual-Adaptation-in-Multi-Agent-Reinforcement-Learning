#!/bin/bash
# Submit a 10-seed job array for ALL agent types.
# Usage:  bash slurm/submit_all.sh [total_episodes]
#
# Creates one SLURM array job per agent type (5 types × 10 seeds = 50 jobs total).
# SEED = SLURM_ARRAY_TASK_ID (0-9) inside run_experiment.sh.

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOTAL_EPS="${1:-500}"
SEEDS="0-9"  # array indices 0-9 become seeds 0-9

AGENT_TYPES=("him_her" "him_only" "vanilla" "bayesian" "static")

mkdir -p "$PROJECT_DIR/slurm_logs"

echo "Submitting ${#AGENT_TYPES[@]} agent types × 10 seeds, ${TOTAL_EPS} episodes each..."
echo "Logs → $PROJECT_DIR/slurm_logs/"
echo ""

JOB_IDS=()
for AGENT in "${AGENT_TYPES[@]}"; do
    JOB_ID=$(
        sbatch \
            --array="${SEEDS}%10" \
            --job-name="himher_${AGENT}" \
            --export="ALL,AGENT_TYPE=${AGENT},TOTAL_EPS=${TOTAL_EPS}" \
            "$PROJECT_DIR/slurm/run_experiment.sh" \
        | awk '{print $NF}'
    )
    echo "  Submitted ${AGENT}: job ${JOB_ID}"
    JOB_IDS+=("$JOB_ID")
done

echo ""
echo "All jobs submitted: ${JOB_IDS[*]}"
echo "Monitor with:  bash slurm/check_status.sh"
echo "Collect with:  bash slurm/collect_results.sh"
