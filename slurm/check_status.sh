#!/bin/bash
# Show status of all HIM-HER SLURM jobs.
# Usage:  bash slurm/check_status.sh

set -euo pipefail

echo "=== Active HIM-HER SLURM jobs ==="
squeue --user="$USER" --format="%-18i %-15j %-8T %-10M %-6D %R" | grep -E "(him_her|JOBID)" || true
echo ""

echo "=== Completed logs (last 20) ==="
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs/slurm"
if [[ -d "$LOG_DIR" ]]; then
    ls -lt "$LOG_DIR"/*.out 2>/dev/null | head -20 || echo "  (no log files yet)"
else
    echo "  (no logs/slurm/ directory yet)"
fi

echo ""
echo "=== Episode CSV counts per run ==="
DATA_DIR="$PROJECT_DIR/logs"
if [[ -d "$DATA_DIR" ]]; then
    find "$DATA_DIR" -name "episodes.csv" | while read -r f; do
        lines=$(wc -l < "$f")
        # subtract 1 for header
        eps=$((lines - 1))
        run=$(echo "$f" | sed "s|$DATA_DIR/||")
        printf "  %-60s %5d episodes\n" "$run" "$eps"
    done
else
    echo "  (no data directory yet)"
fi
