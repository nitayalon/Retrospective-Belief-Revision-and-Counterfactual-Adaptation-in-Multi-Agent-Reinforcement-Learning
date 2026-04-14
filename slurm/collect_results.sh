#!/bin/bash
# Collect results from logs/ into a summary CSV.
# Usage:  bash slurm/collect_results.sh [output_file]
#
# Produces a single merged CSV of all episode records across all runs,
# plus a quick console summary by agent type.

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_FILE="${1:-$PROJECT_DIR/logs/all_results.csv}"

echo "Collecting episode CSVs from $PROJECT_DIR/logs/ ..."

HEADER_WRITTEN=0
> "$OUT_FILE"

find "$PROJECT_DIR/logs" -name "episodes.csv" | sort | while read -r f; do
    if [[ $HEADER_WRITTEN -eq 0 ]]; then
        # Write header from first file
        head -1 "$f" >> "$OUT_FILE"
        HEADER_WRITTEN=1
    fi
    # Append data rows (skip header)
    tail -n +2 "$f" >> "$OUT_FILE"
done

TOTAL=$(wc -l < "$OUT_FILE")
TOTAL=$((TOTAL - 1))
echo "Merged ${TOTAL} episode rows → $OUT_FILE"

echo ""
echo "=== Episode counts by agent_type ==="
if command -v python3 &>/dev/null; then
    python3 - "$OUT_FILE" <<'EOF'
import sys, csv
from collections import Counter, defaultdict
path = sys.argv[1]
counts = Counter()
seeds = defaultdict(set)
with open(path) as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        at = row.get('agent_type', '?')
        s = row.get('seed', '?')
        counts[at] += 1
        seeds[at].add(s)
for at in sorted(counts):
    print(f"  {at:<20} {counts[at]:5d} episodes  {len(seeds[at])} seeds: {sorted(seeds[at])}")
EOF
else
    echo "  (python3 not available; run: python3 scripts/analyze_results.py $OUT_FILE)"
fi
