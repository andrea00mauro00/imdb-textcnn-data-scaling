#!/bin/bash
# Run the full experiment (training + LIME + deletion test) for the two new
# seeds, then aggregate all three seeds (42 is reused from outputs/).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Multi-seed experiment started: $(date) ==="

for SEED in 123 2024; do
  OUT="outputs_seed${SEED}"
  echo ""
  echo "=== Seed ${SEED}: training + LIME -> ${OUT} (start: $(date)) ==="
  python3 run_experiment.py --seed "${SEED}" --output "${OUT}"
  echo ""
  echo "=== Seed ${SEED}: deletion test (start: $(date)) ==="
  python3 run_deletion_test.py --output "${OUT}"
  echo "=== Seed ${SEED} finished: $(date) ==="
done

echo ""
echo "=== Aggregating the 3 seeds (42, 123, 2024) ==="
python3 aggregate_seeds.py

echo ""
echo "=== All done: $(date) ==="
