#!/usr/bin/env bash
set -euo pipefail

python run_robust_family.py \
  --problem data/fixed_problem_reproduction.npz \
  --output-root outputs/robust_family_300 \
  --seed-group confirmation \
  --steps 300 \
  --window-start 40 \
  --window-end 200 \
  --log-every 10 \
  --threads 1
