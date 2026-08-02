#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXP_DIR="outputs/mnist_pca_muon_reproduction"
mkdir -p "$EXP_DIR"
cp data/fixed_problem_reproduction.npz "$EXP_DIR/fixed_problem.npz"
python run_experiment.py run \
  --config configs/reproduction.json \
  --groups development,confirmation
python run_experiment.py analyze \
  --config configs/reproduction.json \
  --groups development,confirmation
