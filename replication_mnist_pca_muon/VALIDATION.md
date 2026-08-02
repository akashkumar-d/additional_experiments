# Validation performed

- The smoke preset was executed end to end: problem construction, two student seeds, two-phase analysis, CSV output, and plots.
- `notebooks/mnist_pca_muon_experiment_executed_smoke.ipynb` was executed successfully from a clean kernel.
- With `data/fixed_problem_reproduction.npz`, seed 61 at the reproduction preset matches the previously reported raw trajectory exactly: maximum absolute differences are zero for dense loss, two-step residual, alignment logs, final weights, and saved checkpoints.
- Rebuilding the PCA problem from the MNIST archive is deterministic at the selected seeds, but small PCA-coordinate differences can occur across scikit-learn/LAPACK versions. Use `run_exact_frozen_problem.sh` when exact numerical matching to the reported frozen problem is important.
