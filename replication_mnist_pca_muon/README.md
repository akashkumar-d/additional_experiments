# Parametric PCA-MNIST exact-polar Muon bundle

This bundle reproduces the fixed-learning-rate, one-hidden-layer teacher-student experiment on an empirical non-Gaussian MNIST input distribution.

## Editable parameters

Edit `configs/reproduction.json` or the first parameter cell of the notebook. Main parameters include:

- `data.n_train`
- `data.pca_dim`
- `data.whiten`
- `data.problem_seed`
- `data.allowed_digits`
- `teacher.rank`, `teacher.seed`, `teacher.head_value`
- `student.width`, `student.head_value`
- `student.learning_rate`
- `student.steps`, `student.log_every`, `student.threads`
- development and confirmation seed lists
- primary and secondary analysis windows

## Setup

```bash
python -m pip install -r requirements.txt
```

Put `mnist_muon.zip` at `data/mnist_muon.zip`, or change `data.mnist_source`.

## Smoke test

```bash
python run_experiment.py all --config configs/smoke.json --groups development --force
```

## Exact reproduction

```bash
python run_experiment.py all \
  --config configs/reproduction.json \
  --groups development,confirmation
```

The exact preset uses 5,000 images, PCA dimension 100, teacher rank 4, student width 50, fixed learning rate 0.60, 1,500 steps, five development seeds, and 30 confirmation seeds.

## Exact frozen-problem reproduction

To bypass PCA-version differences and reuse the exact frozen problem from the reported run:

```bash
bash run_exact_frozen_problem.sh
```

This copies `data/fixed_problem_reproduction.npz` to the expected output location before training.

## Learning-rate sweep

```bash
python sweep_eta.py \
  --config configs/reproduction.json \
  --etas 0.4,0.5,0.6,0.7,0.8 \
  --group development
```

## Notebook

Open `notebooks/mnist_pca_muon_experiment.ipynb`. It defaults to a quick smoke run. Set `PRESET="reproduction"` and `RUN_CONFIRMATION=True` for the full experiment.

## Numerical reproducibility

The exact polar factor uses normalized reduced QR followed by LAPACK `gesvd`. Keep `student.threads=1` for the closest reproducibility near rank deficiency.
