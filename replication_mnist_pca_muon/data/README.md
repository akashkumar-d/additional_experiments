# Included data files

This bundle does **not** require the original 60--260 MB `mnist_muon.zip`.

- `mnist_train_uint8.npz` (about 9.4 MB) contains only the 60,000 MNIST
  training images and labels needed to refit PCA and rebuild the controlled
  teacher task. Keys: `images` with shape `(60000, 784)` and dtype `uint8`,
  and `labels` with shape `(60000,)` and dtype `uint8`.
- `fixed_problem_reproduction.npz` (about 2.1 MB) contains the exact frozen
  PCA covariates, teacher basis, and teacher targets used in the reported
  experiment. Use it for the most exact reproduction.

The original archive also contained old run trajectories, plots, Python files,
and test arrays. None of those are needed by this experiment.
