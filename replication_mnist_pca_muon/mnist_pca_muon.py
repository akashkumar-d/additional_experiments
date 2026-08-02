from __future__ import annotations

import copy
import io
import json
import math
import time
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


DEFAULT_CONFIRMATION_SEEDS = [
    61, 67, 71, 79, 83, 89, 97, 101, 103, 107,
    109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
    167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
]

DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "mnist_source": "data/mnist_train_uint8.npz",
        "archive_member": "mnist_muon/mnist.npz",  # legacy zip support
        "images_key": "images",
        "labels_key": "labels",
        "n_train": 5000,
        "pca_dim": 100,
        "whiten": True,
        "problem_seed": 2026,
        "allowed_digits": None,
        "pixel_scale": 255.0,
    },
    "teacher": {
        "rank": 4,
        "seed": 5,
        "head_value": 1.0,
    },
    "student": {
        "width": 50,
        "head_value": None,
        "learning_rate": 0.60,
        "steps": 1500,
        "log_every": 10,
        "log_both_phases": True,
        "threads": 1,
        "checkpoint_steps": [0, 60, 240, 400, 700, 1000, 1200],
    },
    "seeds": {
        "development": [5, 11, 23, 37, 41],
        "confirmation": DEFAULT_CONFIRMATION_SEEDS,
    },
    "analysis": {
        "primary_window": [60, 240],
        "windows": {
            "60_240": [60, 240],
            "240_400": [240, 400],
            "400_700": [400, 700],
            "700_1000": [700, 1000],
            "1000_1200": [1000, 1200],
            "1200_1490": [1200, 1490],
            "60_400": [60, 400],
            "60_700": [60, 700],
            "60_1000": [60, 1000],
            "60_1490": [60, 1490],
        },
        "bootstrap_repetitions": 20000,
        "plot_xmax": 1490,
    },
    "output": {
        "experiment_name": "mnist_pca_muon_reproduction",
        "output_root": "outputs",
        "problem_filename": "fixed_problem.npz",
    },
}


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    config = deep_update(DEFAULT_CONFIG, raw)
    validate_config(config)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")


def validate_config(config: dict[str, Any]) -> None:
    data, teacher, student, analysis = (
        config["data"], config["teacher"], config["student"], config["analysis"]
    )
    if data["n_train"] < 2:
        raise ValueError("data.n_train must be at least 2")
    if data["pca_dim"] < 1:
        raise ValueError("data.pca_dim must be positive")
    if teacher["rank"] < 1 or teacher["rank"] > data["pca_dim"]:
        raise ValueError("teacher.rank must lie in [1, pca_dim]")
    if student["width"] < teacher["rank"]:
        raise ValueError("student.width must be at least teacher.rank")
    if student["learning_rate"] <= 0:
        raise ValueError("student.learning_rate must be positive")
    if student["steps"] < 3:
        raise ValueError("student.steps must be at least 3")
    if student["log_every"] < 1:
        raise ValueError("student.log_every must be positive")
    start, end = analysis["primary_window"]
    if not (0 <= start < end and end + 1 < student["steps"]):
        raise ValueError("primary_window must satisfy 0 <= start < end and end+1 < steps")
    if student["log_both_phases"]:
        for endpoint in (start, end):
            if endpoint % student["log_every"] != 0:
                raise ValueError(
                    "For two-phase analysis, primary-window endpoints must be multiples of log_every"
                )


def experiment_dir(config: dict[str, Any]) -> Path:
    return Path(config["output"]["output_root"]) / config["output"]["experiment_name"]


def _load_mnist(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data_cfg = config["data"]
    source = Path(data_cfg["mnist_source"])
    if not source.exists():
        raise FileNotFoundError(
            f"MNIST source not found: {source}. The lightweight bundle includes data/mnist_train_uint8.npz; edit config.data.mnist_source if it was moved."
        )
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            payload = archive.read(data_cfg["archive_member"])
        saved = np.load(io.BytesIO(payload))
    else:
        saved = np.load(source)
    images = saved[data_cfg["images_key"]].astype(np.float32)
    labels = saved[data_cfg["labels_key"]].astype(np.int64).reshape(-1)
    if images.ndim > 2:
        images = images.reshape(len(images), -1)
    images = images / float(data_cfg["pixel_scale"])
    allowed = data_cfg.get("allowed_digits")
    if allowed is not None:
        mask = np.isin(labels, np.asarray(allowed, dtype=np.int64))
        images, labels = images[mask], labels[mask]
    return images, labels


def prepare_problem(
    config: dict[str, Any],
    *,
    force: bool = False,
) -> Path:
    validate_config(config)
    out_dir = experiment_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem_path = out_dir / config["output"]["problem_filename"]
    if problem_path.exists() and not force:
        return problem_path

    images, labels = _load_mnist(config)
    data_cfg, teacher_cfg = config["data"], config["teacher"]
    n_train = int(data_cfg["n_train"])
    if n_train > len(images):
        raise ValueError(f"n_train={n_train} exceeds available sample count {len(images)}")
    if data_cfg["pca_dim"] > min(n_train, images.shape[1]):
        raise ValueError("pca_dim exceeds min(n_train, raw input dimension)")

    rng = np.random.default_rng(int(data_cfg["problem_seed"]))
    indices = rng.choice(len(images), size=n_train, replace=False)
    selected_images, selected_labels = images[indices], labels[indices]

    pca = PCA(
        n_components=int(data_cfg["pca_dim"]),
        whiten=bool(data_cfg["whiten"]),
        svd_solver="randomized",
        random_state=int(data_cfg["problem_seed"]),
        iterated_power=3,
    )
    X = pca.fit_transform(selected_images).astype(np.float32)

    teacher_rng = np.random.default_rng(int(teacher_cfg["seed"]))
    raw_basis = teacher_rng.standard_normal((X.shape[1], int(teacher_cfg["rank"])))
    teacher_basis, _ = np.linalg.qr(raw_basis)
    teacher_basis = teacher_basis.astype(np.float32)
    teacher_head = np.full(
        int(teacher_cfg["rank"]), float(teacher_cfg["head_value"]), dtype=np.float32
    )
    y = (np.maximum(X @ teacher_basis, 0.0) @ teacher_head).astype(np.float32)

    np.savez_compressed(
        problem_path,
        X=X,
        y=y,
        U_teacher=teacher_basis,
        teacher_head=teacher_head,
        source_indices=indices,
        source_labels=selected_labels,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        pca_explained_variance=pca.explained_variance_.astype(np.float32),
        config_json=json.dumps(config),
    )
    fourth = np.mean(X.astype(np.float64) ** 4, axis=0)
    pd.DataFrame([{
        "n_train": len(X),
        "raw_dimension": selected_images.shape[1],
        "pca_dimension": X.shape[1],
        "whiten": bool(data_cfg["whiten"]),
        "teacher_rank": int(teacher_cfg["rank"]),
        "mean_coordinate_variance": float(np.mean(np.var(X, axis=0))),
        "mean_coordinate_fourth_moment": float(np.mean(fourth)),
        "median_coordinate_fourth_moment": float(np.median(fourth)),
        "minimum_coordinate_fourth_moment": float(np.min(fourth)),
        "maximum_coordinate_fourth_moment": float(np.max(fourth)),
        "gaussian_fourth_moment": 3.0,
    }]).to_csv(out_dir / "problem_summary.csv", index=False)
    save_config(config, out_dir / "resolved_config.json")
    return problem_path


def load_problem(path: str | Path) -> dict[str, np.ndarray]:
    saved = np.load(Path(path), allow_pickle=False)
    return {name: saved[name] for name in saved.files}


def stable_polar(gradient: np.ndarray) -> np.ndarray:
    matrix = np.asarray(gradient, dtype=np.float32)
    scale = float(np.linalg.norm(matrix))
    if not np.isfinite(scale) or scale <= 1e-30:
        return np.zeros_like(matrix)
    q, reduced = np.linalg.qr(matrix / scale, mode="reduced")
    u, _, vh = scipy.linalg.svd(
        reduced, full_matrices=False, lapack_driver="gesvd", check_finite=False
    )
    return np.ascontiguousarray((q @ (u @ vh)).astype(np.float32))


def initialize_student(config: dict[str, Any], seed: int, input_dimension: int):
    student_cfg = config["student"]
    width = int(student_cfg["width"])
    rng = np.random.default_rng(seed + 17)
    weights = (
        rng.standard_normal((input_dimension, width)) / math.sqrt(input_dimension)
    ).astype(np.float32)
    head_value = student_cfg.get("head_value")
    if head_value is None:
        head_value = 1.0 / math.sqrt(input_dimension)
    head = np.full(width, float(head_value), dtype=np.float32)
    return weights, head


def loss_and_gradient(X, y, weights, head):
    preactivation = X @ weights
    hidden = np.maximum(preactivation, 0.0)
    error = hidden @ head - y
    loss = float(np.mean(error * error))
    gradient = (2.0 / len(X)) * (
        X.T @ (error[:, None] * (preactivation > 0) * head[None, :])
    )
    return loss, gradient.astype(np.float32, copy=False)


def alignment_diagnostics(X, weights, head, teacher_basis):
    preactivation = X @ weights
    derivative_features = (preactivation > 0).astype(np.float32) * head[None, :]
    coefficient = derivative_features.T @ derivative_features / len(X)
    agop = weights @ coefficient @ weights.T
    agop = 0.5 * (agop + agop.T)
    eigenvalues, eigenvectors = np.linalg.eigh(agop)
    rank = teacher_basis.shape[1]
    top_student = eigenvectors[:, np.argsort(eigenvalues)[::-1][:rank]]
    singular = np.linalg.svd(top_student.T @ teacher_basis, compute_uv=False)
    cos2 = np.clip(singular, 0.0, 1.0) ** 2
    return float(cos2.mean()), float(cos2.min())


def run_seed(config, seed, problem_path, raw_dir, force=False):
    validate_config(config)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    steps = int(config["student"]["steps"])
    output_path = raw_dir / f"seed{seed}_T{steps}_biphase.npz"
    if output_path.exists() and not force:
        return output_path

    problem = load_problem(problem_path)
    X = problem["X"].astype(np.float32)
    y = problem["y"].astype(np.float32)
    teacher_basis = problem["U_teacher"].astype(np.float32)
    weights, head = initialize_student(config, seed, X.shape[1])
    student_cfg = config["student"]
    eta = float(student_cfg["learning_rate"])
    log_every = int(student_cfg["log_every"])
    phase_offsets = {0, 1} if student_cfg["log_both_phases"] else {0}
    checkpoint_set = {
        int(s) for s in student_cfg.get("checkpoint_steps", []) if 0 <= int(s) < steps
    }
    checkpoint_set.add(steps - 1)

    dense, r2_trace, logs = [], [], []
    checkpoint_steps, checkpoint_weights = [], []
    previous_update = None
    started = time.time()
    with threadpool_limits(limits=int(student_cfg["threads"])):
        for step in range(steps):
            loss, gradient = loss_and_gradient(X, y, weights, head)
            update = -eta * stable_polar(gradient)
            if previous_update is None:
                residual = float("nan")
            else:
                residual = float(
                    np.linalg.norm(update + previous_update)
                    / (0.5 * (np.linalg.norm(update) + np.linalg.norm(previous_update)) + 1e-30)
                )
            dense.append(loss)
            r2_trace.append(residual)
            if step % log_every in phase_offsets:
                mean_cos2, min_cos2 = alignment_diagnostics(
                    X, weights, head, teacher_basis
                )
                logs.append((step, loss, mean_cos2, min_cos2, residual))
            if step in checkpoint_set:
                checkpoint_steps.append(step)
                checkpoint_weights.append(weights.copy())
            weights += update
            previous_update = update

    np.savez_compressed(
        output_path,
        dense=np.asarray(dense),
        r2=np.asarray(r2_trace),
        logs=np.asarray(logs),
        final_weights=weights,
        head=head,
        checkpoint_steps=np.asarray(checkpoint_steps, dtype=np.int64),
        checkpoint_weights=np.stack(checkpoint_weights),
        runtime=time.time() - started,
        seed=seed,
        config_json=json.dumps(config),
    )
    return output_path


def run_experiment(
    config: dict[str, Any],
    *,
    groups=("development", "confirmation"),
    rebuild_problem=False,
    force_runs=False,
):
    validate_config(config)
    out_dir = experiment_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    problem_path = prepare_problem(config, force=rebuild_problem)
    outputs = {"experiment_dir": out_dir, "problem": problem_path}
    for group in groups:
        seeds = list(config["seeds"][group])
        if not seeds:
            print(f"{group}: no seeds configured; skipping", flush=True)
            continue
        raw_dir = out_dir / f"raw_{group}"
        for index, seed in enumerate(seeds, 1):
            path = run_seed(config, seed, problem_path, raw_dir, force=force_runs)
            saved = np.load(path)
            print(
                f"{group} {index:02d}/{len(seeds)} seed={seed}: "
                f"{float(saved['runtime']):.2f}s",
                flush=True,
            )
        outputs[f"raw_{group}"] = raw_dir
    return outputs


def _exact_log_value(logs, step, column):
    index = int(np.argmin(np.abs(logs[:, 0] - step)))
    if int(logs[index, 0]) != step:
        raise ValueError(
            f"Step {step} was not logged. Choose window endpoints that are multiples of log_every."
        )
    return float(logs[index, column])


def window_metrics(run, start, end):
    dense, logs, residual = run["dense"], run["logs"], run["r2"]
    cycle_mean = 0.5 * (dense[:-1] + dense[1:])
    increments = np.diff(dense[start : end + 2])
    ems, eme = _exact_log_value(logs, start, 2), _exact_log_value(logs, end, 2)
    oms, ome = _exact_log_value(logs, start + 1, 2), _exact_log_value(logs, end + 1, 2)
    eis, eie = _exact_log_value(logs, start, 3), _exact_log_value(logs, end, 3)
    ois, oie = _exact_log_value(logs, start + 1, 3), _exact_log_value(logs, end + 1, 3)
    rho2 = float("nan")
    if len(increments) > 5 and np.std(increments[:-1]) > 0 and np.std(increments[1:]) > 0:
        rho2 = float(-np.corrcoef(increments[:-1], increments[1:])[0, 1])
    return {
        "loss_change": float((cycle_mean[end] - cycle_mean[start]) / cycle_mean[start]),
        "mean_even_start": ems,
        "mean_even_end": eme,
        "mean_even_gain": eme - ems,
        "mean_odd_start": oms,
        "mean_odd_end": ome,
        "mean_odd_gain": ome - oms,
        "mean_phaseavg_start": 0.5 * (ems + oms),
        "mean_phaseavg_end": 0.5 * (eme + ome),
        "mean_phaseavg_gain": 0.5 * ((eme - ems) + (ome - oms)),
        "min_even_start": eis,
        "min_even_end": eie,
        "min_even_gain": eie - eis,
        "min_odd_start": ois,
        "min_odd_end": oie,
        "min_odd_gain": oie - ois,
        "min_phaseavg_start": 0.5 * (eis + ois),
        "min_phaseavg_end": 0.5 * (eie + oie),
        "min_phaseavg_gain": 0.5 * ((eie - eis) + (oie - ois)),
        "rho2": rho2,
        "r2": float(np.nanmedian(residual[start : end + 1])),
    }


def _bootstrap_ci(values, seed, repetitions):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(repetitions, len(x)))
    return np.percentile(np.median(x[idx], axis=1), [2.5, 97.5])


def _median_iqr(array):
    return np.median(array, axis=0), np.percentile(array, 25, axis=0), np.percentile(array, 75, axis=0)


def _load_group(config, group):
    steps = int(config["student"]["steps"])
    raw_dir = experiment_dir(config) / f"raw_{group}"
    runs = {}
    for seed in config["seeds"][group]:
        path = raw_dir / f"seed{seed}_T{steps}_biphase.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        saved = np.load(path)
        runs[int(seed)] = {name: saved[name] for name in saved.files}
    return runs


def analyze_group(config, group):
    out_dir = experiment_dir(config) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = _load_group(config, group)
    rows = []
    for window_name, endpoints in config["analysis"]["windows"].items():
        start, end = map(int, endpoints)
        if end + 1 >= int(config["student"]["steps"]):
            continue
        for seed, run in runs.items():
            rows.append({
                "seed": seed,
                "window": window_name,
                "start": start,
                "end": end,
                **window_metrics(run, start, end),
            })
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_dir / f"{group}_per_seed_windows.csv", index=False)

    metric_columns = [c for c in per_seed.columns if c not in {"seed", "window", "start", "end"}]
    summary_rows = []
    repetitions = int(config["analysis"]["bootstrap_repetitions"])
    for group_index, (window_name, frame) in enumerate(per_seed.groupby("window", sort=False)):
        row = {
            "group": group,
            "window": window_name,
            "start": int(frame["start"].iloc[0]),
            "end": int(frame["end"].iloc[0]),
            "n_seeds": len(frame),
        }
        for column_index, column in enumerate(metric_columns):
            lo, hi = _bootstrap_ci(frame[column], 1000 * (group_index + 1) + column_index, repetitions)
            row[f"median_{column}"] = frame[column].median()
            row[f"{column}_ci_low"] = lo
            row[f"{column}_ci_high"] = hi
        row["positive_mean_even_seeds"] = int((frame["mean_even_gain"] > 0).sum())
        row["positive_mean_odd_seeds"] = int((frame["mean_odd_gain"] > 0).sum())
        row["positive_min_even_seeds"] = int((frame["min_even_gain"] > 0).sum())
        row["positive_min_odd_seeds"] = int((frame["min_odd_gain"] > 0).sum())
        row["loss_within_5pct_seeds"] = int((frame["loss_change"].abs() <= 0.05).sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"{group}_window_summary.csv", index=False)

    _plot_group(config, group, runs, per_seed, out_dir)
    return per_seed, summary


def _plot_group(config, group, runs, per_seed, out_dir):
    steps = int(config["student"]["steps"])
    log_every = int(config["student"]["log_every"])
    primary_start, primary_end = map(int, config["analysis"]["primary_window"])
    plot_xmax = int(config["analysis"].get("plot_xmax") or steps - 1)
    base_steps = np.arange(0, steps, log_every)

    mean_even, mean_odd, min_even, min_odd, dense, residual = [], [], [], [], [], []
    for seed, run in runs.items():
        logs = run["logs"]
        even = logs[logs[:, 0] % log_every == 0]
        odd = logs[logs[:, 0] % log_every == 1]
        expected_even = base_steps
        expected_odd = (base_steps + 1)[base_steps + 1 < steps]
        if not np.array_equal(even[:, 0], expected_even):
            raise ValueError(f"Unexpected even-phase grid for seed {seed}")
        if not np.array_equal(odd[:, 0], expected_odd):
            raise ValueError(f"Unexpected odd-phase grid for seed {seed}")
        pair_count = min(len(even), len(odd))
        mean_even.append(even[:pair_count, 2])
        mean_odd.append(odd[:pair_count, 2])
        min_even.append(even[:pair_count, 3])
        min_odd.append(odd[:pair_count, 3])
        dense.append(run["dense"])
        residual.append(run["r2"])
    paired_steps = base_steps[: len(mean_even[0])]
    mean_even, mean_odd = np.vstack(mean_even), np.vstack(mean_odd)
    min_even, min_odd = np.vstack(min_even), np.vstack(min_odd)
    mean_avg, min_avg = 0.5 * (mean_even + mean_odd), 0.5 * (min_even + min_odd)
    dense, residual = np.vstack(dense), np.vstack(residual)
    cycle_mean = 0.5 * (dense[:, :-1] + dense[:, 1:])
    normalizer = np.median(cycle_mean[:, primary_start])

    plt.figure(figsize=(10, 5.9))
    plt.plot(np.arange(0, steps, 2), np.median(dense[:, 0::2], axis=0) / normalizer, linewidth=1.15, alpha=0.75, label="Even loss phase")
    plt.plot(np.arange(1, steps, 2), np.median(dense[:, 1::2], axis=0) / normalizer, linewidth=1.15, alpha=0.75, label="Odd loss phase")
    plt.plot(np.arange(steps - 1), np.median(cycle_mean, axis=0) / normalizer, linewidth=2.7, label="Cycle mean")
    plt.axhline(1, linestyle="--", alpha=0.65)
    for marker in (primary_start, primary_end):
        plt.axvline(marker, linestyle=":", alpha=0.65)
    plt.xlim(0, plot_xmax)
    plt.xlabel("Training step")
    plt.ylabel(f"Loss / median cycle mean at step {primary_start}")
    plt.title(f"{group.capitalize()}: fixed-LR Muon loss cycle")
    plt.legend(); plt.tight_layout(); plt.savefig(out_dir / f"{group}_loss_cycle.png", dpi=220); plt.close()

    def phase_plot(even_array, odd_array, ylabel, title, filename):
        em, e1, e3 = _median_iqr(even_array)
        om, o1, o3 = _median_iqr(odd_array)
        plt.figure(figsize=(9.8, 5.9))
        plt.fill_between(paired_steps, e1, e3, alpha=0.16)
        plt.plot(paired_steps, em, linewidth=2.5, label="Even parameter phase")
        plt.fill_between(paired_steps + 1, o1, o3, alpha=0.16)
        plt.plot(paired_steps + 1, om, linewidth=2.5, label="Odd parameter phase")
        for marker in (primary_start, primary_end):
            plt.axvline(marker, linestyle=":", alpha=0.65)
        plt.xlim(0, plot_xmax); plt.ylim(0, 1)
        plt.xlabel("Training step"); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_dir / filename, dpi=220); plt.close()

    phase_plot(mean_even, mean_odd, "Mean squared principal-cosine alignment", f"{group.capitalize()}: mean alignment in both phases", f"{group}_mean_alignment_two_phases.png")
    phase_plot(min_even, min_odd, "Minimum squared principal-cosine alignment", f"{group.capitalize()}: weakest-direction alignment in both phases", f"{group}_min_alignment_two_phases.png")

    def avg_plot(array, ylabel, title, filename):
        med, q1, q3 = _median_iqr(array)
        plt.figure(figsize=(9.8, 5.9))
        for trajectory in array:
            plt.plot(paired_steps + 0.5, trajectory, alpha=0.05, linewidth=0.6)
        plt.fill_between(paired_steps + 0.5, q1, q3, alpha=0.20, label="IQR")
        plt.plot(paired_steps + 0.5, med, linewidth=2.7, label="Phase average")
        for marker in (primary_start, primary_end):
            plt.axvline(marker, linestyle=":", alpha=0.65)
        plt.xlim(0, plot_xmax); plt.ylim(0, 1)
        plt.xlabel("Training step"); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_dir / filename, dpi=220); plt.close()

    avg_plot(mean_avg, "Phase-averaged mean alignment", f"{group.capitalize()}: mean alignment growth and saturation", f"{group}_mean_alignment_phase_average.png")
    avg_plot(min_avg, "Phase-averaged minimum alignment", f"{group.capitalize()}: weakest-direction growth and boundary", f"{group}_min_alignment_phase_average.png")

    rmed, rq1, rq3 = _median_iqr(residual)
    plt.figure(figsize=(9.8, 5.9))
    plt.fill_between(np.arange(steps), rq1, rq3, alpha=0.20, label="IQR")
    plt.plot(np.arange(steps), rmed, linewidth=2.7, label="Median")
    for marker in (primary_start, primary_end):
        plt.axvline(marker, linestyle=":", alpha=0.65)
    plt.xlim(0, plot_xmax); plt.ylim(0, min(2.0, max(1.0, float(np.nanpercentile(residual, 99)))))
    plt.xlabel("Training step"); plt.ylabel(r"Parameter two-step residual $R_2^W$"); plt.title(f"{group.capitalize()}: parameter recurrence"); plt.legend(); plt.tight_layout(); plt.savefig(out_dir / f"{group}_two_step_residual.png", dpi=220); plt.close()

    primary_name = next((name for name, endpoints in config["analysis"]["windows"].items() if endpoints == config["analysis"]["primary_window"]), None)
    if primary_name:
        frame = per_seed[per_seed["window"] == primary_name].sort_values("seed")
        positions = np.arange(len(frame))
        plt.figure(figsize=(10, 5.8))
        plt.bar(positions, frame["mean_phaseavg_gain"])
        plt.axhline(frame["mean_phaseavg_gain"].median(), linestyle="--", label=f"Median = {frame['mean_phaseavg_gain'].median():.3f}")
        plt.axhline(0, linewidth=1)
        plt.xticks(positions, frame["seed"].astype(str), rotation=90)
        plt.ylabel("Phase-averaged mean-alignment gain"); plt.title(f"{group.capitalize()}: per-seed primary-window gains"); plt.legend(); plt.tight_layout(); plt.savefig(out_dir / f"{group}_per_seed_mean_gain.png", dpi=220); plt.close()


def analyze_experiment(config, groups=("development", "confirmation")):
    out_dir = experiment_dir(config) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for group in groups:
        if not config["seeds"][group]:
            print(f"{group}: no seeds configured; skipping analysis", flush=True)
            continue
        _, summary = analyze_group(config, group)
        summaries.append(summary)
    if not summaries:
        raise ValueError("No non-empty seed groups were available for analysis")
    combined = pd.concat(summaries, ignore_index=True)
    path = out_dir / "all_window_summaries.csv"
    combined.to_csv(path, index=False)
    return {"analysis_dir": out_dir, "summary": path}


def sweep_learning_rates(base_config, learning_rates, group="development", force=False):
    rows = []
    for eta in learning_rates:
        config = copy.deepcopy(base_config)
        config["student"]["learning_rate"] = float(eta)
        config["output"]["experiment_name"] = f"{base_config['output']['experiment_name']}_eta_{eta:g}"
        run_experiment(config, groups=(group,), rebuild_problem=True, force_runs=force)
        result = analyze_experiment(config, groups=(group,))
        summary = pd.read_csv(result["summary"])
        primary_name = next(name for name, endpoints in config["analysis"]["windows"].items() if endpoints == config["analysis"]["primary_window"])
        row = summary[summary["window"] == primary_name].iloc[0].to_dict()
        row["learning_rate"] = eta
        rows.append(row)
    table = pd.DataFrame(rows)
    output_path = Path(base_config["output"]["output_root"]) / f"{base_config['output']['experiment_name']}_eta_sweep.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table
