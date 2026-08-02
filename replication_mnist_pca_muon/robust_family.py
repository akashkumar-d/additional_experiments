from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mnist_pca_muon as mpm


ROBUST_FAMILY_CONFIGS: list[dict[str, Any]] = [
    {
        "config_id": "C5_w60_e060_h100",
        "label": r"$r_s=60,\ \eta=0.60,\ h=1.00$",
        "width": 60,
        "learning_rate": 0.60,
        "head_multiplier": 1.00,
    },
    {
        "config_id": "C3_w50_e060_h125",
        "label": r"$r_s=50,\ \eta=0.60,\ h=1.25$",
        "width": 50,
        "learning_rate": 0.60,
        "head_multiplier": 1.25,
    },
    {
        "config_id": "C4_w60_e050_h100",
        "label": r"$r_s=60,\ \eta=0.50,\ h=1.00$",
        "width": 60,
        "learning_rate": 0.50,
        "head_multiplier": 1.00,
    },
    {
        "config_id": "C2_w50_e050_h125",
        "label": r"$r_s=50,\ \eta=0.50,\ h=1.25$",
        "width": 50,
        "learning_rate": 0.50,
        "head_multiplier": 1.25,
    },
]

ROBUST_FAMILY_DEVELOPMENT_SEEDS = [5, 11, 23, 37, 41]

# These are the fresh 30 seeds used for the robust-family confirmation.
ROBUST_FAMILY_CONFIRMATION_SEEDS = [
    227, 229, 233, 239, 241, 251, 257, 263, 269, 271,
    277, 281, 283, 293, 307, 311, 313, 317, 331, 337,
    347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
]


def make_config(
    base_config: dict[str, Any],
    spec: dict[str, Any],
    *,
    problem_path: str | Path,
    output_root: str | Path,
    seeds: Iterable[int],
    steps: int = 300,
    window: tuple[int, int] = (40, 200),
    log_every: int = 10,
    threads: int = 1,
) -> dict[str, Any]:
    """Create one fixed-LR family configuration.

    The head multiplier h means that every fixed output coefficient is
    h / sqrt(p), where p is the PCA dimension.
    """
    problem = mpm.load_problem(problem_path)
    input_dimension = int(problem["X"].shape[1])

    config = copy.deepcopy(base_config)
    config["student"].update(
        width=int(spec["width"]),
        head_value=float(spec["head_multiplier"]) / math.sqrt(input_dimension),
        learning_rate=float(spec["learning_rate"]),
        steps=int(steps),
        log_every=int(log_every),
        log_both_phases=True,
        threads=int(threads),
        checkpoint_steps=[0, int(window[0]), int(window[1]), int(steps) - 1],
    )
    config["analysis"]["primary_window"] = [int(window[0]), int(window[1])]
    config["analysis"]["windows"] = {
        f"{int(window[0])}_{int(window[1])}": [int(window[0]), int(window[1])]
    }
    config["analysis"]["plot_xmax"] = int(steps) - 1
    config["output"]["output_root"] = str(Path(output_root))
    config["output"]["experiment_name"] = str(spec["config_id"])
    config["seeds"]["development"] = [int(seed) for seed in seeds]
    config["seeds"]["confirmation"] = []
    mpm.validate_config(config)
    return config


def run_family(
    base_config: dict[str, Any],
    problem_path: str | Path,
    output_root: str | Path,
    *,
    configs: list[dict[str, Any]] | None = None,
    seeds: Iterable[int] = ROBUST_FAMILY_DEVELOPMENT_SEEDS,
    steps: int = 300,
    window: tuple[int, int] = (40, 200),
    log_every: int = 10,
    threads: int = 1,
    force: bool = False,
) -> dict[str, Path]:
    """Run all robust-family configurations on the same fixed problem."""
    specs = ROBUST_FAMILY_CONFIGS if configs is None else configs
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in seeds]
    outputs: dict[str, Path] = {}

    for config_index, spec in enumerate(specs, start=1):
        config = make_config(
            base_config,
            spec,
            problem_path=problem_path,
            output_root=output_root,
            seeds=seeds,
            steps=steps,
            window=window,
            log_every=log_every,
            threads=threads,
        )
        config_dir = output_root / str(spec["config_id"])
        raw_dir = config_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        mpm.save_config(config, config_dir / "resolved_config.json")

        for seed_index, seed in enumerate(seeds, start=1):
            path = mpm.run_seed(
                config,
                seed,
                problem_path,
                raw_dir,
                force=force,
            )
            saved = np.load(path)
            print(
                f"{config_index:02d}/{len(specs)} {spec['config_id']} "
                f"seed {seed_index:02d}/{len(seeds)}={seed}: "
                f"{float(saved['runtime']):.2f}s",
                flush=True,
            )
        outputs[str(spec["config_id"])] = raw_dir

    metadata = {
        "problem_path": str(Path(problem_path).resolve()),
        "steps": int(steps),
        "window": [int(window[0]), int(window[1])],
        "log_every": int(log_every),
        "threads": int(threads),
        "seeds": seeds,
        "configs": specs,
    }
    (output_root / "family_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return outputs


def _load_runs(
    output_root: str | Path,
    spec: dict[str, Any],
    seeds: Iterable[int],
    steps: int,
) -> dict[int, dict[str, np.ndarray]]:
    raw_dir = Path(output_root) / str(spec["config_id"]) / "raw"
    runs: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        path = raw_dir / f"seed{int(seed)}_T{int(steps)}_biphase.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        saved = np.load(path)
        runs[int(seed)] = {name: saved[name] for name in saved.files}
    return runs


def _bootstrap_median_ci(
    values: pd.Series,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    medians = np.median(array[indices], axis=1)
    low, high = np.percentile(medians, [2.5, 97.5])
    return float(low), float(high)


def _phase_average_curves(
    run: dict[str, np.ndarray],
    *,
    steps: int,
    log_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logs = run["logs"]
    base_steps = np.arange(0, steps, log_every)
    even = logs[logs[:, 0] % log_every == 0]
    odd = logs[logs[:, 0] % log_every == 1]
    expected_odd = (base_steps + 1)[base_steps + 1 < steps]
    if not np.array_equal(even[:, 0], base_steps):
        raise ValueError("Unexpected even-phase logging grid")
    if not np.array_equal(odd[:, 0], expected_odd):
        raise ValueError("Unexpected odd-phase logging grid")
    pair_count = min(len(even), len(odd))
    paired_steps = base_steps[:pair_count] + 0.5
    mean_average = 0.5 * (even[:pair_count, 2] + odd[:pair_count, 2])
    min_average = 0.5 * (even[:pair_count, 3] + odd[:pair_count, 3])
    return paired_steps, mean_average, min_average


def analyze_family(
    output_root: str | Path,
    *,
    configs: list[dict[str, Any]] | None = None,
    seeds: Iterable[int] = ROBUST_FAMILY_DEVELOPMENT_SEEDS,
    steps: int = 300,
    window: tuple[int, int] = (40, 200),
    log_every: int = 10,
    bootstrap_repetitions: int = 20_000,
) -> dict[str, Path]:
    """Create one summary table and three aggregated family plots."""
    specs = ROBUST_FAMILY_CONFIGS if configs is None else configs
    seeds = [int(seed) for seed in seeds]
    output_root = Path(output_root)
    analysis_dir = output_root / "family_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    curve_data: dict[str, dict[str, np.ndarray | str]] = {}

    for config_index, spec in enumerate(specs):
        runs = _load_runs(output_root, spec, seeds, steps)
        frame = pd.DataFrame(
            [
                {
                    "config_id": spec["config_id"],
                    "label": spec["label"],
                    "width": int(spec["width"]),
                    "learning_rate": float(spec["learning_rate"]),
                    "head_multiplier": float(spec["head_multiplier"]),
                    "seed": seed,
                    **mpm.window_metrics(run, int(window[0]), int(window[1])),
                }
                for seed, run in runs.items()
            ]
        )
        per_seed_rows.extend(frame.to_dict("records"))

        row: dict[str, Any] = {
            "config_id": spec["config_id"],
            "label": spec["label"],
            "width": int(spec["width"]),
            "learning_rate": float(spec["learning_rate"]),
            "head_multiplier": float(spec["head_multiplier"]),
            "n_seeds": len(frame),
            "window_start": int(window[0]),
            "window_end": int(window[1]),
        }
        columns = [
            "loss_change",
            "mean_phaseavg_start",
            "mean_phaseavg_end",
            "mean_phaseavg_gain",
            "min_phaseavg_start",
            "min_phaseavg_end",
            "min_phaseavg_gain",
            "rho2",
            "r2",
        ]
        for column_index, column in enumerate(columns):
            low, high = _bootstrap_median_ci(
                frame[column],
                seed=10_000 * (config_index + 1) + column_index,
                repetitions=bootstrap_repetitions,
            )
            row[f"median_{column}"] = float(frame[column].median())
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        row.update(
            {
                "positive_mean_gain_seeds": int(
                    (frame["mean_phaseavg_gain"] > 0).sum()
                ),
                "positive_min_gain_seeds": int(
                    (frame["min_phaseavg_gain"] > 0).sum()
                ),
                "loss_within_5pct_seeds": int(
                    (frame["loss_change"].abs() <= 0.05).sum()
                ),
                "strict_seeds": int(
                    (
                        (frame["loss_change"].abs() <= 0.05)
                        & (frame["mean_phaseavg_gain"] >= 0.15)
                        & (frame["rho2"] >= 0.99)
                        & (frame["r2"] <= 0.35)
                    ).sum()
                ),
            }
        )
        summary_rows.append(row)

        cycle_curves = []
        mean_curves = []
        min_curves = []
        paired_steps = None
        for run in runs.values():
            dense = run["dense"]
            cycle_mean = 0.5 * (dense[:-1] + dense[1:])
            cycle_curves.append(cycle_mean / cycle_mean[int(window[0])])
            steps_pair, mean_average, min_average = _phase_average_curves(
                run, steps=steps, log_every=log_every
            )
            paired_steps = steps_pair
            mean_curves.append(mean_average)
            min_curves.append(min_average)
        curve_data[str(spec["config_id"])] = {
            "label": str(spec["label"]),
            "cycle": np.vstack(cycle_curves),
            "mean": np.vstack(mean_curves),
            "minimum": np.vstack(min_curves),
            "paired_steps": np.asarray(paired_steps),
        }

    per_seed = pd.DataFrame(per_seed_rows)
    summary = pd.DataFrame(summary_rows)
    per_seed_path = analysis_dir / "robust_family_per_seed.csv"
    summary_path = analysis_dir / "robust_family_summary.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    def median_iqr(array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.median(array, axis=0),
            np.percentile(array, 25, axis=0),
            np.percentile(array, 75, axis=0),
        )

    def plot_cycle_loss(path: Path, *, zoom: bool) -> None:
        plt.figure(figsize=(9.8, 5.9))
        all_visible = []
        zoom_start = max(0, int(window[0]) - 20)
        for spec in specs:
            curves = np.asarray(curve_data[str(spec["config_id"])]["cycle"])
            median, q1, q3 = median_iqr(curves)
            x = np.arange(curves.shape[1])
            line, = plt.plot(x, median, linewidth=2.4, label=spec["label"])
            plt.fill_between(x, q1, q3, alpha=0.14, color=line.get_color())
            if zoom:
                all_visible.append(curves[:, zoom_start:])
        plt.axhline(1.0, linestyle="--", alpha=0.65)
        for marker in window:
            plt.axvline(marker, linestyle=":", alpha=0.55)
        if zoom:
            visible = np.concatenate(all_visible, axis=0)
            low = float(np.percentile(visible, 1))
            high = float(np.percentile(visible, 99))
            padding = 0.12 * max(high - low, 0.02)
            plt.xlim(zoom_start, steps - 1)
            plt.ylim(low - padding, high + padding)
            title = "Robust fixed-LR family: cycle-mean loss (plateau zoom)"
        else:
            plt.xlim(0, steps - 1)
            title = "Robust fixed-LR family: cycle-mean loss (full trajectory)"
        plt.xlabel("Training step")
        plt.ylabel(f"Cycle-mean loss / value at step {int(window[0])}")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=220)
        plt.close()

    cycle_path = analysis_dir / "robust_family_cycle_mean_loss.png"
    cycle_full_path = analysis_dir / "robust_family_cycle_mean_loss_full.png"
    plot_cycle_loss(cycle_path, zoom=True)
    plot_cycle_loss(cycle_full_path, zoom=False)

    mean_path = analysis_dir / "robust_family_mean_alignment.png"
    plt.figure(figsize=(9.8, 5.9))
    for spec in specs:
        item = curve_data[str(spec["config_id"])]
        curves = np.asarray(item["mean"])
        median, q1, q3 = median_iqr(curves)
        x = np.asarray(item["paired_steps"])
        line, = plt.plot(x, median, linewidth=2.4, label=spec["label"])
        plt.fill_between(x, q1, q3, alpha=0.14, color=line.get_color())
    for marker in window:
        plt.axvline(marker, linestyle=":", alpha=0.55)
    plt.xlim(0, steps - 1)
    plt.ylim(0, 1)
    plt.xlabel("Training step")
    plt.ylabel("Phase-averaged mean squared principal cosine")
    plt.title("Robust fixed-LR family: mean alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(mean_path, dpi=220)
    plt.close()

    min_path = analysis_dir / "robust_family_min_alignment.png"
    plt.figure(figsize=(9.8, 5.9))
    for spec in specs:
        item = curve_data[str(spec["config_id"])]
        curves = np.asarray(item["minimum"])
        median, q1, q3 = median_iqr(curves)
        x = np.asarray(item["paired_steps"])
        line, = plt.plot(x, median, linewidth=2.4, label=spec["label"])
        plt.fill_between(x, q1, q3, alpha=0.14, color=line.get_color())
    for marker in window:
        plt.axvline(marker, linestyle=":", alpha=0.55)
    plt.xlim(0, steps - 1)
    plt.ylim(0, 1)
    plt.xlabel("Training step")
    plt.ylabel("Phase-averaged minimum squared principal cosine")
    plt.title("Robust fixed-LR family: weakest-direction alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(min_path, dpi=220)
    plt.close()

    return {
        "analysis_dir": analysis_dir,
        "summary": summary_path,
        "per_seed": per_seed_path,
        "cycle_plot": cycle_path,
        "cycle_full_plot": cycle_full_path,
        "mean_plot": mean_path,
        "min_plot": min_path,
    }
