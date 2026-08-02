from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd

import mnist_pca_muon as mpm
import robust_family as rf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and plot the four robust PCA-MNIST Muon configurations."
    )
    parser.add_argument(
        "--problem",
        type=Path,
        default=Path("data/fixed_problem_reproduction.npz"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/robust_family_300"),
    )
    parser.add_argument(
        "--seed-group",
        choices=["development", "confirmation"],
        default="confirmation",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--window-start", type=int, default=40)
    parser.add_argument("--window-end", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    seeds = (
        rf.ROBUST_FAMILY_CONFIRMATION_SEEDS
        if args.seed_group == "confirmation"
        else rf.ROBUST_FAMILY_DEVELOPMENT_SEEDS
    )
    base_config = copy.deepcopy(mpm.DEFAULT_CONFIG)
    window = (args.window_start, args.window_end)

    rf.run_family(
        base_config,
        args.problem,
        args.output_root,
        seeds=seeds,
        steps=args.steps,
        window=window,
        log_every=args.log_every,
        threads=args.threads,
        force=args.force,
    )
    outputs = rf.analyze_family(
        args.output_root,
        seeds=seeds,
        steps=args.steps,
        window=window,
        log_every=args.log_every,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    summary = pd.read_csv(outputs["summary"])
    print(
        summary[
            [
                "config_id",
                "width",
                "learning_rate",
                "head_multiplier",
                "median_loss_change",
                "median_mean_phaseavg_start",
                "median_mean_phaseavg_end",
                "median_mean_phaseavg_gain",
                "median_min_phaseavg_gain",
                "median_rho2",
                "median_r2",
                "strict_seeds",
            ]
        ].to_string(index=False)
    )
    print(f"Analysis: {outputs['analysis_dir']}")


if __name__ == "__main__":
    main()
