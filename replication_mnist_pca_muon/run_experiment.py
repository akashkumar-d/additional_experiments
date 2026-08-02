from __future__ import annotations
import argparse
from pathlib import Path
import mnist_pca_muon as mpm


def main():
    parser = argparse.ArgumentParser(description="Run or analyze the PCA-MNIST Muon experiment")
    parser.add_argument("command", choices=["prepare", "run", "analyze", "all"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--groups", default="development,confirmation")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = mpm.load_config(args.config)
    groups = tuple(token.strip() for token in args.groups.split(",") if token.strip())
    if args.command == "prepare":
        print(mpm.prepare_problem(config, force=args.force))
    elif args.command == "run":
        mpm.run_experiment(config, groups=groups, force_runs=args.force)
    elif args.command == "analyze":
        print(mpm.analyze_experiment(config, groups=groups))
    else:
        mpm.run_experiment(config, groups=groups, rebuild_problem=args.force, force_runs=args.force)
        print(mpm.analyze_experiment(config, groups=groups))

if __name__ == "__main__":
    main()
