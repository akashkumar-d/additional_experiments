from __future__ import annotations
import argparse
from pathlib import Path
import mnist_pca_muon as mpm


def main():
    parser = argparse.ArgumentParser(description="Sweep fixed Muon learning rates")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--etas", required=True, help="Comma-separated values")
    parser.add_argument("--group", default="development", choices=["development", "confirmation"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = mpm.load_config(args.config)
    etas = [float(x) for x in args.etas.split(",") if x.strip()]
    print(mpm.sweep_learning_rates(config, etas, group=args.group, force=args.force).to_string(index=False))

if __name__ == "__main__":
    main()
