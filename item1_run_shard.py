#!/usr/bin/env python3
"""
Shard worker for the Item-1 grid (all 33 configs x 5 seeds = 165 runs).

The 165 runs are fully independent, so parallelism is at the run level:
launch K copies of this script with --num-shards K --shard 0..K-1 and each
copy takes the interleaved slice ALL_RUNS[shard::num_shards] (round-robin,
so every shard gets a balanced mix of activations and config sizes).

All shards write per-run .npz files into the same --out-dir. The runner
skips runs whose checkpoint is already complete, so re-launching a crashed
shard (or overlapping shards on a shared filesystem) is safe as long as two
processes do not work on the SAME run simultaneously -- use disjoint shard
indices to guarantee that.

NOTE: this is pure NumPy (CPU). On a GPU node, the GPU is idle; what
matters is one CPU worker (with a couple of BLAS threads) per shard.
Set OMP_NUM_THREADS=2 (or 4) per process to avoid oversubscription when
running many shards on one node.

Examples
--------
# preview shard assignments
python item1_run_shard.py --num-shards 8 --shard 0 --list

# plain shells / tmux on one big node (8 workers):
for i in $(seq 0 7); do
  OMP_NUM_THREADS=2 nohup python item1_run_shard.py \
      --num-shards 8 --shard $i > shard_$i.log 2>&1 &
done

# SLURM array (one worker per task, e.g. across GPU nodes):
#   #SBATCH --array=0-7
#   #SBATCH --cpus-per-task=4
#   export OMP_NUM_THREADS=4
#   python item1_run_shard.py --num-shards 8 --shard $SLURM_ARRAY_TASK_ID

# then merge (if shards wrote to node-local dirs, rsync them into one
# item1_all33/ on the analysis machine) and run the analysis cells of
# item1_all33_seeds.ipynb.
"""
import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "shared"))

from teacher_full_runner import run_one  # noqa: E402

# ------------------------------------------------------------------
# The exact 33-config grid (must match item1_all33_seeds.ipynb)
# ------------------------------------------------------------------
CONFIGS_33 = [
    ('R1', 'relu', 100,  4, 50, 15000, 2.0),
    ('R2', 'relu', 100,  6, 50, 15000, 1.5),
    ('R3', 'relu', 100,  8, 50, 15000, 1.5),
    ('R4', 'relu', 100,  8, 50, 15000, 1.0),
    ('R5', 'relu', 100,  8, 30, 15000, 1.5),
    ('R6', 'relu', 100, 12, 50, 15000, 1.5),
    ('R7', 'relu', 100, 16, 50, 15000, 1.5),
    ('R8', 'relu', 100, 16, 50, 15000, 2.0),
    ('R9', 'relu', 100, 20, 50, 15000, 2.0),
    ('G1', 'gelu', 100,  4, 50, 15000, 0.7),
    ('G2', 'gelu', 100,  6, 50, 15000, 0.7),
    ('G3', 'gelu', 100,  8, 50, 15000, 0.7),
    ('G4', 'gelu', 100,  8, 30, 15000, 0.7),
    ('G5', 'gelu', 100, 12, 50, 15000, 0.7),
    ('G6', 'gelu', 100,  8, 50, 15000, 0.5),
    ('S1', 'silu', 100,  4, 50, 15000, 0.85),
    ('S2', 'silu', 100,  6, 50, 15000, 0.85),
    ('S3', 'silu', 100,  8, 50, 15000, 0.85),
    ('S4', 'silu', 100,  8, 30, 15000, 0.85),
    ('S5', 'silu', 100, 12, 50, 15000, 0.85),
    ('S6', 'silu', 100,  8, 50, 15000, 0.6),
    ('G7',  'gelu', 100, 12, 50, 15000, 0.6),
    ('G8',  'gelu', 100, 12, 50, 15000, 0.85),
    ('G9',  'gelu', 100, 16, 50, 15000, 0.7),
    ('G10', 'gelu', 100, 16, 50, 15000, 0.85),
    ('G11', 'gelu', 100, 20, 50, 15000, 0.7),
    ('G12', 'gelu', 100, 12, 80, 15000, 0.7),
    ('S7',  'silu', 100, 12, 50, 15000, 0.7),
    ('S8',  'silu', 100, 12, 50, 15000, 1.0),
    ('S9',  'silu', 100, 16, 50, 15000, 0.85),
    ('S10', 'silu', 100, 16, 50, 15000, 1.0),
    ('S11', 'silu', 100, 20, 50, 15000, 0.85),
    ('S12', 'silu', 100, 12, 80, 15000, 0.85),
]
assert len(CONFIGS_33) == 33
SEEDS = [5, 11, 23, 37, 41]

ALL_RUNS = [(gid, ta, p, rt, rs, n, eta, seed)
            for (gid, ta, p, rt, rs, n, eta) in CONFIGS_33
            for seed in SEEDS]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out-dir", default=str(_HERE / "item1_all33"))
    ap.add_argument("--n-steps", type=int, default=8000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--chunk-steps", type=int, default=4000)
    ap.add_argument("--budget-s", type=float, default=1e9,
                    help="wall-clock cap for this shard (default: unlimited)")
    ap.add_argument("--list", action="store_true",
                    help="print this shard's assignments and exit")
    args = ap.parse_args()

    if not (0 <= args.shard < args.num_shards):
        raise SystemExit(f"--shard must be in [0, {args.num_shards})")

    mine = ALL_RUNS[args.shard::args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(mine)} of {len(ALL_RUNS)} runs")

    if args.list:
        for gid, ta, p, rt, rs, n, eta, seed in mine:
            print(f"  {gid:>4s} {ta:>5s} rt={rt:>2d} rs={rs:>3d} eta={eta:<5} seed={seed}")
        return

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for i, (gid, ta, p, rt, rs, n, eta, seed) in enumerate(mine):
        elapsed = time.time() - t0
        if elapsed > args.budget_s:
            print(f"[budget] stop after {i}/{len(mine)}")
            break
        name = f"{ta}_p{p}_rt{rt}_rs{rs}_n{n}_eta{eta}_seed{seed}.npz"
        print(f"\n[{i+1}/{len(mine)}] {gid} seed={seed}: {name} "
              f"(elapsed {elapsed/60:.1f} min)")
        run_one(ta, p, rt, rs, n, eta, seed=seed, n_steps=args.n_steps,
                log_every=args.log_every, ckpt_path=out / name,
                chunk_steps=args.chunk_steps,
                budget_s=max(60.0, args.budget_s - elapsed))
    print(f"\n[shard {args.shard} done] {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
