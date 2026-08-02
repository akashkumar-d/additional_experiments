#!/usr/bin/env python3
"""Does the submitted Figure 3(d) survive reseeding ALONE?

Everything from the submitted pipeline is held fixed -- the same 15
configurations, the same late-half secant of cos^2_min, the same <0.005
filter -- and only the seed count changes: the published single seed 5
becomes the 10 reported seeds.

Per config: median secant over its kept seeds, IQR bars; a configuration is
dropped by the filter if its MEDIAN late-half change is < 0.005 (the closest
10-seed analogue of the published per-config rule). Faint dots are individual
kept runs. The published fit (a = +2.12, from seed 5) is the grey dashed
reference; the black line is the same fit re-done at 10 seeds.

Writes fig3d_15config_10seed.png/.csv next to this script.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CKPT = HERE.parent / "rebuttal" / "consolidated_experiments" / "item1_all33"
SEEDS = [5, 11, 23, 37, 41, 53, 67, 71, 83, 97]
FILT, M = 0.005, 50

CONFIGS = [  # identical to the left plot of fig3d_15config_reconstruction
    ("ReLU rt4 e2.0",   "relu_p100_rt4_rs50_n15000_eta2.0",    "relu",  4),
    ("ReLU rt6 e1.5",   "relu_p100_rt6_rs50_n15000_eta1.5",    "relu",  6),
    ("ReLU rt8 e1.5",   "relu_p100_rt8_rs50_n15000_eta1.5",    "relu",  8),
    ("ReLU rt12 e1.5",  "relu_p100_rt12_rs50_n15000_eta1.5",   "relu", 12),
    ("ReLU rt16 e2.0",  "relu_p100_rt16_rs50_n15000_eta2.0",   "relu", 16),
    ("GELU rt4 e0.7",   "gelu_p100_rt4_rs50_n15000_eta0.7",    "gelu",  4),
    ("GELU rt8 e0.7",   "gelu_p100_rt8_rs50_n15000_eta0.7",    "gelu",  8),
    ("GELU rt12 e0.7",  "gelu_p100_rt12_rs50_n15000_eta0.7",   "gelu", 12),
    ("GELU rt16 e0.7",  "gelu_p100_rt16_rs50_n15000_eta0.7",   "gelu", 16),
    ("GELU rt16 e0.85", "gelu_p100_rt16_rs50_n15000_eta0.85",  "gelu", 16),
    ("SiLU rt4 e0.85",  "silu_p100_rt4_rs50_n15000_eta0.85",   "silu",  4),
    ("SiLU rt8 e0.85",  "silu_p100_rt8_rs50_n15000_eta0.85",   "silu",  8),
    ("SiLU rt12 e0.85", "silu_p100_rt12_rs50_n15000_eta0.85",  "silu", 12),
    ("SiLU rt16 e0.85", "silu_p100_rt16_rs50_n15000_eta0.85",  "silu", 16),
    ("SiLU rt20 e0.85", "silu_p100_rt20_rs50_n15000_eta0.85",  "silu", 20),
]
COL = {"relu": "tab:blue", "gelu": "tab:orange", "silu": "tab:green"}


def dedupe(s, y):
    s = np.asarray(s, float); y = np.asarray(y, float)
    _, i = np.unique(s, return_index=True)
    return s[i], y[i]


def secant(f):
    z = np.load(f, allow_pickle=True)
    s, cm = dedupe(z["step"], z["cos2_min_AGOP"])
    i = len(cm) // 2
    return cm[-1] - cm[i], (cm[-1] - cm[i]) / (s[-1] - s[i])


def fit_loglog(x, y):
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)
    r2 = 1 - ((ly - (a * lx + b)) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return a, b, r2


rows = []
for lab, base, act, rt in CONFIGS:
    per = []
    for f in CKPT.glob(f"{base}_seed*.npz"):
        sd = int(re.search(r"seed(\d+)", f.name).group(1))
        if sd in SEEDS:
            ch, sl = secant(f)
            per.append((sd, ch, sl))
    ch_med = float(np.median([c for _, c, _ in per]))
    sls = np.array([s for _, _, s in per])
    rows.append(dict(label=lab, act=act, x=rt / M, n_seeds=len(per),
                     change_med=ch_med, secant_med=float(np.median(sls)),
                     secant_q1=float(np.percentile(sls, 25)),
                     secant_q3=float(np.percentile(sls, 75)),
                     filtered=ch_med < FILT,
                     per_run=[(sd, s) for sd, _, s in per]))

with (HERE / "fig3d_15config_10seed.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=[k for k in rows[0] if k != "per_run"])
    w.writeheader()
    for r in rows:
        w.writerow({k: v for k, v in r.items() if k != "per_run"})

kept = [r for r in rows if not r["filtered"]]
a10, b10, r2_10 = fit_loglog(np.array([r["x"] for r in kept]),
                             np.array([r["secant_med"] for r in kept]))
# per-run version: each run kept if its OWN late-half change >= FILT
xr, yr = [], []
for lab, base, act, rt in CONFIGS:
    for f in CKPT.glob(f"{base}_seed*.npz"):
        sd = int(re.search(r"seed(\d+)", f.name).group(1))
        if sd not in SEEDS:
            continue
        ch, sl = secant(f)
        if ch >= FILT:
            xr.append(rt / M); yr.append(sl)
a_pr, b_pr, r2_pr = fit_loglog(np.array(xr), np.array(yr))

fig, ax = plt.subplots(figsize=(7.6, 5.4))
FLOOR = 1.2e-6
for r in rows:
    xx = r["x"]
    for _, s in r["per_run"]:
        if s > 0:
            ax.scatter([xx], [s], s=9, color=COL[r["act"]], alpha=0.25, zorder=2)
    if r["filtered"]:
        yv = max(abs(r["secant_med"]), FLOOR)
        ax.scatter([xx], [yv], marker="v" if r["secant_med"] < 0 else "^",
                   s=80, facecolors="none", edgecolors=COL[r["act"]],
                   linewidths=1.6, zorder=4)
    else:
        ax.errorbar([xx], [r["secant_med"]],
                    yerr=[[r["secant_med"] - r["secant_q1"]],
                          [r["secant_q3"] - r["secant_med"]]],
                    fmt="o", ms=6, color=COL[r["act"]], ecolor=COL[r["act"]],
                    elinewidth=1.3, capsize=2.6, zorder=3,
                    markeredgecolor="black", markeredgewidth=0.5)

xs = np.linspace(0.07, 0.45, 60)
ax.plot(xs, np.exp(b10) * xs ** a10, "k-", lw=1.6,
        label=rf"10-seed medians, same filter: $a={a10:+.2f}$  ($R^2$={r2_10:.2f}, {len(kept)}/15 kept)")
# the published seed-5 fit as reference
A_PUB, R2_PUB = 2.12, 0.46
b_pub = np.log(np.array([r["secant_med"] for r in kept])).mean() - A_PUB * np.log(np.array([r["x"] for r in kept])).mean()
ax.plot(xs, np.exp(b_pub) * xs ** A_PUB, color="0.55", ls="--", lw=1.5,
        label=rf"published (seed 5): $a=+{A_PUB}$  ($R^2$={R2_PUB})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$r_t/M$")
ax.set_ylabel(r"late-half secant of $\cos^2_{\min}$  [per step]")
ax.set_title("Submitted Fig. 3(d) pipeline, unchanged, at 10 seeds\n"
             "same 15 configurations, same secant, same $<0.005$ filter",
             fontsize=10)
handles, labels = ax.get_legend_handles_labels()
handles += [plt.Line2D([], [], marker="o", ls="", color=c, label=a) for a, c in COL.items()]
handles.append(plt.Line2D([], [], marker="^", ls="", markerfacecolor="none",
                          color="0.3", label="filtered (median change < 0.005)"))
ax.legend(handles=handles, fontsize=7.2, loc="lower right", framealpha=0.92)
ax.grid(alpha=0.25, which="both", lw=0.5)

fig.tight_layout()
out = HERE / "fig3d_15config_10seed.png"
fig.savefig(out, dpi=170)
print("wrote", out)
print(f"\npublished (seed 5)                : a = +2.12 (R^2 0.46) 12/15 kept")
print(f"10-seed medians, same pipeline    : a = {a10:+.2f} (R^2 {r2_10:.2f}) {len(kept)}/15 kept")
print(f"per-run, same per-run filter      : a = {a_pr:+.2f} (R^2 {r2_pr:.2f}) on {len(xr)} runs")
print("filtered configs:", [r["label"] for r in rows if r["filtered"]])
print("seeds found per config:", {r["label"]: r["n_seeds"] for r in rows if r["n_seeds"] != 10} or "10 everywhere")
