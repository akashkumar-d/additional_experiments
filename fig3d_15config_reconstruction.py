#!/usr/bin/env python3
"""Forensic reconstruction of the submitted Figure 3(d).

The submitted panel: 15 configurations (5 per activation), seed 5 only,
late-half secant of cos^2_min, configs with late-half change < 0.005 silently
dropped (the three r_t=4 ones), log-log fit vs r_t/M -> "empirical exponent
~ 2.12". DRIFT_LAW_README.md row 1 documents the same number.

Left panel : the reconstruction, faithful to the submitted pipeline, with the
             three filtered configurations shown instead of hidden.
Right panel: the SAME 15 runs, same seed, but a saturation-proof estimator
             (max slope of cos^2_min over any 500-step window). The fitted
             exponent flips sign. Nothing about the data changed - only the
             estimator's measurement window.

Needs the checkpoint folder (44 configs x 13 seeds); the 11 configs dropped
from the revised 33-grid are exactly the old paper grid, so all 15 resolve.
Writes fig3d_15config_reconstruction.png + .csv next to this script.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CKPT = next(p for p in [
    HERE.parent / "rebuttal" / "consolidated_experiments" / "item1_all33",
    HERE / "item1_all33",
] if p.exists() and list(p.glob("*.npz")))
SEED = 5           # the submitted panel used this single seed
FILT = 0.005       # the submitted filter on late-half change
M = 50

# The 15 configurations, read off the submitted figure's legends.
CONFIGS = [  # (label, file base, activation, r_t, eta)
    ("ReLU rt4 e2.0",   "relu_p100_rt4_rs50_n15000_eta2.0",    "relu",  4, 2.0),
    ("ReLU rt6 e1.5",   "relu_p100_rt6_rs50_n15000_eta1.5",    "relu",  6, 1.5),
    ("ReLU rt8 e1.5",   "relu_p100_rt8_rs50_n15000_eta1.5",    "relu",  8, 1.5),
    ("ReLU rt12 e1.5",  "relu_p100_rt12_rs50_n15000_eta1.5",   "relu", 12, 1.5),
    ("ReLU rt16 e2.0",  "relu_p100_rt16_rs50_n15000_eta2.0",   "relu", 16, 2.0),
    ("GELU rt4 e0.7",   "gelu_p100_rt4_rs50_n15000_eta0.7",    "gelu",  4, 0.7),
    ("GELU rt8 e0.7",   "gelu_p100_rt8_rs50_n15000_eta0.7",    "gelu",  8, 0.7),
    ("GELU rt12 e0.7",  "gelu_p100_rt12_rs50_n15000_eta0.7",   "gelu", 12, 0.7),
    ("GELU rt16 e0.7",  "gelu_p100_rt16_rs50_n15000_eta0.7",   "gelu", 16, 0.7),
    ("GELU rt16 e0.85", "gelu_p100_rt16_rs50_n15000_eta0.85",  "gelu", 16, 0.85),
    ("SiLU rt4 e0.85",  "silu_p100_rt4_rs50_n15000_eta0.85",   "silu",  4, 0.85),
    ("SiLU rt8 e0.85",  "silu_p100_rt8_rs50_n15000_eta0.85",   "silu",  8, 0.85),
    ("SiLU rt12 e0.85", "silu_p100_rt12_rs50_n15000_eta0.85",  "silu", 12, 0.85),
    ("SiLU rt16 e0.85", "silu_p100_rt16_rs50_n15000_eta0.85",  "silu", 16, 0.85),
    ("SiLU rt20 e0.85", "silu_p100_rt20_rs50_n15000_eta0.85",  "silu", 20, 0.85),
]
COL = {"relu": "tab:blue", "gelu": "tab:orange", "silu": "tab:green"}


def dedupe(s, y):
    s = np.asarray(s, float); y = np.asarray(y, float)
    _, i = np.unique(s, return_index=True)
    return s[i], y[i]


def load(base):
    f = CKPT / f"{base}_seed{SEED}.npz"
    z = np.load(f, allow_pickle=True)
    return dedupe(z["step"], z["cos2_min_AGOP"])


def fit_loglog(x, y):
    lx, ly = np.log(x), np.log(y)
    a, b = np.polyfit(lx, ly, 1)
    r2 = 1 - ((ly - (a * lx + b)) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return a, b, r2


rows = []
for lab, base, act, rt, eta in CONFIGS:
    s, cm = load(base)
    i = len(cm) // 2
    change = cm[-1] - cm[i]
    secant = change / (s[-1] - s[i])
    w = max(2, int(round(500 / (s[1] - s[0]))))
    peak = float(np.max((cm[w:] - cm[:-w]) / (s[w:] - s[:-w])))
    rows.append(dict(label=lab, act=act, x=rt / M, change=change,
                     secant=secant, peak=peak, filtered=change < FILT))

import csv
with (HERE / "fig3d_15config_reconstruction.csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

kept = [r for r in rows if not r["filtered"]]
a_sec, b_sec, r2_sec = fit_loglog(np.array([r["x"] for r in kept]),
                                  np.array([r["secant"] for r in kept]))
a_pk, b_pk, r2_pk = fit_loglog(np.array([r["x"] for r in rows]),
                               np.array([r["peak"] for r in rows]))

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 4.7))

# ---------------- left: the submitted pipeline, omissions made visible
FLOOR = 1.2e-6
# the two saturated r_t=4 configs share x=0.08 and the floor; stagger them
STAG = {"GELU rt4 e0.7": (0.985, (7, 5)), "SiLU rt4 e0.85": (1.015, (7, -13))}
for r in rows:
    if r["filtered"]:
        yv = max(abs(r["secant"]), FLOOR)
        jx, off = STAG.get(r["label"], (1.0, (7, -3)))
        axA.scatter([r["x"] * jx], [yv], marker="v" if r["secant"] < 0 else "^",
                    s=75, facecolors="none", edgecolors=COL[r["act"]],
                    linewidths=1.6, zorder=4)
        axA.annotate(r["label"] + ("  (negative)" if r["secant"] < 0 else "  (saturated)"),
                     (r["x"] * jx, yv), textcoords="offset points", xytext=off,
                     fontsize=6.8, color=COL[r["act"]])
    else:
        axA.scatter([r["x"]], [r["secant"]], s=55, color=COL[r["act"]],
                    edgecolors="black", linewidths=0.5, zorder=3)
xs = np.linspace(0.07, 0.45, 40)
axA.plot(xs, np.exp(b_sec) * xs ** a_sec, "k--", lw=1.4,
         label=rf"fit on the 12 kept: $a={a_sec:+.2f}$  ($R^2$={r2_sec:.2f})")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel(r"$r_t/M$")
axA.set_ylabel(r"late-half secant of $\cos^2_{\min}$  [per step]")
axA.set_title("(i) submitted pipeline reconstructed: seed 5,\n"
              rf"late-half secant, filter $<{FILT}$ — submission reported $a\approx+2.12$",
              fontsize=9.5)
axA.legend(fontsize=7.5, loc="lower right")

# ---------------- right: same 15 runs, saturation-proof estimator
for r in rows:
    m = "^" if r["filtered"] else "o"
    axB.scatter([r["x"]], [r["peak"]], marker=m, s=60 if r["filtered"] else 55,
                facecolors="none" if r["filtered"] else COL[r["act"]],
                edgecolors=COL[r["act"]] if r["filtered"] else "black",
                linewidths=1.4 if r["filtered"] else 0.5, zorder=3,
                color=COL[r["act"]])
axB.plot(xs, np.exp(b_pk) * xs ** a_pk, "k-", lw=1.4,
         label=rf"fit on all 15: $a={a_pk:+.2f}$  ($R^2$={r2_pk:.2f})")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlabel(r"$r_t/M$")
axB.set_ylabel(r"peak rate of $\cos^2_{\min}$  [max 500-step slope]")
axB.set_title("(ii) the same 15 runs, saturation-proof estimator:\n"
              "no configuration dropped, exponent flips sign", fontsize=9.5)
axB.legend(fontsize=7.5, loc="lower left")

handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=a) for a, c in COL.items()]
handles.append(plt.Line2D([], [], marker="^", ls="", markerfacecolor="none",
                          color="0.3", label="filtered out of the submitted panel"))
axA.legend(handles=axA.get_legend_handles_labels()[0] + handles, fontsize=6.8,
           loc="lower right", framealpha=0.9)
for ax in (axA, axB):
    ax.grid(alpha=0.25, which="both", lw=0.5)

fig.tight_layout()
out = HERE / "fig3d_15config_reconstruction.png"
fig.savefig(out, dpi=170)
print("wrote", out)
print(f"\nsubmitted pipeline, reconstructed : a = {a_sec:+.2f} (R^2 {r2_sec:.2f}) "
      f"on {len(kept)}/15 after filter  [submission: +2.12, R^2 0.46]")
print(f"same runs, peak-rate estimator    : a = {a_pk:+.2f} (R^2 {r2_pk:.2f}) on 15/15")
print("filtered:", [r["label"] for r in rows if r["filtered"]])
