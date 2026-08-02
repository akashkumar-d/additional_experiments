"""
Shared analysis for the item-2 notebooks (positive configs / NS x minibatch /
NS x minibatch x momentum).

Conventions fixed here so all three notebooks agree:

* Loss is always the CYCLE-MEAN  Lbar_t = (L_t + L_{t+1})/2, which removes the
  intended odd/even period-2 oscillation. Raw consecutive alternation is the
  phenomenon, not instability, and must never be counted as roughness.

* Cycle quality is judged on the PARAMETER ORBIT, not on the loss. rho_2 on the
  loss reaches 1.0000 at beta >= 0.5 while the iterate is not on a period-2
  orbit at all. With momentum the state is (W, buf), so both
  `r2_update` (W-recurrence) and `r2_buf` (buffer recurrence) matter.

* Decoupling is measured from each run's OWN plateau onset, never a fixed
  window. A fixed window penalises fast learners -- the same bias that got
  Figure 3(d) flagged in review. `windowed_orbit` additionally checks for
  metastability: beta=0.30 looks clean on [500, 3000] and breaks by step 8000.

Metric glossary
---------------
dL_pct     change in cycle-mean loss over the window (>= -5% = "not descending")
drift      mean cos^2 gain over the window
R2_W       median ||u_{t-1}+u_t|| / (0.5(||u_{t-1}||+||u_t||))   -> 0 = tight orbit
R2_buf     median ||buf_t-buf_{t-2}|| / (0.5(...))               -> 0 = 2-periodic buffer
J_L        100 * rms(Lbar_{t+2}-Lbar_t) / median(Lbar)           same-phase roughness
drawdown   max peak-to-trough drop of mean cos^2 in the window
path_eff   (A_end - A_start) / sum |dA|                          1 = monotone
disp_rel   mean ||W_{t+1}-W_t||_F / (eta sqrt(M))                1 = polar identity holds
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

# Capability marker. Notebooks check this with getattr(..., set()) so an older
# copy of this file degrades to a clear warning instead of an ImportError or a
# silent wrong result. Bump when adding anything a notebook relies on.
FEATURES = {"mc500", "R2_buf", "facet_plot", "disp_matched_eta"}

# thresholds used for tiering (descriptive, not claimed constants)
TIER_A = dict(R2_W=0.10, drawdown=0.05, J_L=0.10, path_eff=0.70)
TIER_B = dict(R2_W=0.20, drawdown=0.15, J_L=1.00, path_eff=0.50)


# ---------------------------------------------------------------- loading
def load_traj(path):
    """Unified view over our runner schema and the follow-up bundle schema.
    Returns None (with a message) if the file is unreadable."""
    try:
        z = np.load(path, allow_pickle=True)
        for k in z.files:
            _ = z[k]
    except Exception as e:                                  # noqa: BLE001
        print(f"[skip corrupt] {Path(path).name}: {type(e).__name__}")
        return None
    f = z.files
    ours = "mean_cos2_AGOP" in f
    L = np.asarray(z["L_full_dense"] if "L_full_dense" in f else z["loss_full"], float)
    s = np.asarray(z["step"] if ours else z["log_step"], float)
    mc = np.asarray(z["mean_cos2_AGOP"] if ours else z["mean_cos2"], float)
    cm = np.asarray(z["cos2_min_AGOP"] if ours else z["min_cos2"], float)
    if len(s) > 1:                                          # drop chunk-boundary dupes
        keep = np.concatenate([[True], np.diff(s) > 0])
        s, mc, cm = s[keep], mc[keep], cm[keep]
    get = lambda *names: next((np.asarray(z[n], float) for n in names if n in f), None)
    return dict(L=L, s=s, mc=mc, cm=cm,
                r2=get("r2_update_dense", "r2_update"),
                rev=get("rev_cos_dense", "rev_cos"),
                r2buf=get("r2_buf_dense"),
                disp=get("disp_dense"),
                LV=get("L_V_opt_train"),
                z=z)


def _despike(a, k=5):
    """Running median -- removes single-point metric jitter, keeps the trend."""
    a = np.asarray(a, float)
    if len(a) < k or k < 3:
        return a
    pad = k // 2
    ap = np.concatenate([np.full(pad, a[0]), a, np.full(k - 1 - pad, a[-1])])
    return np.array([np.median(ap[i:i + k]) for i in range(len(a))])


def cycle_mean(L):
    """Lbar_t = (L_t + L_{t+1})/2 -- removes the intended parity oscillation."""
    L = np.asarray(L, float)
    return 0.5 * (L[:-1] + L[1:])


# ---------------------------------------------------------------- windows
def plateau_onset(L, win=500, tol=0.02):
    """First step from which the cycle-mean loss never again falls faster than
    `tol` per `win` steps. None if the loss is still descending at the end."""
    Lp = cycle_mean(L)
    T = len(Lp)
    if T <= win + 5:
        return None
    ts = np.arange(win, T)
    ok = (Lp[ts] - Lp[ts - win]) / np.maximum(Lp[ts - win], 1e-30) >= -tol
    idx, run_ok = None, True
    for i in range(len(ok) - 1, -1, -1):
        if ok[i] and run_ok:
            idx = i
        else:
            run_ok = False
    return int(ts[idx]) if idx is not None else None


def windowed_orbit(d, nwin=8):
    """R2_W in `nwin` equal windows -- catches metastability (an orbit that is
    tight early and breaks late). Returns (edges, values)."""
    r2 = d["r2"]
    if r2 is None:
        return np.array([]), np.array([])
    T = len(r2)
    e = np.linspace(0, T, nwin + 1).astype(int)
    v = [float(np.nanmedian(r2[e[i]:e[i + 1]])) if e[i + 1] > e[i] else np.nan
         for i in range(nwin)]
    return e, np.array(v)


def metastable(d, nwin=8, tight=0.10, factor=3.0):
    """True if the orbit starts tight and degrades by >= `factor` -- i.e. a
    short measurement window would have called it clean. (beta=0.30 case.)"""
    _, v = windowed_orbit(d, nwin)
    if len(v) < 4 or not np.isfinite(v[:2]).all():
        return False
    early, late = float(np.nanmedian(v[:2])), float(np.nanmedian(v[-2:]))
    return bool(early < tight and late > max(factor * early, tight))


# ---------------------------------------------------------------- scoring
def score(path, win_lo=None, eta=None, M=None):
    """Full trajectory-quality record. Window = [plateau onset (or win_lo), T]."""
    d = load_traj(path)
    if d is None:
        return None
    L, s, mc, cm = d["L"], d["s"], d["mc"], d["cm"]
    T = len(L) - 1
    tp = plateau_onset(L)
    lo = win_lo if win_lo is not None else (tp if tp is not None else 500)
    lo = min(lo, max(T - 500, 0))

    Lp = cycle_mean(L)
    w = Lp[lo:]
    same_phase = w[2:] - w[:-2] if len(w) > 2 else np.array([0.0])
    m = s >= lo
    mw = mc[m] if m.sum() >= 3 else mc[-3:]
    cw = cm[m] if m.sum() >= 3 else cm[-3:]
    # Drawdown describes the SECULAR shape of the alignment curve, so de-spike
    # first. The raw logged curve carries AGOP eigenvector-swap jitter (the same
    # flicker that makes cos2_min unreliable at a single step); a one-point dip
    # is a metric artefact, not the run backtracking. Without this a condition
    # with R2_W = 0.018 and J_L = 0.003% -- essentially perfect -- can be tiered
    # "C unstable" by a single spike. `drawdown_raw` keeps the unfiltered value.
    mw_s = _despike(mw)
    run = np.maximum.accumulate(mw_s)
    # Path efficiency = net gain / total variation. Computed on the raw logged
    # points this measures LOGGING DENSITY, not trajectory quality: at 400 log
    # points even 2e-3 of jitter inflates TV enough to drive path_eff from 1.00
    # to 0.38 on a perfectly monotone curve. Resample onto a fixed grid first so
    # the statistic is invariant to log_every and run length.
    _NPE = 50
    if len(mw_s) > _NPE:
        _g = np.interp(np.linspace(0, 1, _NPE), np.linspace(0, 1, len(mw_s)), mw_s)
    else:
        _g = mw_s
    tv = float(np.sum(np.abs(np.diff(_g))))
    def late(a):
        """Median over the last half. An all-NaN tail is a legitimate state --
        r2_buf is NaN when momentum is 0, for instance -- so return NaN quietly
        rather than letting np.nanmedian emit an All-NaN RuntimeWarning for
        every such run."""
        if a is None or not len(a):
            return float("nan")
        tail = np.asarray(a[int(0.5 * len(a)):], dtype=float)
        tail = tail[np.isfinite(tail)]
        return float(np.median(tail)) if len(tail) else float("nan")

    # alignment headroom at t=500: the cheap screening statistic. Empirically
    # monotone against remaining drift (mc500 ~ 0.45 -> drift ~ 0.4;
    # mc500 ~ 0.72 -> drift ~ 0.11), so target bases with mc500 ~ 0.45-0.55.
    t_head = min(500.0, float(s[-1]))
    rec = dict(
        T=T, t_plateau=tp if tp is not None else float("nan"), win_lo=lo,
        mc500=float(np.interp(t_head, s, mc)),
        dL_pct=float((w[-1] - w[0]) / max(abs(w[0]), 1e-30) * 100),
        drift=float(mw[-1] - mw[0]), mc_end=float(mw[-1]),
        drift_min=float(cw[-1] - cw[0]), cm_end=float(cw[-1]),
        R2_W=late(d["r2"]), rev_cos=late(d["rev"]), R2_buf=late(d["r2buf"]),
        J_L=float(100 * math.sqrt(float(np.mean(same_phase ** 2)))
                  / max(float(np.median(w)), 1e-30)),
        drawdown=float(np.max(run - mw_s)),
        drawdown_raw=float(np.max(np.maximum.accumulate(mw) - mw)),
        path_eff=float((_g[-1] - _g[0]) / tv) if tv > 1e-12 else float("nan"),
        metastable=metastable(d),
    )
    rec["ratio"] = (float(np.mean(L[int(0.7 * len(L)):])
                          / max(np.mean(d["LV"][int(0.7 * len(d["LV"])):]), 1e-12))
                    if d["LV"] is not None else float("nan"))
    rec["disp_rel"] = (float(np.mean(d["disp"]) / (eta * math.sqrt(M)))
                       if (d["disp"] is not None and eta and M) else float("nan"))
    return rec


def tier(e):
    """A = smooth + tight orbit, B = acceptable, C = unstable. Independent of
    whether the run SHOWS decoupling -- see `works`.

    Accepts either a per-run record (`metastable`: bool) or an aggregated one
    (`n_meta`: count over seeds). Aggregated rows previously carried only
    `n_meta`, so the metastability branch silently never fired and e.g. a
    beta=0.30 condition with 3/3 metastable seeds was labelled "B ok".
    """
    if not np.isfinite(e["R2_W"]):
        return "B? (no orbit data)"
    if e.get("metastable") or e.get("n_meta", 0):
        return "C metastable"
    ok = lambda th: (e["R2_W"] < th["R2_W"] and e["drawdown"] < th["drawdown"]
                     and e["J_L"] < th["J_L"]
                     and (not np.isfinite(e["path_eff"]) or e["path_eff"] > th["path_eff"]))
    if ok(TIER_A):
        return "A stable"
    if ok(TIER_B):
        return "B ok"
    return "C unstable"


def works(e, min_drift=0.05, max_drop=-5.0):
    """Temporal overlap: loss not meaningfully descending AND alignment still
    improving, both on the run-adaptive window."""
    return bool(e["drift"] > min_drift and e["dL_pct"] > max_drop)


# ---------------------------------------------------------------- eta calibration
def disp_matched_eta(path, eta_pilot, M, eta_target=None):
    """NS changes the step length, so a raw-eta comparison confounds spectral
    distortion with a smaller effective learning rate. This returns the eta that
    restores the exact-polar displacement eta_target*sqrt(M):

        eta_k = eta_target * sqrt(M) / median||NS_k(H_t)||_F

    Read from a short pilot run's `disp_dense` (which stores eta*||u||_F)."""
    d = load_traj(path)
    if d is None or d["disp"] is None:
        return None
    q = int(0.5 * len(d["disp"]))
    disp_rel = float(np.median(d["disp"][q:])) / (eta_pilot * math.sqrt(M))
    return (eta_target or eta_pilot) / max(disp_rel, 1e-9), disp_rel


# ---------------------------------------------------------------- plotting
_FIELDS = ("ns", "B", "eta", "beta")
_PRETTY = {"ns": "NS", "B": "B", "eta": r"$\eta$", "beta": r"$\beta$"}


def _fmt(k, v):
    if k == "ns":  return f"NS{v}" if v else "exact"
    if k == "B":   return f"B={'full' if v is None else v}"
    return f"{_PRETTY[k]}={v:g}"


def _varying(rows):
    return [k for k in _FIELDS if len({r[k] for r in rows}) > 1]


def facet_plot(OUT, run_table, seeds, ckpt_name, facet_by="ns", ref_label=None,
               show_seeds=False, groups=None, title="", savepath=None,
               figsize_per_panel=(5.4, 3.4)):
    """Readable multi-panel trajectories.

    Plotting every condition on one axis is unreadable past ~5 curves, so by
    default this facets into one panel per distinct value of `facet_by`, colours
    by whatever else varies *within* that panel, and labels each curve with only
    the varying fields. `ref_label` draws one condition grey-dashed in every
    panel as a common baseline. Set `groups=[(title, [labels]), ...]` for full
    manual control, or `facet_by=None` for a single panel.
    """
    import matplotlib.pyplot as plt

    by_label = {r["label"]: r for r in run_table}
    ref = by_label.get(ref_label) if ref_label else None
    body = [r for r in run_table if r is not ref]

    if groups is None:
        if facet_by is None:
            groups = [("all conditions", [r["label"] for r in body])]
        else:
            vals, seen = [], set()
            for r in body:                              # keep table order
                if r[facet_by] not in seen:
                    seen.add(r[facet_by]); vals.append(r[facet_by])
            groups = [(_fmt(facet_by, v),
                       [r["label"] for r in body if r[facet_by] == v]) for v in vals]

    ncol = max(len(groups), 1)
    fig, axes = plt.subplots(3, ncol, squeeze=False,
                             figsize=(figsize_per_panel[0] * ncol,
                                      figsize_per_panel[1] * 3))
    palette = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set2").colors)

    for col, (gtitle, labels) in enumerate(groups):
        rows = [by_label[L] for L in labels if L in by_label]
        vary = _varying(rows) or ["label"]
        if ref is not None:                             # common baseline first
            files = [OUT / ckpt_name(ref, sd) for sd in seeds
                     if (OUT / ckpt_name(ref, sd)).exists()]
            if files:
                for k, key in enumerate(("Lbar", "mc", "cm")):
                    seed_band(axes[k][col], files, key, color="0.35", lw_seed=0,
                              alpha=0, lw_med=1.4,
                              label=f"{ref['label']} (ref)" if k == 0 else None)
                    axes[k][col].lines[-1].set_linestyle("--")
        for j, r in enumerate(rows):
            files = [OUT / ckpt_name(r, sd) for sd in seeds
                     if (OUT / ckpt_name(r, sd)).exists()]
            if not files:
                continue
            c = palette[j % len(palette)]
            lab = (r["label"] if vary == ["label"]
                   else "  ".join(_fmt(k, r[k]) for k in vary))
            sk, al = (0.6, 0.28) if show_seeds else (0, 0)
            seed_band(axes[0][col], files, "Lbar", color=c, label=lab, lw_seed=sk, alpha=al)
            seed_band(axes[1][col], files, "mc",   color=c, lw_seed=sk, alpha=al)
            seed_band(axes[2][col], files, "cm",   color=c, lw_seed=sk, alpha=al)
        axes[0][col].set_yscale("log")
        axes[0][col].set_title(gtitle, fontsize=11)
        axes[0][col].legend(fontsize=7.5, loc="best")
        for a in axes[:, col]:
            a.grid(alpha=0.3); a.axvline(500, ls="--", c="grey", lw=0.8)
        for a in axes[1:, col]:
            a.set_ylim(-0.02, 1.02)
        axes[2][col].set_xlabel("step")
    axes[0][0].set_ylabel(r"cycle-mean loss $\bar L_t$")
    axes[1][0].set_ylabel(r"mean $\cos^2$")
    axes[2][0].set_ylabel(r"weakest direction $\cos^2_{\min}$")
    if title:
        fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=140)
    return fig


def seed_band(ax, runs, key, loader=load_traj, xkey=None, color="tab:blue",
              label=None, lw_med=2.2, lw_seed=0.7, alpha=0.35):
    """Faint per-seed curves + heavy pointwise median on a common grid."""
    curves = []
    for p in runs:
        d = loader(p)
        if d is None:
            continue
        if key == "Lbar":
            y = cycle_mean(d["L"]); x = np.arange(len(y), dtype=float)
        else:
            y = d[key]; x = d["s"]
        ax.plot(x, y, color=color, lw=lw_seed, alpha=alpha)
        curves.append((x, y))
    if not curves:
        return None
    grid = curves[0][0]
    for x, _ in curves[1:]:
        if len(x) < len(grid):
            grid = x
    Y = np.vstack([np.interp(grid, x, y) for x, y in curves])
    med = np.median(Y, axis=0)
    ax.plot(grid, med, color=color, lw=lw_med, label=label)
    return grid, med
