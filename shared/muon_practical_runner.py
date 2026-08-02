"""
Practical-Muon runner for meta-review Item 2: does the plateau-with-
feature-learning phenomenon survive minibatch noise, Newton-Schulz
orthogonalization, and momentum?

Extends the paper's full-batch exact-polar runner (teacher_full_runner)
with the three ingredients of practical Muon:

  1. Minibatch gradients: batch_size B <= n, fresh uniform sample per step.
  2. Newton-Schulz orthogonalization: the quintic iteration used by the
     Muon reference implementation (Jordan et al.), ns_steps iterations,
     instead of the exact SVD polar factor. ns_steps=0 means exact polar.
  3. Momentum: standard Muon momentum buffer with the nesterov-style
     update from the reference implementation,
         buf   <- beta * buf + g
         g_eff <- g + beta * buf     (nesterov=True)
         W     <- W - eta * Orth(g_eff)
     beta=0 recovers the memoryless update.

Logging matches teacher_full_runner: sparse diagnostics every `log_every`
steps (alignment metrics, oracles) + dense per-step traces. For minibatch
runs the dense loss trace `L_full_dense` is ALWAYS the full-batch training
loss (forward pass on all n points each step), so the period-2 signature
rho_2 is measured on the deterministic loss even when updates are noisy.
`disp_dense` logs the per-step Frobenius displacement ||W_{t+1}-W_t||_F,
which for exact polar equals eta*sqrt(M) exactly and for NS/momentum
quantifies how far practical Muon deviates from the displacement identity.

Checkpoint fields additionally record the practical-Muon settings and the
momentum buffer so runs resume exactly.
"""
from __future__ import annotations

import math
import os
import time
import zipfile
from pathlib import Path

import numpy as np

import sys
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from teacher_activation_sweep import make_problem, init_W                 # noqa: E402

# ---------------------------------------------------------------------------
# Minibatch sampling.
# Standard practice (epoch shuffling) draws WITHOUT replacement, and that is
# what the item-2 follow-up study used. Drawing WITH replacement at B close to
# n is far noisier: gradient-noise variance scales as 1/B (with replacement)
# vs (1/B)(1 - B/n) (without), a 150x ratio at B=14900, n=15000 -- measured
# 12x larger relative gradient error. Runs before 2026-07-25 used
# with-replacement and are labelled `cfg_batch_mode='with_replacement'`
# (or carry no such key); their effective noise is much higher than their
# nominal B suggests. Default is now 'without_replacement'.
# ---------------------------------------------------------------------------
def _sample_idx(rng, n, B, mode="without_replacement"):
    if mode == "with_replacement":
        return rng.integers(0, n, size=B)
    if mode == "without_replacement":
        return rng.choice(n, size=B, replace=False)
    if mode == "paired_without_replacement":
        raise RuntimeError("paired mode is handled in the training loop")
    raise ValueError(f"unknown batch_mode {mode!r}")
# `paired_without_replacement` is handled in the training loop rather than here,
# because it needs the step index: the SAME batch is used on steps 2k and 2k+1.
# Rationale -- a period-2 orbit requires the same objective on both phases of the
# cycle. With a fresh batch every step the objective changes each step, so the
# orbit can never close no matter how small the noise. Pairing removes that
# obstruction specifically, without reducing the noise level. The batch is drawn
# from an RNG seeded by (seed, t//2), so it is deterministic and resume-safe.




# ---------------------------------------------------------------------------
# Crash-safe checkpoint IO.
# Plain np.savez writes the zip in place, so a mid-save kill (walltime limit,
# OOM, crash) leaves a truncated file => "BadZipFile: Bad CRC-32" on the next
# load. Fix: (1) save to tmp + atomic rename, so a visible .npz is always
# complete; (2) verify every entry on load, quarantining unreadable files as
# *.corrupt so the run restarts from scratch instead of crashing.
# Stale *.tmp<pid> files (from killed jobs) are harmless; delete freely.
# ---------------------------------------------------------------------------
_CKPT_ERRORS = (zipfile.BadZipFile, KeyError, EOFError, OSError, ValueError)


def _atomic_savez(path, **arrs):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "wb") as f:        # file handle => numpy won't append .npz
        np.savez(f, **arrs)
    os.replace(tmp, path)


def _load_ckpt(path):
    """Fully-verified np.load; returns None (file -> *.corrupt) if unreadable."""
    try:
        z = np.load(path, allow_pickle=True)
        for k in z.files:
            _ = z[k]                  # CRC errors only surface on entry read
        return z
    except _CKPT_ERRORS as e:
        try:
            path.rename(path.with_name(path.name + ".corrupt"))
        except OSError:
            pass
        print(f"[corrupt] {path.name}: {type(e).__name__}: {e} "
              f"-> quarantined, run restarts fresh")
        return None
from teacher_full_runner import (                                          # noqa: E402
    loss_only, loss_and_grad, alignment_metrics,
    loss_V_optimal, loss_AGOP_optimal,
)

# ------------------------------------------------------------------
# Newton-Schulz orthogonalization (Muon reference quintic)
# ------------------------------------------------------------------

_NS_COEFFS = (3.4445, -4.7750, 2.0315)

# Two odd-polynomial iterations, both of the form  X <- aX + (bA + cA^2)X.
# They differ qualitatively, and the difference is the whole story for item 2:
#
#   'tuned'      Muon's reference quintic. phi(1) = 0.7010, phi'(1) = -0.7230,
#                so 1 is NOT a fixed point: the iteration never converges, it
#                oscillates in ~[0.69, 1.19] forever. Its singular-value spread
#                is therefore bounded away from zero at EVERY depth
#                (sv-CV ~ 0.16 at 5 passes, 0.022 at 8). Fast, deliberately
#                inexact -- which is the right trade for optimisation, but it
#                means the period-2 orbit can never close.
#
#   'convergent' phi(s) = 2s - 1.5s^3 + 0.5s^5. phi(1) = 1 and phi'(1) = 0, a
#                SUPERATTRACTING fixed point, so it converges quadratically and
#                reaches machine precision by ~6 passes on a normalised
#                gradient. At >= 6 passes it IS the exact polar factor, and
#                depth beyond that changes trajectories less than the seed does.
NS_COEFF_SETS = {
    "tuned":      (3.4445, -4.7750, 2.0315),
    "convergent": (2.0, -1.5, 0.5),
}

# 'hybrid' -- the repair for the tuned map, at unchanged cost.
#
# The two maps fail and succeed for opposite reasons. The tuned quintic lifts
# SMALL singular values fast (phi(0.01) -> 0.699 in 5 passes, vs 0.076 for the
# classical cubic) but has no fixed point at 1, so it oscillates forever and its
# sv-CV never drops below ~0.05. The convergent map is slow to lift small
# singular values but converges quadratically once they are near 1.
#
# Using each for what it is good at -- a few tuned passes to reach the basin,
# then convergent passes to polish -- reaches machine-level uniformity within
# Muon's own 5-pass budget (well-conditioned gradient, M=30):
#
#     5 tuned            sv-CV 1.4e-01     (fails; tolerance is ~2e-4)
#     5 convergent       sv-CV 1.3e-05     (marginal)
#     1 tuned + 4 conv   sv-CV 1.5e-08     (passes with 3 orders to spare)
#     2 tuned + 3 conv   sv-CV 5.8e-07     (passes)
#
# CAVEAT: the required depth grows with the gradient's dynamic range. At
# M=100 with sigma spanning [0.01, 1], NO 5-pass schedule reaches tolerance.
# Use `ns_tol` for that case -- adaptive depth is the robust answer.
NS_HYBRID_LEAD = 1          # tuned passes before switching to convergent


# Capability marker. Notebooks check this BEFORE launching so a stale module in
# the kernel fails immediately with an instruction, instead of 180 jobs later
# with a bare KeyError from a worker process. (fork() inherits whatever the
# parent already imported, so re-uploading the file is not enough -- the kernel
# must be restarted.)
FEATURES = {"hybrid", "ns_tol", "shape_cv", "paired_batches", "log_approx",
            "r2_buf", "ns_coeff", "two_phase"}


def ns_schedule(steps, coeff):
    """Coefficient sequence for `steps` passes."""
    if coeff != "hybrid":
        if coeff not in NS_COEFF_SETS:
            raise ValueError(
                f"unknown ns_coeff {coeff!r}; known: "
                f"{sorted(NS_COEFF_SETS) + ['hybrid']}. If you expected this to "
                f"work, the module in memory is stale -- RESTART THE KERNEL.")
        return [NS_COEFF_SETS[coeff]] * steps
    lead = min(NS_HYBRID_LEAD, steps)
    return ([NS_COEFF_SETS["tuned"]] * lead
            + [NS_COEFF_SETS["convergent"]] * (steps - lead))


def newton_schulz(G: np.ndarray, steps: int = 5, eps: float = 1e-7,
                  coeff: str = "tuned", ns_tol: float = 0.0,
                  ns_max: int = 20) -> np.ndarray:
    """Quintic Newton-Schulz iteration approximating Polar(G) = U V^T.

    Follows the Muon reference implementation: normalise by the Frobenius
    norm so all singular values are <= 1, then iterate
        X <- a X + (b A + c A^2) X,   A = X X^T
    with (a, b, c) = (3.4445, -4.7750, 2.0315). After ~5 iterations the
    singular values lie in roughly [0.7, 1.2] -- deliberately NOT exactly 1,
    matching what practical Muon actually applies.

    Works on the smaller Gram side for efficiency (transpose if rows > cols).
    steps=0 returns the exact polar factor via SVD (baseline).
    """
    if steps <= 0:
        U, _, Vh = np.linalg.svd(G, full_matrices=False)
        return U @ Vh
    X = G.astype(np.float64)
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    X = X / (np.linalg.norm(X) + eps)

    if ns_tol > 0:
        # Adaptive depth. Stop when the update is orthogonal enough, measured by
        #     ||X X^T - I||_F / sqrt(M)
        # which costs one matmul (no SVD) and runs ~2x the singular-value CV, so
        # ns_tol = 2e-5 targets sv-CV ~ 1e-5 -- inside the measured safe region.
        # This is the robust option: the passes needed grow with the gradient's
        # dynamic range, so a fixed budget cannot be right for every step.
        M = X.shape[0]
        seq = ns_schedule(ns_max, coeff)
        for a, b, c in seq:
            A = X @ X.T
            if float(np.linalg.norm(A - np.eye(M))) / math.sqrt(M) <= ns_tol:
                break
            X = a * X + (b * A + c * (A @ A)) @ X
        return X.T if transposed else X

    for a, b, c in ns_schedule(steps, coeff):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    return X.T if transposed else X




# ---------------------------------------------------------------------------
# Controlled orthogonalisation SHAPE error.
#
# Every NS variant confounds three things: how many matmuls it costs, what step
# length it produces, and how uniform the resulting singular values are. The
# last one is what the theory cares about (the analysis needs all singular
# values equal to 1). `shape_cv` isolates it: start from the EXACT polar factor
# and impose a prescribed coefficient of variation on its singular values,
#
#     u = U diag(s) V^T,   s_i = 1 + c (i/(M-1) - 1/2),   std(s)/mean(s) = cv
#
# The pattern is systematic (a smooth ramp across modes), not random, because
# that is what an NS iteration actually produces -- a deterministic function of
# the input singular value. Step length is renormalised to eta*sqrt(M) so the
# dial changes ONLY shape, never scale.
#
# Reference points measured on the R5 base: Muon's tuned quintic gives
# cv ~ 0.16 at 5 passes and ~0.022 at 8; a convergent iteration reaches
# cv < 1e-12 by 6 passes. Runs with cv = 0 are bit-identical to exact polar.
# ---------------------------------------------------------------------------
def shape_perturb(u, cv, renormalise=True):
    """Impose singular-value CV `cv` on an already-orthogonalised update."""
    if cv <= 0:
        return u
    U, S, Vh = np.linalg.svd(u, full_matrices=False)
    M = len(S)
    if M < 2:
        return u
    ramp = np.linspace(-0.5, 0.5, M)
    ramp = ramp / (ramp.std() + 1e-30)          # unit std
    s = 1.0 + cv * ramp
    s = np.clip(s, 1e-6, None)
    out = (U * s) @ Vh
    if renormalise:                              # keep ||u||_F = sqrt(M)
        out *= math.sqrt(M) / (np.linalg.norm(out) + 1e-30)
    return out


# ------------------------------------------------------------------
# Practical-Muon trainer
# ------------------------------------------------------------------

KEYS_SCALAR = ("step", "L_full", "L_test", "L_V_opt_train", "L_V_opt_test",
               "L_AGOP_opt_train", "L_AGOP_opt_test",
               "in_V_frac", "mean_cos2_AGOP", "cos2_min_AGOP",
               "mass_in_V_AGOP_p", "thr50_l2", "pr_eff_rank")

# PHASE-2 alignment (log_both_phases=True).
#
# `log_every` is even, so the sparse diagnostics land on steps 0, 20, 40, ...
# -- every one of them the SAME parity of the period-2 cycle. A claim that
# "features improve inside the cycle" is therefore only ever verified on one of
# the two phases, and a run could in principle pass because the sampled parity
# happens to align while the other does not. These keys record the same metrics
# one step later (t+1, the opposite phase) in separate arrays, so the primary
# traces and all existing analysis are unchanged.
KEYS_PHASE1 = ("step_p1", "mean_cos2_AGOP_p1", "cos2_min_AGOP_p1", "L_full_p1")

KEYS_DENSE = ("step_dense", "L_full_dense", "L_batch_dense", "disp_dense",
              "rev_cos_dense", "r2_update_dense", "r2_buf_dense",
              "approx_cv_dense", "approx_cos_dense", "approx_err_dense")

# ---------------------------------------------------------------------------
# APPROXIMATION QUALITY ALONG THE TRAJECTORY (log_approx=True).
#
# A single sv-CV measured statically on one gradient is NOT a reliable proxy
# for the error a run actually experiences: the gradient spectrum changes along
# the trajectory, and an NS run's trajectory diverges from the exact-polar one
# it was calibrated against. Two wrong conclusions came out of relying on the
# static number:
#   * "a 5e-7 shape error destroys the orbit" -- unfounded;
#   * "no, it is the 11% displacement deficit" -- falsified, because exact
#     polar at the SAME effective step (eta 1.34) is perfectly stable
#     (dL +4.3%, R2_W 0.017) while convergent NS5 at eta 1.50 fails
#     (dL -29.1%, R2_W 0.486), and matching eta upward does not rescue it.
# So log it per step instead. With Q the applied update and P = Polar(g_eff):
#   approx_cv  = std(sigma(Q))/mean(sigma(Q))     shape error, 0 = uniform
#   approx_cos = <Q,P>/(|Q||P|)                   direction agreement
#   approx_err = |Q - P|_F / |P|_F                total relative error
# Costs one extra SVD per step, so it is opt-in.
# ---------------------------------------------------------------------------

# NOTE (added after the follow-up study): rho_2 on the LOSS is not a
# sufficient cycle diagnostic. At beta >= 0.5 the loss alternates with
# rho_2 = 1.0000 while the parameter iterate is NOT on a period-2 orbit.
# These two per-step scalars measure the orbit directly, with
# u_t := the applied (orthogonalized) update at step t:
#   rev_cos_t   = -<u_{t-1}, u_t> / (|u_{t-1}| |u_t|)     -> 1 for exact reversal
#   r2_update_t = |u_{t-1} + u_t| / (0.5(|u_{t-1}|+|u_t|)) -> 0 for exact period-2
# With momentum the optimizer state is (W, buf), so W_{t+2} = W_t is NOT
# sufficient for a period-2 orbit -- the buffer must also return. Hence
#   r2_buf_t = |buf_t - buf_{t-2}| / (0.5(|buf_t|+|buf_{t-2}|))  -> 0 for
# 2-periodic buffer. NaN when momentum = 0 (no buffer).
# Reference values (ReLU r_t=8 r_s=30 eta=1.5): exact polar 1.000/0.017;
# NS-5 0.94/0.36 (broadened orbit); momentum beta=0.95 0.45/0.92 (no orbit).


def run_one_practical(teacher_act, p, r_t, r_s, n, eta, seed, n_steps,
                      log_every, ckpt_path: Path, chunk_steps, budget_s,
                      batch_size=None, ns_steps=0, momentum=0.0, batch_mode="without_replacement", shape_cv=0.0, ns_coeff="tuned", ns_tol=0.0,
                      log_approx=False, log_both_phases=False,
                      nesterov=True):
    """One practical-Muon run. batch_size=None => full batch;
    ns_steps=0 => exact polar; momentum=0 => no buffer."""
    X, y, X_te, y_te, U_t, a_t = make_problem(p, r_t, n, n_test=5000,
                                              seed=seed, teacher_act=teacher_act)
    a_s = (1.0 / math.sqrt(p)) * np.ones(r_s)
    P_V = U_t @ U_t.T
    B = int(batch_size) if batch_size else n
    B = min(B, n)
    full_batch = B >= n
    # Stream for minibatch sampling: independent of the init rng, seeded so
    # runs are reproducible and resumable (reseed by (seed, start step)).
    tag = (f"B={'full' if full_batch else B} ns={ns_steps if ns_steps > 0 else 'exact'} "
           f"m={momentum}")

    z = _load_ckpt(ckpt_path) if ckpt_path.exists() else None
    if z is not None:
        if int(z["next_t"]) >= n_steps:
            print(f"[skip] {ckpt_path.name} at {int(z['next_t'])}")
            return False
        W = z["W"].copy()
        buf = z["mom_buf"].copy() if "mom_buf" in z.files else np.zeros_like(W)
        log = {k: list(z[k]) for k in KEYS_SCALAR}
        cos_per_log = list(z["cos_per"])
        per_eigvec_log = list(z["per_eigvec_in_V"])
        top_lams_log = list(z["top_lams"])
        dense = {k: list(z[k]) if k in z.files else [] for k in KEYS_DENSE}
        ph1 = {k: list(z[k]) if k in z.files else [] for k in KEYS_PHASE1}
        # exact cross-chunk continuity for the orbit diagnostics
        prev_update = z["last_update"].copy() if "last_update" in z.files else None
        buf_hist = [z[k].copy() for k in ("buf_m2", "buf_m1") if k in z.files]
        start_t = int(z["next_t"])
        print(f"[resume] {ckpt_path.name}: step {start_t} ({tag})")
    else:
        rng0 = np.random.default_rng(seed + 17)
        W0, ia = init_W(p, r_s, U_t, rng=rng0)
        print(f"[init {teacher_act} {tag}] {ckpt_path.name}: in-V={ia:.3f}")
        W = W0.copy()
        buf = np.zeros_like(W)
        log = {k: [] for k in KEYS_SCALAR}
        cos_per_log = []; per_eigvec_log = []; top_lams_log = []
        dense = {k: [] for k in KEYS_DENSE}
        ph1 = {k: [] for k in KEYS_PHASE1}
        prev_update = None
        buf_hist = []
        start_t = 0

    batch_rng = np.random.default_rng((seed + 1) * 100003 + start_t)

    end_t = min(start_t + chunk_steps, n_steps)
    t_start = time.time()
    for t in range(start_t, end_t + 1):
        if time.time() - t_start > budget_s:
            print(f"[budget] stop at {t}"); end_t = t; break

        # ---- gradient (mini or full batch) ----
        if full_batch:
            L_batch, gW = loss_and_grad(W, X, y, a_s)
            L_full = L_batch
        else:
            if batch_mode == "paired_without_replacement":
                _pr = np.random.default_rng((seed + 1) * 100003 + (t // 2))
                idx = _pr.choice(n, size=B, replace=False)
            else:
                idx = _sample_idx(batch_rng, n, B, batch_mode)
            L_batch, gW = loss_and_grad(W, X[idx], y[idx], a_s)
            L_full = loss_only(W, X, y, a_s)      # deterministic trace for rho_2

        # ---- momentum (Muon reference: nesterov-style) ----
        if momentum > 0.0:
            buf = momentum * buf + gW
            g_eff = gW + momentum * buf if nesterov else buf
        else:
            g_eff = gW

        # ---- orthogonalization ----
        update = newton_schulz(g_eff, steps=ns_steps, coeff=ns_coeff, ns_tol=ns_tol)
        if shape_cv > 0:
            update = shape_perturb(update, shape_cv)

        if t < end_t:
            dense["step_dense"].append(t)
            dense["L_full_dense"].append(L_full)
            dense["L_batch_dense"].append(L_batch)
            dense["disp_dense"].append(float(eta * np.linalg.norm(update)))
            # parameter-space period-2 diagnostics (see KEYS_DENSE note)
            if prev_update is None:
                dense["rev_cos_dense"].append(float("nan"))
                dense["r2_update_dense"].append(float("nan"))
            else:
                n1 = float(np.linalg.norm(prev_update))
                n2 = float(np.linalg.norm(update))
                dense["rev_cos_dense"].append(
                    float(-np.sum(prev_update * update) / (n1 * n2 + 1e-30)))
                dense["r2_update_dense"].append(
                    float(np.linalg.norm(prev_update + update) / (0.5 * (n1 + n2) + 1e-30)))
            if log_approx and (ns_steps > 0 or shape_cv > 0):
                Up, _, Vhp = np.linalg.svd(g_eff, full_matrices=False)
                P = Up @ Vhp
                sv = np.linalg.svd(update, compute_uv=False)
                nQ, nP = np.linalg.norm(update), np.linalg.norm(P)
                dense["approx_cv_dense"].append(float(sv.std() / (sv.mean() + 1e-30)))
                dense["approx_cos_dense"].append(
                    float(np.sum(update * P) / (nQ * nP + 1e-30)))
                dense["approx_err_dense"].append(float(np.linalg.norm(update - P) / (nP + 1e-30)))
            else:
                for _k in ("approx_cv_dense", "approx_cos_dense", "approx_err_dense"):
                    dense[_k].append(float("nan"))
            prev_update = update
            if momentum > 0.0:
                if len(buf_hist) >= 2:
                    d = buf - buf_hist[-2]
                    dn = 0.5 * (np.linalg.norm(buf) + np.linalg.norm(buf_hist[-2]))
                    dense["r2_buf_dense"].append(float(np.linalg.norm(d) / (dn + 1e-30)))
                else:
                    dense["r2_buf_dense"].append(float("nan"))
                buf_hist.append(buf.copy()); buf_hist[:] = buf_hist[-2:]
            else:
                dense["r2_buf_dense"].append(float("nan"))

        if log_both_phases and t > 0 and (t - 1) % log_every == 0:
            am1 = alignment_metrics(W, X, U_t, a_s, P_V)
            ph1["step_p1"].append(t)
            ph1["mean_cos2_AGOP_p1"].append(am1["mean_cos2_AGOP"])
            ph1["cos2_min_AGOP_p1"].append(am1["cos2_min_AGOP"])
            ph1["L_full_p1"].append(L_full)
        if t % log_every == 0 or t == end_t:
            am = alignment_metrics(W, X, U_t, a_s, P_V)
            L_te = loss_only(W, X_te, y_te, a_s)
            L_V_tr, L_V_te = loss_V_optimal(W, X, y, X_te, y_te, U_t, p)
            r_tilde = am["thr50_l2"]
            L_AG_tr, L_AG_te = loss_AGOP_optimal(W, X, y, X_te, y_te, p, a_s, r_tilde)
            log["step"].append(t); log["L_full"].append(L_full); log["L_test"].append(L_te)
            log["L_V_opt_train"].append(L_V_tr); log["L_V_opt_test"].append(L_V_te)
            log["L_AGOP_opt_train"].append(L_AG_tr); log["L_AGOP_opt_test"].append(L_AG_te)
            for k in ("in_V_frac", "mean_cos2_AGOP", "cos2_min_AGOP",
                      "mass_in_V_AGOP_p", "thr50_l2", "pr_eff_rank"):
                log[k].append(am[k])
            cos_per_log.append(am["cos_per"])
            per_eigvec_log.append(am["per_eigvec_in_V"])
            top_lams_log.append(am["top_lams"])
        if t == end_t:
            break
        W = W - eta * update

    save = {k: np.asarray(log[k]) for k in KEYS_SCALAR}
    save["W"] = W; save["next_t"] = end_t
    save["mom_buf"] = buf
    if prev_update is not None:
        save["last_update"] = prev_update
    for nm, arr in zip(("buf_m2", "buf_m1"), buf_hist[-2:]):
        save[nm] = arr
    save["cos_per"] = np.stack(cos_per_log, axis=0)
    save["per_eigvec_in_V"] = np.stack(per_eigvec_log, axis=0)
    save["top_lams"] = np.stack(top_lams_log, axis=0)
    for k in KEYS_DENSE:
        save[k] = np.asarray(dense[k])
    for k in KEYS_PHASE1:
        save[k] = np.asarray(ph1[k])
    save["cfg_batch_size"] = B if not full_batch else -1
    save["cfg_ns_steps"] = ns_steps
    save["cfg_batch_mode"] = np.bytes_(batch_mode.encode())
    save["cfg_shape_cv"] = shape_cv
    save["cfg_ns_coeff"] = np.bytes_(ns_coeff.encode())
    save["cfg_momentum"] = momentum
    save["cfg_eta"] = eta
    _atomic_savez(ckpt_path, **save)
    print(f"[done] {teacher_act} {tag} step={end_t} "
          f"L={log['L_full'][-1]:.3f} cm={log['cos2_min_AGOP'][-1]:.3f}")
    return True
