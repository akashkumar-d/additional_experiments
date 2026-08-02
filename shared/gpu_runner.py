"""
GPU-capable runner for item1/item2 (torch backend, float64), with a numpy
backend for verification.

Design: all training/metric math is written once against a minimal backend
adapter `B` (matmul, svd, eigh, qr, lstsq, relu, ...). Two implementations:

  - NumpyBackend: used to VERIFY the port against the reference runner
    (teacher_full_runner / muon_practical_runner) on any machine.
  - TorchBackend(device): float64 on 'cuda' (A100) or 'cpu'.

Reproducibility notes:
  * Problem generation (X, y, U_t) and the init W0 are ALWAYS produced by
    the same numpy RNG code as the paper runner (make_problem / init_W),
    then transferred to the device -> identical problem instances across
    backends.
  * float64 everywhere. Do NOT run this in float32: the period-2 signature
    (1 - rho_2 ~ 1e-10) and EoS dynamics need f64.
  * GPU trajectories are reproducible run-to-run on the same device, but
    are NOT bit-identical to CPU trajectories (different BLAS); at the edge
    of stability the 1e-16 differences amplify into different
    micro-trajectories with the same statistics. Do not mix CPU and GPU
    runs within one seed set for a statistic if bit-level provenance
    matters; statistically they are interchangeable.

The saved checkpoint format is a superset-compatible match of
muon_practical_runner (same keys), so the item1/item2 analysis cells work
unchanged.

Usage:
    from gpu_runner import run_one_gpu
    run_one_gpu('relu', 100, 8, 50, 15000, 1.5, seed=5, n_steps=8000,
                log_every=20, ckpt_path=Path('out.npz'), chunk_steps=4000,
                budget_s=1e9, device='cuda')                    # item 1
    run_one_gpu(..., batch_size=256, ns_steps=5, momentum=0.95) # item 2
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

from teacher_activation_sweep import make_problem, init_W   # noqa: E402

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



def _shape_perturb(B, u, cv):
    """Controlled orthogonalisation SHAPE error (see muon_practical_runner).
    Imposes singular-value CV `cv` on an already-orthogonalised update and
    renormalises so ||u||_F = sqrt(M) -- changes shape only, never scale."""
    if cv <= 0:
        return u
    U, S, Vh = B.svd(u)
    M = int(np.prod(B.to_numpy(S).shape))
    if M < 2:
        return u
    ramp = np.linspace(-0.5, 0.5, M)
    ramp = ramp / (ramp.std() + 1e-30)
    sv = B.from_numpy(np.clip(1.0 + cv * ramp, 1e-6, None))
    out = B.matmul(U * sv, Vh)
    return out * (math.sqrt(M) / (B.to_float(B.norm(out)) + 1e-30))


# ==================================================================
# Backends
# ==================================================================

class NumpyBackend:
    """Reference backend (verification)."""
    name = "numpy"

    def from_numpy(self, a):  return np.asarray(a, dtype=np.float64)
    def to_numpy(self, a):    return np.asarray(a)
    def to_float(self, a):    return float(a)

    def matmul(self, a, b):   return a @ b
    def relu(self, z):        return np.maximum(z, 0.0)
    def relu_p(self, z):      return (z > 0).astype(np.float64)
    def sum(self, a, axis=None): return a.sum(axis=axis)
    def mean(self, a):        return a.mean()
    def norm(self, a):        return np.linalg.norm(a)
    def svd(self, a):
        U, S, Vh = np.linalg.svd(a, full_matrices=False)
        return U, S, Vh
    def eigh(self, a):
        w, v = np.linalg.eigh(a)
        return w, v
    def qr(self, a):
        q, r = np.linalg.qr(a)
        return q, r
    def lstsq(self, A, b):
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol
    def clip(self, a, lo, hi): return np.clip(a, lo, hi)
    def zeros_like(self, a):  return np.zeros_like(a)
    def take_rows(self, a, idx): return a[idx]
    def einsum_pervec(self, V_eig, P_V):
        return np.einsum("ji,jk,ki->i", V_eig, P_V, V_eig)


class TorchBackend:
    """torch float64 backend; device='cuda' on A100 links, 'cpu' otherwise."""
    name = "torch"

    def __init__(self, device="cuda"):
        import torch
        self.torch = torch
        if device == "cuda" and not torch.cuda.is_available():
            print("[gpu_runner] cuda requested but unavailable -> using cpu")
            device = "cpu"
        self.device = torch.device(device)

    def from_numpy(self, a):
        return self.torch.as_tensor(np.asarray(a, dtype=np.float64),
                                    dtype=self.torch.float64, device=self.device)
    def to_numpy(self, a):    return a.detach().cpu().numpy()
    def to_float(self, a):
        return float(a.item()) if hasattr(a, "item") else float(a)

    def matmul(self, a, b):   return a @ b
    def relu(self, z):        return self.torch.clamp(z, min=0.0)
    def relu_p(self, z):      return (z > 0).to(self.torch.float64)
    def sum(self, a, axis=None):
        return a.sum() if axis is None else a.sum(dim=axis)
    def mean(self, a):        return a.mean()
    def norm(self, a):        return self.torch.linalg.norm(a)
    def svd(self, a):
        U, S, Vh = self.torch.linalg.svd(a, full_matrices=False)
        return U, S, Vh
    def eigh(self, a):
        w, v = self.torch.linalg.eigh(a)
        return w, v
    def qr(self, a):
        q, r = self.torch.linalg.qr(a)
        return q, r
    def lstsq(self, A, b):
        # Device-safe min-norm least squares. torch.linalg.lstsq only supports
        # driver='gels' on CUDA/ROCm, which assumes full rank and misbehaves on
        # the rank-deficient head-refit systems this runner produces (e.g.
        # relu(X @ P_V W) features late in training). Instead: reduced QR, then
        # SVD of the small R factor with the same machine-eps cutoff as
        # np.linalg.lstsq(rcond=None). Verified to match numpy on full-rank,
        # near-rank-1, and zero-column systems.
        t = self.torch
        Q, R = t.linalg.qr(A)                      # Q: (m, r), R: (r, r)
        U, S, Vh = t.linalg.svd(R)
        eps = t.finfo(A.dtype).eps
        cutoff = eps * max(A.shape) * S.max()
        Sinv = t.where(S > cutoff, 1.0 / S, t.zeros_like(S))
        return Vh.transpose(-2, -1) @ (Sinv * (U.transpose(-2, -1) @ (Q.transpose(-2, -1) @ b)))
    def clip(self, a, lo, hi): return self.torch.clamp(a, lo, hi)
    def zeros_like(self, a):  return self.torch.zeros_like(a)
    def take_rows(self, a, idx):
        ii = self.torch.as_tensor(idx, dtype=self.torch.long, device=self.device)
        return a[ii]
    def einsum_pervec(self, V_eig, P_V):
        return self.torch.einsum("ji,jk,ki->i", V_eig, P_V, V_eig)


def get_backend(device: str | None):
    """device None/'numpy' -> NumpyBackend; 'cuda'/'cpu' -> TorchBackend."""
    if device in (None, "numpy"):
        return NumpyBackend()
    return TorchBackend(device)


# ==================================================================
# Math (written once against the backend)
# ==================================================================

def _loss_only(B, W, X, y, a_s):
    z = B.matmul(X, W)
    f = B.matmul(B.relu(z), a_s)
    err = f - y
    return B.to_float(B.mean(err * err) * err.shape[0]) / err.shape[0] * 1.0 \
        if False else B.to_float(B.sum(err * err)) / err.shape[0]


def _loss_and_grad(B, W, X, y, a_s):
    n = X.shape[0]
    z = B.matmul(X, W)
    h = B.relu(z)
    f = B.matmul(h, a_s)
    err = f - y
    L = B.to_float(B.sum(err * err)) / n
    dphi = B.relu_p(z)
    tmp = err[:, None] * dphi * a_s[None, :]
    gW = B.matmul(X.T, tmp) * (2.0 / n)
    return L, gW


def _student_Cs(B, X, W, a_s):
    z = B.matmul(X, W)
    Ss = B.relu_p(z) * a_s[None, :]
    return B.matmul(Ss.T, Ss) / X.shape[0]


def _thr_rank_l2(lam_np, thr_frac=0.5):
    s = np.sort(lam_np)[::-1]
    if len(s) < 2:
        return 1
    return int(np.sum(lam_np >= thr_frac * max(s[1], 1e-30)))


def _alignment_metrics(B, W, X, U_t, a_s, P_V, r_t, n_extra=3, n_top_lams=12):
    PVW = B.matmul(U_t.T, W)
    in_V = B.to_float(B.sum(PVW * PVW)) / B.to_float(B.sum(W * W))
    Cs = _student_Cs(B, X, W, a_s)
    G_s = B.matmul(B.matmul(W, Cs), W.T)
    G_s = 0.5 * (G_s + G_s.T)
    lam, V_eig = B.eigh(G_s)                      # ascending
    lam_np = B.to_numpy(lam)[::-1].copy()
    lam_np = np.clip(lam_np, 0, None)
    # reverse eigvec columns to descending order (backend-side)
    idx = list(range(V_eig.shape[1] - 1, -1, -1))
    V_eig = V_eig[:, idx] if B.name == "numpy" else V_eig[:, B.torch.as_tensor(idx, device=B.device)]
    s_lam = float(lam_np.sum()); s_lam2 = float((lam_np ** 2).sum())
    pr_eff_rank = (s_lam * s_lam) / max(s_lam2, 1e-30)
    V_top = V_eig[:, :r_t]
    QA, _ = B.qr(V_top)
    QB, _ = B.qr(U_t)
    _, sv, _ = B.svd(B.matmul(QA.T, QB))
    cos_per = np.clip(B.to_numpy(sv), 0.0, 1.0)
    n_track = r_t + n_extra
    in_V_per_full = B.to_numpy(B.einsum_pervec(V_eig, P_V))
    per_eigvec_in_V = in_V_per_full[:n_track].copy()
    mass_in_V = float(np.sum(lam_np * in_V_per_full) / max(s_lam, 1e-30))
    K = min(n_top_lams, len(lam_np))
    top_lams = np.zeros(n_top_lams); top_lams[:K] = lam_np[:K]
    return {
        "in_V_frac": in_V,
        "cos_per": cos_per,
        "mean_cos2_AGOP": float((cos_per ** 2).mean()),
        "cos2_min_AGOP": float(cos_per.min() ** 2),
        "per_eigvec_in_V": per_eigvec_in_V,
        "top_lams": top_lams,
        "mass_in_V_AGOP_p": mass_in_V,
        "thr50_l2": _thr_rank_l2(lam_np, 0.5),
        "pr_eff_rank": pr_eff_rank,
    }


def _loss_head_refit(B, W_proj, X_tr, y_tr, X_te, y_te, p):
    sqp = math.sqrt(p)
    H_tr = B.relu(B.matmul(X_tr, W_proj)) / sqp
    a_opt = B.lstsq(H_tr, y_tr)
    r_tr = B.matmul(H_tr, a_opt) - y_tr
    L_tr = B.to_float(B.sum(r_tr * r_tr)) / X_tr.shape[0]
    H_te = B.relu(B.matmul(X_te, W_proj)) / sqp
    r_te = B.matmul(H_te, a_opt) - y_te
    L_te = B.to_float(B.sum(r_te * r_te)) / X_te.shape[0]
    return L_tr, L_te


def _loss_V_optimal(B, W, X_tr, y_tr, X_te, y_te, U_t, P_V, p):
    W_V = B.matmul(P_V, W)
    return _loss_head_refit(B, W_V, X_tr, y_tr, X_te, y_te, p)


def _loss_AGOP_optimal(B, W, X_tr, y_tr, X_te, y_te, p, a_s, r_tilde):
    Cs = _student_Cs(B, X_tr, W, a_s)
    G_s = B.matmul(B.matmul(W, Cs), W.T)
    G_s = 0.5 * (G_s + G_s.T)
    _, V_eig = B.eigh(G_s)
    idx = list(range(V_eig.shape[1] - 1, -1, -1))
    V_eig = V_eig[:, idx] if B.name == "numpy" else V_eig[:, B.torch.as_tensor(idx, device=B.device)]
    r_tilde = max(1, min(int(r_tilde), V_eig.shape[1]))
    V_top = V_eig[:, :r_tilde]
    P = B.matmul(V_top, V_top.T)
    W_hat = B.matmul(P, W)
    return _loss_head_refit(B, W_hat, X_tr, y_tr, X_te, y_te, p)


# Logged keys. Kept in sync with muon_practical_runner so the two backends
# produce interchangeable checkpoints. (The `approx_*_dense` diagnostics are
# CPU-path only; load_traj returns None for absent keys, so analysis is
# unaffected.) These sat between _newton_schulz and run_one_gpu and were once
# lost to a text-splice edit -- if this file ever raises NameError on a KEYS_*
# name, that is what happened.
KEYS_SCALAR = ("step", "L_full", "L_test", "L_V_opt_train", "L_V_opt_test",
               "L_AGOP_opt_train", "L_AGOP_opt_test",
               "in_V_frac", "mean_cos2_AGOP", "cos2_min_AGOP",
               "mass_in_V_AGOP_p", "thr50_l2", "pr_eff_rank")

KEYS_DENSE = ("step_dense", "L_full_dense", "L_batch_dense", "disp_dense",
              "rev_cos_dense", "r2_update_dense", "r2_buf_dense")

_NS_COEFFS = (3.4445, -4.7750, 2.0315)
# See muon_practical_runner.NS_COEFF_SETS for why these two differ qualitatively:
# 'tuned' has no fixed point at 1 (never converges); 'convergent' has a
# superattracting one (exact polar by ~6 passes).
NS_COEFF_SETS = {"tuned": (3.4445, -4.7750, 2.0315), "convergent": (2.0, -1.5, 0.5)}
NS_HYBRID_LEAD = 1   # see muon_practical_runner for the rationale


def _ns_schedule(steps, coeff):
    if coeff != "hybrid":
        return [NS_COEFF_SETS[coeff]] * steps
    lead = min(NS_HYBRID_LEAD, steps)
    return ([NS_COEFF_SETS["tuned"]] * lead
            + [NS_COEFF_SETS["convergent"]] * (steps - lead))


def _newton_schulz(B, G, steps, eps=1e-7, coeff="tuned", ns_tol=0.0, ns_max=20):
    """Backend NS. Supports 'tuned' | 'convergent' | 'hybrid', and adaptive
    depth via ns_tol (stop when ||X X^T - I||_F / sqrt(M) <= ns_tol)."""
    if steps <= 0 and ns_tol <= 0:
        U, _, Vh = B.svd(G)
        return B.matmul(U, Vh)
    X = G
    transposed = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transposed = True
    X = X * (1.0 / (B.to_float(B.norm(X)) + eps))
    if ns_tol > 0:
        M = int(X.shape[0])
        I = B.from_numpy(np.eye(M))
        for a, b, c in _ns_schedule(ns_max, coeff):
            A = B.matmul(X, X.T)
            if B.to_float(B.norm(A - I)) / math.sqrt(M) <= ns_tol:
                break
            X = a * X + B.matmul(b * A + c * B.matmul(A, A), X)
        return X.T if transposed else X
    for a, b, c in _ns_schedule(steps, coeff):
        A = B.matmul(X, X.T)
        X = a * X + B.matmul(b * A + c * B.matmul(A, A), X)
    return X.T if transposed else X

def run_one_gpu(teacher_act, p, r_t, r_s, n, eta, seed, n_steps, log_every,
                ckpt_path: Path, chunk_steps, budget_s,
                batch_size=None, ns_steps=0, momentum=0.0, batch_mode="without_replacement", shape_cv=0.0, ns_coeff="tuned", ns_tol=0.0, nesterov=True,
                device="cuda"):
    """GPU/CPU-torch (or numpy) run. Checkpoint format matches
    muon_practical_runner so all analysis cells work unchanged."""
    B = get_backend(device)

    # --- identical problem generation to the CPU reference (numpy RNG) ---
    Xn, yn, Xten, yten, U_tn, _ = make_problem(p, r_t, n, n_test=5000,
                                               seed=seed, teacher_act=teacher_act)
    a_sn = (1.0 / math.sqrt(p)) * np.ones(r_s)
    X, y = B.from_numpy(Xn), B.from_numpy(yn)
    X_te, y_te = B.from_numpy(Xten), B.from_numpy(yten)
    U_t, a_s = B.from_numpy(U_tn), B.from_numpy(a_sn)
    P_V = B.matmul(U_t, U_t.T)

    Bsz = int(batch_size) if batch_size else n
    Bsz = min(Bsz, n)
    full_batch = Bsz >= n
    tag = (f"[{B.name}:{getattr(B, 'device', 'cpu')}] "
           f"B={'full' if full_batch else Bsz} ns={ns_steps or 'exact'} m={momentum}")

    z = _load_ckpt(ckpt_path) if ckpt_path.exists() else None
    if z is not None:
        if int(z["next_t"]) >= n_steps:
            print(f"[skip] {ckpt_path.name} at {int(z['next_t'])}")
            return False
        W = B.from_numpy(z["W"])
        buf = B.from_numpy(z["mom_buf"]) if "mom_buf" in z.files else B.zeros_like(W)
        log = {k: list(z[k]) for k in KEYS_SCALAR}
        cos_per_log = list(z["cos_per"]); per_eigvec_log = list(z["per_eigvec_in_V"])
        top_lams_log = list(z["top_lams"])
        dense = {k: list(z[k]) if k in z.files else [] for k in KEYS_DENSE}
        prev_update = (B.from_numpy(z["last_update"])
                       if "last_update" in z.files else None)
        buf_hist = [B.from_numpy(z[k]) for k in ("buf_m2", "buf_m1") if k in z.files]
        start_t = int(z["next_t"])
        print(f"[resume {tag}] {ckpt_path.name}: step {start_t}")
    else:
        rng0 = np.random.default_rng(seed + 17)
        W0, ia = init_W(p, r_s, U_tn, rng=rng0)     # identical init to CPU
        print(f"[init {teacher_act} {tag}] {ckpt_path.name}: in-V={ia:.3f}")
        W = B.from_numpy(W0)
        buf = B.zeros_like(W)
        log = {k: [] for k in KEYS_SCALAR}
        cos_per_log = []; per_eigvec_log = []; top_lams_log = []
        dense = {k: [] for k in KEYS_DENSE}
        prev_update = None
        buf_hist = []
        start_t = 0

    batch_rng = np.random.default_rng((seed + 1) * 100003 + start_t)

    end_t = min(start_t + chunk_steps, n_steps)
    t_start = time.time()
    for t in range(start_t, end_t + 1):
        if time.time() - t_start > budget_s:
            print(f"[budget] stop at {t}"); end_t = t; break

        if full_batch:
            L_batch, gW = _loss_and_grad(B, W, X, y, a_s)
            L_full = L_batch
        else:
            if batch_mode == "paired_without_replacement":
                _pr = np.random.default_rng((seed + 1) * 100003 + (t // 2))
                idx = _pr.choice(n, size=Bsz, replace=False)
            else:
                idx = _sample_idx(batch_rng, n, Bsz, batch_mode)
            L_batch, gW = _loss_and_grad(B, W, B.take_rows(X, idx),
                                         B.take_rows(y, idx), a_s)
            L_full = _loss_only(B, W, X, y, a_s)

        if momentum > 0.0:
            buf = momentum * buf + gW
            g_eff = gW + momentum * buf if nesterov else buf
        else:
            g_eff = gW

        update = _newton_schulz(B, g_eff, ns_steps, coeff=ns_coeff, ns_tol=ns_tol)
        if shape_cv > 0:
            update = _shape_perturb(B, update, shape_cv)

        if t < end_t:
            dense["step_dense"].append(t)
            dense["L_full_dense"].append(L_full)
            dense["L_batch_dense"].append(L_batch)
            dense["disp_dense"].append(eta * B.to_float(B.norm(update)))
            if prev_update is None:
                dense["rev_cos_dense"].append(float("nan"))
                dense["r2_update_dense"].append(float("nan"))
            else:
                n1 = B.to_float(B.norm(prev_update))
                n2 = B.to_float(B.norm(update))
                dense["rev_cos_dense"].append(
                    -B.to_float(B.sum(prev_update * update)) / (n1 * n2 + 1e-30))
                dense["r2_update_dense"].append(
                    B.to_float(B.norm(prev_update + update)) / (0.5 * (n1 + n2) + 1e-30))
            prev_update = update
            if momentum > 0.0:
                if len(buf_hist) >= 2:
                    nb = 0.5 * (B.to_float(B.norm(buf)) + B.to_float(B.norm(buf_hist[-2])))
                    dense["r2_buf_dense"].append(
                        B.to_float(B.norm(buf - buf_hist[-2])) / (nb + 1e-30))
                else:
                    dense["r2_buf_dense"].append(float("nan"))
                buf_hist.append(buf); buf_hist[:] = buf_hist[-2:]
            else:
                dense["r2_buf_dense"].append(float("nan"))

        if t % log_every == 0 or t == end_t:
            am = _alignment_metrics(B, W, X, U_t, a_s, P_V, r_t)
            L_te = _loss_only(B, W, X_te, y_te, a_s)
            L_V_tr, L_V_te = _loss_V_optimal(B, W, X, y, X_te, y_te, U_t, P_V, p)
            L_AG_tr, L_AG_te = _loss_AGOP_optimal(B, W, X, y, X_te, y_te, p, a_s,
                                                  am["thr50_l2"])
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
    save["W"] = B.to_numpy(W); save["next_t"] = end_t
    save["mom_buf"] = B.to_numpy(buf)
    if prev_update is not None:
        save["last_update"] = B.to_numpy(prev_update)
    for nm, arr in zip(("buf_m2", "buf_m1"), buf_hist[-2:]):
        save[nm] = B.to_numpy(arr)
    save["cos_per"] = np.stack(cos_per_log, axis=0)
    save["per_eigvec_in_V"] = np.stack(per_eigvec_log, axis=0)
    save["top_lams"] = np.stack(top_lams_log, axis=0)
    for k in KEYS_DENSE:
        save[k] = np.asarray(dense[k])
    save["cfg_batch_size"] = Bsz if not full_batch else -1
    save["cfg_ns_steps"] = ns_steps
    save["cfg_batch_mode"] = np.bytes_(batch_mode.encode())
    save["cfg_shape_cv"] = shape_cv
    save["cfg_ns_coeff"] = np.bytes_(ns_coeff.encode())
    save["cfg_momentum"] = momentum
    save["cfg_eta"] = eta
    save["cfg_backend"] = np.bytes_(f"{B.name}:{getattr(B, 'device', 'cpu')}".encode())
    _atomic_savez(ckpt_path, **save)
    print(f"[done {tag}] step={end_t} L={log['L_full'][-1]:.3f} "
          f"cm={log['cos2_min_AGOP'][-1]:.3f}")
    return True
