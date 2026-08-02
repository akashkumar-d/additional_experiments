"""
Module-level worker for parallel notebook runs (item1 / item2).

Lives in shared/ so ProcessPoolExecutor can pickle it by reference —
functions defined inside notebook cells cannot be pickled reliably.
Each worker process limits its BLAS threads so N_PROC processes do not
oversubscribe the node's cores.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _limit_threads(k: int = 2) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = str(k)
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(k)
    except Exception:
        pass


def run_full(args):
    """Run one config to completion with the standard (paper) runner.

    args = (out_dir, n_steps, log_every, gid, teacher, p, r_t, r_s, n, eta, seed)
    Returns (gid, seed). Restart-safe: skips if already complete.
    """
    (out_dir, n_steps, log_every, gid, ta, p, rt, rs, n, eta, seed) = args
    _limit_threads(2)
    from teacher_full_runner import run_one
    import numpy as np

    ckpt = Path(out_dir) / f"{ta}_p{p}_rt{rt}_rs{rs}_n{n}_eta{eta}_seed{seed}.npz"
    for _ in range(100):
        run_one(ta, p, rt, rs, n, eta, seed=seed, n_steps=n_steps,
                log_every=log_every, ckpt_path=ckpt,
                chunk_steps=4000, budget_s=10**9)
        z = np.load(ckpt, allow_pickle=True)
        if int(z["next_t"]) >= n_steps:
            break
    return gid, seed


def run_full_practical(args):
    """Same, for the practical-Muon runner (item 2).

    args = (out_dir, ckpt_name, n_steps, log_every, teacher, p, r_t, r_s, n,
            eta, seed, batch_size, ns_steps, momentum)
    """
    # Optional trailing elements, so existing 14-tuples still work:
    #   args[14] = shape_cv    orthogonalisation shape-error dial (default 0)
    #   args[15] = batch_mode  default 'without_replacement';
    #                          'paired_without_replacement' reuses the batch
    #                          across both phases of the period-2 cycle
    (out_dir, ckpt_name, n_steps, log_every, ta, p, rt, rs, n,
     eta, seed, batch_size, ns_steps, momentum) = args[:14]
    shape_cv   = args[14] if len(args) > 14 else 0.0
    batch_mode = args[15] if len(args) > 15 else "without_replacement"
    ns_coeff   = args[16] if len(args) > 16 else "tuned"   # args[16] = ns_coeff
    ns_tol     = args[17] if len(args) > 17 else 0.0       # args[17] = ns_tol
    two_phase  = args[18] if len(args) > 18 else False     # args[18] = log_both_phases
    # args[19] = log_approx. Records, at every logged step, how far the APPLIED
    # update is from the exact polar factor of the same gradient:
    #   approx_err = ||Q - Polar(g)||_F / ||Polar(g)||_F
    # Without this an "NS" row is only nominally inexact -- the routine's name
    # is not evidence. Convergent NS7 is exact to machine precision on a
    # well-conditioned gradient but leaves ~15% error on a decaying spectrum,
    # so the error must be measured on the actual trajectory, not calibrated once.
    log_approx = args[19] if len(args) > 19 else False
    _limit_threads(2)
    from muon_practical_runner import run_one_practical
    import numpy as np

    ckpt = Path(out_dir) / ckpt_name
    for _ in range(100):
        run_one_practical(ta, p, rt, rs, n, eta, seed=seed, n_steps=n_steps,
                          log_every=log_every, ckpt_path=ckpt,
                          chunk_steps=3000, budget_s=10**9,
                          batch_size=batch_size, ns_steps=ns_steps,
                          momentum=momentum, shape_cv=shape_cv,
                          batch_mode=batch_mode, ns_coeff=ns_coeff,
                          ns_tol=ns_tol, log_both_phases=two_phase,
                          log_approx=log_approx)
        z = np.load(ckpt, allow_pickle=True)
        if int(z["next_t"]) >= n_steps:
            break
    return ckpt_name


def run_switchon(args):
    """Stage-C switch-on: continue a FORMED exact-polar plateau under a
    practical perturbation.

    The base checkpoint (exact polar, full batch, no momentum, run to
    t_switch) must already exist; it is copied once, then resumed with the
    new settings. Because run_one_practical reads only (W, mom_buf, logs,
    next_t) from the checkpoint and takes ns_steps/batch_size/momentum as
    arguments, resuming a copy with different settings switches the
    optimizer mid-run and keeps both phases in one dense trace.

    args = (out_dir, ckpt_name, base_ckpt_name, n_steps, log_every, teacher,
            p, r_t, r_s, n, eta, seed, batch_size, ns_steps, momentum)

    Optional trailing elements mirror run_full_practical so a switch-on row can
    carry the same diagnostics as an ordinary one:
      args[15] = shape_cv   args[16] = batch_mode   args[17] = ns_coeff
      args[18] = ns_tol     args[19] = log_both_phases
      args[20] = log_approx
    Without these a switch-on row could not record which coefficient set it
    switched TO, nor how far the applied update sat from exact polar -- which is
    the whole point of switching.
    """
    import shutil
    (out_dir, ckpt_name, base_ckpt_name, n_steps, log_every, ta, p, rt, rs, n,
     eta, seed, batch_size, ns_steps, momentum) = args[:15]
    shape_cv   = args[15] if len(args) > 15 else 0.0
    batch_mode = args[16] if len(args) > 16 else "without_replacement"
    ns_coeff   = args[17] if len(args) > 17 else "tuned"
    ns_tol     = args[18] if len(args) > 18 else 0.0
    two_phase  = args[19] if len(args) > 19 else False
    log_approx = args[20] if len(args) > 20 else False
    _limit_threads(2)
    from muon_practical_runner import run_one_practical
    import numpy as np

    out = Path(out_dir)
    ckpt, base = out / ckpt_name, out / base_ckpt_name
    if not base.exists():
        return f"{ckpt_name} SKIPPED (base {base_ckpt_name} missing)"
    # The base must stop AT the switch point. There is no way to rewind: only
    # the final W is stored, so a base that ran to completion cannot be
    # truncated back to t_switch. Copying such a base and "resuming" silently
    # produces a byte-identical duplicate of the base -- the switch never
    # happens and the row looks like a successful control. Fail loudly instead.
    _bt = int(np.load(base, allow_pickle=True)["next_t"])
    if _bt >= n_steps:
        raise ValueError(
            f"{ckpt_name}: base {base_ckpt_name} is already at step {_bt} >= "
            f"n_steps={n_steps}, so the switch would be a no-op. Run the base "
            f"row with n_steps = t_switch, not to completion.")
    if not ckpt.exists():
        shutil.copy(base, ckpt)
    for _ in range(100):
        run_one_practical(ta, p, rt, rs, n, eta, seed=seed, n_steps=n_steps,
                          log_every=log_every, ckpt_path=ckpt,
                          chunk_steps=3000, budget_s=10**9,
                          batch_size=batch_size, ns_steps=ns_steps,
                          momentum=momentum, shape_cv=shape_cv,
                          batch_mode=batch_mode, ns_coeff=ns_coeff,
                          ns_tol=ns_tol, log_both_phases=two_phase,
                          log_approx=log_approx)
        z = np.load(ckpt, allow_pickle=True)
        if int(z["next_t"]) >= n_steps:
            break
    return ckpt_name


def run_full_gpu(args):
    """GPU/torch variant for item 1.

    args = (out_dir, n_steps, log_every, gid, teacher, p, r_t, r_s, n, eta,
            seed, device)   with device in {'cuda', 'cuda:0', 'cpu'}.

    IMPORTANT: torch must NOT be imported in the parent notebook before the
    process pool is created (fork + initialized CUDA breaks children). torch
    is imported here, inside the child, after the fork.
    """
    (out_dir, n_steps, log_every, gid, ta, p, rt, rs, n, eta, seed, device) = args
    _limit_threads(2)
    from gpu_runner import run_one_gpu
    import numpy as np

    ckpt = Path(out_dir) / f"{ta}_p{p}_rt{rt}_rs{rs}_n{n}_eta{eta}_seed{seed}.npz"
    for _ in range(100):
        run_one_gpu(ta, p, rt, rs, n, eta, seed=seed, n_steps=n_steps,
                    log_every=log_every, ckpt_path=ckpt,
                    chunk_steps=4000, budget_s=10**9, device=device)
        z = np.load(ckpt, allow_pickle=True)
        if int(z["next_t"]) >= n_steps:
            break
    return gid, seed


def run_full_practical_gpu(args):
    """GPU/torch variant for item 2.

    args = (out_dir, ckpt_name, n_steps, log_every, teacher, p, r_t, r_s, n,
            eta, seed, batch_size, ns_steps, momentum, device)
    """
    # Optional trailing elements: args[15] = shape_cv, args[16] = batch_mode
    (out_dir, ckpt_name, n_steps, log_every, ta, p, rt, rs, n,
     eta, seed, batch_size, ns_steps, momentum, device) = args[:15]
    shape_cv   = args[15] if len(args) > 15 else 0.0
    batch_mode = args[16] if len(args) > 16 else "without_replacement"
    ns_coeff   = args[17] if len(args) > 17 else "tuned"   # args[17] = ns_coeff
    ns_tol     = args[18] if len(args) > 18 else 0.0       # args[18] = ns_tol
    _limit_threads(2)
    from gpu_runner import run_one_gpu
    import numpy as np

    ckpt = Path(out_dir) / ckpt_name
    for _ in range(100):
        run_one_gpu(ta, p, rt, rs, n, eta, seed=seed, n_steps=n_steps,
                    log_every=log_every, ckpt_path=ckpt,
                    chunk_steps=3000, budget_s=10**9,
                    batch_size=batch_size, ns_steps=ns_steps,
                    momentum=momentum, device=device, shape_cv=shape_cv,
                    batch_mode=batch_mode, ns_coeff=ns_coeff, ns_tol=ns_tol)
        z = np.load(ckpt, allow_pickle=True)
        if int(z["next_t"]) >= n_steps:
            break
    return ckpt_name
