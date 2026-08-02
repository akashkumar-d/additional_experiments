"""
Vary teacher activation in 1-HL teacher-student NN, see if plateau-with-feature-learning
phenomenon survives.

Setup: same as our G config (p=100, r_t=4, r_s=50, n=15000, eta=2.0, full-batch Muon)
       except teacher activation σ ∈ {ReLU, tanh, squared, erf}.
Student keeps ReLU activation.

Track all metrics including L_V_opt, L_AGOP_opt, mean_cos², cos²_min, etc.
"""
import argparse, math, time
from pathlib import Path
import numpy as np


def relu(z): return np.maximum(z, 0.0)
def relu_p(z): return (z > 0).astype(z.dtype)


def erf_np(z):
    # Abramowitz & Stegun 7.1.26 approximation, accurate to ~1.5e-7
    a = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429]
    p = 0.3275911
    sign = np.sign(z); z = np.abs(z)
    t = 1.0 / (1.0 + p * z)
    y = 1.0 - (((((a[4]*t + a[3])*t) + a[2])*t + a[1])*t + a[0])*t * np.exp(-z*z)
    return sign * y


# Teacher activations (only matter for label generation)
def relu_act(z): return np.maximum(z, 0.0)
def tanh_act(z): return np.tanh(z)
def sq_act(z): return z * z
def erf_act(z): return erf_np(z)
def linear_act(z): return z
def gelu_act(z):  # exact GELU
    return 0.5 * z * (1.0 + erf_np(z / math.sqrt(2.0)))
def silu_act(z):  # SiLU/Swish: x * sigmoid(x)
    return z * (1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))
def mish_act(z):  # Mish: x * tanh(softplus(x))
    sp = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)  # numerically stable softplus
    return z * np.tanh(sp)
def elu_act(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(np.minimum(z, 30)) - 1))
def leaky_relu_act(z, alpha=0.1):
    return np.where(z > 0, z, alpha * z)

ACTIVATIONS = {
    "relu":       relu_act,
    "tanh":       tanh_act,
    "squared":    sq_act,
    "erf":        erf_act,
    "linear":     linear_act,
    "gelu":       gelu_act,
    "silu":       silu_act,
    "mish":       mish_act,
    "elu":        elu_act,
    "leakyrelu":  leaky_relu_act,
}


def make_problem(p, r_t, n, n_test, seed, teacher_act):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, r_t)); U_t, _ = np.linalg.qr(A)
    a_t = np.ones(r_t)
    X = rng.standard_normal((n, p))
    sigma = ACTIVATIONS[teacher_act]
    y = (sigma(X @ U_t) * a_t[None, :]).sum(axis=1)
    rng_te = np.random.default_rng(seed + 99999)
    X_te = rng_te.standard_normal((n_test, p))
    y_te = (sigma(X_te @ U_t) * a_t[None, :]).sum(axis=1)
    return X, y, X_te, y_te, U_t, a_t


def loss_only(W, X, y, a_s, chunk=4096):
    n = X.shape[0]; total = 0.0
    for i in range(0, n, chunk):
        Xi = X[i:i+chunk]; yi = y[i:i+chunk]
        z = Xi @ W; h = relu(z); f = h @ a_s; err = f - yi
        total += float(np.sum(err**2))
    return total/n


def loss_and_grad(W, X, y, a_s, chunk=4096):
    n = X.shape[0]; total = 0.0; gW = np.zeros_like(W)
    for i in range(0, n, chunk):
        Xi = X[i:i+chunk]; yi = y[i:i+chunk]
        z = Xi @ W; h = relu(z); f = h @ a_s; err = f - yi
        total += float(np.sum(err**2))
        dphi = relu_p(z); tmp = err[:,None]*dphi*a_s[None,:]
        gW += Xi.T @ tmp
    return total/n, (2.0/n)*gW


def student_Cs(X, W, a_s):
    z = X @ W; dphi = relu_p(z); Ss = dphi * a_s[None, :]
    return (Ss.T @ Ss) / X.shape[0]


def matrix_pearson(A, B, eps=1e-12):
    a = A.ravel().astype(float); b = B.ravel().astype(float)
    a -= a.mean(); b -= b.mean()
    d = (a@a)**0.5 * (b@b)**0.5
    return float((a@b)/d) if d > eps else 0.0


def thr_rank_l2(lam, thr_frac=0.5):
    s = np.sort(lam)[::-1]
    if len(s) < 2: return 1
    return int(np.sum(np.asarray(lam) >= thr_frac * max(s[1], 1e-30)))


def alignment_metrics(W, X, U_t, a_s, P_V):
    p, r_s = W.shape; r_t = U_t.shape[1]
    PVW = U_t.T @ W
    in_V = float(np.sum(PVW*PVW)) / float(np.sum(W*W))
    Cs = student_Cs(X, W, a_s)
    G_s = W @ Cs @ W.T
    G_s = 0.5 * (G_s + G_s.T)
    lam, V_eig = np.linalg.eigh(G_s)
    lam = np.clip(lam[::-1], 0, None); V_eig = V_eig[:, ::-1]
    V_top = V_eig[:, :r_t]
    QA, _ = np.linalg.qr(V_top); QB, _ = np.linalg.qr(U_t)
    cos_per = np.clip(np.linalg.svd(QA.T @ QB, compute_uv=False), 0.0, 1.0)
    return {
        "in_V": in_V,
        "mean_cos2": float((cos_per**2).mean()),
        "cos2_min": float(cos_per.min()**2),
        "thr50_l2": thr_rank_l2(lam, 0.5),
        "lam_top": lam[:8].copy(),
    }


def loss_V_optimal(W, X_tr, y_tr, X_te, y_te, U_t, p, chunk=4096):
    P_V = U_t @ U_t.T; W_V = P_V @ W
    r_s = W_V.shape[1]; sqp = math.sqrt(p)
    def feats(X):
        n = X.shape[0]; H = np.zeros((n, r_s))
        for i in range(0, n, chunk):
            Xi = X[i:i+chunk]; z = Xi @ W_V; H[i:i+chunk] = relu(z)/sqp
        return H
    M_tr = feats(X_tr)
    a_opt, _, _, _ = np.linalg.lstsq(M_tr, y_tr, rcond=None)
    L_tr = float(np.mean((M_tr @ a_opt - y_tr)**2))
    M_te = feats(X_te)
    L_te = float(np.mean((M_te @ a_opt - y_te)**2))
    return L_tr, L_te


def init_W(p, r_s, U, max_align=0.30, max_tries=200, rng=None):
    rng = rng or np.random.default_rng()
    bestW, besta = None, None
    for _ in range(max_tries):
        W0 = rng.standard_normal((p, r_s)) / math.sqrt(p)
        PVW = U.T @ W0
        a = float(np.sum(PVW*PVW)) / float(np.sum(W0*W0))
        if bestW is None or a < besta: bestW, besta = W0, a
        if a <= max_align: return W0, a
    return bestW, besta


def run_one(teacher_act, p, r_t, r_s, n, eta, seed, n_steps, log_every,
            ckpt_path, chunk_steps, budget_s):
    X, y, X_te, y_te, U_t, a_t = make_problem(p, r_t, n, n_test=5000,
                                              seed=seed, teacher_act=teacher_act)
    a_s = (1.0/math.sqrt(p)) * np.ones(r_s)
    P_V = U_t @ U_t.T

    KEYS = ("step", "L_full", "L_test", "L_V_opt_train", "L_V_opt_test",
            "in_V", "mean_cos2", "cos2_min", "thr50_l2")
    if ckpt_path.exists():
        z = np.load(ckpt_path, allow_pickle=True)
        if int(z["next_t"]) >= n_steps:
            print(f"[skip] {ckpt_path.name} at {int(z['next_t'])}")
            return False
        W = z["W"].copy()
        log = {k: list(z[k]) for k in KEYS}
        start_t = int(z["next_t"])
        print(f"[resume] {ckpt_path.name}: step {start_t}")
    else:
        rng = np.random.default_rng(seed + 17)
        W0, ia = init_W(p, r_s, U_t, rng=rng)
        print(f"[init {teacher_act}] {ckpt_path.name}: in-V={ia:.3f}")
        W = W0.copy()
        log = {k: [] for k in KEYS}; start_t = 0

    end_t = min(start_t + chunk_steps, n_steps)
    t_start = time.time()
    for t in range(start_t, end_t + 1):
        if time.time() - t_start > budget_s:
            print(f"[budget] stop at {t}"); end_t = t; break
        L, gW = loss_and_grad(W, X, y, a_s)
        Ug, _, Vhg = np.linalg.svd(gW, full_matrices=False)
        update = Ug @ Vhg
        if t % log_every == 0 or t == end_t:
            am = alignment_metrics(W, X, U_t, a_s, P_V)
            L_te = loss_only(W, X_te, y_te, a_s)
            L_V_tr, L_V_te = loss_V_optimal(W, X, y, X_te, y_te, U_t, p)
            log["step"].append(t); log["L_full"].append(L); log["L_test"].append(L_te)
            log["L_V_opt_train"].append(L_V_tr); log["L_V_opt_test"].append(L_V_te)
            for k in ("in_V", "mean_cos2", "cos2_min", "thr50_l2"):
                log[k].append(am[k])
        if t == end_t: break
        W = W - eta * update

    np.savez(ckpt_path, W=W, next_t=end_t,
             **{k: np.asarray(log[k]) for k in KEYS})
    print(f"[done] {teacher_act} step={end_t} L={log['L_full'][-1]:.3f} "
          f"L_te={log['L_test'][-1]:.3f} L_Vopt(tr)={log['L_V_opt_train'][-1]:.4f} "
          f"in_V={log['in_V'][-1]:.3f} mean_cos²={log['mean_cos2'][-1]:.3f}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="teacher_activation")
    ap.add_argument("--budget-s", type=float, default=38.0)
    ap.add_argument("--chunk-steps", type=int, default=2000)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)

    # G config: p=100, r_t=4, r_s=50, n=15000, eta=2.0
    seed = 5; p = 100; r_t = 4; r_s = 50; n = 15000; eta = 2.0; n_steps = 6000

    teachers = ["relu", "tanh", "squared", "erf", "linear",
                "gelu", "silu", "mish", "elu", "leakyrelu"]

    t_start = time.time()
    for ta in teachers:
        if time.time() - t_start > args.budget_s:
            print("[budget global] stop"); break
        ckpt = out_dir / f"teacher_{ta}_p{p}_rt{r_t}_rs{r_s}_n{n}_eta{eta}_seed{seed}.npz"
        run_one(ta, p, r_t, r_s, n, eta, seed, n_steps, log_every=20,
                ckpt_path=ckpt, chunk_steps=args.chunk_steps,
                budget_s=max(5.0, args.budget_s - (time.time() - t_start)))


if __name__ == "__main__":
    main()
