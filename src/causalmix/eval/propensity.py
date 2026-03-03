# Propensity/overlap utilities

# overlap evaluation class
# A similar class can also be found in the causal_eval.py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------- helpers ----------

def fit_logistic_torch(X_np, T_np, max_iter=200):
    '''Torch logistic regression; returns e_hat = sigmoid(X @ w + b).'''
    X = torch.from_numpy(np.asarray(X_np, np.float32))
    t = torch.from_numpy(np.asarray(T_np, np.float32))

    # Use CPU by default; switch to GPU if desired
    device = torch.device("cpu")
    X = X.to(device); t = t.to(device)

    w = torch.zeros(X.size(1), 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, tolerance_grad=1e-8, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        logits = X @ w + b
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), t)
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        e_hat = torch.sigmoid(X @ w + b).squeeze(1).cpu().numpy()
    return e_hat
'''
# we already calculated AUC in overlap_diagnostics() associated with the model, so commented out here
def auc_from_scores(scores, labels):
    '''Tie-aware ROC AUC via rank statistic; labels in {0,1}. Returns float or NaN.'''
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(scores)
    s = scores[order]; y = labels[order]
    n = len(s)
    ranks = np.empty(n, dtype=float)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and s[j + 1] == s[i]:
            j += 1
        ranks[i:j+1] = 0.5 * (i + j) + 1.0  # ranks start at 1
        i = j + 1

    n_pos = y.sum()
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-12))
'''
def _silverman_bandwidth(u):
    '''Silverman's rule on ℝ with robust scale (min(std, IQR/1.34)).'''
    u = np.asarray(u, dtype=float).ravel()
    n = max(len(u), 1)
    if n < 2:
        return 0.5
    std = np.std(u, ddof=1)
    q75, q25 = np.percentile(u, [75, 25])
    iqr = q75 - q25
    scale = max(min(std, iqr / 1.34), 1e-3)
    h = 0.9 * scale * n ** (-1/5)
    return max(h, 1e-3)

def _kde_gaussian_R(grid_u, u, bw=None):
    '''Gaussian KDE on the real line. Returns (density_on_grid, used_bandwidth).'''
    u = np.asarray(u, dtype=float).ravel()
    n = len(u)
    if n == 0:
        return np.zeros_like(grid_u), 1.0
    if bw is None:
        bw = _silverman_bandwidth(u)
    z = (grid_u[:, None] - u[None, :]) / bw  # [G, n]
    dens_u = np.exp(-0.5 * z * z).sum(axis=1) / (np.sqrt(2 * np.pi) * bw * n)
    return np.maximum(dens_u, 0.0), bw

# ---------- main function ----------

def plot_propensity_overlap(
    X_gen,
    T_gen,
    gridsize: int = 600,
    eps: float = 1e-6,
    bw0: float | None = None,
    bw1: float | None = None,
    savepath: str | None = None,
    e_hat_override: np.ndarray | None = None,
    shared_bw = True
):
    '''
    Smooth overlap plot via KDE on logit scale.

    Steps:
      1) Fit ê(x)=P(T=1|X) (or use e_hat_override).
      2) Transform s=ê to u=logit(s), do Gaussian KDE on u∈ℝ for each group T=0/1.
      3) Map back to s-grid with Jacobian: p_S(s) = p_U(logit(s)) / (s*(1-s)).
      4) Overlap coefficient = ∫ min(p0(s), p1(s)) ds on s∈[0,1].

    Returns:
      dict: {'auc', 'overlap_coeff', 'bw0', 'bw1', 'path'}
    '''
    X_np = np.asarray(X_gen, dtype=np.float32)
    T_np = np.asarray(T_gen, dtype=int).reshape(-1)
    if X_np.ndim != 2 or T_np.ndim != 1 or X_np.shape[0] != T_np.shape[0]:
        raise ValueError("Shapes must be X_gen: [N, V], T_gen: [N]")

    # 1) get propensities
    if e_hat_override is None:
        e_hat = fit_logistic_torch(X_np, T_np, max_iter=250)
    else:
        e_hat = np.asarray(e_hat_override, dtype=float).reshape(-1)

    # clip to avoid logit overflow
    e_hat = np.clip(e_hat, eps, 1.0 - eps)
    # we already calculated AUC in overlap_diagnostics() 
    # auc = auc_from_scores(e_hat, T_np)

    # split by treatment
    e0 = e_hat[T_np == 0]
    e1 = e_hat[T_np == 1]
    u0 = np.log(e0) - np.log(1.0 - e0)
    u1 = np.log(e1) - np.log(1.0 - e1)

    # 2) KDE on real line
    grid_s = np.linspace(eps, 1.0 - eps, gridsize)
    grid_u = np.log(grid_s) - np.log(1.0 - grid_s)

    # d0_u, bw0 = _kde_gaussian_R(grid_u, u0, bw=bw0)
    # d1_u, bw1 = _kde_gaussian_R(grid_u, u1, bw=bw1)
    # update 11/10/2025: use a shared bandwidth for both groups when reporting overlap 
    #                    pooled silverman is used to compute bandwidth on the combined logit samples
    if shared_bw:
        u_all = np.concatenate([u0, u1])
        bw = _silverman_bandwidth(u_all)
        d0_u, bw0 = _kde_gaussian_R(grid_u, u0, bw=bw)
        d1_u, bw1 = _kde_gaussian_R(grid_u, u1, bw=bw)
    else:
        d0_u, bw0 = _kde_gaussian_R(grid_u, u0, bw=bw0)
        d1_u, bw0 = _kde_gaussian_R(grid_u, u1, bw=bw1)

    # 3) change of variables back to s in (0,1)
    jac = grid_s * (1.0 - grid_s)
    d0_s = d0_u / (jac + 1e-12)
    d1_s = d1_u / (jac + 1e-12)

    # numeric normalization on [0,1]
    area0 = np.trapz(d0_s, grid_s);  area1 = np.trapz(d1_s, grid_s)
    if area0 > 0: d0_s = d0_s / area0
    if area1 > 0: d1_s = d1_s / area1

    # 4) overlap coefficient
    overlap_coeff = float(np.trapz(np.minimum(d0_s, d1_s), grid_s))

    # plot
    plt.figure(figsize=(7, 4.5))
    plt.plot(grid_s, d0_s, label="T=0: density of ê(x)")
    plt.plot(grid_s, d1_s, label="T=1: density of ê(x)")
    plt.fill_between(grid_s, np.minimum(d0_s, d1_s), alpha=0.3)
    plt.xlabel("ê(x) = P(T=1 | X)")
    plt.ylabel("Density")
    # plt.title(
    #     f"Propensity Overlap (Logit-KDE)\n"
    #     f"AUC={auc:.3f} | Overlap Coeff={overlap_coeff:.3f} | bw₀={bw0:.3f}, bw₁={bw1:.3f}"
    # )
    plt.legend(loc="best")
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()
    # return the metrics as a sanity check compared to those from 
    return {
            #"auc": auc, 
            "overlap_coeff": overlap_coeff, 
            "bw0": float(bw0),
            "bw1": float(bw1), 
            "path": savepath}
