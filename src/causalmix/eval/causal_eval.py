# causal_eval.py
 
from dataclasses import dataclass, asdict
from typing import Optional, Dict
 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, pearsonr
 
import torch
import torch.nn.functional as F
 
 
# ----------------------------------------------------------------------
# Helpers for propensity-based overlap (your code, slightly organized)
# ----------------------------------------------------------------------
 
def fit_logistic_torch(X_np, T_np, max_iter=200):
    """Torch logistic regression; returns e_hat = sigmoid(X @ w + b)."""
    X = torch.from_numpy(np.asarray(X_np, np.float32))
    t = torch.from_numpy(np.asarray(T_np, np.float32))
 
    device = torch.device("cpu")
    X = X.to(device)
    t = t.to(device)
 
    w = torch.zeros(X.size(1), 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
 
    opt = torch.optim.LBFGS(
        [w, b],
        max_iter=max_iter,
        tolerance_grad=1e-8,
        tolerance_change=1e-9,
    )
 
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
 
 
def _silverman_bandwidth(u):
    """Silverman's rule on ℝ with robust scale (min(std, IQR/1.34))."""
    u = np.asarray(u, dtype=float).ravel()
    n = max(len(u), 1)
    if n < 2:
        return 0.5
    std = np.std(u, ddof=1)
    q75, q25 = np.percentile(u, [75, 25])
    iqr = q75 - q25
    scale = max(min(std, iqr / 1.34), 1e-3)
    h = 0.9 * scale * n ** (-1 / 5)
    return max(h, 1e-3)
 
 
def _kde_gaussian_R(grid_u, u, bw=None):
    """Gaussian KDE on the real line. Returns (density_on_grid, used_bandwidth)."""
    u = np.asarray(u, dtype=float).ravel()
    n = len(u)
    if n == 0:
        return np.zeros_like(grid_u), 1.0
    if bw is None:
        bw = _silverman_bandwidth(u)
    z = (grid_u[:, None] - u[None, :]) / bw  # [G, n]
    dens_u = np.exp(-0.5 * z * z).sum(axis=1) / (np.sqrt(2 * np.pi) * bw * n)
    return np.maximum(dens_u, 0.0), bw
 
 
@dataclass
class CausalEvalResults:
    # Treatment effect metrics
    mae_tau: Optional[float] = None
    corr_tau: Optional[float] = None
    ate_pred: Optional[float] = None
    ate_target: Optional[float] = None
    ate_abs_error: Optional[float] = None
    tau_wasserstein: Optional[float] = None
 
    # Confounding metrics
    mae_kappa: Optional[float] = None
    mae_kappa_t0: Optional[float] = None
    mae_kappa_t1: Optional[float] = None
    kappa_wasserstein: Optional[float] = None
 
    # Overlap metrics (propensity-based)
    overlap_coeff: Optional[float] = None
    bw0: Optional[float] = None
    bw1: Optional[float] = None
 
 
class CausalEval:
    """
    Minimal causal-structure evaluation:
 
      - Treatment effect fidelity (τ):
          * MAE(τ_pred, τ_target)
          * Corr(τ_pred, τ_target)
          * ATE_pred vs ATE_target
          * Wasserstein distance between τ distributions
 
      - Unmeasured confounding fidelity (κ):
          * MAE(κ_pred, κ_target)
          * Group-wise MAE by T (if T is provided)
          * Wasserstein distance between κ distributions
 
      - Propensity-based overlap:
          * KDE-based overlap coefficient for ê(X) in treated vs control groups,
            with smooth densities on the logit scale.
    """
 
    def __init__(
        self,
        tau_pred: Optional[np.ndarray] = None,
        tau_target: Optional[np.ndarray] = None,
        kappa_pred: Optional[np.ndarray] = None,
        kappa_target: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None
    ):
        """
        Parameters
        ----------
        tau_pred : np.ndarray [N] or [N, d]
            Predicted CATE/ITE τ_θ(X). Flattened internally.
        tau_target : np.ndarray [N] or [N, d]
            Target CATE τ(X) on the same units.
        kappa_pred : np.ndarray [N] or [N, d]
            Predicted confounding bias κ_θ(X, T).
        kappa_target : np.ndarray [N] or [N, d]
            Target confounding κ(X, T).
        T : np.ndarray [N], optional
            Treatment indicator for group-wise κ metrics (0/1).
        """
        self.tau_pred = None if tau_pred is None else np.asarray(tau_pred).reshape(-1)
        self.tau_target = None if tau_target is None else np.asarray(tau_target).reshape(-1)
        self.kappa_pred = None if kappa_pred is None else np.asarray(kappa_pred).reshape(-1)
        self.kappa_target = None if kappa_target is None else np.asarray(kappa_target).reshape(-1)
        self.T = None if T is None else np.asarray(T).reshape(-1)
        self.X = None if X is None else np.asarray(X, dtype=np.float32) 
 
        # place-holders for overlap metrics (to be filled by propensity_overlap)
        self._overlap_metrics: Dict[str, float] = {}
 
    # ------------------------------------------------------------------
    # Treatment effect metrics
    # ------------------------------------------------------------------
    def treatment_effect_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
 
        if self.tau_pred is None or self.tau_target is None:
            return out
 
        diff = self.tau_pred - self.tau_target
        out["mae_tau"] = float(np.mean(np.abs(diff)))
 
        # correlation
        if np.std(self.tau_pred) > 0 and np.std(self.tau_target) > 0:
            corr, _ = pearsonr(self.tau_pred, self.tau_target)
            out["corr_tau"] = float(corr)
        else:
            out["corr_tau"] = None
 
        # ATE
        ate_pred = float(np.mean(self.tau_pred))
        out["ate_pred"] = ate_pred
 
        out["ate_target"] = float(np.mean(self.tau_target))
 
        out["ate_abs_error"] = float(abs(out["ate_pred"] - out["ate_target"]))

        # Wasserstein distance between τ distributions
        out["tau_wasserstein"] = float(
            wasserstein_distance(self.tau_pred, self.tau_target)
        )
        return out
 
    # ------------------------------------------------------------------
    # Confounding metrics
    # ------------------------------------------------------------------
    def confounding_metrics(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if self.kappa_pred is None or self.kappa_target is None:
            return out
 
        diff = self.kappa_pred - self.kappa_target
        out["mae_kappa"] = float(np.mean(np.abs(diff)))
 
        if self.T is not None:
            mask0 = (self.T == 0)
            mask1 = (self.T == 1)
            if mask0.any():
                out["mae_kappa_t0"] = float(np.mean(np.abs(diff[mask0])))
            else:
                out["mae_kappa_t0"] = None
            if mask1.any():
                out["mae_kappa_t1"] = float(np.mean(np.abs(diff[mask1])))
            else:
                out["mae_kappa_t1"] = None
        else:
            out["mae_kappa_t0"] = None
            out["mae_kappa_t1"] = None
 
        out["kappa_wasserstein"] = float(
            wasserstein_distance(self.kappa_pred, self.kappa_target)
        )
 
        return out
 
    # ------------------------------------------------------------------
    # Propensity-based overlap (your KDE-based plot)
    # ------------------------------------------------------------------
    def propensity_overlap(
        self,
        X_gen,
        T_gen,
        plot = True,
        gridsize: int = 600,
        eps: float = 1e-6,
        bw0: Optional[float] = None,
        bw1: Optional[float] = None,
        savepath: Optional[str] = None, # needs to be pdf vector format
        e_hat_override: Optional[np.ndarray] = None,
        shared_bw: bool = True,
    ) -> Dict[str, float]:
        """
        Smooth overlap plot via KDE on logit scale and return overlap metrics.
        This does NOT depend on the decoder; it is purely model-agnostic.
 
        Returns dict:
            {
              'overlap_coeff': float,
              'bw0': float,
              'bw1': float,
              'path': savepath or None
            }
        """
        X_np = np.asarray(X_gen, dtype=np.float32)
        T_np = np.asarray(T_gen, dtype=int).reshape(-1)
        if X_np.ndim != 2 or T_np.ndim != 1 or X_np.shape[0] != T_np.shape[0]:
            raise ValueError("Shapes must be X_gen: [N, V], T_gen: [N]")
 
        # 1) propensities
        if e_hat_override is None:
            e_hat = fit_logistic_torch(X_np, T_np, max_iter=250)
        else:
            e_hat = np.asarray(e_hat_override, dtype=float).reshape(-1)
 
        e_hat = np.clip(e_hat, eps, 1.0 - eps)
 
        # split by T
        e0 = e_hat[T_np == 0]
        e1 = e_hat[T_np == 1]
        u0 = np.log(e0) - np.log(1.0 - e0)
        u1 = np.log(e1) - np.log(1.0 - e1)
 
        # 2) KDE on ℝ
        grid_s = np.linspace(eps, 1.0 - eps, gridsize)
        grid_u = np.log(grid_s) - np.log(1.0 - grid_s)
 
        if shared_bw:
            u_all = np.concatenate([u0, u1])
            bw = _silverman_bandwidth(u_all)
            d0_u, bw0_used = _kde_gaussian_R(grid_u, u0, bw=bw)
            d1_u, bw1_used = _kde_gaussian_R(grid_u, u1, bw=bw)
        else:
            d0_u, bw0_used = _kde_gaussian_R(grid_u, u0, bw=bw0)
            d1_u, bw1_used = _kde_gaussian_R(grid_u, u1, bw=bw1)
 
        # 3) change of variables back to s in (0,1)
        jac = grid_s * (1.0 - grid_s)
        d0_s = d0_u / (jac + 1e-12)
        d1_s = d1_u / (jac + 1e-12)
 
        # normalize numerically
        area0 = np.trapz(d0_s, grid_s)
        area1 = np.trapz(d1_s, grid_s)
        if area0 > 0:
            d0_s = d0_s / area0
        if area1 > 0:
            d1_s = d1_s / area1
 
        # 4) overlap coefficient
        overlap_coeff = float(np.trapz(np.minimum(d0_s, d1_s), grid_s))
 
        # plot
        plt.figure(figsize=(5, 4))# 7, 4.5
        plt.plot(grid_s, d0_s, label="T=0: density of ê(x)")
        plt.plot(grid_s, d1_s, label="T=1: density of ê(x)")
        plt.fill_between(grid_s, np.minimum(d0_s, d1_s), alpha=0.3)
        plt.legend(loc="upper right")
        plt.xlabel("ê(x) = P(T=1 | X)")
        plt.ylabel("Density")
        plt.legend(loc="best")
        plt.tight_layout()
        if savepath:
            plt.savefig(savepath + "overlap_plot.pdf", format="pdf", bbox_inches="tight")
        if plot:
            plt.show()
 
        metrics = {
            "overlap_coeff": overlap_coeff,
            "bw0": float(bw0_used),
            "bw1": float(bw1_used),
            "path": savepath,
        }
        # store internally so all_metrics can include it
        self._overlap_metrics = metrics
        return metrics
 
    # ------------------------------------------------------------------
    # Combined metrics
    # ------------------------------------------------------------------
    
    def all_metrics(self, plot, savepath = None) -> CausalEvalResults: # add varaibles here to show the choices of evaluation
        te = self.treatment_effect_metrics()
        cf = self.confounding_metrics()
        ov = self.propensity_overlap(T_gen = self.T, X_gen = self.X, plot = plot, savepath=savepath)
        if plot:  # Plots
            fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4)) # 10, 4
            self.plot_treatment_effect_scatter(ax=axes1[0])
            self.plot_treatment_effect_distributions(ax=axes1[1])
            plt.tight_layout()
            if savepath:
                fig1.savefig(savepath + "treatment_effect_plots.pdf", format="pdf", bbox_inches="tight")

            fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4)) #10,4
            self.plot_confounding_scatter(ax=axes2[0])
            self.plot_confounding_distributions(ax=axes2[1])
            plt.tight_layout()
            if savepath:
                fig2.savefig(savepath + "confounding_plots.pdf", format="pdf", bbox_inches="tight")
            
        metrics = CausalEvalResults(
            mae_tau=te.get("mae_tau"),
            corr_tau=te.get("corr_tau"),
            ate_pred=te.get("ate_pred"),
            ate_target=te.get("ate_target"),
            ate_abs_error=te.get("ate_abs_error"),
            tau_wasserstein=te.get("tau_wasserstein"),
            mae_kappa=cf.get("mae_kappa"),
            mae_kappa_t0=cf.get("mae_kappa_t0"),
            mae_kappa_t1=cf.get("mae_kappa_t1"),
            kappa_wasserstein=cf.get("kappa_wasserstein"),
            overlap_coeff=ov.get("overlap_coeff"),
            bw0=ov.get("bw0"),
            bw1=ov.get("bw1"),
        )
        def round_dict(d, n=3):
            return {k: round(v, n) if v is not None else None for k, v in d.items()}
        return CausalEvalResults(**round_dict(asdict(metrics), 3))
 
    # ------------------------------------------------------------------
    # Visualization: τ
    # ------------------------------------------------------------------
    def plot_treatment_effect_scatter(self, ax: Optional[plt.Axes] = None, alpha: float = 0.4):
        if self.tau_pred is None or self.tau_target is None:
            raise ValueError("tau_pred and tau_target must be provided for plotting.")
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(self.tau_target, self.tau_pred, alpha=alpha)
        mn = min(self.tau_target.min(), self.tau_pred.min())
        mx = max(self.tau_target.max(), self.tau_pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        ax.set_xlabel(r"Target $\tau(X)$")
        ax.set_ylabel(r"Predicted $\tau_\theta(X)$")
        ax.set_title("CATE fidelity: scatter")
 
    # def plot_treatment_effect_distributions(self, ax: Optional[plt.Axes] = None, bins: int = 30):
    #     if self.tau_pred is None or self.tau_target is None:
    #         raise ValueError("tau_pred and tau_target must be provided for plotting.")
    #     if ax is None:
    #         _, ax = plt.subplots()
    #     ax.hist(self.tau_target, bins=bins, density=True, alpha=0.5, label="Target τ")
    #     ax.hist(self.tau_pred, bins=bins, density=True, alpha=0.5, label=r"Predicted $\tau_\theta$")
    #     ax.set_xlabel("Treatment effect value")
    #     ax.set_ylabel("Density")
    #     ax.set_title("CATE distribution: target vs predicted")
    #     ax.legend()
    
    # updated Jan 11 2026: to deal with the case where the target is constant
    def plot_treatment_effect_distributions(self, ax: Optional[plt.Axes] = None, bins: int = 30):
        if self.tau_pred is None or self.tau_target is None:
            raise ValueError("tau_pred and tau_target must be provided for plotting.")
        if ax is None:
            _, ax = plt.subplots()

        tp = self.tau_pred
        tt = self.tau_target

        tt_val = float(np.mean(tt))
        tp_mean = float(np.mean(tp))
        tp_std = float(np.std(tp))

        is_constant_target = np.std(tt) < 1e-12 * max(1.0, abs(tt_val))

        if is_constant_target:
            half_width = 5.0 * tp_std if tp_std > 0 else 1e-3 * max(1.0, abs(tt_val))
            x_min, x_max = tt_val - half_width, tt_val + half_width
            ax.axvline(tt_val, linestyle="--", linewidth=2, label="Target τ")
        else:
            z = np.concatenate([tt, tp])
            lo, hi = np.percentile(z, [0.5, 99.5])
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            x_min, x_max = lo - pad, hi + pad
            ax.hist(tt, bins=bins, range=(x_min, x_max), density=True, alpha=0.5, label="Target τ")

        ax.hist(tp, bins=bins, range=(x_min, x_max), density=True, alpha=0.5, label=r"Predicted $\tau_\theta$")

        ax.text(
            0.02, 0.98,
            f"target τ = {tt_val:.4g}\npred mean = {tp_mean:.4g}\npred std = {tp_std:.2g}",
            transform=ax.transAxes, va="top", ha="left",
        )

        ax.set_xlabel("Treatment effect value")
        ax.set_ylabel("Density")
        ax.set_title("CATE distribution: target vs predicted")
        ax.set_xlim(x_min, x_max)
        ax.legend()


    # ------------------------------------------------------------------
    # Visualization: κ
    # ------------------------------------------------------------------
    def plot_confounding_scatter(self, ax: Optional[plt.Axes] = None, alpha: float = 0.4):
        if self.kappa_pred is None or self.kappa_target is None:
            raise ValueError("kappa_pred and kappa_target must be provided for plotting.")
        if ax is None:
            _, ax = plt.subplots()
        ax.scatter(self.kappa_target, self.kappa_pred, alpha=alpha)
        mn = min(self.kappa_target.min(), self.kappa_pred.min())
        mx = max(self.kappa_target.max(), self.kappa_pred.max())
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        ax.set_xlabel(r"Target $\kappa(X,T)$")
        ax.set_ylabel(r"Predicted $\kappa_\theta(X,T)$")
        ax.set_title("Confounding fidelity: scatter")
 
    # def plot_confounding_distributions(self, ax: Optional[plt.Axes] = None, bins: int = 30):
    #     if self.kappa_pred is None or self.kappa_target is None:
    #         raise ValueError("kappa_pred and kappa_target must be provided for plotting.")
    #     if ax is None:
    #         _, ax = plt.subplots()
    #     ax.hist(self.kappa_target, bins=bins, density=True, alpha=0.5, label="Target κ")
    #     ax.hist(self.kappa_pred, bins=bins, density=True, alpha=0.5, label=r"Predicted $\kappa_\theta$")
    #     ax.set_xlabel("Confounding bias value")
    #     ax.set_ylabel("Density")
    #     ax.set_title("Confounding distribution: target vs predicted")
    #     ax.legend()

    # updated Jan 11 2026: to deal with the case where the target is constant
    def plot_confounding_distributions(self, ax: Optional[plt.Axes] = None, bins: int = 30):
        if self.kappa_pred is None or self.kappa_target is None:
            raise ValueError("kappa_pred and kappa_target must be provided for plotting.")
        if ax is None:
            _, ax = plt.subplots()

        kp = self.kappa_pred
        kt = self.kappa_target

        kt_val = float(np.mean(kt))
        kp_mean = float(np.mean(kp))
        kp_std = float(np.std(kp))

        is_constant_target = np.std(kt) < 1e-12 * max(1.0, abs(kt_val))

        if is_constant_target:
            # zoom around the constant target to reveal tiny pred variation
            half_width = 5.0 * kp_std if kp_std > 0 else 1e-3 * max(1.0, abs(kt_val))
            x_min, x_max = kt_val - half_width, kt_val + half_width
            ax.axvline(kt_val, linestyle="--", linewidth=2, label="Target κ")
        else:
            # show the full spread of both target and pred
            z = np.concatenate([kt, kp])
            lo, hi = np.percentile(z, [0.5, 99.5])
            pad = 0.1 * (hi - lo) if hi > lo else 1.0
            x_min, x_max = lo - pad, hi + pad
            ax.hist(kt, bins=bins, range=(x_min, x_max), density=True, alpha=0.5, label="Target κ")

        ax.hist(kp, bins=bins, range=(x_min, x_max), density=True, alpha=0.5, label=r"Predicted $\kappa_\theta$")

        ax.text(
            0.02, 0.98,
            f"target κ = {kt_val:.4g}\npred mean = {kp_mean:.4g}\npred std = {kp_std:.2g}",
            transform=ax.transAxes, va="top", ha="left",
        )

        ax.set_xlabel("Confounding bias value")
        ax.set_ylabel("Density")
        ax.set_title("Confounding distribution: target vs predicted")
        ax.set_xlim(x_min, x_max)
        #ax.legend()
        ax.legend(loc="lower left")
