# functions for producing tables for the evaluation of distributional fidelity, causal structure, and privacy.
import numpy as np
import pandas as pd
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional


# -------------------------
# Helpers
# -------------------------
def _as_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {}

def _to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

def _safe_get(d: Any, *keys, default=np.nan):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _models_to_dict(
    models: Any,
    *,
    single_default_name: str = "BGMM",
) -> Dict[str, Any]:
    """
    Accept either:
      - a single result dict  -> returns {"BGMM": result}
      - a dict of model_name -> result
    """
    if isinstance(models, dict) and all(isinstance(k, str) for k in models.keys()):
        # Heuristic: could be either "result dict" or "name->result" dict.
        # If it looks like SynthEval.run_all keys, treat as single result.
        synth_keys = {
            "marginal_continuous", "marginal_discrete", "pairwise_mi",
            "pairwise_discrete", "energy", "c2st", "privacy_dcr"
        }
        if any(k in models for k in synth_keys) or "causal_metrics" in models or "overlap" in models:
            return {single_default_name: models}
        return models
    return {single_default_name: models}


# =========================================================
# Table A: Distributional Fidelity (from SynthEval.run_all)
# =========================================================
def build_table_distributional_fidelity(
    synth_results: Any,
    *,
    include_directions: bool = True,
    single_default_name: str = "BGMM",
) -> pd.DataFrame:
    """
    synth_results:
      - single SynthEval.run_all dict
      - OR dict: {model_name: SynthEval.run_all dict}
    """
    models = _models_to_dict(synth_results, single_default_name=single_default_name)

    rows = [
        ("Marginal (cont.)", "Normalized Wasserstein (mean)",
         "↓ better",
         ("marginal_continuous", "aggregates", "mean_norm_wasserstein")),

        ("Marginal (cont.)", "KSComplement (mean)",
         "↑ better",
         ("marginal_continuous", "aggregates", "mean_KSComplement")),

        ("Marginal (disc.)", "TVComplement (mean)",
         "↑ better",
         ("marginal_discrete", "aggregates", "mean_TVComplement")),

        ("Pairwise (cont–cont.)", "CorrelationSimilarity",
         "↑ better",
         ("pairwise_continuous", "CorrelationSimilarity")),

        ("Pairwise (all vars)", "SU similarity (mean)",
         "↑ better",
         ("pairwise_mi", "SU_similarity_mean")),

        ("Pairwise (disc–disc.)", "ContingencySimilarity (mean)",
         "↑ better",
         ("pairwise_discrete", "ContingencySimilarity_mean")),

        ("Conditional (all except C)", "Weighted MMD$^2$",
         "↓ better",
         ("conditional_mmd2", "weighted_mean_mmd2")),

        ("Conditional (all except C)", "Normalized MMD$^2$ ratio vs real",
         "↓ better; 1 = real-data baseline",
         ("conditional_mmd2", "normalized_ratio_vs_real")),

        ("Joint (all vars)", "Normalized Energy Distance",
         "↓ better",
         ("energy", "normalized_energy")),

        ("Joint (all vars)", "C2ST (AUC complement)",
         "↑ better",
         ("c2st", "auc_complement")),
    ]

    out_rows = []
    for category, metric, direction, path in rows:
        row = {"Category": category, "Metric": metric}
        if include_directions:
            row["Direction"] = direction
        for name, res in models.items():
            row[name] = _to_float(_safe_get(res, *path))
        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    return df


# =========================================================
# Table B: Causal-Structure Fidelity
# TE+Confounding from causal_metrics; Overlap from overlap_diagnostics
# =========================================================
def build_table_causal_structure(
    causal_results: Any,
    *,
    include_directions: bool = True,
    single_default_name: str = "BGMM",
) -> pd.DataFrame:
    """
    causal_results:
      - single dict output of CausalMix.causal_bgmm()/causal_gauss()
      - OR dict: {model_name: causal_output_dict}
    Each causal_output_dict should include:
      - "causal_metrics": dataclass or dict
      - "overlap": dict from overlap_diagnostics (namespaced keys)
    """
    models = _models_to_dict(causal_results, single_default_name=single_default_name)

    rows = [
        # ---- Treatment effect ----
        ("Treatment Effect", "CATE/ITE MAE",
         "↓ better",
         ("ce", "mae_tau")),
        ("Treatment Effect", "CATE Correlation",
         "↑ better",
         ("ce", "corr_tau")),
        ("Treatment Effect", "ATE Error",
         "↓ better",
         ("ce", "ate_abs_error")),
        ("Treatment Effect", "TE Distribution Distance (W1)",
         "↓ better",
         ("ce", "tau_wasserstein")),

        # ---- Confounding ----
        ("Confounding", "Confounding MAE",
         "↓ better",
         ("ce", "mae_kappa")),
        ("Confounding", "Group-wise MAE (T=0)",
         "↓ better",
         ("ce", "mae_kappa_t0")),
        ("Confounding", "Group-wise MAE (T=1)",
         "↓ better",
         ("ce", "mae_kappa_t1")),
        ("Confounding", "Confounding Dist. (W1)",
         "↓ better",
         ("ce", "kappa_wasserstein")),

        # ---- Overlap (decoder) ----
        ("Overlap (decoder)", "MSE",
         "↓ better",
         ("ov", "dec/mse_to_target")),
        ("Overlap (decoder)", "Fraction within tolerance",
         "↑ better",
         ("ov", "dec/fraction_within_tol")),

        # ---- Overlap (propensity) ----
        ("Overlap (propensity)", "Propensity AUC",
         "NA",
         ("ov", "ps/auc")),
        ("Overlap (propensity)", "Histogram overlap coefficient",
         "↑ better",
         ("ov", "ps/hist_overlap_coeff")),
        ("Overlap (propensity)", "Common support fraction",
         "↑ better",
         ("ov", "ps/frac_common_support")),
        ("Overlap (propensity)", "Common support fraction (T=0)",
         "↑ better",
         ("ov", "ps/frac_common_support_t0")),
        ("Overlap (propensity)", "Common support fraction (T=1)",
         "↑ better",
         ("ov", "ps/frac_common_support_t1")),
    ]

    out_rows = []
    for category, metric, direction, (src, key) in rows:
        row = {"Category": category, "Metric": metric}
        if include_directions:
            row["Direction"] = direction

        for name, out in models.items():
            ce = _as_dict(out.get("causal_metrics", {}))
            ov = _as_dict(out.get("overlap", {}))
            if src == "ce":
                row[name] = _to_float(ce.get(key, np.nan))
            else:
                row[name] = _to_float(ov.get(key, np.nan))

        out_rows.append(row)

    return pd.DataFrame(out_rows)


# =========================================================
# Table C: Privacy (from SynthEval.run_all -> privacy_dcr)
# =========================================================
def build_table_privacy(
    synth_results: Any,
    *,
    include_directions: bool = True,
    single_default_name: str = "BGMM",
) -> pd.DataFrame:
    """
    synth_results:
      - single SynthEval.run_all dict
      - OR dict: {model_name: SynthEval.run_all dict}
    """
    models = _models_to_dict(synth_results, single_default_name=single_default_name)

    rows = [
        ("DCR", "Protection Fraction", 
         "↑ better",
         ("privacy_dcr", "protection_fraction")),
        ("DCR", "Distance Ratio (mean)",
         "↑ better",
         ("privacy_dcr", "ratio_mean")),
        ("DCR", "Distance Ratio (p5)",
         "↑ better",
         ("privacy_dcr", "ratio_p5")),
        ("DCR", "Distance Ratio (p50)",
         "↑ better",
         ("privacy_dcr", "ratio_p50")),
        ("DCR", "Distance Ratio (p95)",
         "↑ better",
         ("privacy_dcr", "ratio_p95")),
        ("DCR", "Standardized Distance Ratio",
         "↑ better",
         ("privacy_dcr", "sdmetrics_DCRBaseline")),
    ]

    out_rows = []
    for category, metric, direction, path in rows:
        row = {"Category": category, "Metric": metric}
        if include_directions:
            row["Direction"] = direction

        for name, res in models.items():
            row[name] = _to_float(_safe_get(res, *path))
        out_rows.append(row)

    return pd.DataFrame(out_rows)

# ==============================================================================================
# Tables of Mean (sd) across n generated datasets for the three tables distribution, privacy, and causal structure
# ==============================================================================================
# -------------------------
# Mean (sd) aggregator (concise)
# -------------------------
def table_mean_sd(tables, decimals=3):
    """tables: list[pd.DataFrame] with same shape/index/cols. Returns mean(sd) strings."""
    idx, cols = tables[0].index, tables[0].columns
    num_cols = [c for c in cols if c not in ("Category", "Metric", "Direction")]  # keep your meta cols as-is

    out = tables[0][[c for c in cols if c not in num_cols]].copy()  # keep Category/Metric/Direction
    for c in num_cols:
        X = np.vstack([pd.to_numeric(t[c], errors="coerce").to_numpy() for t in tables])  # [R, n_rows]
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=1) if X.shape[0] > 1 else np.zeros_like(mu)
        out[c] = [f"{m:.{decimals}f} ({s:.{decimals}f})" if np.isfinite(m) else "" for m, s in zip(mu, sd)]
    return out

# -------------------------
# Run R times and build mean(sd) tables
# -------------------------
def eval_tables_mean(
    model,
    real_df: pd.DataFrame,
    schema,
    *,
    R: int = 10,
    conditional_col: str = "exp",
    n_samples: int = None,
    deterministic: bool = False,
    return_probs: bool = False,
    plot_each_run: bool = False,  # set True only if you really want plots every run
    random_state: int = 0,
):
    n_samples = len(real_df) if n_samples is None else n_samples
    rng = np.random.RandomState(random_state)

    t_dist_runs, t_priv_runs, t_causal_runs = [], [], []

    for r in range(R):
        # optional: change randomness per run if you want reproducibility across reps
        # (only effective if your sampling uses numpy random_state somewhere)
        _ = rng.randint(0, 10**9)

        out_bgmm = model.causal_bgmm(n_samples=n_samples, deterministic=deterministic,
                                 return_probs=return_probs, plot=plot_each_run)
        out_gauss = model.causal_gauss(n_samples=n_samples, deterministic=deterministic,
                                    return_probs=return_probs, plot=plot_each_run)

        # distributional + privacy
        dist_all = {}
        for name, out in {"BGMM": out_bgmm, "Gaussian": out_gauss}.items():
            dist_all[name] = SynthEval(real_df, out["df_gen"], schema).run_all(conditional_col=conditional_col)

        t_dist_runs.append(build_table_distributional_fidelity(dist_all))
        t_priv_runs.append(build_table_privacy(dist_all))

        # causal table needs causal outputs (with causal_metrics + overlap inside)
        causal_all = {"BGMM": out_bgmm, "Gaussian": out_gauss}
        t_causal_runs.append(build_table_causal_structure(causal_all))

    t_dist = table_mean_sd(t_dist_runs, decimals=3)
    t_priv = table_mean_sd(t_priv_runs, decimals=3)
    t_causal = table_mean_sd(t_causal_runs, decimals=3)
    return t_causal, t_dist, t_priv
