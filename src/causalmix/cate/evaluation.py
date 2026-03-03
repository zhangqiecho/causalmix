"""Evaluation utilities for CATE experiments."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# ---- Metrics / evaluation ----

def evaluate_estimator(results, true_cate, true_ate, estimator_name="Estimator"):
    """
    Compute evaluation metrics for a causal inference estimator.
    
    Parameters
    ----------
    results : dict
        Output from xlearner_binary() or similar estimator
    true_cate : array-like
        True CATE values for each observation
    true_ate : float
        True ATE value
    estimator_name : str
        Name of the estimator for reporting
        
    Returns
    -------
    dict with evaluation metrics
    """
    import numpy as np
    
    # ATE metrics
    ate_bias = results["ATE"] - true_ate
    ate_coverage = 1 if (results["ATE_lower"] <= true_ate <= results["ATE_upper"]) else 0
    
    # CATE metrics
    cate_errors = results["CATE"] - true_cate
    cate_rmse = np.sqrt(np.mean(cate_errors ** 2))
    cate_covered = (results["CATE_lower"] <= true_cate) & (true_cate <= results["CATE_upper"])
    cate_coverage = np.mean(cate_covered)

    # --- Runtime ---
    runtime_sec = results.get("runtime_sec", np.nan)
    
    return {
        "estimator": estimator_name,
        "runtime_sec": runtime_sec,
        "ate_bias": ate_bias,
        "ate_stderr": results["ATE_stderr"],
        "ate_coverage": ate_coverage,
        "cate_rmse": cate_rmse,
        "cate_coverage": cate_coverage,
    }

def compare_estimators(estimator_results_list):
    """
    Compare multiple estimators and create a summary table.
    
    Parameters
    ----------
    estimator_results_list : list of dict
        List of evaluation results from evaluate_estimator_performance()
        
    Returns
    -------
    pd.DataFrame
        Comparison table with all metrics
    """
    return pd.DataFrame(estimator_results_list)

def evaluate_estimator_rep(
    results: Dict[str, Any],
    true_cate: np.ndarray,
    true_ate: float,
    estimator_name: str
) -> Dict[str, Any]:
    """
    Per-rep evaluation metrics (one dataset).

    Added Feb 23 2026:
      - cate_ci_width_mean: mean_i (CATE_upper_i - CATE_lower_i)
      - cate_bias_abs_mean: mean_i |CATE_hat_i - true_cate_i|
    """
    # Ensure arrays
    cate_hat = np.asarray(results["CATE"], dtype=float).reshape(-1)
    cate_lb  = np.asarray(results["CATE_lower"], dtype=float).reshape(-1)
    cate_ub  = np.asarray(results["CATE_upper"], dtype=float).reshape(-1)
    true_cate = np.asarray(true_cate, dtype=float).reshape(-1)

    # ATE
    ate_hat = float(results["ATE"])
    ate_lb  = float(results["ATE_lower"])
    ate_ub  = float(results["ATE_upper"])
    ate_bias = ate_hat - float(true_ate)
    ate_coverage = int(ate_lb <= true_ate <= ate_ub)

    # CATE
    cate_err = cate_hat - true_cate
    cate_rmse = float(np.sqrt(np.mean(cate_err ** 2)))
    cate_bias_abs_mean = float(np.mean(np.abs(cate_err)))

    # CATE CI width (mean)
    cate_ci_width_mean = float(np.mean(cate_ub - cate_lb))

    # CATE coverage
    cate_covered = (cate_lb <= true_cate) & (true_cate <= cate_ub)
    cate_coverage = float(np.mean(cate_covered))

    return {
        "estimator": estimator_name,
        "runtime_sec": float(results.get("runtime_sec", np.nan)),

        # ATE per-dataset quantities
        "ate_hat": ate_hat,
        "ate_bias": float(ate_bias),
        "ate_stderr": float(results.get("ATE_stderr", np.nan)),
        "ate_coverage": int(ate_coverage),

        # CATE per-dataset quantities
        "cate_rmse": cate_rmse,
        "cate_bias_abs_mean": cate_bias_abs_mean,
        "cate_ci_width_mean": cate_ci_width_mean,
        "cate_coverage": cate_coverage,
    }

def summarize_results(per_rep_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate across reps (datasets) for each estimator.

    ATE summary requested:
      - RMSE
      - mean bias
      - variance/std of bias
      - coverage probability (mean coverage indicator)

    CATE summary requested:
      - mean RMSE
      - variance/std of the bias (we interpret as per-dataset mean CATE bias)
      - mean/SD of mean absolute CATE bias # added 2/23/2026
      - mean/SD of mean CATE CI width # added 2/23/2026
      - mean coverage probability
    """
    g = per_rep_df.groupby("estimator", as_index=False)

    summary = g.agg(
        n_reps=("estimator", "size"),

        # runtime
        runtime_mean_sec=("runtime_sec", "mean"),
        runtime_sd_sec=("runtime_sec", "std"),

        # ATE
        ate_rmse=("ate_bias", lambda x: np.sqrt(np.mean((x ** 2)))), # added RMSE of ATE
        ate_bias_mean=("ate_bias", "mean"),
        ate_bias_sd=("ate_bias", "std"),
        ate_stderr_mean=("ate_stderr", "mean"), # added mean standard error of ATE
        ate_coverage=("ate_coverage", "mean"),

        # CATE
        cate_rmse_mean=("cate_rmse", "mean"),
        cate_rmse_sd=("cate_rmse", "std"),
        cate_coverage_mean=("cate_coverage", "mean"),

        # CATE (added 2/23/2026)
        cate_bias_abs_mean_mean=("cate_bias_abs_mean", "mean"),
        cate_bias_abs_mean_sd=("cate_bias_abs_mean", "std"),
        cate_ci_width_mean_mean=("cate_ci_width_mean", "mean"),
        cate_ci_width_mean_sd=("cate_ci_width_mean", "std"),
    )

    cols = [
        "estimator", "n_reps",
        "runtime_mean_sec", "runtime_sd_sec", "ate_rmse",
        "ate_bias_mean", "ate_bias_sd", "ate_stderr_mean", "ate_coverage",
        "cate_rmse_mean", "cate_rmse_sd", "cate_bias_abs_mean_mean", "cate_bias_abs_mean_sd",
        "cate_ci_width_mean_mean", "cate_ci_width_mean_sd", "cate_coverage_mean",
    ]
    return summary[cols]
