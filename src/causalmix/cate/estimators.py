"""CATE estimator wrappers used in the paper experiments.

Each estimator returns a dict containing individual-level CATE estimates aligned to
the input dataframe. Some estimators also provide uncertainty intervals or ATE summaries.

Heavy dependencies are imported inside functions where possible.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---- Estimators ----

def bayesian_causal_forest(
    data,
    outcome,
    treatment,
    covariates=None,
    nburn=1000,
    nsim=2000,
    alpha=0.05,
    random_state=36
):
    """
    Bayesian Causal Forest (bcf R package) via rpy2.

    Returns dict with:
      - CATE (pd.Series) and pointwise credible intervals (np arrays)
      - ATE mean, posterior SD, and credible interval
    """
    # Import rpy2 inside the function to avoid pickling issues
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter
    import rpy2.robjects.numpy2ri as numpy2ri
    from rpy2.robjects.vectors import FloatVector
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Add R library path before importing bcf
    ro.r('.libPaths(c(.libPaths(), "/local_disk0/.ephemeral_nfs/envs/rEnv-d45f9e43-bbb7-4f6d-827c-87c730d96e2f"))')
    
    bcf = importr("bcf")
    base = importr("base")

    n = data.shape[0]

    # --- Prepare numeric inputs ---
    y_np = data[outcome].to_numpy(dtype=float).reshape(-1)
    z_np = data[treatment].to_numpy(dtype=float).reshape(-1)

    # check binary treatment coding
    if not set(np.unique(z_np[~np.isnan(z_np)])).issubset({0.0, 1.0}):
        raise ValueError("Treatment z must be coded as 0/1 for bcf().")

    # X must be numeric matrix
    if covariates is None:
        X_df = data.drop(columns=[outcome, treatment]).copy()
    else:
        X_df = data[covariates].copy()
    # If you have label-encoded categoricals, consider one-hot encoding instead:
    # X_df = pd.get_dummies(X_df, drop_first=True)

    X_np = X_df.to_numpy(dtype=float)

    if np.isnan(X_np).any() or np.isnan(y_np).any() or np.isnan(z_np).any():
        raise ValueError("Missing values detected; handle NAs before calling bcf().")

    # --- Estimate propensity pihat in Python ---
    #ps_model = LogisticRegression()
    ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100))
    ps_model.fit(X_np, z_np)
    pihat_np = ps_model.predict_proba(X_np)[:, 1].astype(float)

    # Avoid exactly 0 or 1 
    eps = 1e-6
    pihat_np = np.clip(pihat_np, eps, 1 - eps)

    # --- Convert to R objects ---
    # y, z as true vectors
    y_r = base.as_numeric(FloatVector(y_np))
    z_r = base.as_numeric(FloatVector(z_np))
    pihat_r = base.as_numeric(FloatVector(pihat_np))

    # X as R matrix
    with localconverter(default_converter + numpy2ri.converter):
        X_r = ro.conversion.py2rpy(X_np)
    X_r = base.matrix(X_r, nrow=X_np.shape[0], ncol=X_np.shape[1])

    # reproducibility
    ro.r["set.seed"](random_state)
    # force burnin an nsim to R integers, otherwise they were identified as tuples
    nburn_r = base.as_integer(int(np.array(nburn).reshape(-1)[0]))
    nsim_r   = base.as_integer(int(np.array(nsim).reshape(-1)[0]))

    # --- Fit BCF ---
    fit = bcf.bcf(
        y=y_r,
        z=z_r,
        x_control=X_r,
        x_moderate=X_r,
        pihat=pihat_r,
        nburn=nburn_r,
        nsim=nsim_r,
        n_threads=1 # to avoid oversubscription
    )

    # --- Extract tau draws ---
    # tau is usually an R matrix: (nsim, n) or (n, nsim)
    with localconverter(default_converter + numpy2ri.converter):
        tau = np.array(ro.conversion.rpy2py(fit.rx2("tau")))

    # Ensure shape is (n_draws, n_units)
    if tau.ndim != 2:
        raise ValueError(f"Unexpected tau ndim={tau.ndim}, shape={tau.shape}")
    if tau.shape[1] == n:
        tau_samples = tau
    elif tau.shape[0] == n:
        tau_samples = tau.T
    else:
        raise ValueError(f"tau shape {tau.shape} doesn't match n={n} in either dimension.")

    # --- CATE posterior mean + pointwise credible intervals ---
    cate_mean = tau_samples.mean(axis=0)
    alpha = float(np.array(alpha).reshape(-1)[0])
    cate_ci_lower = np.quantile(tau_samples, alpha/ 2, axis=0)
    cate_ci_upper = np.quantile(tau_samples, 1 - alpha / 2, axis=0)

    cate_mean = pd.Series(cate_mean, index=data.index, name="CATE")

    # --- ATE posterior (average across units within each draw) ---
    ate_samples = tau_samples.mean(axis=1)
    ate_mean = float(ate_samples.mean())
    ate_sd = float(ate_samples.std(ddof=1))
    ate_ci_lower = float(np.quantile(ate_samples, alpha / 2))
    ate_ci_upper = float(np.quantile(ate_samples, 1 - alpha / 2))

    return {
        "CATE": cate_mean,
        "CATE_lower": cate_ci_lower,
        "CATE_upper": cate_ci_upper,
        "ATE": ate_mean,
        "ATE_lower": ate_ci_lower,
        "ATE_upper": ate_ci_upper,
        "ATE_stderr": ate_sd,
    }

def causal_forest(data, 
                  outcome, 
                  treatment, 
                  covariates=None, 
                  alpha=0.05, 
                  num_trees=2000,
                  min_node_size=5,
                  random_state=36):
    """
    Estimate CATE and ATE using GRF causal forest.

    Parameters
    ----------
    outcome : str
        Name of outcome variable
    treatment : str
        Name of treatment variable
    data : pd.DataFrame
        Dataset containing outcome, treatment, and covariates
    covariates : list of str, optional
        List of covariates to use. If None, all other columns are used.
    alpha : float
        Significance level for 100*(1-alpha)% confidence intervals

    Returns
    -------
    dict with:
        - cate_df: DataFrame with CATE, standard error, CI lower/upper
        - ate: average treatment effect
        - ate_se: standard error of ATE
        - ate_ci: tuple with 95% CI of ATE
    """
    # Import rpy2 inside the function to avoid pickling issues
    from scipy.stats import norm
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import default_converter
    import rpy2.robjects.numpy2ri as numpy2ri
    from rpy2.robjects.vectors import FloatVector
    
    # Import GRF
    grf = importr("grf")
    base = importr("base")
    if covariates is None:
        covariates = [c for c in data.columns if c not in [outcome, treatment]]

    # Explicit conversion to R objects
    # --- Force clean numeric arrays ---
    X_np = data[covariates].to_numpy(dtype=float)          # numeric matrix
    Y_np = data[outcome].to_numpy(dtype=float).ravel()     # numeric vector
    W_np = data[treatment].to_numpy(dtype=float).ravel()   # numeric 0/1

    # Optional: sanity checks
    if not set(np.unique(W_np)).issubset({0.0, 1.0}):
        raise ValueError("Treatment W must be coded as 0/1.")
    if np.isnan(X_np).any() or np.isnan(Y_np).any() or np.isnan(W_np).any():
        raise ValueError("Missing values detected; grf does not like NAs without handling.")

    # Explicit conversion to R objects
    # --- Force clean numeric arrays ---
    X_np = data[covariates].to_numpy(dtype=float)          # numeric matrix
    Y_np = data[outcome].to_numpy(dtype=float).ravel()     # numeric vector
    W_np = data[treatment].to_numpy(dtype=float).ravel()   # numeric 0/1

    # Optional: sanity checks
    if not set(np.unique(W_np)).issubset({0.0, 1.0}):
        raise ValueError("Treatment W must be coded as 0/1.")
    if np.isnan(X_np).any() or np.isnan(Y_np).any() or np.isnan(W_np).any():
        raise ValueError("Missing values detected; grf does not like NAs without handling.")

    # ---- EXPLICIT input conversion ----
    with localconverter(default_converter + numpy2ri.converter):
        Xr = ro.conversion.py2rpy(X_np)
    # Ensure X is an R matrix (sometimes comes through as array; this is safe)
    Xr = base.matrix(Xr, nrow=X_np.shape[0], ncol=X_np.shape[1])
    Yr = base.as_numeric(FloatVector(Y_np))
    Wr = base.as_numeric(FloatVector(W_np))

    # Reproducibility
    ro.r["set.seed"](random_state)

    # Fit causal forest (honest splitting internally)
    cf = grf.causal_forest(Xr, Yr, Wr, num_trees=num_trees, min_node_size=min_node_size, **{"num.threads": 1}) # set the num.threads = 1 to avoid oversubscription
    # Predict CATE with variance estimates enabled
    pred = ro.r["predict"](cf, Xr, **{"estimate.variance": True})

    # Extract predictions + variances
    with localconverter(default_converter + numpy2ri.converter):
        tau_hat = np.array(ro.conversion.rpy2py(pred.rx2("predictions"))).ravel()
        tau_var = np.array(ro.conversion.rpy2py(pred.rx2("variance.estimates"))).ravel()

    tau_se = np.sqrt(np.maximum(tau_var, 0.0))
    # Pointwise 100*(1-alpha)% CI
    z = norm.ppf(1 - alpha/2)
    cate_lb = tau_hat - z * tau_se
    cate_ub = tau_hat + z * tau_se

    # Average treatment effect using GRF's built-in function
    ate_res = grf.average_treatment_effect(cf)  # returns (estimate, SE)
    ate = float(ate_res.rx2("estimate")[0])
    ate_se = float(ate_res.rx2("std.err")[0])
    ate_lb = ate - z*ate_se
    ate_ub = ate + z*ate_se
    # ate_ci = (ate - z*ate_se, ate + z*ate_se)

    return {
            # CATE results (individual-level)
            "CATE": tau_hat,
            "CATE_lower": cate_lb,
            "CATE_upper": cate_ub,
            
            # ATE results (population-level)
            "ATE": ate,
            "ATE_lower": ate_lb,
            "ATE_upper": ate_ub,
            "ATE_stderr": ate_se,
    }

def xlearner_binary(
    df,
    outcome_col="Y",
    treatment_col="T",
    feature_cols=None,
    method="GBR",
    inference_method="bootstrap",
    n_bootstrap=100,
    ci_level=0.95,
    random_state=42
):
    """
    X-learner for binary treatment and binary outcome.
    Returns ATE, ATE standard error, ATE CI, CATE, CATE CI using econml's built-in inference.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    outcome_col : str
        Binary outcome column
    treatment_col : str
        Binary treatment column
    feature_cols : list
        Covariate columns
    method : {"GBR", "linear"}
        Base learner choice
    inference_method : {"bootstrap", "None"}
        Inference method to use
    n_bootstrap : int
        Number of bootstrap resamples (if using bootstrap inference)
    ci_level : float
        Confidence level for intervals
    random_state : int
        Random seed
        
    Returns
    -------
    dict with keys:
        - "CATE": array of conditional average treatment effects
        - "CATE_lower": array of CATE lower bounds of confidence intervals
        - "CATE_upper": array of CATE upper bounds of confidence intervals
        - "ATE": scalar average treatment effect
        - "ATE_lower": array of ATE lower bounds of confidence intervals
        - "ATE_upper": array of ATE upper bounds of confidence intervals
        - "ATE_stderr": ATE standard error
    """

    from econml.metalearners import XLearner
    from econml.inference import BootstrapInference
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegressionCV, LassoCV

    if feature_cols is None:
        feature_cols = [
            c for c in df.columns if c not in [outcome_col, treatment_col]
        ]
    
    X = df[feature_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    n = df.shape[0]
    # ----------------------------
    # Model specification
    # ----------------------------
    outcome_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100))
    propensity_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100))
    if method == "linear":
        # outcome_model = LogisticRegressionCV() #LogisticRegression()
        # propensity_model = LogisticRegressionCV() #LogisticRegression()
        # cate_model = StatsModelsLinearRegression()
        cate_model = LassoCV(cv=3, random_state=random_state) # use lasso when effect heterogeneity is sparse: only a few categories/variables truly modify treatment effect.
        #cate_model = RidgeCV() # compared to lasso, ridge handles correlated covariates more smoothly, producing lower variance (of cate) across replications
    elif method == "GBR":
        # outcome_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100))
        # propensity_model = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100))
        cate_model=GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100)) # consider using RandomForestClassifier() if too slow
    else:
        raise ValueError("method must be 'linear' or 'GBR'")
    
    # ----------------------------
    # Fit X-learner with inference
    # ----------------------------
    # Configure inference
    if inference_method == "bootstrap":
        inference = BootstrapInference(
            n_bootstrap_samples=n_bootstrap,
            n_jobs=1 # to avoid oversubscription
            #n_jobs=-1 # use all available cpu cores (Maximum parallelization)
        )
    else:
        inference = inference_method
    
    x_learner = XLearner(
        models=outcome_model,
        cate_models=cate_model, # need to specify cate_models when outcome is binary
        propensity_model=propensity_model,
    )
    
    # Fit with inference enabled
    x_learner.fit(Y=Y, T=T, X=X, inference=inference)
    # alpha calculated from Confidence intervals
    alpha = 1 - ci_level
    # ----------------------------
    # CATE estimates and inference
    # ----------------------------
    # # Full CATE inference (includes standard errors, p-values, etc.)
    cate_inference = x_learner.effect_inference(X)
    # Point estimates
    cate_hat = cate_inference.point_estimate  # alternatively, using: cate_hat = x_learner.effect(X)
    # cate_stderr = cate_inference.stderr       # Standard errors, removed due to the large amount of values
    cate_lb, cate_ub = cate_inference.conf_int(alpha=alpha)  # Confidence intervals
    # alternatively, to get CIs directly:
    # cate_lb, cate_ub = x_learner.effect_interval(X, alpha=alpha)
    
    # ----------------------------
    # ATE estimates and inference
    # ----------------------------
    # Point estimate
    # Get ATE inference results
    ate_inference = x_learner.ate_inference(X)
    ate_hat = ate_inference.mean_point # point ate estimate,alternatively, using: ate_hat = x_learner.ate(X)
    ate_stderr = ate_inference.stderr_mean    # Standard error
    ate_lb, ate_ub = ate_inference.conf_int_mean(alpha=alpha)  # Confidence interval
    # alternately, to get CIs directly:
    # ate_lb, ate_ub = x_learner.ate_interval(X, alpha=alpha)
    # ----------------------------
    # Return results
    # ----------------------------
    return {
        # CATE results (individual-level)
        "CATE": cate_hat,
        "CATE_lower": cate_lb,
        "CATE_upper": cate_ub,
        
        # ATE results (population-level)
        "ATE": ate_hat,
        "ATE_lower": ate_lb,
        "ATE_upper": ate_ub,
        "ATE_stderr": ate_stderr,
    }

def dml_binary(
    df,
    outcome_col="Y",
    treatment_col="T",
    feature_cols=None,
    method="auto",
    method_cate="GBR",
    #inference_method="auto",
    n_bootstrap=100,
    ci_level=0.95,
    random_state=42
):
    """
    Double Machine Learning (DML) for binary treatment and binary outcome.
    Returns ATE, ATE standard error, ATE CI, CATE, CATE CI using econml's built-in inference.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    outcome_col : str
        Binary outcome column
    treatment_col : str
        Binary treatment column
    feature_cols : list
        Covariate columns
    method : {"auto", "GBR", "linear"}
        Base learner choice for outcome and propensity models: "auto": the model will be the best-fitting of a set of linear and forest models according to the NonparamDML and LinearDML package
    method_cate : {"GBR", "lasso", "linear"}
        Different cate stage models corrresponds to different DML estimators and inference methods
        Type of DML estimator:
        - "nonparam": NonParamDML (flexible, allows heterogeneous effects)，depends on final model (model_final) specification
        - "linear": LinearDML (assumes linear treatment effect model), better for lower dimensional settings
        Inference method to use
        - "auto": Use DML's asymptotic inference (default, faster)
        - "bootstrap": Use bootstrap inference (when using NonParamDML, we may need to specify bootstrap inference to get valid CIs)
    n_bootstrap : int
        Number of bootstrap resamples (if using bootstrap inference)
    ci_level : float
        Confidence level for intervals
    random_state : int
        Random seed
        
    Returns
    -------
    dict with keys:
        - "CATE": array of conditional average treatment effects
        - "CATE_lower": array of CATE lower confidence bounds
        - "CATE_upper": array of CATE upper confidence bounds
        - "ATE": scalar average treatment effect
        - "ATE_lower": ATE lower confidence bound
        - "ATE_upper": ATE upper confidence bound
        - "ATE_stderr": ATE standard error
    """
    from econml.dml import LinearDML, NonParamDML
    from econml.inference import BootstrapInference
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LassoCV
    
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns if c not in [outcome_col, treatment_col]
        ]
    
    X = df[feature_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    n = df.shape[0]
    
    # ----------------------------
    # Model specification
    # ----------------------------
    if method == "auto":
        model_y = 'auto'
        model_t = 'auto'
    elif method == "linear":
        model_y = LogisticRegression()
        model_t = LogisticRegression()
    elif method == "GBR":
        model_y = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100), random_state=random_state)
        model_t = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100), random_state=random_state)
    else:
        raise ValueError("method must be 'auto','linear', or 'GBR'")
    

    
    # ----------------------------
    # DML estimator selection and fitting
    # ----------------------------
    if method_cate == "lasso":
        inference_method="bootstrap"
        # NonParamDML allows flexible heterogeneous treatment effects
        # Uses a final model to learn CATE as a function of features
        model_final = LassoCV(cv=3, random_state=random_state)
        dml_estimator = NonParamDML(
            model_y=model_y,
            model_t=model_t,
            model_final=model_final,
            discrete_treatment=True,
            discrete_outcome=True,
            cv=3,
            random_state=random_state
        )
    elif method_cate == "GBR":
        inference_method="bootstrap"
        model_final = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=int(n/100), random_state=random_state) # doesn't provide `prediction_stderr` method for "auto" inference
        
        dml_estimator = NonParamDML(
            model_y=model_y,
            model_t=model_t,
            model_final=model_final,
            discrete_treatment=True,
            discrete_outcome=True,
            cv=3,
            random_state=random_state
        )
    elif method_cate == "linear":
        inference_method="auto"
        # LinearDML assumes treatment effect is linear in features
        dml_estimator = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            discrete_outcome=True,
            cv=3,
            random_state=random_state
        )
    else:
        raise ValueError("dml_type must be 'nonparam' or 'linear'")

    # ----------------------------
    # Configure inference
    # ----------------------------
    if inference_method == "bootstrap":
        inference = BootstrapInference(
            n_bootstrap_samples=n_bootstrap,
            n_jobs=1 # to avoid oversubscription
            #n_jobs=-1 
        )
    else:
        inference = inference_method  # use "auto" for asymptotic inference of linear DML
    
    # Fit with inference enabled
    dml_estimator.fit(Y=Y, T=T, X=X, inference=inference)
    
    # Calculate alpha from confidence level
    alpha = 1 - ci_level
    
    # ----------------------------
    # CATE estimates and inference
    # ----------------------------
    cate_inference = dml_estimator.effect_inference(X)
    cate_hat = cate_inference.point_estimate.ravel()
    cate_lb, cate_ub = cate_inference.conf_int(alpha=alpha)
    cate_lb = cate_lb.ravel()  # Flatten to 1D
    cate_ub = cate_ub.ravel()  # Flatten to 1D
    
    # ----------------------------
    # ATE estimates and inference
    # ----------------------------
    ate_inference = dml_estimator.ate_inference(X)
    ate_hat = float(ate_inference.mean_point)
    ate_stderr = float(ate_inference.stderr_mean)
    ate_lb, ate_ub = ate_inference.conf_int_mean(alpha=alpha)
    ate_lb = float(ate_lb) # to scalar
    ate_ub = float(ate_ub) 
    
    # ----------------------------
    # Return results
    # ----------------------------
    return {
        # CATE results (individual-level)
        "CATE": cate_hat,
        "CATE_lower": cate_lb,
        "CATE_upper": cate_ub,
        
        # ATE results (population-level)
        "ATE": ate_hat,
        "ATE_lower": ate_lb,
        "ATE_upper": ate_ub,
        "ATE_stderr": ate_stderr,
    }
# "nonparam" vs. "linear", both with "auto" or "GBR"

def drlearner_binary(
    df,
    outcome_col="Y",
    treatment_col="T",
    feature_cols=None,
    method="auto",
    method_cate="GBR",
    #inference_method="auto",
    n_bootstrap=100,
    ci_level=0.95,
    random_state=42
):
    """
    Doubly Robust Learner (DRLearner) for binary treatment and binary outcome.
    Returns ATE, ATE standard error, ATE CI, CATE, CATE CI using econml's built-in inference.
    
    DRLearner combines outcome regression and propensity score weighting for
    doubly robust estimation - consistent if either the outcome model OR the
    propensity model is correctly specified.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    outcome_col : str
        Binary outcome column
    treatment_col : str
        Binary treatment column
    feature_cols : list
        Covariate columns
    method : {"auto", "GBR", "linear"}
        Base learner choice for outcome and propensity models
        - "auto": the model will be the best-fitting of a set of linear and forest models
        - "GBR": Gradient Boosting
        - "linear": Logistic Regression
    inference_method : {"auto", "bootstrap"}
        Inference method to use
        - "auto": Use asymptotic inference (default, faster), this is only valid if method is "auto"
        - "bootstrap": Use bootstrap inference (more robust)
    n_bootstrap : int
        Number of bootstrap resamples (if using bootstrap inference)
    ci_level : float
        Confidence level for intervals
    random_state : int
        Random seed
        
    Returns
    -------
    dict with keys:
        - "CATE": array of conditional average treatment effects
        - "CATE_lower": array of CATE lower confidence bounds
        - "CATE_upper": array of CATE upper confidence bounds
        - "ATE": scalar average treatment effect
        - "ATE_lower": ATE lower confidence bound
        - "ATE_upper": ATE upper confidence bound
        - "ATE_stderr": ATE standard error
    """
    from econml.dr import DRLearner
    from econml.inference import BootstrapInference
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, LassoCV
    
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns if c not in [outcome_col, treatment_col]
        ]
    
    X = df[feature_cols].values
    T = df[treatment_col].values
    Y = df[outcome_col].values
    n = df.shape[0]
    
    # ----------------------------
    # Model specification
    # ----------------------------
    if method == "auto":
        model_regression = 'auto' # takes too long
        model_propensity = 'auto'
    elif method == "linear":
        model_regression = LogisticRegression()
        model_propensity = LogisticRegression()
    elif method == "GBR":
        model_regression = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)), random_state=random_state)
        model_propensity = GradientBoostingClassifier(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)), random_state=random_state)
    else:
        raise ValueError("method must be 'auto', 'linear', or 'GBR'")
    
    if method_cate == "linear":
        model_final = StatsModelsLinearRegression() # default final model for DRLearner
        inference_method="auto"
    elif method_cate == "lasso":
        model_final = LassoCV(cv=3, random_state=random_state)
        inference_method="bootstrap"
    elif method_cate == "GBR":
        model_final = GradientBoostingRegressor(n_estimators=100, max_depth=6, min_samples_leaf=max(1, int(n/100)), random_state=random_state)
        inference_method="bootstrap"
    else:
        raise ValueError("method_cate must be 'linear', 'lasso', or 'GBR'")
    # ----------------------------
    # Configure inference
    # ----------------------------
    if inference_method == "bootstrap":
        inference = BootstrapInference(
            n_bootstrap_samples=n_bootstrap,
            n_jobs=1 # to avoid oversubscription
            #n_jobs=-1
        )
    else:
        inference = inference_method  # use "auto" for asymptotic inference
    
    # ----------------------------
    # DRLearner fitting
    # ----------------------------
    dr_learner = DRLearner(
        model_regression=model_regression,
        model_propensity=model_propensity,
        model_final=model_final,
        discrete_outcome=True,
        cv=3,
        random_state=random_state
    )
    
    # Fit with inference enabled
    dr_learner.fit(Y=Y, T=T, X=X, inference=inference)
    
    # Calculate alpha from confidence level
    alpha = 1 - ci_level
    
    # ----------------------------
    # CATE estimates and inference
    # ----------------------------
    cate_inference = dr_learner.effect_inference(X)
    cate_hat = cate_inference.point_estimate.ravel()
    cate_lb, cate_ub = cate_inference.conf_int(alpha=alpha)
    cate_lb = cate_lb.ravel()  # Flatten to 1D
    cate_ub = cate_ub.ravel()  # Flatten to 1D
    
    # ----------------------------
    # ATE estimates and inference
    # ----------------------------
    ate_inference = dr_learner.ate_inference(X)
    ate_hat = float(ate_inference.mean_point)
    ate_stderr = float(ate_inference.stderr_mean)
    ate_lb, ate_ub = ate_inference.conf_int_mean(alpha=alpha)
    ate_lb = float(ate_lb)  # to scalar
    ate_ub = float(ate_ub)
    
    # ----------------------------
    # Return results
    # ----------------------------
    return {
        # CATE results (individual-level)
        "CATE": cate_hat,
        "CATE_lower": cate_lb,
        "CATE_upper": cate_ub,
        
        # ATE results (population-level)
        "ATE": ate_hat,
        "ATE_lower": ate_lb,
        "ATE_upper": ate_ub,
        "ATE_stderr": ate_stderr,
    }


# ---- Preprocessing ----

def preprocess_data(
    df: pd.DataFrame,
    *,
    categorical_cols: list = ['mets_site', 'trt_prev'],
    continuous_cols: list = ['age', 'Charlson'],
    binary_cols: list = ['Abiraterone_prev', 'Enzalutamide_prev', 'anti_arr_pre', 'anti_diab_pre', 'any_infect_pre', 'arf_renal_pre', 'bld_thinner_pre', 'cvd_pre', 'dementia_pre', 'dm_pre', 'exp', 'hosp_ed_any', 'htn_pre', 'opiods_pre', 'race_cat', 'smoker_copd_pre'],
    dummy_code: bool = False,
) -> pd.DataFrame:
    """
    Preprocess covariates for CATE estimators.

    - One-hot encode categorical variables (keep all levels)
    - Standardize continuous variables
    - Keep binary variables (0/1) unchanged

    Returns
    -------
    X_proc : pd.DataFrame
        Processed design matrix with meaningful column names
    """

    X_parts = []

    # --- Binary variables: keep as-is ---
    if binary_cols:
        X_bin = df[binary_cols].astype(float)
        X_parts.append(X_bin)

    # --- Continuous variables: standardize ---
    if continuous_cols:
        X_cont = df[continuous_cols].astype(float)
        cont_mean = X_cont.mean(axis=0)
        cont_std = X_cont.std(axis=0, ddof=0).replace(0, 1.0)
        X_cont_std = (X_cont - cont_mean) / cont_std
        X_parts.append(X_cont_std)

    # --- Categorical variables: one-hot encode ---
    if categorical_cols:
        X_cat = pd.get_dummies(
            df[categorical_cols],
            drop_first=dummy_code,
            dummy_na=False
        ).astype(float)
        X_parts.append(X_cat)

    # --- Concatenate into a single DataFrame ---
    X_proc = pd.concat(X_parts, axis=1)

    return X_proc
