"""CATE benchmarking utilities (estimators + evaluation)."""

from .estimators import (
    bayesian_causal_forest,
    causal_forest,
    xlearner_binary,
    dml_binary,
    drlearner_binary,
    preprocess_data,
)

from .evaluation import (
    evaluate_estimator,
    compare_estimators,
    evaluate_estimator_rep,
    summarize_results,
)
