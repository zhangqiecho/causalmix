"""
CausalMix: calibrated causal synthetic data generator and evaluation suite.

Public API for the CausalMix package.

This file re-exports the main classes and utilities used in the validation notebooks 
so notebooks can rely on short imports
"""

# Core generator
from .core.causalmix import CausalMix

# Models
from .models.convae import conVAE

# Data / schema / preprocessing
from .data.schema import DataSchema
from .data.preprocess import VarIndex, fit_metadata, preprocess_with_meta, postprocess_generated


# Evaluation
from .eval.synth_eval import SynthEval
from .eval.causal_eval import CausalEval

# Reporting helpers used in CausalMix_validate.ipynb
from .reporting.tables import (
    build_table_distributional_fidelity,
    build_table_privacy,
    build_table_causal_structure,
    eval_tables_mean,
)

# plotting 
from .viz.plots import(
    single_column_plot,
    pair_column_plot,
    plot_joint_embedding_2d,
)

__all__ = [
    # core
    "CausalMix",
    # models
    "conVAE",
    # data / schema
    "DataSchema",
    "VarIndex",
    "fit_metadata",
    "preprocess_with_meta", 
    "postprocess_generated",
    # eval
    "SynthEval",
    "CausalEval",
    # table reporting
    "build_table_distributional_fidelity",
    "build_table_privacy",
    "build_table_causal_structure",
    "eval_tables_mean",
    # distribution plots
    "single_column_plot",
    "pair_column_plot",
    "plot_joint_embedding_2d",
]