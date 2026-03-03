# helpers for data preprocessing, postprocessing and variable indexing
import numpy as np
import pandas as pd

# ---------- 1) Fit & store preprocessing metadata ----------
def fit_metadata(
    df: pd.DataFrame,
    binary_var=None, categorical_var=None, numerical_var=None,
    integer_numerical_var=None,
    ddof: int = 1,              # use 0 if you standardized with pandas default .std(ddof=0); change to 1 to match your choice
    eps: float = 1e-8
):
    """
    Collects everything needed to invert your preprocess later.
    Returns:
      meta: dict with means/stds and categorical category orders,
            plus the output column order your preprocess produced.
    """
    binary_var = list(binary_var or [])
    categorical_var = list(categorical_var or [])
    numerical_var = list(numerical_var or [])
    integer_numerical_var = list(integer_numerical_var or [])

    # category lists in the EXACT order pandas 'cat.codes' used during preprocess
    cat_categories = {}
    for col in binary_var + categorical_var:
        # use observed categories order from the training df
        cat_categories[col] = list(pd.Categorical(df[col]).categories)

    # means/stds for continuous
    num_mean = {}
    num_std  = {}
    for col in numerical_var:
        m = float(df[col].mean())
        s = float(df[col].std(ddof=ddof))
        if s == 0 or not np.isfinite(s):
            s = 1.0  # avoid divide-by-zero during preprocess; inverse will be x*s + m
        num_mean[col] = m
        num_std[col]  = max(s, eps)

    meta = {
        "binary_var": binary_var,
        "categorical_var": categorical_var,
        "numerical_var": numerical_var,
        "integer_numerical_var": integer_numerical_var,
        "cat_categories": cat_categories,     # {col: [cat0, cat1, ...]}
        "num_mean": num_mean,                 # {col: mean}
        "num_std": num_std,                   # {col: std > 0}
        # This is the column order your preprocess returns: cat/bin first, then numeric
        "output_order": list(binary_var + categorical_var + numerical_var),
        "ddof": ddof,
    }
    return meta


# ---------- 2) (Optional) Transform using saved metadata (matches your preprocess) ----------
def preprocess_with_meta(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Applies the same transformation as your preprocess(), but using the saved metadata
    (so you can transform val/test consistently).
    """
    bin_cols = meta["binary_var"]
    cat_cols = meta["categorical_var"]
    num_cols = meta["numerical_var"]

    # categorical/binary to codes using the same category order
    df_cat = pd.DataFrame(index=df.index)
    for col in bin_cols + cat_cols:
        cats = meta["cat_categories"][col]
        df_cat[col] = pd.Categorical(df[col], categories=cats).codes

    # standardize numericals using saved mean/std
    df_num = pd.DataFrame(index=df.index)
    for col in num_cols:
        m = meta["num_mean"][col]
        s = meta["num_std"][col]
        df_num[col] = (df[col].astype(float) - m) / s

    # join in the same order as preprocess
    out = pd.concat([df_cat, df_num], axis=1)[meta["output_order"]]
    return out


# ---------- 3) Postprocess (invert to original labels/scales) ---------

def postprocess_generated(
    y_gen,               # np.ndarray or pd.DataFrame of generated Y
    meta: dict,
    return_as_category: bool = True,  # if False, returns plain object dtype for cats
):
    """
    Inverts your preprocess:
      - categorical/binary codes -> original labels (using saved category order)
      - continuous -> de-standardize (x*std + mean)
      - integer continuous -> round to int

    Works even if some columns are missing or out of order in y_gen.

    Args:
      y_gen: array-like (N, D) or DataFrame. If ndarray, assume columns correspond to meta['output_order'] subset.
    Returns:
      DataFrame with original dtypes restored as closely as possible.
    """
    cols = meta["output_order"]

    # Normalize input to DataFrame
    if isinstance(y_gen, pd.DataFrame):
        df = y_gen.copy()
    else:
        df = pd.DataFrame(y_gen, columns=cols[:y_gen.shape[1]])

    out = pd.DataFrame(index=df.index)

    # 1) Categorical + Binary: codes -> labels
    for col in meta["binary_var"] + meta["categorical_var"]:
        if col in df.columns:
            cats = meta["cat_categories"][col]
            codes = np.rint(df[col].to_numpy()).astype(int)  # round codes if needed
            codes = np.clip(codes, 0, len(cats) - 1)         # clip to valid range

            if return_as_category:
                out[col] = pd.Categorical.from_codes(codes, categories=cats)
            else:
                out[col] = np.array(cats, dtype=object)[codes]

    # 2) Continuous: inverse standardize; round integer vars
    for col in meta["numerical_var"]:
        if col in df.columns:
            m = meta["num_mean"][col]
            s = meta["num_std"][col]
            vals = df[col].astype(float).to_numpy()
            vals = vals * s + m

            if col in set(meta.get("integer_numerical_var", [])):
                vals = np.rint(vals).astype(int)
            out[col] = vals

    # 3) Ensure consistent column order (only for columns present in output)
    valid_cols = [c for c in cols if c in out.columns]
    out = out[valid_cols]

    return out

# Added January 13, 2026 to identify index of covariates for function specification
class VarIndex:
    def __init__(self, var_names):
        self.names = list(var_names)
        self.index = {n: i for i, n in enumerate(self.names)}

    def idx(self, *names):
        return [self.index[n] for n in names]

    # def slice(self, X, *names):
    #     return X[:, self.idx(*names)]
    def slice(self, X, *names, unbind=False):
        cols = self.idx(*names)
        Xs = X[:, cols].squeeze(-1)  # works for torch.Tensor and np.ndarray; squeeze(-1) is useful to return a (N,) Xs if X include only one column

        if not unbind:
            return Xs

        # torch: match torch.unbind(dim=-1)
        if torch is not None and isinstance(Xs, torch.Tensor):
            return Xs.unbind(dim=-1)

        # numpy: return column vectors (N,)
        Xs = np.asarray(Xs)  # safe; no-op if already ndarray
        return tuple(Xs[:, i] for i in range(Xs.shape[1]))
