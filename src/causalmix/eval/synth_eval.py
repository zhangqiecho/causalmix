# SynthEval.py
# need to import plots.py if in standalone scripts
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


from ..data.schema import DataSchema
from ..viz.plots import(
    single_column_plot,
    pair_column_plot,
    plot_joint_embedding_2d,
)
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.special import digamma
from sklearn.metrics import pairwise_distances, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Optional sdmetrics (used for KS/TV/Correlation/Contingency/DCR when available)
_SD_OK = True
try: # update January 12, 2026
    from sdmetrics.single_table import (
        KSComplement, TVComplement, 
        DCRBaselineProtection
    )
    from sdmetrics.column_pairs import CorrelationSimilarity, ContingencySimilarity
    from sdv.metadata import SingleTableMetadata
except Exception:
    _SD_OK = False


# ------------------------------- SynthEval ------------------------------- #

class SynthEval:
    """
    Fidelity + Privacy evaluator with:
      • Marginals: normalized Wasserstein (cont), TVComplement (disc)
      • Pairwise dependence:
          - CorrelationSimilarity for cont–cont (optional, for parity)
          - Normalized MI (Symmetric Uncertainty, SU) for ALL pair types (cat–cat, cont–cat, cont–cont)
          - ContingencySimilarity per cat–cat pair (sdmetrics) via pairwise_discrete()
      • Conditional fidelity: MMD^2 with mixed kernel (RBF + Hamming) + optional normalization
      • Joint fidelity: Energy distance (normalized) + C2ST (AUC + complement)
      • Privacy: DCR Baseline Protection
    """

    def __init__(self, real: pd.DataFrame, synth: pd.DataFrame, schema: DataSchema, random_state: int = 0):
        self.real = real.reset_index(drop=True)
        self.synth = synth.reset_index(drop=True)
        self.schema = schema
        self.rs = random_state

        need = set(self.columns)
        miss_r, miss_s = need - set(self.real.columns), need - set(self.synth.columns)
        if miss_r or miss_s:
            raise ValueError(f"Both dataframes must include all schema columns. "
                             f"Missing in real: {miss_r}; missing in synth: {miss_s}")

    # ---------------------------- helpers / encoders ---------------------------- #

    @property
    def columns(self) -> List[str]:
        return list(self.schema.numeric) + list(self.schema.binary) + list(self.schema.categorical)

    def _ct_enc(self) -> ColumnTransformer:
        num, cat = self.schema.numeric, self.schema.categorical + self.schema.binary
        trs = []
        if num: trs.append(("num", StandardScaler(), num))
        if cat: trs.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat))
        return ColumnTransformer(trs, remainder="drop")

    @staticmethod
    def _tv(counts_r: np.ndarray, counts_s: np.ndarray) -> float:
        p = counts_r / max(counts_r.sum(), 1e-12)
        q = counts_s / max(counts_s.sum(), 1e-12)
        return 0.5 * np.abs(p - q).sum()

    # ---------------------------- Marginals ---------------------------- #

    def marginal_continuous(self) -> Dict:
        Wnorm, KScomp = {}, {}
        for c in self.schema.numeric:
            r = self.real[c].to_numpy(float); s = self.synth[c].to_numpy(float)
            r, s = r[~np.isnan(r)], s[~np.isnan(s)]
            if r.size == 0 or s.size == 0: continue
            Wnorm[c] = float(wasserstein_distance(r, s) / (np.std(r) + 1e-12))
            if _SD_OK:
                try: KScomp[c] = float(KSComplement.compute(self.real[[c]], self.synth[[c]]))
                except: KScomp[c] = float(1 - ks_2samp(r, s).statistic)
            else:
                KScomp[c] = float(1 - ks_2samp(r, s).statistic)
        return {
            "normalized_wasserstein": Wnorm,                   # lower = better
            "KSComplement": KScomp,                            # higher = better
            "aggregates": {
                "mean_norm_wasserstein": np.mean(list(Wnorm.values())) if Wnorm else None,
                "mean_KSComplement": np.mean(list(KScomp.values())) if KScomp else None
            }
        }

    def marginal_discrete(self) -> Dict:
        cols = self.schema.binary + self.schema.categorical
        tvc = {}
        for c in cols:
            cr = self.real[c].value_counts(dropna=False).sort_index()
            cs = self.synth[c].value_counts(dropna=False).sort_index()
            idx = cr.index.union(cs.index)
            cr, cs = cr.reindex(idx, fill_value=0).to_numpy(), cs.reindex(idx, fill_value=0).to_numpy()
            tvc[c] = float(1.0 - self._tv(cr, cs))
        if _SD_OK:
            try: tvc = {c: float(TVComplement.compute(self.real[[c]], self.synth[[c]])) for c in cols}
            except Exception: pass
        return {"TVComplement": tvc, "aggregates": {"mean_TVComplement": np.mean(list(tvc.values())) if tvc else None}}

    # ---------------------------- Pairwise continuous (optional, corr) ---------------------------- #

    def pairwise_continuous(self) -> Dict:
        cols = self.schema.numeric
        if len(cols) < 2: return {"CorrelationSimilarity": None}
        if _SD_OK:
            try: return {"CorrelationSimilarity": float(CorrelationSimilarity.compute(self.real[cols], self.synth[cols]))}
            except Exception: pass
        R, S = self.real[cols].corr("spearman").to_numpy(), self.synth[cols].corr("spearman").to_numpy()
        diffs = [abs(R[i, j] - S[i, j]) / 2.0 for i in range(len(cols)) for j in range(i + 1, len(cols))]
        return {"CorrelationSimilarity": (1 - float(np.mean(diffs))) if diffs else None}

    # ---------------------------- Normalized MI (SU) for ALL pairs ---------------------------- #

    @staticmethod
    def _zscore_block(df: pd.DataFrame, cols: List[str]) -> Dict[str, np.ndarray]:
        if not cols: return {}
        Z = StandardScaler().fit_transform(df[cols].astype(float).to_numpy())
        return {c: Z[:, i : i + 1] for i, c in enumerate(cols)}

    @staticmethod
    def _codes_from_levels(df: pd.DataFrame, col: str, categories: List) -> np.ndarray:
        return pd.Categorical(df[col], categories=categories).codes.astype(int)

    @staticmethod
    def _disc_entropy_bits_from_counts(counts: np.ndarray, laplace: float = 0.0) -> float:
        p = counts.astype(float)
        if laplace > 0: p += laplace
        s = p.sum()
        if s <= 0: return 0.0
        p = p / s; p = p[p > 0]
        return float(-(p * np.log2(p)).sum())

    @staticmethod
    def _contingency_from_codes(x: np.ndarray, y: np.ndarray, kx: int, ky: int, laplace: float = 1e-6) -> np.ndarray:
        M = np.zeros((kx, ky), dtype=float)
        m = (x >= 0) & (y >= 0)
        np.add.at(M, (x[m], y[m]), 1.0)
        if laplace > 0: M += laplace
        return M

    @staticmethod
    def _mi_bits_from_contingency(M: np.ndarray) -> float:
        N = M.sum()
        if N <= 0: return 0.0
        px = M.sum(1) / N; py = M.sum(0) / N; pxy = M / N
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = pxy / (px[:, None] * py[None, :]); ratio[~np.isfinite(ratio)] = 1.0
            return float(np.sum(pxy * np.log2(ratio)))

    @staticmethod
    def _knn_kth_eps(X: np.ndarray, k: int, metric: str = "chebyshev") -> np.ndarray:
        if X.shape[0] <= k:
            raise ValueError(f"k={k} requires at least k+1 samples; got n={X.shape[0]}")
        nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm="brute")
        nn.fit(X)
        dists, _ = nn.kneighbors(X, n_neighbors=k + 1, return_distance=True)
        return dists[:, -1]

    @staticmethod
    def _ksg_mi_bits(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
        n = x.shape[0]
        if n <= k: raise ValueError("Not enough samples for KSG MI.")
        XY = np.c_[x, y]
        eps = SynthEval._knn_kth_eps(XY, k=k, metric="chebyshev")
        nx = np.empty(n, dtype=int); ny = np.empty(n, dtype=int)
        nnx = NearestNeighbors(metric="chebyshev", algorithm="brute").fit(x)
        nny = NearestNeighbors(metric="chebyshev", algorithm="brute").fit(y)
        for i in range(n):
            #update Jan 11 2025: to resolve the problem of negaive radius when the variation of continuous data is small
            # ex = eps[i] - 1e-12
            ex = max(np.nextafter(eps[i], -np.inf), 0.0)
            nx[i] = nnx.radius_neighbors(x[i:i+1], radius=ex, return_distance=False)[0].size - 1
            ny[i] = nny.radius_neighbors(y[i:i+1], radius=ex, return_distance=False)[0].size - 1
        mi_nats = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
        return float(mi_nats / np.log(2))

    @staticmethod
    def _kl_entropy_bits(X: np.ndarray, k: int = 5, metric: str = "chebyshev") -> float:
        n, d = X.shape
        eps = SynthEval._knn_kth_eps(X, k=k, metric=metric)
        log_cd = d * np.log(2.0)  # ln(2^d)
        H_nats = digamma(n) - digamma(k) + log_cd + d * np.mean(np.log(eps + 1e-12))
        return float(H_nats / np.log(2))

    def _pairwise_su_all(
        self,
        df: pd.DataFrame,
        k: int = 5,
        n_bins: int = 10,
        laplace: float = 1e-6,
        mixed_method: str = "knn",   # "knn" (recommended) or "discretize"
        standardize_continuous: bool = True,
    ) -> Dict[Tuple[str, str], Dict[str, float]]:
        """Compute MI_bits and SU for all pairs in df (cat-cat, cont-cat, cont-cont)."""
        numeric = list(self.schema.numeric)
        discrete = list(self.schema.binary) + list(self.schema.categorical)
        cols = numeric + discrete

        # continuous prep
        z_map = self._zscore_block(df, numeric) if standardize_continuous else {c: df[[c]].to_numpy() for c in numeric}
        # discretizers for continuous (if discretize path used)
        bin_edges = {c: KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile").fit(df[[c]].astype(float))
                     for c in numeric} if n_bins and numeric else {}
        # category vocabularies
        levels = {c: pd.Index(df[c]).unique().tolist() for c in discrete}

        out: Dict[Tuple[str, str], Dict[str, float]] = {}
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                if (a in discrete) and (b in discrete):
                    xa = self._codes_from_levels(df, a, levels[a]); xb = self._codes_from_levels(df, b, levels[b])
                    kx, ky = int(xa.max() + 1), int(xb.max() + 1)
                    M = self._contingency_from_codes(xa, xb, max(kx, 1), max(ky, 1), laplace=laplace)
                    mi_bits = self._mi_bits_from_contingency(M)
                    Hx = self._disc_entropy_bits_from_counts(M.sum(axis=1), laplace=laplace)
                    Hy = self._disc_entropy_bits_from_counts(M.sum(axis=0), laplace=laplace)
                    su = float(np.clip(2.0 * mi_bits / max(Hx + Hy, 1e-12), 0.0, 1.0))

                elif (a in numeric) and (b in numeric):
                    xa, xb = z_map[a], z_map[b]
                    mi_bits = self._ksg_mi_bits(xa, xb, k=k)
                    Hx = self._kl_entropy_bits(xa, k=k); Hy = self._kl_entropy_bits(xb, k=k)
                    su = float(np.clip(2.0 * mi_bits / max(Hx + Hy, 1e-12), 0.0, 1.0))

                else:
                    cont, cat = (a, b) if (a in numeric) else (b, a)
                    x = z_map[cont]; y_codes = self._codes_from_levels(df, cat, levels[cat])
                    if mixed_method == "knn":
                        Hx = self._kl_entropy_bits(x, k=min(k, max(2, len(x) - 1)))
                        counts = np.bincount(y_codes[y_codes >= 0])
                        N = x.shape[0]; Hx_given_y = 0.0
                        for y_val, cnt in enumerate(counts):
                            if cnt <= 1: continue
                            mask = (y_codes == y_val)
                            k_eff = min(k, cnt - 1)
                            Hxy = self._kl_entropy_bits(x[mask], k=k_eff)
                            Hx_given_y += (cnt / N) * Hxy
                        mi_bits = max(Hx - Hx_given_y, 0.0)
                        Hy = self._disc_entropy_bits_from_counts(counts, laplace=laplace)
                        su = float(np.clip(2.0 * mi_bits / max(Hx + Hy, 1e-12), 0.0, 1.0))
                    else:
                        disc = bin_edges[cont].transform(df[[cont]].astype(float)).astype(int).ravel()
                        kx = int(disc.max() + 1) if disc.size else 0
                        ky = len(levels[cat])
                        M = self._contingency_from_codes(disc, self._codes_from_levels(df, cat, levels[cat]),
                                                         max(kx, 1), max(ky, 1), laplace=laplace)
                        mi_bits = self._mi_bits_from_contingency(M)
                        Hx = self._disc_entropy_bits_from_counts(M.sum(axis=1), laplace=laplace)
                        Hy = self._disc_entropy_bits_from_counts(M.sum(axis=0), laplace=laplace)
                        su = float(np.clip(2.0 * mi_bits / max(Hx + Hy, 1e-12), 0.0, 1.0))
                out[(a, b)] = {"MI_bits": float(mi_bits), "SU": su}
        return out

    def _pairwise_su_only(
        self,
        df: pd.DataFrame,
        k: int,
        n_bins: int,
        laplace: float,
        mixed_method: str,
        standardize_continuous: bool,
    ) -> Dict[Tuple[str, str], float]:
        res = self._pairwise_su_all(df, k=k, n_bins=n_bins, laplace=laplace,
                                    mixed_method=mixed_method, standardize_continuous=standardize_continuous)
        return {pair: d["SU"] for pair, d in res.items()}

    # ---- public: SU fidelity summary (and optional table) ----

    def pairwise_mi_table(
        self,
        k: int = 5,
        n_bins: int = 10,
        laplace: float = 1e-6,
        mixed_method: str = "knn",
        standardize_continuous: bool = True,
    ) -> pd.DataFrame:
        """Tidy per-pair table with MI (bits) & SU for real vs synthetic."""
        res_r = self._pairwise_su_all(self.real, k=k, n_bins=n_bins, laplace=laplace,
                                      mixed_method=mixed_method, standardize_continuous=standardize_continuous)
        res_s = self._pairwise_su_all(self.synth, k=k, n_bins=n_bins, laplace=laplace,
                                      mixed_method=mixed_method, standardize_continuous=standardize_continuous)

        num = set(self.schema.numeric)
        disc = set(self.schema.binary + self.schema.categorical)

        rows = []
        for (i, j), dr in res_r.items():
            if (i, j) not in res_s:
                continue
            ds = res_s[(i, j)]
            if i in num and j in num:
                ptype = "cont-cont"
            elif i in disc and j in disc:
                ptype = "cat-cat"
            else:
                ptype = "cont-cat"
            su_delta = abs(dr["SU"] - ds["SU"])
            rows.append({
                "var_i": i, "var_j": j, "pair_type": ptype,
                "MI_bits_real": dr["MI_bits"], "SU_real": dr["SU"],
                "MI_bits_synth": ds["MI_bits"], "SU_synth": ds["SU"],
                "SU_delta": su_delta, "SU_similarity": 1.0 - su_delta,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["pair_type", "SU_delta", "var_i", "var_j"], ascending=[True, True, True, True])
        return df

    def pairwise_mi_fidelity(
        self,
        k: int = 5,
        n_bins: int = 10,
        laplace: float = 1e-6,
        mixed_method: str = "knn",
        standardize_continuous: bool = True,
        return_table: bool = False,
    ) -> Dict:
        """Summary of MI/SU fidelity; optionally include the tidy table."""
        su_r = self._pairwise_su_only(self.real, k, n_bins, laplace, mixed_method, standardize_continuous)
        su_s = self._pairwise_su_only(self.synth, k, n_bins, laplace, mixed_method, standardize_continuous)

        sims, per_pair = [], {}
        for pair in su_r.keys():
            if pair not in su_s: 
                continue
            sim = 1.0 - abs(su_r[pair] - su_s[pair])
            sims.append(sim)
            per_pair[pair] = sim

        out = {
            "SU_similarity_mean": float(np.mean(sims)) if sims else None,
            "SU_similarity_per_pair": per_pair if per_pair else None,
            "n_pairs": len(sims),
        }
        if return_table:
            out["table"] = self.pairwise_mi_table(
                k=k, n_bins=n_bins, laplace=laplace,
                mixed_method=mixed_method,
                standardize_continuous=standardize_continuous,
            )
        return out

    # ---------------------------- Pairwise discrete via sdmetrics ---------------------------- #

    def pairwise_discrete(self) -> Dict:
        """
        ContingencySimilarity over ALL cat–cat pairs (pairwise), averaged.
        Falls back to SU similarity on that pair if sdmetrics is unavailable or errors.
        """
        disc = self.schema.binary + self.schema.categorical
        if len(disc) < 2:
            return {"ContingencySimilarity_mean": None, "ContingencySimilarity_per_pair": None, "n_pairs": 0}

        # SU fallback helpers (use SU similarity on that cat–cat pair if needed)
        su_real = self._pairwise_su_all(self.real)
        su_synth = self._pairwise_su_all(self.synth)

        per_pair = {}
        scores = []
        for i in range(len(disc)):
            for j in range(i + 1, len(disc)):
                a, b = disc[i], disc[j]
                score = None
                if _SD_OK:
                    try:
                        score = float(ContingencySimilarity.compute(
                            real_data=self.real[[a, b]],
                            synthetic_data=self.synth[[a, b]]
                        ))
                    except Exception:
                        score = None
                if score is None:
                    key = (a, b) if (a, b) in su_real else (b, a)
                    if key in su_real and key in su_synth:
                        # SU similarity on this pair
                        su_r = su_real[key]["SU"]
                        su_s = su_synth[key]["SU"]
                        score = 1.0 - abs(su_r - su_s)
                if score is not None:
                    per_pair[(a, b)] = score
                    scores.append(score)

        return {
            "ContingencySimilarity_mean": float(np.mean(scores)) if scores else None,
            "ContingencySimilarity_per_pair": per_pair if per_pair else None,
            "n_pairs": len(scores)
        }

    # ---------------------------- Conditional MMD² ---------------------------- #

    def _numZ(self, df: pd.DataFrame) -> np.ndarray:
        if not self.schema.numeric: return np.empty((len(df), 0))
        return StandardScaler().fit_transform(df[self.schema.numeric].to_numpy(float))

    def _cat_codes(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in (self.schema.categorical + self.schema.binary) if c in df.columns]
        if not cols: return np.empty((len(df), 0), int)
        return np.vstack([pd.factorize(df[c], sort=True)[0].astype(int) for c in cols]).T

    def _rbf(self, X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        D = pairwise_distances(X, Y, metric="euclidean")
        if sigma is None:
            Dfull = pairwise_distances(np.vstack([X, Y]), metric="euclidean")
            med = np.median(Dfull[np.triu_indices_from(Dfull, 1)])
            sigma = med if med > 1e-12 else 1.0
        return np.exp(-(D ** 2) / (2 * sigma ** 2))

    def _hamm(self, Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
        if Xc.size == 0 or Yc.size == 0: return np.ones((len(Xc), len(Yc)))
        return (Xc[:, None, :] == Yc[None, :, :]).mean(axis=2)

    def _mixedK(self, Xn, Yn, Xc, Yc, sigma=None, a_num=1.0, a_cat=1.0) -> np.ndarray:
        parts, w = [], []
        if Xn is not None and Yn is not None and Xn.shape[1] > 0: parts.append(self._rbf(Xn, Yn, sigma)); w.append(a_num)
        if Xc is not None and Yc is not None and Xc.shape[1] > 0: parts.append(self._hamm(Xc, Yc)); w.append(a_cat)
        if not parts: raise ValueError("No features to build kernel.")
        K = np.zeros_like(parts[0]); ws = 0.0
        for P, wi in zip(parts, w): K += wi * P; ws += wi
        return K / ws

    def conditional_mmd2(
        self,
        cond_col: str,
        bins_for_cont: int = 5,
        a_num: float = 1.0,
        a_cat: float = 1.0,
        sigma: Optional[float] = None,
        normalize_by_within_real: bool = True,
        min_per_stratum: int = 3,         # NEW: require at least this many per stratum in both sets
        ref_epsilon: float = 1e-8,        # NEW: floor to avoid 0/near-0 baseline
        stratified_split: bool = True,    # NEW: strive for stratified baseline split on cond_col
    ) -> Dict:
        assert cond_col in self.real and cond_col in self.synth, f"{cond_col} missing"
    
        feats = [c for c in self.columns if c != cond_col]
    
        def make_strata(df: pd.DataFrame):
            z = df[cond_col]
            if pd.api.types.is_numeric_dtype(z) and z.nunique() > 2 * bins_for_cont:
                return pd.qcut(z, bins_for_cont, duplicates="drop").astype(str)
            return z.astype(str)
    
        strata_real = make_strata(self.real)
        strata_synth = make_strata(self.synth)
        levels = sorted(set(strata_real.unique()).union(set(strata_synth.unique())))
    
        # Encodings once
        Xnr = self._numZ(self.real[feats]); Xns = self._numZ(self.synth[feats])
        Xcr = self._cat_codes(self.real[feats]); Xcs = self._cat_codes(self.synth[feats])
    
        def mmd2(Kxx, Kyy, Kxy):
            n, m = Kxx.shape[0], Kyy.shape[0]
            if n <= 1 or m <= 1: return 0.0
            np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
            term_xx = Kxx.sum() / (n * (n - 1))
            term_yy = Kyy.sum() / (m * (m - 1))
            term_xy = Kxy.mean()
            return term_xx + term_yy - 2 * term_xy
    
        per_stratum = {}
        weights, vals = [], []
    
        # --- real vs synth per-stratum MMD^2 ---
        for v in levels:
            idx_r = np.where(strata_real == v)[0]
            idx_s = np.where(strata_synth == v)[0]
            # skip strata that are too small in either set
            if len(idx_r) < min_per_stratum or len(idx_s) < min_per_stratum:
                continue
    
            Knn = self._mixedK(
                Xnr[idx_r] if Xnr.size else None, Xnr[idx_r] if Xnr.size else None,
                Xcr[idx_r] if Xcr.size else None, Xcr[idx_r] if Xcr.size else None,
                sigma=sigma, a_num=a_num, a_cat=a_cat,
            )
            Kmm = self._mixedK(
                Xns[idx_s] if Xns.size else None, Xns[idx_s] if Xns.size else None,
                Xcs[idx_s] if Xcs.size else None, Xcs[idx_s] if Xcs.size else None,
                sigma=sigma, a_num=a_num, a_cat=a_cat,
            )
            Knm = self._mixedK(
                Xnr[idx_r] if Xnr.size else None, Xns[idx_s] if Xns.size else None,
                Xcr[idx_r] if Xcr.size else None, Xcs[idx_s] if Xcs.size else None,
                sigma=sigma, a_num=a_num, a_cat=a_cat,
            )
            m2 = float(mmd2(Knn, Kmm, Knm))
            per_stratum[v] = m2
            weights.append(len(idx_r) / len(self.real))
            vals.append(m2)
    
        weighted = float(np.average(vals, weights=weights)) if vals else None
    
        # --- normalized ratio vs within-real baseline (robust) ---
        ratio = None
        baseline_ref = None
        if normalize_by_within_real and weighted is not None:
            # Build a *stratified* half/half split of REAL on the same strata labels (when possible)
            rng = np.random.RandomState(self.rs)
            sr = strata_real.to_numpy()
    
            # Identify strata with at least 2 samples (so both halves get >=1)
            valid_mask = np.zeros(len(sr), dtype=bool)
            for v in np.unique(sr):
                idx = np.where(sr == v)[0]
                if len(idx) >= 2:
                    valid_mask[idx] = True
            # Work only with valid rows for baseline
            real_base = self.real.loc[valid_mask].reset_index(drop=True)
            sr_base = sr[valid_mask]
    
            if stratified_split and len(np.unique(sr_base)) >= 1 and len(real_base) >= 2:
                # stratified split: keep proportions per stratum
                try:
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=self.rs)
                    (i1, i2), = sss.split(np.zeros(len(sr_base)), sr_base)
                    r1, r2 = real_base.iloc[i1], real_base.iloc[i2]
                except Exception:
                    # fallback: plain permutation split
                    idx = rng.permutation(len(real_base)); mid = len(idx) // 2
                    r1, r2 = real_base.iloc[idx[:mid]], real_base.iloc[idx[mid:]]
            else:
                idx = rng.permutation(len(self.real)); mid = len(idx) // 2
                r1, r2 = self.real.iloc[idx[:mid]], self.real.iloc[idx[mid:]]
    
            # Compute baseline on the halves, using the SAME parameters but no renormalization
            ref_out = SynthEval(r1, r2, self.schema, self.rs).conditional_mmd2(
                cond_col=cond_col, bins_for_cont=bins_for_cont,
                a_num=a_num, a_cat=a_cat, sigma=sigma,
                normalize_by_within_real=False,
                min_per_stratum=min_per_stratum,
            )
            baseline_ref = ref_out["weighted_mean_mmd2"]
    
            if baseline_ref is not None:
                denom = baseline_ref if abs(baseline_ref) > ref_epsilon else ref_epsilon
                ratio = float(weighted / denom)
    
        return {
            "per_stratum_mmd2": per_stratum,
            "weighted_mean_mmd2": weighted,
            "normalized_ratio_vs_real": ratio,     # now robustly populated
            "baseline_ref": baseline_ref,          # NEW: return baseline so you can inspect it
        }


    # ---------------------------- Joint fidelity ---------------------------- #

    def energy_distance_normalized(self) -> Dict:
        enc = self._ct_enc(); Zr, Zs = enc.fit_transform(self.real), enc.transform(self.synth)
        d_xy = pairwise_distances(Zr, Zs).mean()
        Drr, Dss = pairwise_distances(Zr, Zr), pairwise_distances(Zs, Zs)
        ed = 2 * d_xy - (Drr.sum() - np.trace(Drr)) / max(len(Zr) * (len(Zr) - 1), 1) \
                       - (Dss.sum() - np.trace(Dss)) / max(len(Zs) * (len(Zs) - 1), 1)
        d_within = (Drr.sum() - np.trace(Drr)) / max(len(Zr) * (len(Zr) - 1), 1)
        return {"energy_distance": float(ed), "normalized_energy": float(ed / (d_within + 1e-12))}

    def c2st_auc(self, test_size: float = 0.3, model: Optional[Pipeline] = None) -> Dict:
        X = pd.concat([self.real[self.columns], self.synth[self.columns]], ignore_index=True)
        y = np.r_[np.ones(len(self.real)), np.zeros(len(self.synth))]
        if model is None: model = Pipeline([("pre", self._ct_enc()), ("clf", LogisticRegression(max_iter=500, random_state=self.rs))])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=self.rs, stratify=y)
        auc = roc_auc_score(yte, model.fit(Xtr, ytr).predict_proba(Xte)[:, 1])
        return {"auc": float(auc), "n_test": int(len(yte)), "auc_complement": float(1 - 2 * abs(auc - 0.5))}

    # ---------------------------- Privacy ---------------------------- #

    def dcr_baseline_protection(self, metric: str = "euclidean") -> Dict:
        enc = self._ct_enc(); Zr, Zs = enc.fit_transform(self.real), enc.transform(self.synth)
        rs = NearestNeighbors(n_neighbors=1, metric=metric, algorithm="brute").fit(Zs).kneighbors(Zr, n_neighbors=1, return_distance=True)[0].ravel()
        rr = NearestNeighbors(n_neighbors=2, metric=metric, algorithm="brute").fit(Zr).kneighbors(Zr, n_neighbors=2, return_distance=True)[0][:, 1]
        # prot = float(np.mean(rs > rr)); ratio = rs / np.maximum(rr, 1e-12)
        # out = {"protection_fraction": prot, "ratio_mean": float(np.mean(ratio)),
        #        "ratio_p5": float(np.percentile(ratio, 5)), "ratio_p50": float(np.percentile(ratio, 50)),
        #        "ratio_p95": float(np.percentile(ratio, 95))}
        #Update January 12 2025: to resolve the problem of ratio being extremely high when rr==0
        rr_pos = rr[rr > 0]
        eps = 1e-3 * (np.median(rr_pos) if len(rr_pos) else 1.0)

        dup = (rr == 0)
        ratio = rs[~dup] / np.maximum(rr[~dup], eps)

        prot = float(np.mean(rs > np.maximum(rr, eps)))  # avoids rr==0 inflating prot

        out = {
        "protection_fraction": prot,
        "rr_zero_fraction": float(np.mean(dup)),
        "ratio_mean": float(np.mean(ratio)),
        "ratio_p5": float(np.percentile(ratio, 5)),
        "ratio_p50": float(np.percentile(ratio, 50)),
        "ratio_p95": float(np.percentile(ratio, 95)),
        }

        if _SD_OK:
            try: # update January 12, 2026
                metadata = SingleTableMetadata()
                metadata.detect_from_dataframe(self.real)
                metadata = metadata.to_dict()
                out["sdmetrics_DCRBaseline"] = float(DCRBaselineProtection.compute_breakdown(self.real, self.synth, metadata = metadata)["score"])
            except Exception: pass
        return out

    # ---------------------------- Run all ---------------------------- #
    
    def run_all(
        self,
        conditional_col: Optional[str] = None,
        bins_for_continuous_cond: int = 5,
        mi_bins: int = 10,
        mi_k: int = 5,
        mi_method_mixed: str = "knn",     # or "discretize"
        include_pairwise_table: bool = False,
    ) -> Dict:
        out = {
            "marginal_continuous": self.marginal_continuous(),
            "marginal_discrete": self.marginal_discrete(),
            "pairwise_continuous": self.pairwise_continuous(),             # correlation view (optional)
            "pairwise_mi": self.pairwise_mi_fidelity(k=mi_k, n_bins=mi_bins, mixed_method=mi_method_mixed,
                                                     return_table=include_pairwise_table),
            "pairwise_discrete": self.pairwise_discrete(),                 # ContingencySimilarity (cat–cat)
            "energy": self.energy_distance_normalized(),
            "c2st": self.c2st_auc(),
            "privacy_dcr": self.dcr_baseline_protection(),
        }
        if conditional_col is not None:
            out["conditional_mmd2"] = self.conditional_mmd2(conditional_col, bins_for_continuous_cond)
        def round_dict(d, n=3):
            out_dict = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out_dict[k] = round_dict(v, n)
                elif v is None:
                    out_dict[k] = None
                else:
                    out_dict[k] = round(v, n)
            return out_dict

        out = round_dict(out)
        return out

    # ---------------------------- Visualization ---------------------------- #
    # -----------------------------
    # 1D column plot
    # -----------------------------
    def plot_column(
        self,
        column_name: str,
        *,
        col_plot_type: ColPlotType = "auto",
        sample_size: Optional[int] = None,
        fig_size: Tuple[float, float] = (6, 4),
        main_title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        real_label: str = "Real",
        real_color: str = "#5E3C99",
        synth_label: str = "Synthetic",
        max_categories: int = 30,
        legend_outside: bool = True,
        show_hist: bool = False,
        num_bins: int = 40,
        kde_bw_adjust: float = 1.0,
        density: bool = True,
        fill_real_kde: bool = True,
    ):
        """
        Visualize marginal distribution of one column (real vs synthetic):
          - categorical: grouped bars (Real + all synthetics)
          - numerical: KDE + optional hist
        """
        return single_column_plot(
            real_data=self.real,
            synthetic_data=self.synth,
            schema=self.schema,
            column_name=column_name,
            col_plot_type=col_plot_type,
            sample_size=sample_size,
            fig_size=fig_size,
            main_title=main_title,
            x_label=x_label,
            y_label=y_label,
            real_label=real_label,
            synth_label=synth_label,
            real_color=real_color,
            max_categories=max_categories,
            legend_outside=legend_outside,
            show_hist=show_hist,
            num_bins=num_bins,
            kde_bw_adjust=kde_bw_adjust,
            density=density,
            fill_real_kde=fill_real_kde,
        )

    # -----------------------------------
    # 2D column-pair plot
    # -----------------------------------
    def plot_column_pair(
        self,
        col_x: str,
        col_y: str,
        *,
        plot_type: PlotType = None,
        sample_size: Optional[int] = None,
        fig_size: Tuple[float, float] = (10, 4),
        main_title: Optional[str] = None,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        real_label: str = "Real",
        real_color: str = "#5E3C99",
        synth_label: str = "Synthetic",
        max_categories: int = 20,
        point_alpha: float = 0.5,
        point_size: float = 12.0,
        legend_outside: bool = True,
    ):
        """
        Visualize joint distribution of two columns (real vs synthetic):
          - num–num: scatter
          - num–cat: grouped boxplots
          - cat–cat: heatmaps (real + one panel per synthetic)
        """
        return pair_column_plot(
            real_data=self.real,
            synthetic_data=self.synth,
            schema=self.schema,
            column_names=(col_x, col_y),
            plot_type=plot_type,
            sample_size=sample_size,
            fig_size=fig_size,
            main_title=main_title,
            x_label=x_label,
            y_label=y_label,
            real_label=real_label,
            real_color=real_color,
            synth_label=synth_label,
            max_categories=max_categories,
            point_alpha=point_alpha,
            point_size=point_size,
            legend_outside=legend_outside,
        )

    # -----------------------------------
    # 2D joint embedding
    # -----------------------------------
    def plot_joint_embedding(
        self,
        *,
        label_col: Optional[str] = None,
        method: str = "auto",       # "auto", "umap", or "tsne"
        fig_size: Tuple[float, float] = (6, 5),
        main_title: Optional[str] = "2D Joint Embedding: Real vs Synthetic",
        real_label: str = "Real",
        real_color: str = "#5E3C99",
        alpha_real: float = 0.65,
        alpha_synth: float = 0.65,
        point_size: float = 20.0,
        random_state: Optional[int] = None,
        legend_outside: bool = True,
    ):
        """
        2D embedding of joint distribution:
          - encodes numeric + categorical with ColumnTransformer
          - applies UMAP/t-SNE
          - real = large filled markers
          - each synthetic dataset = hollow colored markers
        """
        rs = self.random_state if random_state is None else random_state

        return plot_joint_embedding_2d(
            real_df=self.real,
            synth_df=self.synth,
            schema=self.schema,
            label_col=label_col,
            method=method,
            fig_size=fig_size,
            main_title=main_title,
            real_label=real_label,
            real_color=real_color,
            alpha_real=alpha_real,
            alpha_synth=alpha_synth,
            point_size=point_size,
            random_state=rs,
            legend_outside=legend_outside,
        )
