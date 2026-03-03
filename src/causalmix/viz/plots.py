# SynthPlots: 
# separate plotting module for comprison of marginal distributions, pairwise relationships, and joint distributions
# updated 1/26/2026: changed single column plot by adding options for part of a panel figure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Literal, Union, Dict

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from mpl_toolkits.axes_grid1 import make_axes_locatable # for heatmap bar adjustment

ColPlotType = Literal["auto", "categorical", "numerical"]


def single_column_plot(
    real_data: pd.DataFrame,
    synthetic_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    schema,
    column_name: str,
    *,
    col_plot_type: ColPlotType = "auto",    # 'auto', 'categorical', or 'numerical'
    sample_size: Optional[int] = None,
    fig_size: Tuple[float, float] = (6, 4),
    main_title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    label_fontsize: int = 9,
    tick_fontsize: int = 8,      # tick labels
    title_fontsize: int = 9,     # per-axes titles
    real_label: str = "Real",
    synth_label: str = "Synthetic",
    real_color: str = "#5E3C99",   # deep purple for real data
    max_categories: int = 30,
    legend_outside: bool = True,
    show_legend: bool = True,
    # numerical options
    show_hist: bool = False,       # histograms optional (off by default)
    num_bins: int = 40,
    kde_bw_adjust: float = 1.0,
    density: bool = True,          # y-axis = density (True) or counts (False)
    linewidth: float = 2.0, # for numericals
    fill_real_kde: bool = True,    # fill area under real KDE curve
    ax: Optional[plt.Axes] = None,  
    add_title: bool = True, 
) -> plt.Figure:
    """
    1D column plot similar to SDV's get_column_plot, but with:
      - schema-based type detection (numeric vs categorical)
      - support for one or multiple synthetic datasets
        * synthetic_data: DataFrame
        * synthetic_data: dict[name -> DataFrame]
      - bar plots for categorical/boolean
      - KDE curves for numerical (always shown), hist optional
      - customizable title, axis labels, colors
      - no grid (clean publication style)
    """

    # -----------------------------------------------------
    # 0. Normalize synthetic_data to dict[name -> DataFrame]
    # -----------------------------------------------------
    if isinstance(synthetic_data, pd.DataFrame):
        synth_dict: Dict[str, pd.DataFrame] = {synth_label: synthetic_data}
    else:
        synth_dict = dict(synthetic_data)
        if not synth_dict:
            raise ValueError("synthetic_data dict is empty.")

    synth_names = list(synth_dict.keys())
    n_synth = len(synth_names)

    # publication-quality palette for multiple synthetic datasets
    PUB_COLORS = [
        "#1b9e77", "#d95f02", "#7570b3",
        "#e7298a", "#66a61e", "#e6ab02",
        "#a6761d", "#666666", "#1f78b4", "#b2df8a",
    ]
    synth_colors = {
        name: PUB_COLORS[i % len(PUB_COLORS)]
        for i, name in enumerate(synth_names)
    }

    # -----------------------------------------------------
    # 1. Determine column type from schema
    # -----------------------------------------------------
    num_cols = set(getattr(schema, "numeric", []))
    cat_cols = set(getattr(schema, "categorical", [])) | set(getattr(schema, "binary", []))

    if column_name not in real_data.columns:
        raise ValueError(f"Column '{column_name}' not found in real_data.")

    if col_plot_type == "auto":
        if column_name in num_cols:
            resolved_type = "numerical"
        elif column_name in cat_cols:
            resolved_type = "categorical"
        else:
            raise ValueError(
                f"Column '{column_name}' is not in schema.numeric or schema.categorical/binary; "
                f"cannot auto-detect type."
            )
    else:
        resolved_type = col_plot_type

    # -----------------------------------------------------
    # 2. Optional sampling (per dataset)
    # -----------------------------------------------------
    rng = np.random.RandomState(0)

    def _sample_series(df: pd.DataFrame) -> pd.Series:
        s = df[column_name]
        if sample_size is None or len(s) <= sample_size:
            return s
        return s.sample(sample_size, random_state=rng)

    real_col = _sample_series(real_data)
    synth_cols = {name: _sample_series(df) for name, df in synth_dict.items()}

    # -----------------------------------------------------
    # 3. Build figure
    # -----------------------------------------------------
    #fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=True)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=True)
        created_fig = True
    else:
        fig = ax.figure

    # =====================================================
    # CATEGORICAL / BOOLEAN → GROUPED BAR PLOT
    # =====================================================
    if resolved_type == "categorical":
        real_cat = real_col.astype(str)
        synth_cat = {name: s.astype(str) for name, s in synth_cols.items()}

        # All levels across real + all synthetic
        all_levels = [real_cat] + list(synth_cat.values())
        all_levels_concat = pd.concat(all_levels, ignore_index=True).dropna()
        levels = sorted(all_levels_concat.unique().tolist())[:max_categories]

        # counts per dataset
        real_counts = real_cat.value_counts(dropna=False).reindex(levels, fill_value=0)
        synth_counts = {
            name: s.value_counts(dropna=False).reindex(levels, fill_value=0)
            for name, s in synth_cat.items()
        }

        # frequency or count
        if density:
            real_vals = (
                real_counts / real_counts.sum() if real_counts.sum() > 0 else real_counts
            )
            synth_vals = {
                name: (cnt / cnt.sum() if cnt.sum() > 0 else cnt)
                for name, cnt in synth_counts.items()
            }
            ylab = y_label or "Frequency"
        else:
            real_vals = real_counts
            synth_vals = synth_counts
            ylab = y_label or "Count"

        idx = np.arange(len(levels))

        # ----- NEW: compact group, bars touching within level -----
        n_groups = 1 + n_synth      # Real + synthetic datasets
        cluster_width = 0.8         # total width per category (< 1 leaves gap between categories)

        # total width reserved per category
        if len(levels) <= 2:
            cluster_width = 0.45   # binary: thinner bars + more whitespace
        elif len(levels) <= 4:
            cluster_width = 0.60   # small categorical
        else:
            cluster_width = 0.80   # default

        bar_width = cluster_width / n_groups
        # Optional hard cap so bars never get too fat (helps when n_groups is small)
        bar_width = min(bar_width, 0.22)

        # centers relative to category index: start at -cluster_width/2 + bar_width/2
        offsets = -cluster_width / 2 + bar_width * (0.5 + np.arange(n_groups))
        # ----------------------------------------------------------

        # group 0 = real
        ax.bar(
            idx + offsets[0],
            real_vals.to_numpy(),
            width=bar_width,
            label=real_label,
            color=real_color,
            alpha=0.8,
        )

        # groups 1..n_groups-1 = synthetic datasets
        for j, name in enumerate(synth_names):
            vals = synth_vals[name]
            ax.bar(
                idx + offsets[j + 1],
                vals.to_numpy(),
                width=bar_width,
                label=name,
                color=synth_colors[name],
                alpha=0.7,
            )

        ax.set_xticks(idx)
        ax.set_xticklabels(levels, rotation=35, ha="right")
        # reduce gap between binary categories
        if len(levels) == 2:
            ax.set_xlim(-0.35, 1.35)
        ax.set_xlabel(x_label or column_name, fontsize=label_fontsize)
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        ax.grid(False)
        
        if show_legend:
            if legend_outside and created_fig:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            elif not legend_outside:
                ax.legend(loc="best")

        if add_title:
            fig.suptitle(main_title or f"Distribution of {column_name} ({resolved_type})", fontsize=title_fontsize)
        return fig

    # =====================================================
    # NUMERICAL → KDE (always) + optional HIST
    # =====================================================
    elif resolved_type == "numerical":
        # coerce to numeric
        r = pd.to_numeric(real_col, errors="coerce").dropna().to_numpy()
        s_dict = {
            name: pd.to_numeric(col, errors="coerce").dropna().to_numpy()
            for name, col in synth_cols.items()
        }

        # Ensure we have at least some data
        if len(r) == 0 and all(len(s) == 0 for s in s_dict.values()):
            raise ValueError(f"No valid numeric data for column '{column_name}'.")

        # Axis range based on all data
        mins = [r.min()] if len(r) > 0 else []
        maxs = [r.max()] if len(r) > 0 else []
        for s in s_dict.values():
            if len(s) > 0:
                mins.append(s.min())
                maxs.append(s.max())
        xmin = np.nanmin(mins)
        xmax = np.nanmax(maxs)
        if xmin == xmax:
            xmin -= 0.5
            xmax += 0.5

        bins = np.linspace(xmin, xmax, num_bins + 1)

        # Optional histograms: real + each synthetic
        if show_hist:
            if len(r) > 0:
                ax.hist(
                    r,
                    bins=bins,
                    density=density,
                    alpha=0.25,
                    facecolor=real_color,
                    edgecolor=real_color,
                    linewidth=1.0,
                    label=None,  # legend from KDE curves
                )
            for name in synth_names:
                s = s_dict[name]
                if len(s) == 0:
                    continue
                ax.hist(
                    s,
                    bins=bins,
                    density=density,
                    alpha=0.2,
                    facecolor=synth_colors[name],
                    edgecolor=synth_colors[name],
                    linewidth=1.0,
                    label=None,
                )

        # KDE curves (always shown) if scipy available
        try:
            from scipy.stats import gaussian_kde

            xs = np.linspace(xmin, xmax, 512)

            # Real KDE
            if len(r) > 0:
                kde_r = gaussian_kde(r)
                if kde_bw_adjust != 1.0:
                    kde_r.set_bandwidth(bw_method=kde_r.factor * kde_bw_adjust)
                yr = kde_r(xs)
                if not density:
                    bin_width = bins[1] - bins[0]
                    yr *= len(r) * bin_width
                ax.plot(xs, yr, color=real_color, linewidth=linewidth, label=real_label)
                if fill_real_kde:
                    ax.fill_between(xs, yr, 0, color=real_color, alpha=0.15)

            # Synthetic KDEs
            for name in synth_names:
                s = s_dict[name]
                if len(s) == 0:
                    continue
                kde_s = gaussian_kde(s)
                if kde_bw_adjust != 1.0:
                    kde_s.set_bandwidth(bw_method=kde_s.factor * kde_bw_adjust)
                ys = kde_s(xs)
                if not density:
                    bin_width = bins[1] - bins[0]
                    ys *= len(s) * bin_width
                ax.plot(
                    xs,
                    ys,
                    color=synth_colors[name],
                    linewidth=linewidth,
                    linestyle="--",
                    label=name,
                )

        except ImportError:
            if not show_hist:
                raise ImportError(
                    "scipy is required for KDE; set show_hist=True to plot only histograms."
                )

        if density:
            ylab = y_label or "Density"
        else:
            ylab = y_label or "Count"

        ax.set_xlabel(x_label or column_name, fontsize=label_fontsize)
        ax.set_ylabel(ylab, fontsize=label_fontsize)
        ax.grid(False)
        if show_legend:
            if legend_outside and created_fig:
                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
            elif not legend_outside:
                ax.legend(loc="best")
        
        if add_title:
            fig.suptitle(main_title or f"Distribution of {column_name} ({resolved_type})", fontsize=title_fontsize)
        return fig

    else:
        raise ValueError(f"Unknown resolved_type: {resolved_type}")
    ax.tick_params(axis="both", labelsize=tick_fontsize)



# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from typing import Optional, Tuple, Literal, Union, Dict

PlotType = Literal["scatter", "box", "heatmap", None]


def pair_column_plot(
    real_data: pd.DataFrame,
    synthetic_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    schema,
    column_names: Tuple[str, str],
    *,
    plot_type: PlotType = None,         # None → infer using schema
    sample_size: Optional[int] = None,
    fig_size: Tuple[float, float] = (10, 4),
    main_title: Optional[str] = None,
    add_title: bool = True,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    real_label: str = "Real",
    real_color: str = "#5E3C99",        # deep purple for real data
    synth_label: str = "Synthetic",
    max_categories: int = 20,
    point_alpha: float = 0.5,
    point_size: float = 12.0,
    legend_outside: bool = True,
) -> plt.Figure:
    """
    Pairwise plot for (col_x, col_y) comparing real vs synthetic data.

    - Uses a schema with attributes:
        * schema.numeric      : list of numeric columns
        * schema.categorical  : list of categorical columns
        * schema.binary       : list of binary columns
    - plot_type:
        * 'scatter' → (num, num)
        * 'box'     → (num, cat) or (cat, num)
        * 'heatmap' → (cat, cat)
        * None      → infer from schema
    - synthetic_data:
        * DataFrame                → single synthetic dataset
        * dict[name → DataFrame]   → multiple synthetic datasets
    - No grids; publication-oriented styling.
    """

    col_x, col_y = column_names

    # -----------------------------------------------------
    # 0. Normalize synthetic_data to a dict[name -> DataFrame]
    # -----------------------------------------------------
    if isinstance(synthetic_data, pd.DataFrame):
        synth_dict: Dict[str, pd.DataFrame] = {synth_label: synthetic_data}
    else:
        synth_dict = dict(synthetic_data)
        if not synth_dict:
            raise ValueError("synthetic_data dict is empty.")

    synth_names = list(synth_dict.keys())
    n_synth = len(synth_names)

    # publication-quality palette for multiple synthetic datasets
    PUB_COLORS = [
        "#1b9e77", "#d95f02", "#7570b3",
        "#e7298a", "#66a61e", "#e6ab02",
        "#a6761d", "#666666", "#1f78b4", "#b2df8a",
    ]
    synth_colors = {
        name: PUB_COLORS[i % len(PUB_COLORS)]
        for i, name in enumerate(synth_names)
    }

    # -----------------------------------------------------
    # 1. Type inference from schema
    # -----------------------------------------------------
    num_cols = set(getattr(schema, "numeric", []))
    cat_cols = set(getattr(schema, "categorical", [])) | set(getattr(schema, "binary", []))

    def _col_type(col: str) -> str:
        if col in num_cols:
            return "num"
        if col in cat_cols:
            return "cat"
        raise ValueError(f"Column '{col}' not found in schema.numeric/categorical/binary.")

    t1 = _col_type(col_x)
    t2 = _col_type(col_y)

    if plot_type is None:
        if t1 == "num" and t2 == "num":
            plot_type_resolved = "scatter"
        elif t1 == "cat" and t2 == "cat":
            plot_type_resolved = "heatmap"
        else:
            plot_type_resolved = "box"
    else:
        plot_type_resolved = plot_type

    # -----------------------------------------------------
    # 2. Optional sampling helper
    # -----------------------------------------------------
    rng = np.random.RandomState(0)

    def _sample(df: pd.DataFrame) -> pd.DataFrame:
        if sample_size is None or len(df) <= sample_size:
            return df
        return df.sample(sample_size, random_state=rng)

    # sample once for real
    real = _sample(real_data[[col_x, col_y]])
    # sample each synthetic dataset
    synth_sampled = {name: _sample(df[[col_x, col_y]]) for name, df in synth_dict.items()}

    # =====================================================
    # SCATTER (numeric–numeric)
    # =====================================================
    if plot_type_resolved == "scatter":
        fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=True)

        x_r, y_r = real[col_x].to_numpy(), real[col_y].to_numpy()

        xs_all, ys_all = [], []
        for name in synth_names:
            df_s = synth_sampled[name]
            xs_all.append(df_s[col_x].to_numpy())
            ys_all.append(df_s[col_y].to_numpy())
        xs_all = np.concatenate(xs_all) if xs_all else np.array([])
        ys_all = np.concatenate(ys_all) if ys_all else np.array([])

        xmin = np.nanmin([x_r.min(), xs_all.min()]) if xs_all.size else x_r.min()
        xmax = np.nanmax([x_r.max(), xs_all.max()]) if xs_all.size else x_r.max()
        ymin = np.nanmin([y_r.min(), ys_all.min()]) if ys_all.size else y_r.min()
        ymax = np.nanmax([y_r.max(), ys_all.max()]) if ys_all.size else y_r.max()

        # Real: larger filled circles
        real_point_size = point_size * 1.8
        ax.scatter(
            x_r, y_r,
            s=real_point_size,
            alpha=point_alpha,
            color=real_color,
            marker="o",
            edgecolor="white",
            linewidth=0.6,
            label=real_label,
        )

        # Synthetic: hollow circles with distinct colors
        for name in synth_names:
            df_s = synth_sampled[name]
            x_s = df_s[col_x].to_numpy()
            y_s = df_s[col_y].to_numpy()
            ax.scatter(
                x_s, y_s,
                s=point_size,
                alpha=point_alpha,
                facecolor="none",
                edgecolor=synth_colors[name],
                marker="o",
                linewidth=1.0,
                label=name,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(x_label or col_x)
        ax.set_ylabel(y_label or col_y)
        ax.grid(False)

        if legend_outside:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.legend(loc="best")
        if add_title:
            fig.suptitle(main_title or f"{col_x} vs {col_y}")
        return fig

    # =====================================================
    # BOX (mixed numeric–categorical)
    # =====================================================
    if plot_type_resolved == "box":
        fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=True)

        # determine which is numeric vs categorical
        if t1 == "num" and t2 == "cat":
            num_col, cat_col = col_x, col_y
        elif t1 == "cat" and t2 == "num":
            num_col, cat_col = col_y, col_x
        else:
            # fallback (should be rare if schema is consistent)
            num_col, cat_col = col_y, col_x

        num_r = real[num_col]
        cat_r = real[cat_col].astype(str)

        # all category levels across real + synth
        all_cat_vals = [cat_r]
        for name in synth_names:
            all_cat_vals.append(synth_sampled[name][cat_col].astype(str))
        all_cat = pd.concat(all_cat_vals, ignore_index=True).dropna()
        levels = sorted(all_cat.unique().tolist())[:max_categories]

        cat_r = cat_r.where(cat_r.isin(levels))
        num_s_dict = {}
        cat_s_dict = {}
        for name in synth_names:
            num_s = synth_sampled[name][num_col]
            cat_s = synth_sampled[name][cat_col].astype(str)
            cat_s = cat_s.where(cat_s.isin(levels))
            num_s_dict[name] = num_s
            cat_s_dict[name] = cat_s

        idx = np.arange(len(levels))

        # ----- FIXED: evenly spaced group positions *within* category span -----
        # total groups = 1 (real) + n_synth
        n_groups = 1 + n_synth
        # cluster each category's groups inside this span (< 1 to avoid overlap between categories)
        group_span = 0.4           # you can tweak (0.4–0.7 works well)
        # group centers offsets relative to category index
        if n_groups == 1:
            group_offsets = np.array([0.0])
        else:
            group_offsets = np.linspace(-group_span / 2, group_span / 2, n_groups)
        # choose box width small enough so groups don't touch each other *or* next category
        box_width = group_span / (n_groups + 0.5)
        # ----------------------------------------------------------------------

        # Real boxes (group index 0)
        data_r = [num_r[cat_r == lv] for lv in levels]
        ax.boxplot(
            data_r,
            positions=idx + group_offsets[0],
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor=real_color, alpha=0.6),
            medianprops=dict(color="black"),
        )

        # Synthetic boxes (groups 1 .. n_groups-1)
        for j, name in enumerate(synth_names):
            pos = idx + group_offsets[j + 1]
            num_s = num_s_dict[name]
            cat_s = cat_s_dict[name]
            data_s = [num_s[cat_s == lv] for lv in levels]
            ax.boxplot(
                data_s,
                positions=pos,
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=synth_colors[name], alpha=0.5),
                medianprops=dict(color="black"),
            )

        ax.set_xticks(idx)
        ax.set_xticklabels([str(lv) for lv in levels], rotation=35, ha="right")
        ax.set_xlabel(x_label or cat_col)
        ax.set_ylabel(y_label or num_col)
        ax.grid(False)

        # legend: real + each synthetic
        ax.scatter([], [], color=real_color, label=real_label)
        for name in synth_names:
            ax.scatter([], [], color=synth_colors[name], label=name)

        if legend_outside:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.legend(loc="best")
        
        if add_title:
            fig.suptitle(main_title or f"{col_x} vs {col_y}")
        return fig

    # =====================================================
    # HEATMAP (categorical–categorical)
    # =====================================================
    # plot_type_resolved == "heatmap"
    n_cols = 1 + n_synth
    fig, axes = plt.subplots(
        1, n_cols,
        figsize=fig_size,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_cols == 1:
        axes = [axes]

    ax_real = axes[0]
    ax_synth_list = axes[1:]

    xr = real[col_x].astype(str)
    yr = real[col_y].astype(str)

    xs_list = [synth_sampled[name][col_x].astype(str) for name in synth_names]
    ys_list = [synth_sampled[name][col_y].astype(str) for name in synth_names]

    # levels across real + all synth
    all_x_vals = [xr] + xs_list
    all_y_vals = [yr] + ys_list
    all_x = pd.concat(all_x_vals, ignore_index=True).dropna()
    all_y = pd.concat(all_y_vals, ignore_index=True).dropna()

    levels_x = sorted(all_x.unique().tolist())[:max_categories]
    levels_y = sorted(all_y.unique().tolist())[:max_categories]

    # Real crosstab
    xr2 = xr.where(xr.isin(levels_x))
    yr2 = yr.where(yr.isin(levels_y))
    ct_r = pd.crosstab(xr2, yr2).reindex(index=levels_x, columns=levels_y, fill_value=0)
    Pr = ct_r.values.astype(float)
    if Pr.sum() > 0:
        Pr /= Pr.sum()
    vmax = Pr.max() or 1e-6

    # Synthetic crosstabs
    Ps_list = []
    for xs, ys in zip(xs_list, ys_list):
        xs2 = xs.where(xs.isin(levels_x))
        ys2 = ys.where(ys.isin(levels_y))
        ct_s = pd.crosstab(xs2, ys2).reindex(index=levels_x, columns=levels_y, fill_value=0)
        Ps = ct_s.values.astype(float)
        if Ps.sum() > 0:
            Ps /= Ps.sum()
        Ps_list.append(Ps)
        vmax = max(vmax, Ps.max() or 0.0)

    extent = [0, len(levels_y), 0, len(levels_x)]

    # Real heatmap
    im_real = ax_real.imshow(
        Pr, cmap="Blues", vmin=0, vmax=vmax,
        interpolation="nearest", aspect="equal", extent=extent
    )

    centers_y = np.arange(len(levels_y)) + 0.5
    centers_x = np.arange(len(levels_x)) + 0.5

    ax_real.set_xticks(centers_y)
    ax_real.set_xticklabels(levels_y, rotation=35, ha="right")
    ax_real.set_yticks(centers_x)
    ax_real.set_yticklabels(levels_x)
    ax_real.set_xlabel(x_label or col_y)
    ax_real.set_ylabel(y_label or col_x)
    ax_real.set_title(real_label)

    # Synthetic heatmaps
    for ax_s, Ps, name in zip(ax_synth_list, Ps_list, synth_names):
        ax_s.imshow(
            Ps, cmap="Blues", vmin=0, vmax=vmax,
            interpolation="nearest", aspect="equal", extent=extent
        )
        ax_s.set_xticks(centers_y)
        ax_s.set_xticklabels(levels_y, rotation=35, ha="right")
        ax_s.set_yticks(centers_x)
        ax_s.set_yticklabels(levels_x)
        ax_s.set_xlabel(x_label or col_y)
        ax_s.set_title(name)

    for axh in axes:
        axh.set_xlim(0, len(levels_y))
        axh.set_ylim(0, len(levels_x))

    # cbar = fig.colorbar(im_real, ax=axes)
    # cbar.set_label("Joint probability")
    cbar = fig.colorbar(
        im_real,
        ax=axes,
        fraction=0.035,   # thinner bar
        pad=0.02,         # closer to axes
        shrink=0.75,      # shorter bar
        aspect=25,        # slender look
    )
    
    cbar.set_label("Joint probability")
    cbar.ax.tick_params(labelsize=8)
    cbar.locator = plt.MaxNLocator(4)   # fewer tick labels
    cbar.update_ticks()


    if add_title:
        fig.suptitle(main_title or f"{col_x} vs {col_y}")
    return fig

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from typing import Optional, Literal, Tuple, Union, Dict

# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder

EmbedMethod = Literal["auto", "umap", "tsne"]


def plot_joint_embedding_2d(
    real_df: pd.DataFrame,
    synth_df: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    schema,
    *,
    label_col: Optional[str] = None,
    method: EmbedMethod = "auto",
    fig_size: Tuple[float, float] = (6, 5),
    main_title: Optional[str] = "2D Joint Embedding: Real vs Synthetic",
    real_label: str = "Real",
    real_color: str = "#1F3C88",
    alpha_real: float = 0.65,
    alpha_synth: float = 0.65,
    point_size: float = 20.0,
    random_state: int = 0,
    legend_outside: bool = True,
    # NEW
    sample_size: Optional[int] = None,   # e.g., 2000
) -> plt.Figure:
    # normalize synth input
    if isinstance(synth_df, pd.DataFrame):
        synth_dict = {"Synthetic": synth_df}
    else:
        synth_dict = dict(synth_df)

    num_cols = list(schema.numeric)
    cat_cols = list(schema.categorical) + list(schema.binary)
    feat_cols = num_cols + cat_cols

    # guards
    missing_real = set(feat_cols) - set(real_df.columns)
    if missing_real:
        raise ValueError(f"Missing columns in real_df: {missing_real}")
    for name, sdf in synth_dict.items():
        missing_synth = set(feat_cols) - set(sdf.columns)
        if missing_synth:
            raise ValueError(f"Missing columns in synthetic dataset '{name}': {missing_synth}")

    # -------------------------
    # NEW: subsample
    # -------------------------
    rng = np.random.RandomState(random_state)

    def _sample(df: pd.DataFrame) -> pd.DataFrame:
        if sample_size is None or len(df) <= sample_size:
            return df
        return df.sample(sample_size, random_state=rng)

    real_df = _sample(real_df)
    synth_dict = {name: _sample(sdf) for name, sdf in synth_dict.items()}

    # -------------------------
    # encode into common space
    # -------------------------
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )
    Z_real = pre.fit_transform(real_df[feat_cols])
    Z_synth_dict = {name: pre.transform(sdf[feat_cols]) for name, sdf in synth_dict.items()}

    # reducer
    reducer, used_method = None, None
    if method in ("auto", "umap"):
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=random_state)
            used_method = "UMAP"
        except ImportError:
            if method == "umap":
                raise ImportError("umap-learn not installed for method='umap'.")
    if reducer is None:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=random_state, init="pca")
        used_method = "t-SNE"

    # stack and embed
    Z_list = [Z_real]
    sizes = [Z_real.shape[0]]
    for name in synth_dict.keys():
        Z_list.append(Z_synth_dict[name])
        sizes.append(Z_synth_dict[name].shape[0])

    Z_all = np.vstack(Z_list)
    X_all_2d = reducer.fit_transform(Z_all)

    idx = 0
    X_real_2d = X_all_2d[idx: idx + sizes[0]]
    idx += sizes[0]
    X_synth_2d_dict = {}
    for name, size in zip(synth_dict.keys(), sizes[1:]):
        X_synth_2d_dict[name] = X_all_2d[idx: idx + size]
        idx += size

    # labels (optional)
    if label_col is not None:
        labels_real = real_df[label_col].to_numpy()
        labels_synth_dict = {name: sdf[label_col].to_numpy() for name, sdf in synth_dict.items()}
    else:
        labels_real = None
        labels_synth_dict = {name: None for name in synth_dict.keys()}

    fig, ax = plt.subplots(1, 1, figsize=fig_size, constrained_layout=True)

    PUB_COLORS = [
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
        "#e6ab02", "#a6761d", "#666666", "#1f78b4", "#b2df8a",
    ]
    synth_names = list(synth_dict.keys())
    synth_colors = {name: PUB_COLORS[i % len(PUB_COLORS)] for i, name in enumerate(synth_names)}
    if len(synth_names) == 1:
        synth_colors[synth_names[0]] = "#DD8452"

    real_point_size = point_size * 1.8

    if labels_real is None:
        ax.scatter(
            X_real_2d[:, 0], X_real_2d[:, 1],
            s=real_point_size, alpha=alpha_real,
            color=real_color, marker="o",
            edgecolor="black", linewidth=0.4,
            label=f"{real_label} (n={len(X_real_2d)})",
        )
        for name in synth_names:
            Xs = X_synth_2d_dict[name]
            ax.scatter(
                Xs[:, 0], Xs[:, 1],
                s=point_size, alpha=alpha_synth,
                facecolor="none", edgecolor=synth_colors[name],
                marker="o", linewidth=0.9,
                label=f"{name} (n={len(Xs)})",
            )
    else:
        labels_real = np.asarray(labels_real)
        all_label_vals = np.unique(
            np.concatenate([labels_real] + [np.asarray(labels_synth_dict[n]) for n in synth_names])
        )
        color_map = plt.cm.get_cmap("tab10", len(all_label_vals))
        label_to_idx = {lab: i for i, lab in enumerate(all_label_vals)}

        for lab in all_label_vals:
            mr = labels_real == lab
            if mr.any():
                color = color_map[label_to_idx[lab]]
                ax.scatter(
                    X_real_2d[mr, 0], X_real_2d[mr, 1],
                    s=real_point_size, alpha=alpha_real,
                    color=color, marker="o",
                    edgecolor="k", linewidth=0.4,
                    label=f"{real_label} (label={lab})",
                )

        for name in synth_names:
            labels_s = np.asarray(labels_synth_dict[name])
            Xs = X_synth_2d_dict[name]
            for lab in all_label_vals:
                ms = labels_s == lab
                if ms.any():
                    color = color_map[label_to_idx[lab]]
                    ax.scatter(
                        Xs[ms, 0], Xs[ms, 1],
                        s=point_size, alpha=alpha_synth,
                        facecolor="none", edgecolor=color,
                        marker="o", linewidth=0.9,
                        label=f"{name} (label={lab})",
                    )

    ax.set_xlabel(f"{used_method} dim 1")
    ax.set_ylabel(f"{used_method} dim 2")
    if main_title is not None:
        ax.set_title(main_title)

    # dedupe legend
    handles, labs = ax.get_legend_handles_labels()
    seen, uh, ul = set(), [], []
    for h, lab in zip(handles, labs):
        if lab and lab not in seen:
            seen.add(lab); uh.append(h); ul.append(lab)

    if legend_outside:
        ax.legend(uh, ul, loc="center left", bbox_to_anchor=(1.02, 0.5))
    else:
        ax.legend(uh, ul)

    ax.grid(False)
    return fig
