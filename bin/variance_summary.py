#!/usr/bin/env python3
"""
Variance summary for cell-type annotation benchmarking.

Produces three sets of outputs from label_results.tsv(.gz):

  1. Study-variance heatmaps (F1 / Precision / Recall × cell type × study)
     study_variance_{key}.png/.pdf  +  study_variance_{key}_summary.tsv

  2. Cross-study stability summaries
     method_stability_summary.tsv
     method_reference_stability_summary.tsv
     celltype_stability_summary.tsv

  3. Per-cell-type stability dot plot
     celltype_stability.png

Usage:
    python variance_summary.py \\
        --label_results label_results.tsv.gz \\
        --organism mus_musculus \\
        --cutoff 0.0 \\
        --subsample_ref 100 \\
        --outdir variance_summary
"""

import argparse
import gzip
import os
import sys

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import METHOD_COLORS, METHOD_NAMES, set_pub_style

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LABELS = (
    "/space/grp/rschwartz/rschwartz/evaluation_summary.nf"
    "/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false"
    "/aggregated_results/files/label_results.tsv"
)

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
_CENSUS_BY_ORGANISM = {
    "mus_musculus": os.path.join(_ASSETS, "census_map_mouse_author.tsv"),
    "homo_sapiens": os.path.join(_ASSETS, "census_map_human.tsv"),
}

FAMILY_COLORS = {
    "GABAergic":          "#4e79a7",
    "Glutamatergic":      "#f28e2b",
    "Astrocyte":          "#59a14f",
    "Oligodendrocyte":    "#76b7b2",
    "OPC":                "#edc948",
    "Vascular":           "#e15759",
    "Microglia":          "#ff9da7",
    "CNS macrophage":     "#ff9da7",
    "Neural stem cell":   "#b07aa1",
    "Hippocampal neuron": "#9c755f",
    "Leukocyte":          "#bab0ac",
    "Neuron":             "#d4b483",
    "Non-neuron":         "#999999",
}

METRICS = ["f1_score", "precision", "recall"]
METRIC_LABELS = {"f1_score": "F1", "precision": "Precision", "recall": "Recall"}
ALL_KEYS = ["subclass", "class", "family", "global"]
KEY_ORDER = ["global", "family", "class", "subclass"]

SYSTEMATIC_FAILURE_THRESHOLD = 0.5
MIN_STUDIES_FOR_FAILURE = 3

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_label_results(path):
    opener = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with opener as f:
        df = pd.read_csv(f, sep="\t")
    for col in ["f1_score", "precision", "recall"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["study"] = df["study"].str.split().str[0]
    return df


def load_family_map(census_path):
    df = pd.read_csv(census_path, sep="\t")
    return (
        df.drop_duplicates(subset="subclass")
        .set_index("subclass")["family"]
        .to_dict()
    )

# ---------------------------------------------------------------------------
# Heatmap helpers
# ---------------------------------------------------------------------------

def make_pivot(df, metric, row_order, col_order):
    pivot = (
        df.groupby(["label", "study"])[metric]
        .mean()
        .unstack("study")
    )
    return pivot.reindex(index=row_order, columns=col_order)


def _metric_stats(pivot, prefix):
    return {
        f"mean_{prefix}": pivot.mean(axis=1).values,
        f"std_{prefix}":  pivot.std(axis=1).values,
        f"min_{prefix}":  pivot.min(axis=1).values,
        f"max_{prefix}":  pivot.max(axis=1).values,
    }


def write_heatmap_summary(pivots, outdir, prefix, key, cutoff):
    f1_pivot = pivots["f1_score"]
    n_total = f1_pivot.shape[1]
    data = {
        "label":           f1_pivot.index,
        "key":             key,
        "cutoff":          cutoff,
        "n_studies":       f1_pivot.notna().sum(axis=1).values,
        "n_studies_total": n_total,
        "frac_studies":    (f1_pivot.notna().sum(axis=1) / n_total).values,
    }
    for metric, col_prefix in [("f1_score", "f1"), ("precision", "precision"), ("recall", "recall")]:
        data.update(_metric_stats(pivots[metric], col_prefix))

    summary = (
        pd.DataFrame(data)
        .sort_values("mean_f1", ascending=False)
        .reset_index(drop=True)
    )
    out = os.path.join(outdir, f"{prefix}_summary.tsv")
    summary.to_csv(out, sep="\t", index=False)
    print(f"Saved {out}")


def draw_heatmap(ax, pivot, title, cbar_label, family_map=None):
    nan_mask = pivot.isna()
    ax.pcolormesh(
        np.ones(pivot.shape),
        cmap=ListedColormap(["#cccccc"]),
        vmin=0, vmax=1,
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar_kws={"label": cbar_label, "shrink": 0.7},
        mask=nan_mask,
        yticklabels=True,
    )
    ax.set_title(title, fontweight="bold", loc="left", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("Cell type", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=40)
    ax.tick_params(axis="y", labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    if family_map:
        for tick in ax.get_yticklabels():
            lbl = tick.get_text()
            family = family_map.get(lbl) or (lbl if lbl in FAMILY_COLORS else None)
            tick.set_color(FAMILY_COLORS.get(family, "black"))


def make_family_legend(ax, families_present):
    handles = [
        mpatches.Patch(color=FAMILY_COLORS.get(f, "black"), label=f)
        for f in sorted(families_present)
        if f in FAMILY_COLORS
    ]
    ax.legend(
        handles=handles,
        title="Cell-type family",
        loc="center",
        frameon=False,
        fontsize=9,
        title_fontsize=10,
        ncol=2,
    )
    ax.axis("off")


def run_heatmap_key(df_all, key, cutoff, outdir, organism, family_map=None):
    df = df_all[(df_all["key"] == key) & (df_all["cutoff"] == cutoff)].copy()
    if df.empty:
        print(f"  No data for key={key}, skipping.")
        return

    f1_base = df.groupby(["label", "study"])["f1_score"].mean().unstack("study")
    row_order = f1_base.mean(axis=1).sort_values().index.tolist()
    col_order = f1_base.mean(axis=0).sort_values().index.tolist()
    print(f"  {key}: {len(row_order)} cell types × {len(col_order)} studies")

    pivots = {m: make_pivot(df, m, row_order, col_order) for m in METRICS}
    prefix = f"study_variance_{key}"
    write_heatmap_summary(pivots, outdir, prefix, key, cutoff)

    panel_h = max(3, len(row_order) * 0.35)
    fig_w = 3 + len(col_order) * 0.75
    legend_h = 2.0 if family_map else 0.0
    nrows = 4 if family_map else 3
    height_ratios = ([panel_h, panel_h, panel_h, legend_h] if family_map
                     else [panel_h, panel_h, panel_h])
    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(fig_w, panel_h * 3 + legend_h + 0.5),
        gridspec_kw={"height_ratios": height_ratios},
    )
    for ax, metric in zip(axes[:3], METRICS):
        title = f"{METRIC_LABELS[metric]} by study — {organism} ({key}, cutoff={cutoff})"
        draw_heatmap(ax, pivots[metric], title, METRIC_LABELS[metric], family_map=family_map)
    axes[2].set_xlabel("Study", fontsize=9)

    if family_map:
        families_present = {
            family_map.get(lbl) or (lbl if lbl in FAMILY_COLORS else None)
            for lbl in row_order
        } - {None}
        make_family_legend(axes[3], families_present)

    plt.tight_layout(pad=1.5, h_pad=2.0)
    out = os.path.join(outdir, f"{prefix}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved {out}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Stability summary helpers
# ---------------------------------------------------------------------------

def per_study_stats(df):
    group_cols = ["label", "key", "method", "reference", "subsample_ref", "cutoff", "study"]
    return (
        df.groupby(group_cols)["f1_score"]
        .mean()
        .reset_index()
        .rename(columns={"f1_score": "mean_f1_study"})
    )


def cell_type_stats(study_df):
    group_cols = ["label", "key", "method", "reference", "subsample_ref", "cutoff"]
    return (
        study_df.groupby(group_cols)
        .agg(
            mean_f1=("mean_f1_study", "mean"),
            std_f1_across_studies=("mean_f1_study", "std"),
            n_studies=("mean_f1_study", "count"),
        )
        .reset_index()
    )


def _rollup(ct_stats, group_cols):
    fail_mask = (
        (ct_stats["mean_f1"] < SYSTEMATIC_FAILURE_THRESHOLD)
        & (ct_stats["n_studies"] >= MIN_STUDIES_FOR_FAILURE)
    )
    fail_counts = (
        ct_stats[fail_mask]
        .groupby(group_cols)
        .size()
        .reset_index(name="n_systematic_failures")
    )
    summary = (
        ct_stats.groupby(group_cols)
        .agg(
            n_celltypes=("label", "nunique"),
            mean_f1=("mean_f1", "mean"),
            mean_std_f1_across_studies=("std_f1_across_studies", "mean"),
            median_std_f1_across_studies=("std_f1_across_studies", "median"),
            std_f1_across_celltypes=("mean_f1", "std"),
        )
        .reset_index()
    )
    summary = summary.merge(fail_counts, on=group_cols, how="left")
    summary["n_systematic_failures"] = summary["n_systematic_failures"].fillna(0).astype(int)
    col_order = group_cols + [
        "n_celltypes", "n_systematic_failures",
        "mean_f1", "mean_std_f1_across_studies",
        "median_std_f1_across_studies", "std_f1_across_celltypes",
    ]
    return summary[col_order].sort_values(group_cols).reset_index(drop=True)


def method_summary(ct_stats):
    ref_marginal = (
        ct_stats.groupby(["label", "key", "method", "cutoff", "subsample_ref"])
        .agg(
            mean_f1=("mean_f1", "mean"),
            std_f1_across_studies=("std_f1_across_studies", "mean"),
            n_studies=("n_studies", "mean"),
        )
        .reset_index()
    )
    return _rollup(ref_marginal, ["method", "key", "cutoff", "subsample_ref"])


def method_reference_summary(ct_stats):
    return _rollup(ct_stats, ["method", "reference", "key", "cutoff", "subsample_ref"])


def celltype_stability_summary(ct_stats):
    ref_marginal = (
        ct_stats.groupby(["label", "key", "method", "cutoff", "subsample_ref"])
        .agg(
            mean_f1=("mean_f1", "mean"),
            std_f1_across_studies=("std_f1_across_studies", "mean"),
            n_studies=("n_studies", "mean"),
        )
        .reset_index()
    )
    ref_marginal["systematic_failure"] = (
        (ref_marginal["mean_f1"] < SYSTEMATIC_FAILURE_THRESHOLD)
        & (ref_marginal["n_studies"] >= MIN_STUDIES_FOR_FAILURE)
    )
    return ref_marginal.sort_values(
        ["key", "cutoff", "subsample_ref", "mean_f1"]
    ).reset_index(drop=True)

# ---------------------------------------------------------------------------
# Stability dot plot
# ---------------------------------------------------------------------------

def plot_celltype_stability(df, outpath, cutoff, subsample_ref):
    df = df[(df["cutoff"] == cutoff) & (df["subsample_ref"] == subsample_ref)].copy()
    keys = [k for k in KEY_ORDER if k in df["key"].unique()]
    methods = [m for m in ["scvi_knn", "scvi_rf", "seurat"] if m in df["method"].unique()]
    offsets = {"scvi_knn": -0.2, "scvi_rf": 0.0, "seurat": 0.2}

    ncols = len(keys)
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(7 * ncols, max(4, 0.4 * df["label"].nunique())),
        sharey=False,
    )
    if ncols == 1:
        axes = [axes]

    for ci, key in enumerate(keys):
        ax = axes[ci]
        df_key = df[df["key"] == key]
        label_order = (
            df_key.groupby("label")["std_f1_across_studies"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        y = np.arange(len(label_order))

        for method in methods:
            df_m = df_key[df_key["method"] == method].set_index("label")
            vals = [
                df_m.loc[lbl, "std_f1_across_studies"] if lbl in df_m.index else np.nan
                for lbl in label_order
            ]
            n_studies = [
                df_m.loc[lbl, "n_studies"] if lbl in df_m.index else np.nan
                for lbl in label_order
            ]
            single_study = [ns == 1 for ns in n_studies]
            plot_vals = [0.0 if (np.isnan(v) and ss) else v
                         for v, ss in zip(vals, single_study)]

            multi_mask = [not ss for ss in single_study]
            ax.scatter(
                [v for v, m in zip(plot_vals, multi_mask) if m],
                [yi + offsets[method] for yi, m in zip(y, multi_mask) if m],
                color=METHOD_COLORS[method], s=60, zorder=3,
                label=METHOD_NAMES[method],
            )
            ax.scatter(
                [0.0 for ss in single_study if ss],
                [yi + offsets[method] for yi, ss in zip(y, single_study) if ss],
                color=METHOD_COLORS[method], s=60, zorder=3,
                facecolors="none", edgecolors=METHOD_COLORS[method], linewidths=1.5,
            )

        for yi, lbl in enumerate(label_order):
            if df_key.loc[df_key["label"] == lbl, "systematic_failure"].any():
                ax.axhspan(yi - 0.5, yi + 0.5, color="mistyrose", alpha=0.4, zorder=0)

        ax.set_yticks(y)
        ax.set_yticklabels(label_order, fontsize=9)
        ax.tick_params(axis="x", labelsize=11)
        ax.set_xlabel("Std(F1) across studies", fontsize=12)
        ax.axvline(0, color="lightgrey", lw=0.8, ls="--")
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(key, fontsize=14, fontweight="bold")

    handles = [
        mlines.Line2D([], [], color=METHOD_COLORS[m], marker="o",
                      linestyle="None", markersize=8, label=METHOD_NAMES[m])
        for m in methods
    ]
    handles.append(
        mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
                      markersize=8, markerfacecolor="none", markeredgewidth=1.5,
                      label="single study (std undefined)")
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Per-cell-type cross-study variance (cutoff={cutoff}, subsample_ref={subsample_ref})\n"
        "Pink shading = systematic failure (mean F1 < 0.5 in ≥3 studies)",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label_results", default=DEFAULT_LABELS)
    p.add_argument("--organism",      default="mus_musculus")
    p.add_argument("--census",        default=None,
                   help="Census map TSV; auto-detected from --organism if omitted")
    p.add_argument("--cutoff",        default=0.0, type=float)
    p.add_argument("--subsample_ref", default=100, type=int)
    p.add_argument("--outdir",        default="variance_summary")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    census_path = args.census or _CENSUS_BY_ORGANISM.get(args.organism)
    family_map = (
        load_family_map(census_path)
        if census_path and os.path.exists(census_path)
        else None
    )
    if family_map:
        print(f"Loaded family map: {len(family_map)} entries")
    else:
        print("No census map found; y-tick labels will not be coloured.")

    print("Loading data...")
    df = load_label_results(args.label_results)
    print(f"  {len(df):,} rows loaded")

    # 1. Study-variance heatmaps
    print("Generating study-variance heatmaps...")
    for key in ALL_KEYS:
        run_heatmap_key(df, key, args.cutoff, args.outdir, args.organism, family_map)

    # 2. Cross-study stability summaries
    print("Computing cross-study stability summaries...")
    study_df = per_study_stats(df)
    ct_stats = cell_type_stats(study_df)

    meth = method_summary(ct_stats)
    meth.to_csv(os.path.join(args.outdir, "method_stability_summary.tsv"), sep="\t", index=False)
    print(f"  method_stability_summary: {len(meth)} rows")

    meth_ref = method_reference_summary(ct_stats)
    meth_ref.to_csv(os.path.join(args.outdir, "method_reference_stability_summary.tsv"), sep="\t", index=False)
    print(f"  method_reference_stability_summary: {len(meth_ref)} rows")

    ct_stab = celltype_stability_summary(ct_stats)
    ct_stab.to_csv(os.path.join(args.outdir, "celltype_stability_summary.tsv"), sep="\t", index=False)
    print(f"  celltype_stability_summary: {len(ct_stab)} rows")

    # 3. Stability dot plot
    print("Plotting celltype stability...")
    plot_celltype_stability(
        ct_stab,
        os.path.join(args.outdir, "celltype_stability.png"),
        args.cutoff, args.subsample_ref,
    )

    print("Done.")


if __name__ == "__main__":
    main()
