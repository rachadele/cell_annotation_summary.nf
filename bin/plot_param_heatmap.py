#!/usr/bin/env python3
"""
plot_best_param_heatmap.py

Clustered heatmaps of mean F1 across studies, faceted by taxonomy level (key).
Rows = parameter combinations (reference × method × subsample_ref),
columns = cell types. Row annotation bars encode reference, method, and
subsample_ref; column annotation bars encode n_studies and mean_support.

Usage:
    python plot_best_param_heatmap.py \
        --input rankings_detailed.tsv \
        --outdir param_heatmaps
"""

import argparse
import os
import string

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import METHOD_COLORS, METHOD_NAMES, KEY_ORDER, set_pub_style


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clustered heatmaps of parameter performance per cell type"
    )
    parser.add_argument(
        "--input",
        default="rankings_detailed.tsv",
        help="Path to rankings_detailed.tsv",
    )
    parser.add_argument(
        "--outdir", default="param_heatmaps",
        help="Output directory for per-key heatmap PNGs",
    )
    return parser.parse_args()


def build_ref_short_names(references):
    """Assign single-letter codes (A, B, C, …) to references, sorted."""
    refs_sorted = sorted(references)
    return {ref: string.ascii_uppercase[i] for i, ref in enumerate(refs_sorted)}


def make_clustermap(key, key_df, ref_short, ref_palette, sub_palette, outdir):
    """Create and save a clustered heatmap for one taxonomy level."""

    # Build concise row labels: "A:seurat:500"
    key_df = key_df.copy()
    key_df["row_label"] = (
        key_df["reference"].map(ref_short)
        + ":" + key_df["method"]
        + ":" + key_df["subsample_ref"].astype(str)
    )

    # Pivot: rows = param combos, cols = cell types
    heatmap_data = key_df.pivot_table(
        index="row_label", columns="label", values="mean_f1_across_studies"
    )

    # Build lookup from row_label → original columns
    combo_lookup = (
        key_df.drop_duplicates("row_label")
        .set_index("row_label")[["reference", "method", "subsample_ref"]]
    )

    # --- Row annotation colors ---
    row_colors = pd.DataFrame(index=heatmap_data.index)
    row_colors["Reference"] = [
        ref_palette[combo_lookup.loc[rl, "reference"]] for rl in heatmap_data.index
    ]
    row_colors["Method"] = [
        METHOD_COLORS.get(combo_lookup.loc[rl, "method"], "#333333")
        for rl in heatmap_data.index
    ]
    row_colors["Subsample"] = [
        sub_palette[combo_lookup.loc[rl, "subsample_ref"]]
        for rl in heatmap_data.index
    ]

    # --- Column annotation colors ---
    # n_studies per cell type (same across param combos for a given label+key)
    label_meta = key_df.groupby("label").agg(
        n_studies=("n_studies", "first"),
        mean_support=("mean_support", "first"),
    )

    n_studies_vals = label_meta["n_studies"].reindex(heatmap_data.columns)
    support_vals = label_meta["mean_support"].reindex(heatmap_data.columns)

    # Normalize for colormaps
    n_max = n_studies_vals.max() if n_studies_vals.max() > 0 else 1
    s_max = support_vals.max() if support_vals.max() > 0 else 1
    cmap_n = plt.cm.Blues
    cmap_s = plt.cm.Oranges

    col_colors = pd.DataFrame(index=heatmap_data.columns)
    col_colors["N Studies"] = [cmap_n(v / n_max) for v in n_studies_vals]
    col_colors["Support"] = [cmap_s(v / s_max) for v in support_vals]

    # --- Fill NaN for clustering (shouldn't happen, but defensive) ---
    mask = heatmap_data.isna()
    heatmap_data_filled = heatmap_data.fillna(0)

    # --- Plot ---
    n_rows, n_cols = heatmap_data.shape
    fig_w = max(8, n_cols * 0.6 + 4)
    fig_h = max(6, n_rows * 0.35 + 3)

    g = sns.clustermap(
        heatmap_data_filled,
        mask=mask.reindex(index=heatmap_data_filled.index, columns=heatmap_data_filled.columns),
        cmap="viridis",
        row_colors=row_colors,
        col_colors=col_colors,
        linewidths=0.3,
        linecolor="gray",
        figsize=(fig_w, fig_h),
        cbar_kws={"label": "Mean F1 Across Studies"},
        dendrogram_ratio=(0.12, 0.12),
        xticklabels=True,
        yticklabels=False,
    )

    g.ax_heatmap.set_xlabel("Cell Type", fontsize=14)
    g.ax_heatmap.set_ylabel("")
    g.ax_heatmap.tick_params(axis="x", labelsize=12, rotation=90)

    # Shrink annotation bar labels ("N Studies", "Support", "Reference", etc.)
    g.ax_col_colors.tick_params(axis="y", labelsize=8)
    g.ax_row_colors.tick_params(axis="x", labelsize=8)

    # Title
    g.fig.suptitle(f"Parameter Performance — {key}", fontsize=14, fontweight="bold", y=1.02)

    # --- Move colorbar to right side ---
    g.cax.set_visible(False)
    hm_pos = g.ax_heatmap.get_position()
    cbar_ax = g.fig.add_axes([
        hm_pos.x1 + 0.02,
        hm_pos.y0 + hm_pos.height * 0.3,
        0.015,
        hm_pos.height * 0.4,
    ])
    g.fig.colorbar(
        g.ax_heatmap.collections[0], cax=cbar_ax,
        label="Mean F1",
    )
    cbar_ax.yaxis.label.set_size(11)
    cbar_ax.tick_params(labelsize=9)

    outpath = os.path.join(outdir, f"{key}_param_heatmap.png")
    g.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(g.fig)
    print(f"  Saved {outpath} ({n_rows} combos × {n_cols} cell types)")


def save_legend(ref_short, ref_palette, sub_palette, outdir):
    """Save a standalone legend figure for the heatmap annotations."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axis("off")

    handles = []

    # Reference
    handles.append(mpatches.Patch(color="none", label="Reference"))
    for ref in sorted(ref_short.keys()):
        letter = ref_short[ref]
        handles.append(mpatches.Patch(color=ref_palette[ref],
                                      label=f"  {letter}: {ref}"))

    # Method
    handles.append(mpatches.Patch(color="none", label=""))
    handles.append(mpatches.Patch(color="none", label="Method"))
    for method in sorted(METHOD_COLORS.keys()):
        display = METHOD_NAMES.get(method, method)
        handles.append(mpatches.Patch(color=METHOD_COLORS[method],
                                      label=f"  {display}"))

    # Subsample
    handles.append(mpatches.Patch(color="none", label=""))
    handles.append(mpatches.Patch(color="none", label="Subsample Ref"))
    for sub in sorted(sub_palette.keys()):
        handles.append(mpatches.Patch(color=sub_palette[sub], label=f"  {sub}"))

    ax.legend(
        handles=handles, loc="center", fontsize=20,
        frameon=False, handlelength=1.5, handleheight=1.2,
    )

    fig.tight_layout()
    outpath = os.path.join(outdir, "param_heatmap_legend.png")
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved legend: {outpath}")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    df = pd.read_csv(args.input, sep="\t")
    print(f"Loaded {len(df)} rows from {args.input}")

    # --- Global color palettes ---
    references = sorted(df["reference"].unique())
    ref_short = build_ref_short_names(references)

    ref_palette = dict(zip(references, sns.color_palette("Set2", len(references))))

    subsample_refs = sorted(df["subsample_ref"].unique())
    sub_palette = dict(zip(subsample_refs, sns.color_palette("YlOrRd_d", len(subsample_refs))))

    # --- Standalone legend ---
    save_legend(ref_short, ref_palette, sub_palette, args.outdir)

    # --- One heatmap per key level ---
    keys = [k for k in KEY_ORDER if k in df["key"].unique()]
    for key in keys:
        print(f"\n{key}:")
        key_df = df[df["key"] == key]
        make_clustermap(key, key_df, ref_short, ref_palette, sub_palette, args.outdir)

    print(f"\nDone. Figures in {args.outdir}/")


if __name__ == "__main__":
    main()
