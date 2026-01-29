#!/usr/bin/env python3
"""
Per-study F1 heatmaps showing sample-level F1 scores by cell type.

For each combination of (study, key), creates a heatmap faceted by
method (columns) and reference (rows). Within each facet, rows are
query samples and columns are cell type labels. Values are F1 scores.

Usage:
    python plot_label_heatmap.py \
        --label_f1_results label_f1_results.tsv \
        --organism mus_musculus \
        --outdir heatmaps
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from plot_utils import set_pub_style


# Experimental factors to annotate on rows, by organism
ORGANISM_FACTORS = {
    "homo_sapiens": ["sex", "disease_state", "dev_stage"],
    "mus_musculus": ["sex", "treatment_state", "genotype"],
}

# Colors for factor annotations
FACTOR_PALETTES = {
    "sex": {"male": "#4393c3", "female": "#d6604d", "None": "#cccccc"},
    "disease_state": {"control": "#66c2a5", "disease": "#fc8d62", "None": "#cccccc"},
    "treatment_state": {"no treatment": "#66c2a5", "treatment": "#fc8d62", "None": "#cccccc"},
    "dev_stage": None,  # auto-generate
    "genotype": None,  # auto-generate
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-study F1 heatmaps")
    parser.add_argument("--label_f1_results", type=str, required=True,
                        help="Path to label_f1_results.tsv")
    parser.add_argument("--organism", type=str, required=True,
                        help="homo_sapiens or mus_musculus")
    parser.add_argument("--outdir", type=str, default="heatmaps",
                        help="Output directory")
    return parser.parse_args()


def get_factor_color(factor, value, palette_cache):
    """Get color for a factor value, generating palette if needed."""
    if factor not in palette_cache:
        palette_cache[factor] = {}

    if value in palette_cache[factor]:
        return palette_cache[factor][value]

    # Use predefined palette if available
    predefined = FACTOR_PALETTES.get(factor)
    if predefined and value in predefined:
        palette_cache[factor][value] = predefined[value]
        return predefined[value]

    # Auto-generate color
    n = len(palette_cache[factor])
    colors = sns.color_palette("Set2", max(8, n + 1))
    palette_cache[factor][value] = colors[n % len(colors)]
    return palette_cache[factor][value]


def build_row_annotations(meta_df, factors, palette_cache):
    """Build color arrays for row annotations.

    Parameters
    ----------
    meta_df : DataFrame
        One row per query sample with factor columns.
    factors : list
        Factor column names to include.
    palette_cache : dict
        Mutable palette cache.

    Returns
    -------
    ann_colors : list of list
        One list of colors per factor, aligned with meta_df rows.
    ann_labels : list of str
        Factor names for legend.
    """
    ann_colors = []
    ann_labels = []
    for factor in factors:
        if factor not in meta_df.columns:
            continue
        vals = meta_df[factor].astype(str).str.lower().tolist()
        colors = [get_factor_color(factor, v, palette_cache) for v in vals]
        ann_colors.append(colors)
        ann_labels.append(factor.replace("_", " "))
    return ann_colors, ann_labels


def plot_study_heatmap(study_df, study, key, factors, outdir):
    """Plot a single heatmap for one study x key combination."""
    methods = sorted(study_df["method"].unique())
    references = sorted(study_df["reference"].unique())

    n_methods = len(methods)
    n_refs = len(references)

    if n_methods == 0 or n_refs == 0:
        return

    # Order cell type columns by mean F1 across all facets
    label_means = study_df.groupby("label")["f1_score"].mean().sort_values(ascending=False)
    labels_ordered = label_means.index.tolist()

    # Order sample rows by experimental factors, then query name
    sort_cols = [f for f in factors if f in study_df.columns] + ["query"]
    sample_meta = (study_df.drop_duplicates(subset=["query"])
                   .sort_values(sort_cols)
                   .reset_index(drop=True))
    queries_ordered = sample_meta["query"].tolist()

    n_labels = len(labels_ordered)
    n_queries = len(queries_ordered)

    if n_labels == 0 or n_queries == 0:
        return

    # Build annotation colors
    palette_cache = {}
    ann_colors, ann_labels = build_row_annotations(sample_meta, factors, palette_cache)
    n_ann = len(ann_colors)

    # Figure layout: annotation columns on the left, then one heatmap per method column
    # Rows = references
    ann_width_ratio = 0.3
    heatmap_width_ratio = max(1, n_labels * 0.35)
    width_ratios = [ann_width_ratio] * n_ann + [heatmap_width_ratio] * n_methods

    cell_h = max(0.25, min(0.5, 15.0 / n_queries))
    fig_height = max(4, n_queries * cell_h * n_refs + n_refs * 1.5 + 2)
    fig_width = max(6, sum(width_ratios) + 2)

    fig = plt.figure(figsize=(fig_width, fig_height))
    outer_gs = GridSpec(n_refs, 1, figure=fig, hspace=0.4)

    for r_idx, ref in enumerate(references):
        ref_df = study_df[study_df["reference"] == ref]

        inner_gs = outer_gs[r_idx].subgridspec(
            1, n_ann + n_methods, width_ratios=width_ratios, wspace=0.05
        )

        # Draw annotation strips
        for a_idx in range(n_ann):
            ax_ann = fig.add_subplot(inner_gs[0, a_idx])
            ann_arr = np.array(ann_colors[a_idx]).reshape(-1, 1)
            # Draw colored rectangles manually
            for q_idx in range(n_queries):
                ax_ann.add_patch(plt.Rectangle(
                    (0, q_idx), 1, 1,
                    facecolor=ann_colors[a_idx][q_idx], edgecolor="white", linewidth=0.5
                ))
            ax_ann.set_xlim(0, 1)
            ax_ann.set_ylim(0, n_queries)
            ax_ann.invert_yaxis()
            ax_ann.set_xticks([0.5])
            ax_ann.set_xticklabels([ann_labels[a_idx]], rotation=90, fontsize=7)
            ax_ann.tick_params(left=False, bottom=False)
            if a_idx == 0:
                ax_ann.set_yticks(np.arange(n_queries) + 0.5)
                ax_ann.set_yticklabels(queries_ordered, fontsize=6)
            else:
                ax_ann.set_yticks([])

        # Draw heatmaps for each method
        for m_idx, method in enumerate(methods):
            ax = fig.add_subplot(inner_gs[0, n_ann + m_idx])
            method_df = ref_df[ref_df["method"] == method]

            # Pivot to matrix
            pivot = method_df.pivot_table(
                index="query", columns="label", values="f1_score", aggfunc="mean"
            )
            # Reindex to ordered rows/columns, fill missing with NaN
            pivot = pivot.reindex(index=queries_ordered, columns=labels_ordered)

            sns.heatmap(
                pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                cbar=(m_idx == n_methods - 1),
                cbar_kws={"label": "F1 score", "shrink": 0.6} if m_idx == n_methods - 1 else {},
                linewidths=0.3, linecolor="white",
                xticklabels=True, yticklabels=False,
            )

            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(f"{method}" if r_idx == 0 else "", fontsize=10)
            ax.tick_params(axis="x", labelsize=6, rotation=90)

            # Add reference label on right side of last method column
            if m_idx == n_methods - 1:
                ax.annotate(
                    ref[:50] + ("..." if len(ref) > 50 else ""),
                    xy=(1.02, 0.5), xycoords="axes fraction",
                    fontsize=7, rotation=270, va="center", ha="left",
                )

    fig.suptitle(f"{study} — {key}", fontsize=12, y=1.01)

    # Add factor legends
    if palette_cache:
        from matplotlib.patches import Patch
        legend_handles = []
        for factor, vals in palette_cache.items():
            for val, color in vals.items():
                legend_handles.append(Patch(
                    facecolor=color, edgecolor="gray", linewidth=0.5,
                    label=f"{factor.replace('_', ' ')}: {val}"
                ))
        if legend_handles:
            fig.legend(
                handles=legend_handles, loc="lower center",
                ncol=min(6, len(legend_handles)),
                fontsize=7, frameon=False,
                bbox_to_anchor=(0.5, -0.02),
            )

    # Save
    study_dir = os.path.join(outdir, study)
    os.makedirs(study_dir, exist_ok=True)
    outpath = os.path.join(study_dir, f"{key}_f1_heatmap.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    args = parse_args()
    set_pub_style()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine organism factors
    organism_key = args.organism
    # Handle organism names with suffixes like mus_musculus_tabulamuris
    for key in ORGANISM_FACTORS:
        if organism_key.startswith(key):
            organism_key = key
            break
    factors = ORGANISM_FACTORS.get(organism_key, ORGANISM_FACTORS["homo_sapiens"])

    # Load data
    print(f"Loading {args.label_f1_results}...")
    df = pd.read_csv(args.label_f1_results, sep="\t", low_memory=False)
    print(f"  {len(df)} rows, {df['study'].nunique()} studies, {df['key'].nunique()} keys")

    # Filter to cutoff == 0
    df = df[df["cutoff"] == 0].copy()
    print(f"  {len(df)} rows after cutoff == 0 filter")

    if df.empty:
        print("No data after filtering. Exiting.")
        return

    # Generate heatmaps per (study, key)
    for (study, key), group_df in df.groupby(["study", "key"]):
        print(f"Plotting: study={study}, key={key} ({len(group_df)} rows)")
        plot_study_heatmap(group_df, study, key, factors, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
