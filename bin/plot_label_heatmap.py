#!/usr/bin/env python3
"""
Per-study F1 heatmaps showing sample-level F1 scores by cell type.

For each study, creates a single figure with:
- Rows = granularity levels (subclass, class, family, global)
- Columns = references × methods (e.g. ref1:scVI, ref1:Seurat, ref2:scVI, ...)
- Within each panel: rows = query samples, columns = cell type labels
- Values = F1 scores

Usage:
    python plot_label_heatmap.py \
        --label_f1_results label_f1_results.tsv \
        --organism mus_musculus \
        --outdir heatmaps
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

from plot_utils import set_pub_style, METHOD_NAMES


# Granularity levels in display order (finest → coarsest, one per row)
KEY_ORDER = ["subclass", "class", "family", "global"]

# Experimental factors to annotate on rows, by organism
ORGANISM_FACTORS = {
    "homo_sapiens": ["sex", "disease_state", "dev_stage"],
    "mus_musculus": ["sex", "treatment", "genotype"],
}

# Colors for factor annotations
FACTOR_PALETTES = {
    "sex": {"male": "#4393c3", "female": "#d6604d", "None": "#cccccc"},
    "disease_state": {"control": "#66c2a5", "disease": "#fc8d62", "None": "#cccccc"},
    "treatment": None,
    "dev_stage": None,
    "genotype": None,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-study F1 heatmaps")
    parser.add_argument("--label_f1_results", type=str, required=True,
                        help="Path to label_f1_results.tsv")
    parser.add_argument("--organism", type=str, required=True,
                        help="homo_sapiens or mus_musculus")
    parser.add_argument("--cutoff", type=float, default=0,
                        help="Confidence cutoff to filter data (default: 0)")
    parser.add_argument("--subsample_ref", type=int, required=True,
                        help="Subsample reference level to filter to")
    parser.add_argument("--outdir", type=str, default="heatmaps",
                        help="Output directory")
    return parser.parse_args()


def sanitize_filename(name, max_len=80):
    """Sanitize a string for use as a filename."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:max_len]


def get_factor_color(factor, value, palette_cache):
    """Get color for a factor value, generating palette if needed."""
    if factor not in palette_cache:
        palette_cache[factor] = {}

    if value in palette_cache[factor]:
        return palette_cache[factor][value]

    predefined = FACTOR_PALETTES.get(factor)
    if predefined and value in predefined:
        palette_cache[factor][value] = predefined[value]
        return predefined[value]

    n = len(palette_cache[factor])
    colors = sns.color_palette("Set2", max(8, n + 1))
    palette_cache[factor][value] = colors[n % len(colors)]
    return palette_cache[factor][value]


def build_row_annotations(meta_df, factors, palette_cache):
    """Build color arrays for row annotations."""
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


def draw_annotation_strip(ax, ann_colors, ann_labels, n_queries, queries_ordered,
                          show_ylabels=True):
    """Draw factor annotation strips on the left side of a panel."""
    n_ann = len(ann_colors)
    for a_idx in range(n_ann):
        for q_idx in range(n_queries):
            ax.add_patch(plt.Rectangle(
                (a_idx, q_idx), 1, 1,
                facecolor=ann_colors[a_idx][q_idx],
                edgecolor="white", linewidth=0.3
            ))
    ax.set_xlim(0, n_ann)
    ax.set_ylim(0, n_queries)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_ann) + 0.5)
    ax.set_xticklabels(ann_labels, rotation=90, fontsize=13, fontweight="bold")
    if show_ylabels:
        ax.set_yticks(np.arange(n_queries) + 0.5)
        ax.set_yticklabels(queries_ordered, fontsize=13)
    else:
        ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def get_reference_display(row):
    """Get short display name for a reference, preferring acronym."""
    acronym = row.get("reference_acronym", None)
    if pd.notna(acronym) and str(acronym).strip():
        return str(acronym).strip()
    ref = str(row.get("reference", ""))
    return ref if len(ref) <= 30 else ref[:27] + "..."


def plot_study_heatmap(study_df, study, factors, outdir):
    """Plot a keys (rows) × references × methods (cols) grid of heatmaps for one study."""
    available_keys = [k for k in KEY_ORDER if k in study_df["key"].unique()]
    methods = sorted(study_df["method"].unique())

    # Build reference list with display names
    ref_display = {}
    for ref in study_df["reference"].unique():
        row = study_df[study_df["reference"] == ref].iloc[0]
        ref_display[ref] = get_reference_display(row)
    references = sorted(ref_display.keys())

    n_keys = len(available_keys)
    n_refs = len(references)
    n_methods = len(methods)

    if n_keys == 0 or n_refs == 0 or n_methods == 0:
        return

    # Shared sample (row) ordering across all panels
    sort_cols = [f for f in factors if f in study_df.columns] + ["query"]
    sample_meta = (study_df.drop_duplicates(subset=["query"])
                   .sort_values(sort_cols)
                   .reset_index(drop=True))
    queries_ordered = sample_meta["query"].tolist()
    n_queries = len(queries_ordered)
    if n_queries == 0:
        return

    # Build annotation colors (shared across panels)
    palette_cache = {}
    ann_colors, ann_labels = build_row_annotations(sample_meta, factors, palette_cache)
    n_ann = len(ann_colors)

    # Determine max cell types across all (key, reference) for column sizing
    max_n_labels = 0
    for key in available_keys:
        for ref in references:
            n_labels = study_df[
                (study_df["key"] == key) & (study_df["reference"] == ref)
            ]["label"].nunique()
            max_n_labels = max(max_n_labels, n_labels)

    if max_n_labels == 0:
        return

    # --- Figure sizing ---
    total_heatmap_cols = n_refs * n_methods
    row_height = max(0.4, min(0.6, 25.0 / n_queries))
    panel_height = n_queries * row_height + 3.0
    header_height = 1.5
    fig_height = header_height + panel_height * n_keys + 4.0

    ann_width = max(2.5, n_ann * 1.0)
    col_width_per_label = max(0.45, min(0.85, 35.0 / max_n_labels))
    method_col_width = max(4, max_n_labels * col_width_per_label)
    fig_width = ann_width + method_col_width * total_heatmap_cols + 3

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Grid: (1 header + n_keys) rows × (1 annotation + n_refs*n_methods) cols
    n_grid_rows = n_keys + 1
    width_ratios = [ann_width] + [method_col_width] * total_heatmap_cols
    height_ratios = [header_height] + [panel_height] * n_keys

    outer_gs = gridspec.GridSpec(
        n_grid_rows, 1 + total_heatmap_cols, figure=fig,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.4, wspace=0.15
    )

    # --- Row 0: reference group headers ---
    for ref_idx, reference in enumerate(references):
        col_start = 1 + ref_idx * n_methods
        col_end = col_start + n_methods
        ax_ref = fig.add_subplot(outer_gs[0, col_start:col_end])
        ax_ref.text(
            0.5, 0.5, ref_display[reference],
            transform=ax_ref.transAxes, ha="center", va="center",
            fontsize=16, fontweight="bold", style="italic",
        )
        ax_ref.set_axis_off()

    # Pre-compute cell type ordering per key level (shared across all references)
    key_labels_ordered = {}
    for key in available_keys:
        key_df = study_df[study_df["key"] == key]
        if key_df.empty:
            key_labels_ordered[key] = []
        else:
            label_means = (key_df.groupby("label")["f1_score"]
                           .mean().sort_values(ascending=False))
            key_labels_ordered[key] = label_means.index.tolist()

    # --- Key rows (rows 1..n_keys) ---
    for key_idx, key in enumerate(available_keys):
        grid_row = key_idx + 1
        key_df = study_df[study_df["key"] == key]
        if key_df.empty:
            continue

        labels_ordered = key_labels_ordered[key]

        # Annotation strip (leftmost column)
        ax_ann = fig.add_subplot(outer_gs[grid_row, 0])
        draw_annotation_strip(ax_ann, ann_colors, ann_labels,
                              n_queries, queries_ordered, show_ylabels=True)
        ax_ann.set_ylabel(key.capitalize(), fontsize=18, fontweight="bold",
                          labelpad=15)

        for ref_idx, reference in enumerate(references):
            ref_key_df = key_df[key_df["reference"] == reference]

            for m_idx, method in enumerate(methods):
                col_idx = 1 + ref_idx * n_methods + m_idx
                ax = fig.add_subplot(outer_gs[grid_row, col_idx])

                method_df = ref_key_df[ref_key_df["method"] == method]

                if method_df.empty or len(labels_ordered) == 0:
                    ax.set_visible(False)
                    continue

                pivot = method_df.pivot_table(
                    index="query", columns="label", values="f1_score",
                    aggfunc="mean",
                )
                pivot = pivot.reindex(index=queries_ordered, columns=labels_ordered)

                # Colorbar only on the very last panel
                is_last = (key_idx == n_keys - 1
                           and ref_idx == n_refs - 1
                           and m_idx == n_methods - 1)

                sns.heatmap(
                    pivot, ax=ax, cmap="YlOrRd", vmin=0, vmax=1,
                    cbar=is_last,
                    cbar_kws={"label": "F1 score", "shrink": 0.6,
                              "aspect": 30} if is_last else {},
                    linewidths=0.3, linecolor="white",
                    xticklabels=True, yticklabels=False,
                )

                # Method title on the first key row only
                if key_idx == 0:
                    method_display = METHOD_NAMES.get(method, method)
                    ax.set_title(method_display, fontsize=16,
                                 fontweight="bold", pad=8)

                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.tick_params(axis="x", labelsize=11, rotation=90)
                ax.tick_params(axis="y", left=False, labelleft=False)

    # Suptitle
    fig.suptitle(study, fontsize=22, fontweight="bold", y=1.01)

    # Factor legend at bottom
    if palette_cache:
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
                ncol=min(8, len(legend_handles)),
                fontsize=14, frameon=False,
                bbox_to_anchor=(0.5, -0.02),
            )

    # Save
    os.makedirs(outdir, exist_ok=True)
    study_safe = sanitize_filename(study)
    outpath = os.path.join(outdir, f"{study_safe}_f1_heatmap.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    args = parse_args()
    set_pub_style()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine organism factors
    organism_key = args.organism
    for key in ORGANISM_FACTORS:
        if organism_key.startswith(key):
            organism_key = key
            break
    factors = ORGANISM_FACTORS.get(organism_key, ORGANISM_FACTORS["homo_sapiens"])

    # Load data
    print(f"Loading {args.label_f1_results}...")
    df = pd.read_csv(args.label_f1_results, sep="\t", low_memory=False)
    print(f"  {len(df)} rows, {df['study'].nunique()} studies, "
          f"{df['key'].nunique()} keys, {df['reference'].nunique()} references")

    # Filter to specified subsample_ref and cutoff
    df = df[df["subsample_ref"] == args.subsample_ref].copy()
    print(f"  {len(df)} rows after subsample_ref == {args.subsample_ref} filter")

    df = df[df["cutoff"] == args.cutoff].copy()
    print(f"  {len(df)} rows after cutoff == {args.cutoff} filter")

    if df.empty:
        print("No data after filtering. Exiting.")
        return

    # Generate one heatmap per study
    for study, group_df in df.groupby("study"):
        n_refs = group_df["reference"].nunique()
        n_keys = group_df["key"].nunique()
        print(f"Plotting: study={study} "
              f"({n_refs} references, {n_keys} keys, {len(group_df)} rows)")
        plot_study_heatmap(group_df, study, factors, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
