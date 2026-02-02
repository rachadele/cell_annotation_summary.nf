#!/usr/bin/env python3
"""
Per-study forest plots showing cell type F1 scores with support-scaled dots.

For each study, creates a single figure with:
- Rows = granularity levels (subclass, class, family, global)
- Columns = references
- Each panel: forest plot with y = cell types, x = mean F1 score
- Dots colored by method, sized by mean support
- Error bars = ± 1 SEM across query samples

Usage:
    python plot_label_forest.py \
        --label_f1_results label_f1_results.tsv \
        --organism mus_musculus \
        --outdir forest_plots
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_utils import set_pub_style, METHOD_NAMES, METHOD_COLORS


KEY_ORDER = ["subclass", "class", "family", "global"]

# Size mapping: support value → marker area (points²)
SIZE_MIN = 15
SIZE_MAX = 250


def parse_args():
    parser = argparse.ArgumentParser(description="Plot per-study F1 forest plots")
    parser.add_argument("--label_f1_results", type=str, required=True,
                        help="Path to label_f1_results.tsv")
    parser.add_argument("--organism", type=str, required=True,
                        help="homo_sapiens or mus_musculus")
    parser.add_argument("--cutoff", type=float, default=0,
                        help="Confidence cutoff to filter data (default: 0)")
    parser.add_argument("--subsample_ref", type=int, required=True,
                        help="Subsample reference level to filter to")
    parser.add_argument("--outdir", type=str, default="forest_plots",
                        help="Output directory")
    return parser.parse_args()


def sanitize_filename(name, max_len=80):
    """Sanitize a string for use as a filename."""
    name = re.sub(r'[^\w\s-]', '', name)
    name = re.sub(r'\s+', '_', name.strip())
    return name[:max_len]


def get_reference_display(row):
    """Get short display name for a reference, preferring acronym."""
    acronym = row.get("reference_acronym", None)
    if pd.notna(acronym) and str(acronym).strip():
        return str(acronym).strip()
    ref = str(row.get("reference", ""))
    return ref if len(ref) <= 30 else ref[:27] + "..."


def support_to_size(support, s_min, s_max):
    """Map support value to marker size (area in points²)."""
    if s_max == s_min:
        return (SIZE_MIN + SIZE_MAX) / 2
    frac = (support - s_min) / (s_max - s_min)
    return SIZE_MIN + frac * (SIZE_MAX - SIZE_MIN)


def plot_study_forest(study_df, study, outdir):
    """Plot a keys (rows) × references (cols) grid of forest plots for one study."""
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

    if n_keys == 0 or n_refs == 0:
        return

    # Pre-compute cell type ordering per key level (ascending for bottom-to-top)
    key_labels_ordered = {}
    for key in available_keys:
        key_df = study_df[study_df["key"] == key]
        if key_df.empty:
            key_labels_ordered[key] = []
        else:
            label_means = (key_df.groupby("label")["f1_score"]
                           .mean().sort_values(ascending=True))
            key_labels_ordered[key] = label_means.index.tolist()

    max_n_labels = max(len(v) for v in key_labels_ordered.values())
    if max_n_labels == 0:
        return

    # Global support range for consistent dot sizing
    support_stats = (study_df.groupby(["key", "reference", "method", "label"])
                     ["support"].mean())
    s_min = support_stats.min()
    s_max = support_stats.max()

    # --- Figure sizing ---
    panel_height = max(4, max_n_labels * 0.45 + 2)
    panel_width = 8
    fig_height = panel_height * n_keys + 3
    fig_width = panel_width * n_refs + 2

    fig, axes = plt.subplots(
        n_keys, n_refs, figsize=(fig_width, fig_height),
        squeeze=False,
    )

    n_methods = len(methods)
    offsets = np.linspace(-0.15, 0.15, n_methods) if n_methods > 1 else [0.0]

    for key_idx, key in enumerate(available_keys):
        labels_ordered = key_labels_ordered[key]
        n_labels = len(labels_ordered)
        key_df = study_df[study_df["key"] == key]

        for ref_idx, reference in enumerate(references):
            ax = axes[key_idx, ref_idx]
            ref_key_df = key_df[key_df["reference"] == reference]

            if ref_key_df.empty or n_labels == 0:
                ax.set_visible(False)
                continue

            # Compute stats per (label, method)
            stats = (ref_key_df.groupby(["label", "method"])
                     .agg(
                         mean_f1=("f1_score", "mean"),
                         sem_f1=("f1_score", "sem"),
                         mean_support=("support", "mean"),
                     )
                     .reset_index())
            stats["sem_f1"] = stats["sem_f1"].fillna(0)

            for m_idx, method in enumerate(methods):
                method_stats = stats[stats["method"] == method]
                if method_stats.empty:
                    continue

                color = METHOD_COLORS.get(method, "#333333")

                for _, row in method_stats.iterrows():
                    label = row["label"]
                    if label not in labels_ordered:
                        continue

                    y_pos = labels_ordered.index(label) + offsets[m_idx]
                    size = support_to_size(row["mean_support"], s_min, s_max)

                    ax.errorbar(
                        row["mean_f1"], y_pos,
                        xerr=row["sem_f1"],
                        fmt="none",
                        ecolor=color, elinewidth=1.5, capsize=3, capthick=1,
                        alpha=0.7,
                    )
                    ax.scatter(
                        row["mean_f1"], y_pos,
                        s=size, c=color, edgecolors="white", linewidth=0.5,
                        zorder=3,
                    )

            ax.set_yticks(range(n_labels))
            if ref_idx == 0:
                ax.set_yticklabels(labels_ordered, fontsize=11)
            else:
                ax.set_yticklabels([])
            ax.set_ylim(-0.5, n_labels - 0.5)
            ax.set_xlim(-0.05, 1.05)
            ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

            if key_idx == 0:
                ax.set_title(ref_display[reference], fontsize=16, fontweight="bold")
            if ref_idx == 0:
                ax.set_ylabel(key.capitalize(), fontsize=16, fontweight="bold")
            if key_idx == n_keys - 1:
                ax.set_xlabel("F1 Score", fontsize=13)

    # --- Legend ---
    legend_handles = []

    # Method colors
    for method in methods:
        color = METHOD_COLORS.get(method, "#333333")
        display = METHOD_NAMES.get(method, method)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                   markeredgecolor="white", markersize=10, label=display)
        )

    # Support sizes — pick a few representative values
    support_ticks = [s_min, (s_min + s_max) / 2, s_max]
    for sv in support_ticks:
        sz = support_to_size(sv, s_min, s_max)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="white",
                   markersize=np.sqrt(sz),
                   label=f"support={sv:.2f}")
        )

    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=len(legend_handles), fontsize=12, frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(study, fontsize=20, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])

    # Save
    os.makedirs(outdir, exist_ok=True)
    study_safe = sanitize_filename(study)
    outpath = os.path.join(outdir, f"{study_safe}_f1_forest.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main():
    args = parse_args()
    set_pub_style()

    os.makedirs(args.outdir, exist_ok=True)

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

    # Generate one figure per study
    for study, group_df in df.groupby("study"):
        n_refs = group_df["reference"].nunique()
        n_keys = group_df["key"].nunique()
        print(f"Plotting: study={study} "
              f"({n_refs} references, {n_keys} keys, {len(group_df)} rows)")
        plot_study_forest(group_df, study, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
