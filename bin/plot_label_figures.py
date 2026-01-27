#!/usr/bin/env python3
"""
Plot label-level model results across all cell types.

Reads emmeans summary TSVs from label model output directories and creates
a forest plot showing reference × method estimated marginal means per cell type.

Usage:
    python plot_label_figures.py \
        --label_models_dir path/to/label_models \
        --outdir figures
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_utils import (
    set_pub_style,
    METHOD_COLORS,
    METHOD_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot label-level model results")
    parser.add_argument("--label_models_dir", type=str, required=True,
                        help="Directory containing label model outputs (one subdir per label)")
    parser.add_argument("--outdir", type=str, default="figures",
                        help="Output directory for figures")
    return parser.parse_args()


def load_label_emmeans(label_models_dir, contrast_name):
    """
    Load a specific *_emmeans_summary.tsv across all labels.

    Returns a DataFrame with a 'label' column, or None if nothing loaded.
    """
    dfs = []
    label_dirs = sorted(glob.glob(os.path.join(label_models_dir, "*")))
    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)

        formula_dirs = glob.glob(os.path.join(label_dir, "f1_score_*"))
        if not formula_dirs:
            continue
        fpath = os.path.join(formula_dirs[0], "files", f"{contrast_name}_emmeans_summary.tsv")
        if not os.path.isfile(fpath):
            continue

        try:
            df = pd.read_csv(fpath, sep="\t")
            if "note" in df.columns or df.empty:
                continue
            df["label"] = label.replace("_", " ")
            dfs.append(df)
        except Exception:
            continue

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def plot_reference_method_forest(df, outdir):
    """
    Forest plot: one row per label, points for each reference × method combination.
    Labels on y-axis sorted by mean F1. Points colored by method, dodged by reference.
    Each reference gets its own panel column.
    """
    references = sorted(df["reference"].unique())
    n_refs = len(references)

    # Truncate long reference names for display
    def shorten(name, maxlen=35):
        return (name[:maxlen] + "…") if len(name) > maxlen else name

    # Sort labels by overall mean F1
    label_order = (df.groupby("label")["response"]
                     .mean()
                     .sort_values(ascending=True)
                     .index.tolist())
    n_labels = len(label_order)

    methods = sorted(df["method"].unique())
    n_methods = len(methods)
    offsets = np.linspace(-0.15, 0.15, n_methods)

    # One column per reference
    fig_width = 3.5 * n_refs + 1.5
    fig_height = max(5, n_labels * 0.45 + 1.5)
    fig, axes = plt.subplots(1, n_refs, figsize=(fig_width, fig_height),
                              sharey=True)
    if n_refs == 1:
        axes = [axes]

    for r_idx, ref in enumerate(references):
        ax = axes[r_idx]
        rdf = df[df["reference"] == ref]

        for m_idx, method in enumerate(methods):
            mdf = rdf[rdf["method"] == method]
            color = METHOD_COLORS.get(method, "#333333")
            display_name = METHOD_NAMES.get(method, method)

            for label in label_order:
                row = mdf[mdf["label"] == label]
                if row.empty:
                    continue
                row = row.iloc[0]
                y = label_order.index(label) + offsets[m_idx]
                ax.hlines(y, row["asymp.LCL"], row["asymp.UCL"],
                          colors=color, linewidth=1.5, zorder=1)
                ax.scatter(row["response"], y, s=30, c=[color], zorder=2,
                           edgecolors="white", linewidth=0.3)

        ax.set_title(shorten(ref), fontsize=9)
        ax.set_xlabel("Est. F1", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(-0.05, 1.05)

    # y-axis labels on leftmost panel only
    axes[0].set_yticks(range(n_labels))
    axes[0].set_yticklabels(label_order, fontsize=10)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=METHOD_COLORS.get(m, "#333"),
               markerfacecolor=METHOD_COLORS.get(m, "#333"), markersize=6,
               linewidth=1.5, label=METHOD_NAMES.get(m, m))
        for m in methods
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=n_methods, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.savefig(os.path.join(outdir, "label_reference_method_emmeans.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_pub_style()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_label_emmeans(args.label_models_dir, "reference_method")
    if df is None:
        print("No valid reference_method emmeans summary files found.")
        return

    print(f"Loaded reference_method: {df['label'].nunique()} labels, "
          f"{df['reference'].nunique()} references, {len(df)} rows")

    plot_reference_method_forest(df, args.outdir)
    print(f"Figure saved to {args.outdir}")


if __name__ == "__main__":
    main()
