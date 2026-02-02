#!/usr/bin/env python3
"""
Plot label-level model results across all cell types.

Reads reference_method emmeans summary TSVs and creates a forest plot
showing estimated F1 per cell type label, faceted by reference, colored by method.
Dot size represents median cell count (support) per label.

Usage:
    python plot_label_figures.py \
        --emmeans_files path/to/*.tsv \
        --label_f1_results path/to/label_f1_results.tsv \
        --outdir figures

    OR (directory mode):

    python plot_label_figures.py \
        --label_models_dir path/to/label_models \
        --label_f1_results path/to/label_f1_results.tsv \
        --outdir figures
"""

import argparse
import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_utils import (
    set_pub_style,
    METHOD_COLORS,
    METHOD_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot label-level model results")
    parser.add_argument("--emmeans_files", type=str, nargs="+", default=None,
                        help="Paths to reference_method_emmeans_summary.tsv files")
    parser.add_argument("--label_models_dir", type=str, default=None,
                        help="Directory containing label model outputs (one subdir per label)")
    parser.add_argument("--label_f1_results", type=str, default=None,
                        help="Path to aggregated label F1 results TSV (for support/cell counts)")
    # Removed --outdir argument; output will be saved to current directory
    return parser.parse_args()


def extract_label_from_path(fpath):
    """
    Extract label name from file path.
    Expected patterns:
      .../LabelName/formula_dir/files/reference_method_emmeans_summary.tsv
    """
    parts = fpath.split(os.sep)
    # Walk up from file: files/ -> formula_dir/ -> label/
    for i, part in enumerate(parts):
        if part == "files" and i >= 2:
            return parts[i - 2].replace("_", " ")
    if len(parts) >= 4:
        return parts[-4].replace("_", " ")
    return os.path.basename(os.path.dirname(fpath)).replace("_", " ")


def load_from_files(emmeans_files):
    """Load reference_method emmeans from explicit file paths."""
    dfs = []
    for fpath in emmeans_files:
        try:
            df = pd.read_csv(fpath, sep="\t")
            if "note" in df.columns or df.empty:
                continue
            if "reference" not in df.columns:
                continue
            label = extract_label_from_path(os.path.abspath(fpath))
            df["label"] = label
            dfs.append(df)
        except Exception:
            continue
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return None


def load_from_dir(label_models_dir):
    """Load reference_method emmeans by scanning label model directory tree."""
    dfs = []
    label_dirs = sorted(glob.glob(os.path.join(label_models_dir, "*")))
    for label_dir in label_dirs:
        if not os.path.isdir(label_dir):
            continue
        label = os.path.basename(label_dir)

        formula_dirs = glob.glob(os.path.join(label_dir, "f1_score_*"))
        if not formula_dirs:
            continue
        fpath = os.path.join(formula_dirs[0], "files", "reference_method_emmeans_summary.tsv")
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


def load_support(label_f1_results_path):
    """
    Load median support (cell proportion) per label from the raw label F1 results.
    Returns a dict: label -> median support value.
    """
    if label_f1_results_path is None or not os.path.isfile(label_f1_results_path):
        return {}
    try:
        raw = pd.read_csv(label_f1_results_path, sep="\t")
        if "support" not in raw.columns or "label" not in raw.columns:
            return {}
        support = raw.groupby("label")["support"].sum().to_dict()
        # Normalize keys: replace _ with space to match emmeans labels
        return {k.replace("_", " "): v for k, v in support.items()}
    except Exception:
        return {}


def plot_reference_method_forest(df, outdir, support_map=None):
    """
    Forest plot: one row per label, faceted by reference.
    Points colored by method (scVI / Seurat) with CI bars.
    Dot size proportional to median support (cell count) if available.
    """
    references = sorted(df["reference"].unique())
    n_refs = len(references)

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

    # Compute dot sizes from support
    if support_map:
        support_vals = [support_map.get(label, 0) for label in label_order]
        max_support = max(support_vals) if max(support_vals) > 0 else 1
        # Scale: min 15, max 120
        size_map = {
            label: 15 + 105 * (support_map.get(label, 0) / max_support)
            for label in label_order
        }
    else:
        size_map = {label: 30 for label in label_order}

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

            for label in label_order:
                row = mdf[mdf["label"] == label]
                if row.empty:
                    continue
                row = row.iloc[0]
                y = label_order.index(label) + offsets[m_idx]
                ax.hlines(y, row["asymp.LCL"], row["asymp.UCL"],
                          colors=color, linewidth=1.5, zorder=1)
                ax.scatter(row["response"], y, s=size_map[label], c=[color],
                           zorder=2, edgecolors="white", linewidth=0.3)

        ax.set_title(shorten(ref), fontsize=9)
        ax.set_xlabel("Est. F1", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlim(-0.05, 1.05)

    # Y-axis: label names with support annotation
    ytick_labels = []
    for label in label_order:
        if support_map and label in support_map:
            ytick_labels.append(f"{label}  (n={support_map[label]:,.0f})")
        else:
            ytick_labels.append(label)

    axes[0].set_yticks(range(n_labels))
    axes[0].set_yticklabels(ytick_labels, fontsize=10)

    # Legend: method colors + size guide
    legend_elements = [
        Line2D([0], [0], marker="o", color=METHOD_COLORS.get(m, "#333"),
               markerfacecolor=METHOD_COLORS.get(m, "#333"), markersize=6,
               linewidth=1.5, label=METHOD_NAMES.get(m, m))
        for m in methods
    ]

    if support_map:
        legend_elements.append(
            Line2D([0], [0], marker="o", color="gray", markerfacecolor="gray",
                   markersize=4, linewidth=0, label="dot size = total cells")
        )

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.savefig("label_reference_method_emmeans.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_pub_style()
    # Output directory creation removed; saving to current directory

    if args.emmeans_files:
        df = load_from_files(args.emmeans_files)
    elif args.label_models_dir:
        df = load_from_dir(args.label_models_dir)
    else:
        print("Error: provide either --emmeans_files or --label_models_dir")
        return

    if df is None:
        print("No valid reference_method emmeans summary files found.")
        return

    support_map = load_support(args.label_f1_results)

    print(f"Loaded: {df['label'].nunique()} labels, "
          f"{df['reference'].nunique()} references, {len(df)} rows")
    if support_map:
        print(f"Support data: {len(support_map)} labels")

    plot_reference_method_forest(df, None, support_map)
    print("Figure saved to ./label_reference_method_emmeans.png")


if __name__ == "__main__":
    main()
