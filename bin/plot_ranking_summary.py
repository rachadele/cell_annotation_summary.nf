#!/usr/bin/env python3
"""
plot_ranking_summary.py

Annotated horizontal dot plot showing the recommended parameter combo for each
cell type, with F1 performance and reliability metrics.  One panel per taxonomy
level (key).

Usage:
    python plot_ranking_summary.py \
        --input rankings_best.tsv \
        --outdir ranking_summary
"""

import argparse
import os
import string

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import METHOD_COLORS, METHOD_NAMES, KEY_ORDER, set_pub_style


DEFAULT_INPUT = "rankings_best.tsv"

# Dot size range for mean_support encoding
SIZE_MIN = 30
SIZE_MAX = 300


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotated dot plot of best parameter combo per cell type"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to rankings_best.tsv")
    parser.add_argument("--outdir", default="ranking_summary",
                        help="Output directory")
    return parser.parse_args()


def build_ref_short_names(references):
    """Assign single-letter codes (A, B, C, ...) to references, sorted."""
    refs_sorted = sorted(references)
    return {ref: string.ascii_uppercase[i] for i, ref in enumerate(refs_sorted)}


def support_to_size(support, s_min, s_max):
    """Map support value to marker area (points^2)."""
    if s_max == s_min:
        return (SIZE_MIN + SIZE_MAX) / 2
    frac = np.clip((support - s_min) / (s_max - s_min), 0, 1)
    return SIZE_MIN + frac * (SIZE_MAX - SIZE_MIN)


def plot_key_panel(key, key_df, ref_short, ref_palette, sub_palette, outdir,
                   s_min, s_max):
    """Create and save the annotated dot plot for one taxonomy level."""

    # Sort by mean_f1 descending (highest at top)
    key_df = key_df.sort_values("mean_f1_across_studies", ascending=True).reset_index(drop=True)
    n = len(key_df)

    if n == 0:
        return

    # --- Figure layout using gridspec ---
    # Columns: ref_strip | method_strip | subsample_strip | dot_plot
    fig_height = max(4.0, n * 0.45 + 2.5)
    fig_width = 12

    fig, axes = plt.subplots(
        1, 4, figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [0.02, 0.02, 0.02, 1], "wspace": 0.05},
    )
    ax_ref, ax_method, ax_sub, ax = axes

    y_positions = np.arange(n)

    # --- Annotation strip axes ---
    for strip_ax, col, palette_func in [
        (ax_ref, "reference", lambda row: ref_palette.get(row["reference"], "#cccccc")),
        (ax_method, "method", lambda row: METHOD_COLORS.get(row["method"], "#333333")),
        (ax_sub, "subsample_ref", lambda row: sub_palette.get(row["subsample_ref"], "#cccccc")),
    ]:
        for i, (_, row) in enumerate(key_df.iterrows()):
            strip_ax.barh(i, 1, color=palette_func(row), edgecolor="none")
        strip_ax.set_ylim(-0.5, n - 0.5)
        strip_ax.set_xlim(0, 1)
        strip_ax.set_xticks([])
        strip_ax.set_yticks([])
        strip_ax.spines["top"].set_visible(False)
        strip_ax.spines["right"].set_visible(False)
        strip_ax.spines["bottom"].set_visible(False)
        strip_ax.spines["left"].set_visible(False)

    # Strip labels at bottom
    ax_ref.set_xlabel("Ref", fontsize=7, labelpad=2)
    ax_method.set_xlabel("Meth", fontsize=7, labelpad=2)
    ax_sub.set_xlabel("Sub", fontsize=7, labelpad=2)

    # Cell type labels on the leftmost strip
    ax_ref.set_yticks(y_positions)
    ax_ref.set_yticklabels(key_df["label"], fontsize=10)

    # --- Dots with error bars on main axes ---
    for i, (_, row) in enumerate(key_df.iterrows()):
        mean_f1 = row["mean_f1_across_studies"]
        std_f1 = row["std_f1_across_studies"] if pd.notna(row["std_f1_across_studies"]) else 0
        size = support_to_size(row["mean_support"], s_min, s_max)
        color = METHOD_COLORS.get(row["method"], "#333333")

        ax.errorbar(
            mean_f1, i, xerr=std_f1,
            fmt="none", ecolor=color, elinewidth=1.5, capsize=3, capthick=1,
            alpha=0.7, zorder=2,
        )
        ax.scatter(
            mean_f1, i, s=size, c=color,
            edgecolors="white", linewidth=0.5, zorder=3,
        )

    # --- Right-side text annotations ---
    for i, (_, row) in enumerate(key_df.iterrows()):
        n_wins = int(row["n_wins"]) if pd.notna(row["n_wins"]) else 0
        n_studies = int(row["n_studies"]) if pd.notna(row["n_studies"]) else 0
        win_frac = f"{n_wins}/{n_studies}"
        ax.annotate(
            f"  {win_frac}  (n={n_studies})",
            xy=(1.02, i), xycoords=("axes fraction", "data"),
            fontsize=8, va="center", ha="left", clip_on=False,
        )

    # Column headers for right-side text
    ax.annotate(
        "  win    evidence",
        xy=(1.02, n - 0.5 + 0.5), xycoords=("axes fraction", "data"),
        fontsize=8, va="bottom", ha="left", fontweight="bold", clip_on=False,
    )

    # --- Main axes formatting ---
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])  # labels are on the strip axes
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.set_xlabel("Mean F1 Across Studies")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(f"Best Parameter Combo — {key.capitalize()}", fontsize=14,
                 fontweight="bold")

    # --- Legend ---
    legend_handles = []

    # Reference legend
    legend_handles.append(mpatches.Patch(color="none", label="Reference"))
    for ref in sorted(ref_short.keys()):
        letter = ref_short[ref]
        color = ref_palette[ref]
        legend_handles.append(mpatches.Patch(color=color, label=f"  {letter}: {ref}"))

    # Method legend
    legend_handles.append(mpatches.Patch(color="none", label=""))
    legend_handles.append(mpatches.Patch(color="none", label="Method"))
    for method in sorted(METHOD_COLORS.keys()):
        display = METHOD_NAMES.get(method, method)
        legend_handles.append(mpatches.Patch(color=METHOD_COLORS[method], label=f"  {display}"))

    # Subsample legend
    legend_handles.append(mpatches.Patch(color="none", label=""))
    legend_handles.append(mpatches.Patch(color="none", label="Subsample Ref"))
    for sub in sorted(sub_palette.keys()):
        legend_handles.append(mpatches.Patch(color=sub_palette[sub], label=f"  {sub}"))

    # Support size legend
    legend_handles.append(Line2D([], [], color="none", label=""))
    legend_handles.append(Line2D([], [], color="none", label="Support (dot size)"))
    for frac, lbl in [(0, "Low"), (0.5, "Mid"), (1.0, "High")]:
        sz = SIZE_MIN + frac * (SIZE_MAX - SIZE_MIN)
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="white", markersize=np.sqrt(sz),
                   label=f"  {lbl}")
        )

    fig.legend(
        handles=legend_handles, loc="center left",
        bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False,
        handlelength=1.2, handleheight=1.0,
    )

    outpath = os.path.join(outdir, f"ranking_summary_{key}.png")
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {outpath} ({n} cell types)")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    df = pd.read_csv(args.input, sep="\t")
    print(f"Loaded {len(df)} rows from {args.input}")

    # --- Global palettes ---
    references = sorted(df["reference"].unique())
    ref_short = build_ref_short_names(references)
    ref_palette = dict(zip(references, sns.color_palette("Set2", len(references))))

    subsample_refs = sorted(df["subsample_ref"].unique())
    sub_palette = dict(zip(subsample_refs, sns.color_palette("YlOrRd_d", len(subsample_refs))))

    # Global support range
    s_min = df["mean_support"].min()
    s_max = df["mean_support"].max()

    # --- One panel per key level ---
    keys = [k for k in KEY_ORDER if k in df["key"].unique()]
    for key in keys:
        print(f"\n{key}:")
        key_df = df[df["key"] == key].copy()
        plot_key_panel(key, key_df, ref_short, ref_palette, sub_palette,
                       args.outdir, s_min, s_max)

    print(f"\nDone. Figures in {args.outdir}/")


if __name__ == "__main__":
    main()
