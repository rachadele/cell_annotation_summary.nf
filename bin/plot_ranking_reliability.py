#!/usr/bin/env python3
"""
plot_ranking_reliability.py

Scatter plot of win_fraction vs mean_f1_across_studies from rankings_best.tsv.
Cell types in the upper-right are both high-performing and consistently best
across studies.  Faceted by key level.

Usage:
    python plot_ranking_reliability.py \
        --input rankings_best.tsv \
        --outdir ranking_reliability
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

import sys
sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import KEY_ORDER, set_pub_style


DEFAULT_INPUT = "rankings_best.tsv"

# Key level colors
KEY_COLORS = {
    "subclass": "#e41a1c",
    "class": "#377eb8",
    "family": "#4daf4a",
    "global": "#984ea3",
}

# Size range for n_studies encoding
SIZE_MIN = 30
SIZE_MAX = 250


def parse_args():
    parser = argparse.ArgumentParser(
        description="Win fraction vs F1 scatter plot"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to rankings_best.tsv")
    parser.add_argument("--outdir", default="ranking_reliability",
                        help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    df = pd.read_csv(args.input, sep="\t")
    print(f"Loaded {len(df)} rows from {args.input}")

    keys = [k for k in KEY_ORDER if k in df["key"].unique()]
    n_keys = len(keys)

    if n_keys == 0:
        print("No data. Exiting.")
        return

    # --- Faceted scatter: one panel per key ---
    n_cols = min(n_keys, 2)
    n_rows = (n_keys + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows),
                              squeeze=False)

    # Global n_studies range for sizing
    ns_min = df["n_studies"].min()
    ns_max = df["n_studies"].max()

    for idx, key in enumerate(keys):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        key_df = df[df["key"] == key].copy()
        color = KEY_COLORS.get(key, "#333333")

        # Compute sizes from n_studies
        if ns_max > ns_min:
            sizes = SIZE_MIN + (key_df["n_studies"] - ns_min) / (ns_max - ns_min) * (SIZE_MAX - SIZE_MIN)
        else:
            sizes = (SIZE_MIN + SIZE_MAX) / 2

        ax.scatter(
            key_df["mean_f1_across_studies"],
            key_df["win_fraction"],
            s=sizes, c=color, alpha=0.7,
            edgecolors="white", linewidth=0.5,
        )

        # Label points
        for _, row in key_df.iterrows():
            ax.annotate(
                row["label"], (row["mean_f1_across_studies"], row["win_fraction"]),
                fontsize=6, alpha=0.8,
                xytext=(4, 4), textcoords="offset points",
            )

        # Reference lines
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.4)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel("Mean F1 Across Studies")
        ax.set_ylabel("Win Fraction")
        ax.set_title(key.capitalize(), fontsize=14, fontweight="bold")

    # Hide unused axes
    for idx in range(n_keys, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].set_visible(False)

    # --- Size legend ---
    legend_handles = []
    for ns_val, lbl in [(ns_min, f"n={ns_min}"), (ns_max, f"n={ns_max}")]:
        if ns_max > ns_min:
            sz = SIZE_MIN + (ns_val - ns_min) / (ns_max - ns_min) * (SIZE_MAX - SIZE_MIN)
        else:
            sz = (SIZE_MIN + SIZE_MAX) / 2
        legend_handles.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markeredgecolor="white", markersize=np.sqrt(sz),
                   label=lbl)
        )
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=len(legend_handles), fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.03), title="N Studies",
    )

    fig.suptitle("Reliability vs Performance", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    outpath = os.path.join(args.outdir, "ranking_reliability.png")
    fig.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved {outpath}")
    print(f"\nDone. Figures in {args.outdir}/")


if __name__ == "__main__":
    main()
