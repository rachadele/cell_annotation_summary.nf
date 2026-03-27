#!/usr/bin/env python3
"""Plot macro F1 distributions by method and taxonomy level."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import set_pub_style, KEY_ORDER, METHOD_COLORS, METHOD_NAMES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_results",
                        default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus_tabulamuris/100/dataset_id/SCT/gap_false/aggregated_results/files/sample_results.tsv")
    parser.add_argument("--label_results",
                        default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus_tabulamuris/100/dataset_id/SCT/gap_false/aggregated_results/files/label_results.tsv")
    parser.add_argument("--outdir", default="f1_distributions")
    args, _ = parser.parse_known_args()
    return args


def plot_f1_by_method_and_level(df, f1_col, title, outpath):
    set_pub_style()
    keys = [k for k in KEY_ORDER if k in df["key"].unique()]
    methods = sorted(df["method"].unique())
    n_methods = len(methods)

    fig, axes = plt.subplots(n_methods, len(keys),
                             figsize=(5 * len(keys), 4 * n_methods),
                             sharey=False)
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    if len(keys) == 1:
        axes = axes.reshape(-1, 1)

    for row, method in enumerate(methods):
        for col, key in enumerate(keys):
            ax = axes[row, col]
            vals = df[(df["key"] == key) & (df["method"] == method)][f1_col]
            ax.hist(vals, bins=50, color=METHOD_COLORS.get(method, "gray"),
                    edgecolor="black", linewidth=0.5, alpha=0.8)
            ax.axvline(0, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.axvline(1, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_xlim(-0.05, 1.05)
            if row == 0:
                ax.set_title(key)
            if col == 0:
                ax.set_ylabel(METHOD_NAMES.get(method, method))
            if row == n_methods - 1:
                ax.set_xlabel("F1")

    fig.suptitle(title, fontsize=16, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.sample_results, sep="\t")
    plot_f1_by_method_and_level(
        df, "macro_f1",
        "Macro F1 Distribution by Method and Taxonomy Level",
        os.path.join(args.outdir, "macro_f1_by_method.png"),
    )

    df_label = pd.read_csv(args.label_results, sep="\t")
    plot_f1_by_method_and_level(
        df_label, "f1_score",
        "Per-Label F1 Distribution by Method and Taxonomy Level",
        os.path.join(args.outdir, "label_f1_by_method.png"),
    )
