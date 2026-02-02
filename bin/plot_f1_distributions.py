#!/usr/bin/env python3
"""
Plot distributions of F1 scores at each taxonomy level.
Highlights boundary pile-up at 0 and 1 to motivate ordinal beta regression.

Supports both sample-level macro F1 (weighted_f1_results.tsv) and
per-cell-type F1 (label_f1_results.tsv).
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import set_pub_style, KEY_ORDER, METHOD_COLORS, METHOD_NAMES


def parse_args():
    parser = argparse.ArgumentParser(description="Plot F1 distributions by taxonomy level")
    parser.add_argument("--weighted_f1_results",
                        default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus_tabulamuris/100/dataset_id/SCT/gap_false/aggregated_results/files/weighted_f1_results.tsv",
                        help="Path to weighted_f1_results.tsv")
    parser.add_argument("--label_f1_results",
                        default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mus_musculus_tabulamuris/100/dataset_id/SCT/gap_false/aggregated_results/files/label_f1_results.tsv",
                        help="Path to label_f1_results.tsv")
    parser.add_argument("--outdir", default="f1_distributions", help="Output directory")
    args, _ = parser.parse_known_args()
    return args


def _boundary_stats(series):
    """Return counts and percentages for 0/1 boundaries."""
    n_total = len(series)
    if n_total == 0:
        return 0, 0, 0
    n_zero = (series <= 1e-6).sum()
    n_one = (series >= 1 - 1e-6).sum()
    return n_zero, n_one, n_total


def plot_distributions(df, outdir):
    """Plot macro F1 histograms (sample-level) split by method and key."""
    set_pub_style()
    os.makedirs(outdir, exist_ok=True)

    keys = [k for k in KEY_ORDER if k in df['key'].unique()]
    methods = sorted(df['method'].unique())

    # --- Histograms split by method ---
    fig, axes = plt.subplots(2, len(keys), figsize=(5 * len(keys), 8), sharey=False)
    if len(keys) == 1:
        axes = axes.reshape(-1, 1)

    for row, method in enumerate(methods):
        for col, key in enumerate(keys):
            ax = axes[row, col]
            subset = df[(df['key'] == key) & (df['method'] == method)]['macro_f1']
            color = METHOD_COLORS.get(method, 'gray')
            ax.hist(subset, bins=50, edgecolor='black', linewidth=0.5, color=color, alpha=0.8)
            ax.axvline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(1, color='red', linestyle='--', alpha=0.5, linewidth=1)

            n_zero, n_one, n_total = _boundary_stats(subset)
            ax.text(0.02, 0.95, f'= 0: {n_zero} ({100*n_zero/n_total:.1f}%)',
                    transform=ax.transAxes, va='top', fontsize=10, color='red')
            ax.text(0.02, 0.85, f'= 1: {n_one} ({100*n_one/n_total:.1f}%)',
                    transform=ax.transAxes, va='top', fontsize=10, color='red')

            if row == 0:
                ax.set_title(key)
            if col == 0:
                ax.set_ylabel(METHOD_NAMES.get(method, method))
            if row == len(methods) - 1:
                ax.set_xlabel('Macro F1')

    fig.suptitle('Macro F1 Distribution by Method and Taxonomy Level', fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'macro_f1_histograms_by_method.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- Summary stats table ---
    summary_rows = []
    for key in keys:
        for method in methods:
            subset = df[(df['key'] == key) & (df['method'] == method)]['macro_f1']
            n_zero, n_one, n_total = _boundary_stats(subset)
            summary_rows.append({
                'key': key,
                'method': method,
                'n': n_total,
                'mean': subset.mean(),
                'median': subset.median(),
                'std': subset.std(),
                'n_zero': n_zero,
                'pct_zero': 100 * n_zero / n_total if n_total else 0,
                'n_one': n_one,
                'pct_one': 100 * n_one / n_total if n_total else 0,
                'n_boundary': n_zero + n_one,
                'pct_boundary': 100 * (n_zero + n_one) / n_total if n_total else 0,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(outdir, 'macro_f1_distribution_summary.tsv'), sep='\t', index=False)
    print(summary_df.to_string(index=False))


def plot_label_distributions(df, outdir):
    """Plot per-label F1 histograms, one figure per taxonomy level, faceted by label."""
    set_pub_style()
    os.makedirs(outdir, exist_ok=True)

    keys = [k for k in KEY_ORDER if k in df['key'].unique()]
    methods = sorted(df['method'].unique())

    for key in keys:
        df_key = df[df['key'] == key]
        labels = sorted(df_key['label'].unique())
        n_labels = len(labels)
        if n_labels == 0:
            continue

        ncols = min(5, n_labels)
        nrows = math.ceil(n_labels / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=False)
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        for idx, label in enumerate(labels):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            for method in methods:
                subset = df_key[(df_key['label'] == label) & (df_key['method'] == method)]['f1_score']
                color = METHOD_COLORS.get(method, 'gray')
                ax.hist(subset, bins=30, edgecolor='black', linewidth=0.3,
                        color=color, alpha=0.6, label=METHOD_NAMES.get(method, method))

            ax.axvline(0, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.axvline(1, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
            ax.set_title(label, fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            if row == nrows - 1:
                ax.set_xlabel('F1')

        # hide unused axes
        for idx in range(n_labels, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        # single legend for the figure
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc='upper right', fontsize=10)

        fig.suptitle(f'Per-Label F1 Distribution — {key}', fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'label_f1_histograms_{key}.png'), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # --- Summary stats table per label ---
    summary_rows = []
    for key in keys:
        df_key = df[df['key'] == key]
        for label in sorted(df_key['label'].unique()):
            for method in methods:
                subset = df_key[(df_key['label'] == label) & (df_key['method'] == method)]['f1_score']
                n_zero, n_one, n_total = _boundary_stats(subset)
                if n_total == 0:
                    continue
                summary_rows.append({
                    'key': key,
                    'label': label,
                    'method': method,
                    'n': n_total,
                    'mean': subset.mean(),
                    'median': subset.median(),
                    'std': subset.std(),
                    'n_zero': n_zero,
                    'pct_zero': 100 * n_zero / n_total,
                    'n_one': n_one,
                    'pct_one': 100 * n_one / n_total,
                    'n_boundary': n_zero + n_one,
                    'pct_boundary': 100 * (n_zero + n_one) / n_total,
                })
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(outdir, 'label_f1_distribution_summary.tsv'), sep='\t', index=False)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()

    if args.weighted_f1_results:
        df = pd.read_csv(args.weighted_f1_results, sep='\t')
        plot_distributions(df, args.outdir)

    if args.label_f1_results:
        df_label = pd.read_csv(args.label_f1_results, sep='\t')
        plot_label_distributions(df_label, args.outdir)
