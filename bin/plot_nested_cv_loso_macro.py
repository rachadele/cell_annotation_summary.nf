#!/usr/bin/env python3
"""
Plot nested-LOSO macro-F1 CV results.

Reads `outer_fold_results.tsv`, `unbiased_summary.tsv`, and
`inner_selection_log.tsv` from one or more `nested_cv_macro/` directories
and produces:

1. Per-fold outer score dot plot (2x2 facet over keys), with the full-data
   mean of the full-data-selected config drawn as a horizontal line.
   Optimism = (line) − (dot mean). One figure per organism, plus an
   overlay if multiple organisms are passed.
2. Selected-config grid heatmap per key: rows = held-out study,
   columns = config axes, cell = level picked.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use the project's shared style.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_utils import set_pub_style, KEY_ORDER, save_figure

CONFIG_COLS = ["method", "reference", "cutoff", "subsample_ref"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cv_dir", action="append", required=True,
                   help="Path to a nested_cv_macro directory. Pass multiple "
                        "times to overlay organisms (e.g. --cv_dir mouse/... "
                        "--cv_dir human/...).")
    p.add_argument("--label", action="append", default=None,
                   help="Human-readable label per --cv_dir (same order). "
                        "Default: derive from path.")
    p.add_argument("--outdir", required=True,
                   help="Where to write the figures.")
    return p.parse_args()


def load_run(cv_dir: str):
    """Read per-key outputs from one nested_cv_macro/ directory.
    Returns (outer_long, summary_long) — both with an added `key` column.
    """
    outer_rows, summary_rows = [], []
    for key in KEY_ORDER:
        key_dir = os.path.join(cv_dir, key)
        outer_p = os.path.join(key_dir, "outer_fold_results.tsv")
        sum_p = os.path.join(key_dir, "unbiased_summary.tsv")
        if not (os.path.exists(outer_p) and os.path.exists(sum_p)):
            continue
        outer_rows.append(pd.read_csv(outer_p, sep="\t").assign(key=key))
        summary_rows.append(pd.read_csv(sum_p, sep="\t"))
    if not outer_rows:
        raise FileNotFoundError(f"No per-key outputs found under {cv_dir}")
    return pd.concat(outer_rows, ignore_index=True), pd.concat(summary_rows, ignore_index=True)


def plot_per_fold_facets(label: str, outer_df: pd.DataFrame,
                          summary_df: pd.DataFrame, outpath: str):
    """2x2 small-multiples of per-fold outer score per key for ONE organism.

    Panels: subclass / class / family / global.
    Per panel: dot per held-out study, full-data mean (dashed) and CV mean
    (dotted) horizontal lines, selection-bias gap annotated.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=False)
    axes = axes.ravel()

    for ax, key in zip(axes, KEY_ORDER):
        sub = outer_df[outer_df["key"] == key].sort_values("held_out_study").reset_index(drop=True)
        srow = summary_df[summary_df["key"] == key]
        if sub.empty or srow.empty:
            ax.set_visible(False)
            continue

        x = np.arange(len(sub))
        ax.scatter(x, sub["outer_score"], color="#1f77b4", s=180,
                   edgecolor="black", linewidth=0.8, zorder=3,
                   label="per-fold outer score")

        full_mean = float(srow["full_data_mean"].iloc[0])
        cv_mean = float(sub["outer_score"].mean())
        bias = full_mean - cv_mean

        ax.axhline(full_mean, color="#d62728", linestyle="--", linewidth=2.5,
                   label=f"full-data mean = {full_mean:.3f}")
        ax.axhline(cv_mean, color="#2ca02c", linestyle=":", linewidth=2.5,
                   label=f"CV mean = {cv_mean:.3f}")

        # Bias annotation in the top corner
        ax.text(0.98, 0.04, f"selection bias = {bias:+.3f}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=15,
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.85,
                          boxstyle="round,pad=0.3"))

        ax.set_xticks(x)
        ax.set_xticklabels([s if len(s) <= 18 else s[:15] + "..." for s in sub["held_out_study"]],
                           rotation=40, ha="right", fontsize=14)
        ax.set_title(key)
        ax.set_ylabel("macro F1")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="lower left", fontsize=13, framealpha=0.9)

    fig.suptitle(f"Nested LOSO outer-fold macro F1 — {label}", y=1.0)
    fig.tight_layout()
    save_figure(fig, outpath, formats=["png"])
    plt.close(fig)


def plot_summary_bars(runs, outpath: str):
    """Compact summary across organisms: full-data mean vs CV mean,
    grouped bars per (key, organism). Selection bias = visible gap."""
    fig, ax = plt.subplots(figsize=(13, 7))
    n_runs = len(runs)
    width = 0.8 / (n_runs * 2)
    palette = sns.color_palette("Set1", n_colors=n_runs)
    x_base = np.arange(len(KEY_ORDER))

    for i, (label, _, summary_df) in enumerate(runs):
        rows = summary_df.set_index("key").reindex(KEY_ORDER)
        full = rows["full_data_mean"].values
        cv = rows["outer_mean"].values
        offset_full = (2 * i - (n_runs - 0.5)) * width
        offset_cv = offset_full + width
        ax.bar(x_base + offset_full, full, width, color=palette[i],
               alpha=0.5, edgecolor="black", label=f"{label} full-data")
        ax.bar(x_base + offset_cv, cv, width, color=palette[i],
               alpha=1.0, edgecolor="black", label=f"{label} CV mean")

    ax.set_xticks(x_base)
    ax.set_xticklabels(KEY_ORDER)
    ax.set_ylabel("macro F1")
    ax.set_title("Selection bias: full-data vs CV mean (gap = bias)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=15, ncol=n_runs * 2, frameon=False)
    fig.tight_layout()
    save_figure(fig, outpath, formats=["png"])
    plt.close(fig)


def plot_selected_config_grid(label: str, outer_df: pd.DataFrame, outpath: str):
    """Heatmap-ish grid of (held-out study) × (config axis) showing the
    level picked by inner CV for each fold. One panel per key."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    axes = axes.ravel()
    for ax, key in zip(axes, KEY_ORDER):
        sub = outer_df[outer_df["key"] == key].copy()
        if sub.empty:
            ax.set_visible(False); continue
        sub = sub.sort_values("held_out_study").reset_index(drop=True)
        # Build a string display of each cell so we can use a categorical heatmap.
        # Encode each column independently with a color per level.
        ax.set_xticks(np.arange(len(CONFIG_COLS)))
        ax.set_xticklabels(CONFIG_COLS, rotation=30, ha="right", fontsize=14)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["held_out_study"], fontsize=12)
        ax.set_xlim(-0.5, len(CONFIG_COLS) - 0.5)
        ax.set_ylim(len(sub) - 0.5, -0.5)
        ax.set_title(key)
        # For each column, map levels to colours
        for ci, col in enumerate(CONFIG_COLS):
            levels = sorted(sub[col].astype(str).unique())
            colors = sns.color_palette("Set2", n_colors=max(3, len(levels)))
            cmap = {lv: colors[i] for i, lv in enumerate(levels)}
            for ri, lv in enumerate(sub[col].astype(str)):
                ax.add_patch(plt.Rectangle((ci - 0.45, ri - 0.45), 0.9, 0.9,
                                            color=cmap[lv], ec="black", lw=0.5))
                # Truncate long reference names for label
                disp = lv if len(lv) <= 14 else lv[:11] + "..."
                ax.text(ci, ri, disp, ha="center", va="center", fontsize=10)

    fig.suptitle(f"Selected config per outer fold — {label}", y=1.0)
    fig.tight_layout()
    save_figure(fig, outpath, formats=["png"])
    plt.close(fig)


def main():
    args = parse_args()
    set_pub_style()
    os.makedirs(args.outdir, exist_ok=True)

    labels = args.label or [os.path.basename(os.path.dirname(d.rstrip("/")))
                            for d in args.cv_dir]
    if len(labels) != len(args.cv_dir):
        sys.exit("--label count must match --cv_dir count")

    runs = []
    for label, cv_dir in zip(labels, args.cv_dir):
        outer_df, summary_df = load_run(cv_dir)
        runs.append((label, outer_df, summary_df))
        plot_selected_config_grid(label, outer_df,
            os.path.join(args.outdir, f"selected_config_grid_{label}"))
        plot_per_fold_facets(label, outer_df, summary_df,
            os.path.join(args.outdir, f"per_fold_outer_score_{label}"))

    if len(runs) > 1:
        plot_summary_bars(runs, os.path.join(args.outdir, "summary_bias_comparison"))

    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
