#!/usr/bin/env python3
"""
Cross-species F1 heatmaps across studies.

Two-panel figure (Human / Mouse) showing per-cell-type mean F1 score
by study, illustrating that study of origin is a major driver of
annotation quality across species.

Rows: cell type labels (sorted by mean F1 across studies, ascending)
Columns: studies (sorted by mean F1 across labels, ascending)
Colour: mean F1 score (averaged over method, 0-1)

Usage:
    python plot_study_variance_combined.py \\
        --hs_label_results .../homo_sapiens_new/.../label_results.tsv \\
        --mm_label_results .../mus_musculus/.../label_results.tsv \\
        --outdir figures
"""

import argparse
import os
import sys

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import set_pub_style

_BASE = (
    "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01"
)
DEFAULT_HS = os.path.join(
    _BASE,
    "homo_sapiens_new/100/dataset_id/SCT/gap_false",
    "aggregated_results/files/label_results.tsv",
)
DEFAULT_MM = os.path.join(
    _BASE,
    "mus_musculus/100/dataset_id/SCT/gap_false",
    "aggregated_results/files/label_results.tsv",
)

CMAP = "YlOrRd"
NAN_COLOR = "#cccccc"


def load_data(path: str, key: str, cutoff: float) -> pd.DataFrame:
    """Load label_results filtered to key/cutoff, returning F1 scores."""
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["study", "label", "f1_score", "key", "cutoff"],
        na_values=["None", ""],
    )
    df = df[(df["key"] == key) & (df["cutoff"] == cutoff)].copy()
    df["f1_score"] = pd.to_numeric(df["f1_score"], errors="coerce")
    # Shorten study labels to first token (GEO accession or short name)
    df["study"] = df["study"].str.split().str[0]
    return df


def make_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to label×study; sort rows/cols by mean F1 ascending."""
    pivot = df.groupby(["label", "study"])["f1_score"].mean().unstack("study")
    row_order = pivot.mean(axis=1).sort_values().index.tolist()
    col_order = pivot.mean(axis=0).sort_values().index.tolist()
    return pivot.reindex(index=row_order, columns=col_order)


def draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str) -> None:
    """Draw a single F1 heatmap panel (no internal colorbar)."""
    nan_mask = pivot.isna()
    # Grey background for missing cells
    ax.pcolormesh(
        np.ones(pivot.shape),
        cmap=ListedColormap([NAN_COLOR]),
        vmin=0,
        vmax=1,
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap=CMAP,
        vmin=0,
        vmax=1,
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar=False,
        mask=nan_mask,
        yticklabels=True,
    )
    ax.set_title(title, fontweight="bold", loc="left", fontsize=18, pad=8)
    ax.set_xlabel("")
    ax.set_ylabel("Cell type", fontsize=16)
    ax.tick_params(axis="x", labelsize=14, rotation=40)
    ax.tick_params(axis="y", labelsize=14)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hs_label_results", default=DEFAULT_HS)
    p.add_argument("--mm_label_results", default=DEFAULT_MM)
    p.add_argument("--key",    default="subclass")
    p.add_argument("--cutoff", default=0.0, type=float)
    p.add_argument("--outdir", default="study_variance_combined")
    p.add_argument("--prefix", default="study_variance_combined")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    print("Loading data...")
    df_hs = load_data(args.hs_label_results, args.key, args.cutoff)
    df_mm = load_data(args.mm_label_results, args.key, args.cutoff)

    pivot_hs = make_pivot(df_hs)
    pivot_mm = make_pivot(df_mm)
    n_hs, w_hs = pivot_hs.shape
    n_mm, w_mm = pivot_mm.shape
    print(f"  Human: {n_hs} cell types × {w_hs} studies")
    print(f"  Mouse: {n_mm} cell types × {w_mm} studies")

    # Side-by-side landscape layout: width scales with study counts, height with cell types
    col_w = 0.45
    row_h = 0.25
    hs_w  = 5.5 + w_hs * col_w
    mm_w  = 5.5 + w_mm * col_w
    fig_h = max(4.0, max(n_hs, n_mm) * row_h + 3.5)

    fig = plt.figure(figsize=(hs_w + mm_w + 1.5, fig_h + 1.0))
    gs = GridSpec(
        1, 2,
        width_ratios=[hs_w, mm_w],
        wspace=0.55,
    )
    ax_hs = fig.add_subplot(gs[0])
    ax_mm = fig.add_subplot(gs[1])

    draw_heatmap(
        ax_hs, pivot_hs,
        f"Human — F1 by study ({args.key}, cutoff={args.cutoff})",
    )
    draw_heatmap(
        ax_mm, pivot_mm,
        f"Mouse — F1 by study ({args.key}, cutoff={args.cutoff})",
    )
    ax_hs.set_xlabel("Study", fontsize=16)
    ax_mm.set_xlabel("Study", fontsize=16)

    # Shared colorbar on the far right
    sm = cm.ScalarMappable(
        cmap=CMAP, norm=mcolors.Normalize(vmin=0, vmax=1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_hs, ax_mm], shrink=0.6, pad=0.02)
    cbar.set_label("Mean F1 score", fontsize=14)
    cbar.ax.tick_params(labelsize=13)

    out = os.path.join(args.outdir, f"{args.prefix}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
