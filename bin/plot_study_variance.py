#!/usr/bin/env python3
"""
Cell-type F1, precision, and recall heatmaps across studies.

Shows that every cell type's performance varies substantially across studies,
illustrating that study of origin is a major driver of annotation quality.
Produces a three-panel figure (F1 / Precision / Recall) and an extended
summary TSV that includes per-metric statistics.

Rows: cell type labels (sorted by mean F1 across studies, ascending)
Columns: studies (sorted by mean F1 across labels, ascending)
Colour: mean metric value (averaged over method, 0–1)

Usage:
    python plot_study_variance.py \\
        --label_results 2024-07-01/mus_musculus/.../label_results.tsv \\
        --organism mus_musculus \\
        --outdir figures
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import set_pub_style

DEFAULT_LABELS = (
    "/space/grp/rschwartz/rschwartz/evaluation_summary.nf"
    "/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false"
    "/aggregated_results/files/label_results.tsv"
)

METRICS = ["f1_score", "precision", "recall"]
METRIC_LABELS = {"f1_score": "F1", "precision": "Precision", "recall": "Recall"}


def load_data(path: str, key: str, cutoff: float) -> pd.DataFrame:
    """Load label_results filtered to key/cutoff; cast precision/recall to float."""
    df = pd.read_csv(
        path, sep="\t",
        usecols=["study", "label", "f1_score", "precision", "recall", "key", "cutoff"],
        na_values=["None", ""],
    )
    df = df[(df["key"] == key) & (df["cutoff"] == cutoff)].copy()
    for col in ["f1_score", "precision", "recall"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Shorten study labels to GEO accession
    df["study"] = df["study"].str.split().str[0]
    return df


def make_pivot(df: pd.DataFrame, metric: str,
               row_order: list, col_order: list) -> pd.DataFrame:
    """Return a pivot of `metric` reindexed to the given label/study order."""
    pivot = (
        df.groupby(["label", "study"])[metric]
        .mean()
        .unstack("study")
    )
    return pivot.reindex(index=row_order, columns=col_order)


def _metric_stats(pivot: pd.DataFrame, prefix: str) -> dict:
    return {
        f"mean_{prefix}": pivot.mean(axis=1).values,
        f"std_{prefix}":  pivot.std(axis=1).values,
        f"min_{prefix}":  pivot.min(axis=1).values,
        f"max_{prefix}":  pivot.max(axis=1).values,
    }


def write_summary(pivots: dict, outdir: str, prefix: str, key: str, cutoff: float) -> None:
    """Write summary TSV with F1, precision, and recall statistics per label."""
    f1_pivot = pivots["f1_score"]
    n_total = f1_pivot.shape[1]

    data = {
        "label":           f1_pivot.index,
        "key":             key,
        "cutoff":          cutoff,
        "n_studies":       f1_pivot.notna().sum(axis=1).values,
        "n_studies_total": n_total,
        "frac_studies":    (f1_pivot.notna().sum(axis=1) / n_total).values,
    }
    for metric, col_prefix in [("f1_score", "f1"), ("precision", "precision"), ("recall", "recall")]:
        data.update(_metric_stats(pivots[metric], col_prefix))

    summary = (
        pd.DataFrame(data)
        .sort_values("mean_f1", ascending=False)
        .reset_index(drop=True)
    )
    out = os.path.join(outdir, f"{prefix}_summary.tsv")
    summary.to_csv(out, sep="\t", index=False)
    print(f"Saved {out}")


def draw_heatmap(ax: plt.Axes, pivot: pd.DataFrame, title: str, cbar_label: str) -> None:
    """Draw a single metric heatmap onto ax."""
    nan_mask = pivot.isna()
    ax.pcolormesh(
        np.ones(pivot.shape),
        cmap=ListedColormap(["#cccccc"]),
        vmin=0, vmax=1,
    )
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar_kws={"label": cbar_label, "shrink": 0.7},
        mask=nan_mask,
        yticklabels=True,
    )
    ax.set_title(title, fontweight="bold", loc="left", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.set_ylabel("Cell type", fontsize=9)
    ax.tick_params(axis="x", labelsize=8, rotation=40)
    ax.tick_params(axis="y", labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label_results", default=DEFAULT_LABELS)
    p.add_argument("--organism", default="mus_musculus")
    p.add_argument("--key",    default="subclass")
    p.add_argument("--cutoff", default=0.0, type=float)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--prefix", default="study_variance")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    print("Loading data...")
    df = load_data(args.label_results, args.key, args.cutoff)

    # Determine row/column order from F1 pivot
    f1_base = (
        df.groupby(["label", "study"])["f1_score"]
        .mean()
        .unstack("study")
    )
    row_order = f1_base.mean(axis=1).sort_values().index.tolist()
    col_order = f1_base.mean(axis=0).sort_values().index.tolist()
    print(f"  {len(row_order)} cell types × {len(col_order)} studies")

    # Build all three pivots in the same order
    pivots = {m: make_pivot(df, m, row_order, col_order) for m in METRICS}

    write_summary(pivots, args.outdir, args.prefix, args.key, args.cutoff)

    # --- Three-panel figure: F1 / Precision / Recall ---
    panel_h = max(3, len(row_order) * 0.35)
    fig_w   = 3 + len(col_order) * 0.75
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, panel_h * 3 + 0.5))

    for ax, metric in zip(axes, METRICS):
        label = METRIC_LABELS[metric]
        title = (
            f"{label} by study — {args.organism} "
            f"({args.key}, cutoff={args.cutoff})"
        )
        draw_heatmap(ax, pivots[metric], title, label)

    axes[-1].set_xlabel("Study", fontsize=9)
    plt.tight_layout(pad=1.5, h_pad=2.0)

    for ext in ("png", "pdf"):
        out = os.path.join(args.outdir, f"{args.prefix}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
