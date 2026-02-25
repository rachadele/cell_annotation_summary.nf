#!/usr/bin/env python3
"""
Cell-type F1 score heatmap across studies.

Shows that every cell type's F1 varies substantially across studies,
illustrating that study of origin is a major driver of performance.

Rows: cell type labels (sorted by mean F1 across studies)
Columns: studies (sorted by mean F1 across labels)
Colour: mean F1 score (averaged over method)

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
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import set_pub_style

DEFAULT_LABELS = (
    "/space/grp/rschwartz/rschwartz/evaluation_summary.nf"
    "/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false"
    "/aggregated_results/files/label_results.tsv"
)


def load_pivot(path: str, key: str, cutoff: float) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t",
                     usecols=["study", "label", "f1_score", "key", "cutoff"])
    df = df[(df["key"] == key) & (df["cutoff"] == cutoff)]
    pivot = (
        df.groupby(["label", "study"])["f1_score"]
        .mean()
        .unstack("study")
    )
    # Shorten study labels to GEO accession
    pivot.columns = [c.split()[0] for c in pivot.columns]
    # Sort rows by mean F1 ascending, columns by mean F1 ascending
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index,
                      pivot.mean(axis=0).sort_values().index]
    return pivot


def write_summary(pivot: pd.DataFrame, outdir: str, prefix: str, key: str, cutoff: float) -> None:
    n_total = pivot.shape[1]
    summary = pd.DataFrame({
        "label":           pivot.index,
        "key":             key,
        "cutoff":          cutoff,
        "n_studies":       pivot.notna().sum(axis=1).values,
        "n_studies_total": n_total,
        "frac_studies":    (pivot.notna().sum(axis=1) / n_total).values,
        "mean_f1":         pivot.mean(axis=1).values,
        "std_f1":          pivot.std(axis=1).values,
        "min_f1":          pivot.min(axis=1).values,
        "max_f1":          pivot.max(axis=1).values,
    }).sort_values("mean_f1", ascending=False).reset_index(drop=True)

    out = os.path.join(outdir, f"{prefix}_summary.tsv")
    summary.to_csv(out, sep="\t", index=False)
    print(f"Saved {out}")


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
    pivot = load_pivot(args.label_results, args.key, args.cutoff)
    print(f"  {pivot.shape[0]} cell types × {pivot.shape[1]} studies")

    write_summary(pivot, args.outdir, args.prefix, args.key, args.cutoff)

    fig_h = max(5, pivot.shape[0] * 0.3)
    fig_w = 3 + pivot.shape[1] * 0.7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Draw a grey background for NaN cells (cell type absent from that study)
    import numpy as np
    from matplotlib.colors import ListedColormap
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
        cbar_kws={"label": "Mean F1", "shrink": 0.6},
        mask=nan_mask,
    )
    ax.set_title(
        f"Cell type F1 by study — {args.organism} ({args.key}, cutoff={args.cutoff})",
        fontweight="bold", loc="left", fontsize=11, pad=8,
    )
    ax.set_xlabel("Study", fontsize=10)
    ax.set_ylabel("Cell type", fontsize=10)
    ax.tick_params(axis="x", labelsize=9, rotation=40)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")

    plt.tight_layout(pad=1.5)

    for ext in ("png", "pdf"):
        out = os.path.join(args.outdir, f"{args.prefix}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
