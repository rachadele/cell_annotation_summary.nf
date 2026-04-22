#!/usr/bin/env python3
"""
Variance summary for cell-type annotation benchmarking.

For each taxonomy level (global, family, class, subclass), produces:

  variance_dotplot_{key}.png
      Std(F1) across studies per cell type × method, faceted by reference.
      Pink shading = systematic failure (mean F1 < 0.5 in ≥3 studies).

  variance_summary.tsv
      Per (label, key, method, reference): mean_f1, std_f1, n_studies,
      systematic_failure.  Filtered to --cutoff and --subsample_ref.

Usage:
    python variance_summary.py \\
        --label_results label_results.tsv.gz \\
        --organism mus_musculus \\
        --cutoff 0.0 \\
        --subsample_ref 100 \\
        --outdir variance_summary
"""

import argparse
import gzip
import os
import sys
import textwrap

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import METHOD_COLORS, METHOD_NAMES, set_pub_style

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LABELS = (
    "/space/grp/rschwartz/rschwartz/evaluation_summary.nf"
    "/2024-07-01/mus_musculus/100/dataset_id/SCT/gap_false"
    "/aggregated_results/files/label_results.tsv.gz"
)

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
_CENSUS_BY_ORGANISM = {
    "mus_musculus": os.path.join(_ASSETS, "census_map_mouse_author.tsv"),
    "homo_sapiens": os.path.join(_ASSETS, "census_map_human.tsv"),
}

METHOD_ORDER = ["scvi_knn", "scvi_rf", "seurat"]
KEY_ORDER    = ["global", "family", "class", "subclass"]

SYSTEMATIC_FAILURE_THRESHOLD = 0.5
MIN_STUDIES_FOR_FAILURE = 3

# ---------------------------------------------------------------------------
# Data loading and prep
# ---------------------------------------------------------------------------

def load_label_results(path):
    opener = gzip.open(path, "rt") if path.endswith(".gz") else open(path)
    with opener as f:
        df = pd.read_csv(f, sep="\t")
    df["f1_score"] = pd.to_numeric(df["f1_score"], errors="coerce")
    df["study"] = df["study"].str.split().str[0]
    return df


def compute_variance(df):
    """
    Per (label, key, method, reference, cutoff, subsample_ref):
    mean F1 and std F1 across studies, n_studies, systematic_failure flag.
    """
    group_cols = ["label", "key", "method", "reference", "cutoff", "subsample_ref", "study"]
    study_means = (
        df.groupby(group_cols)["f1_score"]
        .mean()
        .reset_index()
        .rename(columns={"f1_score": "mean_f1_study"})
    )

    agg_cols = ["label", "key", "method", "reference", "cutoff", "subsample_ref"]
    ct = (
        study_means.groupby(agg_cols)
        .agg(
            mean_f1=("mean_f1_study", "mean"),
            std_f1=("mean_f1_study", "std"),
            n_studies=("mean_f1_study", "count"),
        )
        .reset_index()
    )
    ct["std_f1"] = ct["std_f1"].fillna(0.0)
    ct["systematic_failure"] = (
        (ct["mean_f1"] < SYSTEMATIC_FAILURE_THRESHOLD)
        & (ct["n_studies"] >= MIN_STUDIES_FOR_FAILURE)
    )
    return ct


def abbreviate_refs(refs):
    """Shorten reference names to first 3 words; disambiguate collisions with last word."""
    short = {r: " ".join(r.split()[:3]) for r in refs}
    seen = {}
    for r, s in short.items():
        seen.setdefault(s, []).append(r)
    for s, rs in seen.items():
        if len(rs) > 1:
            for r in rs:
                short[r] = s + " " + r.split()[-1]
    return short


def label_order_for_key(df_key):
    """Sort cell types by mean std(F1) across methods and references."""
    return (
        df_key.groupby("label")["std_f1"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )

# ---------------------------------------------------------------------------
# Dot plot: std(F1) by cell type × method, faceted by reference
# ---------------------------------------------------------------------------

def plot_dotplot(df_key, key, outdir, ref_labels):
    refs    = sorted(df_key["reference"].unique(), key=lambda r: ref_labels[r])
    methods = [m for m in METHOD_ORDER if m in df_key["method"].unique()]
    offsets = {"scvi_knn": -0.2, "scvi_rf": 0.0, "seurat": 0.2}

    label_ord = label_order_for_key(df_key)
    y = np.arange(len(label_ord))
    n_refs = len(refs)

    fig, axes = plt.subplots(
        1, n_refs,
        figsize=(8 * n_refs, max(8, 1.1 * len(label_ord))),
    )
    if n_refs == 1:
        axes = [axes]

    for ci, ref in enumerate(refs):
        ax = axes[ci]
        df_ref = df_key[df_key["reference"] == ref]

        labels_in_ref = set(df_ref["label"].unique())

        for yi, lbl in enumerate(label_ord):
            if lbl not in labels_in_ref:
                ax.text(0.02, yi, "*", fontsize=26, color="#888888",
                        ha="left", va="center", zorder=4,
                        transform=ax.get_yaxis_transform())
                continue
            row = df_ref[df_ref["label"] == lbl]
            if row["systematic_failure"].any():
                ax.axhspan(yi - 0.45, yi + 0.45, color="mistyrose", alpha=0.5, zorder=0)

        # Collect per-method points, draw lines connecting same cell type across methods
        method_pts = {}
        for method in methods:
            df_m = df_ref[df_ref["method"] == method].set_index("label")
            xs, ys, xs_single, ys_single = [], [], [], []
            for yi, lbl in enumerate(label_ord):
                if lbl not in df_m.index:
                    continue
                row = df_m.loc[lbl]
                ypos = yi + offsets[method]
                if row["n_studies"] <= 1:
                    xs_single.append(0.0)
                    ys_single.append(ypos)
                    method_pts.setdefault(yi, []).append((0.0, ypos))
                else:
                    xs.append(row["std_f1"])
                    ys.append(ypos)
                    method_pts.setdefault(yi, []).append((row["std_f1"], ypos))

            if xs:
                ax.scatter(xs, ys, color=METHOD_COLORS[method], s=150, zorder=3)
            if xs_single:
                ax.scatter(xs_single, ys_single, color=METHOD_COLORS[method], s=150,
                           zorder=3, facecolors="none",
                           edgecolors=METHOD_COLORS[method], linewidths=2.0)

        for pts in method_pts.values():
            if len(pts) > 1:
                pts_sorted = sorted(pts, key=lambda p: p[1])
                px = [p[0] for p in pts_sorted]
                py = [p[1] for p in pts_sorted]
                ax.plot(px, py, color="grey", lw=1.2, alpha=0.4, zorder=2)

        ax.set_yticks(y)
        ax.set_yticklabels(label_ord if ci == 0 else [], fontsize=22)
        ax.set_xlabel("Std(F1) across studies", fontsize=24)
        ax.set_title("\n".join(textwrap.wrap(ref_labels[ref], width=20)), fontsize=26, fontweight="bold")
        ax.axvline(0, color="lightgrey", lw=0.8, ls="--")
        ax.grid(axis="x", alpha=0.3)
        ax.set_xlim(left=-0.02)
        ax.tick_params(axis="x", labelsize=22)

    handles = [
        mlines.Line2D([], [], color=METHOD_COLORS[m], marker="o",
                      linestyle="None", markersize=12, label=METHOD_NAMES[m])
        for m in methods
    ]
    handles.append(
        mlines.Line2D([], [], color="grey", marker="o", linestyle="None",
                      markersize=12, markerfacecolor="none", markeredgewidth=1.5,
                      label="single study")
    )
    handles.append(
        mlines.Line2D([], [], color="#888888", marker="$*$", linestyle="None",
                      markersize=14, label="absent from reference")
    )
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=26, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(
        f"Cross-study F1 variance by cell type, method, and reference  ({key})\n"
        "Pink = systematic failure (mean F1 < 0.5 in ≥3 studies)  |  * = absent from reference",
        fontsize=32, y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(outdir, f"variance_dotplot_{key}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# ---------------------------------------------------------------------------
# Scatter: mean F1 vs std F1 per cell type × method (averaged over references)
# ---------------------------------------------------------------------------

def plot_scatter(df_key, key, outdir):
    methods  = [m for m in METHOD_ORDER if m in df_key["method"].unique()]
    labels   = sorted(df_key["label"].unique())
    n_labels = len(labels)

    ncols = min(4, n_labels)
    nrows = int(np.ceil(n_labels / ncols))

    # Global axis limits for comparability across panels
    xmax = max(df_key["mean_f1"].max() * 1.05, 1.0)
    ymax = max(df_key["std_f1"].max()  * 1.15, 0.05)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 6.5 * nrows),
    )
    axes_flat = np.array(axes).flatten()

    for i, lbl in enumerate(labels):
        ax = axes_flat[i]
        df_lbl = df_key[df_key["label"] == lbl]

        ax.axvline(SYSTEMATIC_FAILURE_THRESHOLD, color="lightgrey", lw=1.2, ls="--", zorder=0)
        ax.axhline(0, color="lightgrey", lw=0.8, ls="--", zorder=0)
        ax.grid(alpha=0.2)

        # Average over references: one point per method
        avg = df_lbl.groupby("method")[["mean_f1", "std_f1"]].mean()

        # Line connecting the 3 methods
        pts = [(avg.loc[m, "mean_f1"], avg.loc[m, "std_f1"]) for m in methods if m in avg.index]
        if len(pts) > 1:
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color="grey", lw=1.5, alpha=0.5, zorder=2)

        for method in methods:
            if method not in avg.index:
                continue
            ax.scatter(
                avg.loc[method, "mean_f1"], avg.loc[method, "std_f1"],
                color=METHOD_COLORS[method], s=250, zorder=3,
                alpha=0.9,
            )

        ax.set_title(lbl, fontsize=36, fontweight="bold")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.01, ymax)
        ax.set_xlabel("Mean F1", fontsize=30)
        ax.set_ylabel("Std(F1)", fontsize=30)
        ax.tick_params(labelsize=26)

    # Hide unused panels
    for ax in axes_flat[n_labels:]:
        ax.set_visible(False)

    handles = [
        mlines.Line2D([], [], color=METHOD_COLORS[m], marker="o",
                      linestyle="None", markersize=14, label=METHOD_NAMES[m])
        for m in methods
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(methods),
               fontsize=30, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"Mean F1 vs cross-study variance per cell type  ({key})\n"
        "Each point = one method, averaged over references. Dashed line = F1 = 0.5.",
        fontsize=36, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = os.path.join(outdir, f"variance_scatter_{key}.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label_results", default=DEFAULT_LABELS)
    p.add_argument("--organism",      default="mus_musculus")
    p.add_argument("--census",        default=None)
    p.add_argument("--cutoff",        default=0.0, type=float)
    p.add_argument("--subsample_ref", default=100, type=int)
    p.add_argument("--outdir",        default="variance_summary")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    print("Loading data...")
    df = load_label_results(args.label_results)
    df = df[(df["cutoff"] == args.cutoff) & (df["subsample_ref"] == args.subsample_ref)].copy()
    print(f"  {len(df):,} rows after filtering to cutoff={args.cutoff}, subsample_ref={args.subsample_ref}")

    print("Computing cross-study variance...")
    ct = compute_variance(df)

    out_tsv = os.path.join(args.outdir, "variance_summary.tsv")
    ct.to_csv(out_tsv, sep="\t", index=False)
    print(f"  Saved {out_tsv} ({len(ct)} rows)")

    ref_labels = abbreviate_refs(ct["reference"].unique().tolist())
    print(f"  Reference labels: {ref_labels}")

    for key in KEY_ORDER:
        df_key = ct[ct["key"] == key].copy()
        if df_key.empty:
            print(f"  No data for key={key}, skipping.")
            continue
        print(f"  {key}: {df_key['label'].nunique()} cell types")
        plot_dotplot(df_key, key, args.outdir, ref_labels)
        plot_scatter(df_key, key, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
