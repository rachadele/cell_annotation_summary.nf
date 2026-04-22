import argparse
import gzip
import os
import textwrap

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

METHOD_COLORS = {"scvi_knn": "#2ca02c", "scvi_rf": "#1f77b4", "seurat": "#ff7f0e"}
METHOD_LABELS = {"scvi_knn": "scVI kNN", "scvi_rf": "scVI RF", "seurat": "Seurat"}


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


def plot(df, key, outpath, cutoff=0, subsample_ref=100):
    methods = ["scvi_knn", "scvi_rf", "seurat"]
    methods = [m for m in methods if m in df["method"].unique()]

    df = df[
        (df["key"] == key)
        & (df["cutoff"] == cutoff)
        & (df["subsample_ref"] == subsample_ref)
    ].copy()

    labels = sorted(df["label"].unique())

    refs = sorted(df["reference"].unique())
    study_order = sorted(df["study"].unique())

    nrows = len(labels)
    ncols = len(refs)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(9 * ncols, 6 * nrows),
        sharey="row",
        sharex=True,
    )

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    offsets = {"scvi_knn": -0.2, "scvi_rf": 0.0, "seurat": 0.2}
    x = np.arange(len(study_order))

    for ri, label in enumerate(labels):
        df_label = df[df["label"] == label]
        for ci, ref in enumerate(refs):
            ax = axes[ri][ci]
            df_sub = df_label[df_label["reference"] == ref]

            tbl = df_sub.pivot_table(
                index="study", columns="method", values="f1_score", aggfunc="mean"
            ).reindex(study_order)

            for method in methods:
                if method not in tbl.columns:
                    continue
                vals = tbl[method].values
                ax.scatter(
                    x + offsets[method], vals,
                    color=METHOD_COLORS[method], s=200, zorder=3,
                    label=METHOD_LABELS[method],
                )

            for si, study in enumerate(study_order):
                if study not in tbl.index:
                    continue
                row = tbl.loc[study]
                pts = [
                    (si + offsets[m], row[m])
                    for m in methods
                    if m in row and not np.isnan(row[m])
                ]
                if len(pts) > 1:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color="grey", lw=1.5, alpha=0.4, zorder=1)

            ax.set_xticks(x)
            ax.set_xticklabels(study_order, fontsize=16, rotation=45, ha="right")
            ax.tick_params(axis="y", labelsize=16)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="lightgrey", lw=1.0, ls="--", zorder=0)
            ax.grid(axis="y", alpha=0.3)

            if ri == 0:
                ax.set_title(
                    "\n".join(textwrap.wrap(ref, width=30)),
                    fontsize=18, fontweight="bold",
                )
            if ci == 0:
                ax.set_ylabel(f"{label}\nF1", fontsize=18)

    handles = [
        mlines.Line2D([], [], color=METHOD_COLORS[m], marker="o",
                      linestyle="None", markersize=14, label=METHOD_LABELS[m])
        for m in methods
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=18,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"Per-cell-type F1 by study × reference × method\n"
        f"(key={key}, cutoff={cutoff}, subsample_ref={subsample_ref})",
        fontsize=20, y=1.01,
    )
    plt.tight_layout()
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to label_results.tsv.gz")
    parser.add_argument("--outpath", required=True, help="Output PNG path")
    parser.add_argument("--key", default="class", help="Taxonomy level (family, class, subclass)")
    parser.add_argument("--cutoff", type=float, default=0)
    parser.add_argument("--subsample_ref", type=int, default=100)
    args = parser.parse_args()

    df = load(args.results)
    plot(df, args.key, args.outpath, args.cutoff, args.subsample_ref)
