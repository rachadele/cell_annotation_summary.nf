import argparse
import gzip
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

METHOD_COLORS = {"scvi_knn": "#2ca02c", "scvi_rf": "#1f77b4", "seurat": "#ff7f0e"}
METHOD_LABELS = {"scvi_knn": "scVI kNN", "scvi_rf": "scVI RF", "seurat": "Seurat"}


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


def write_tsv(df, keys, outpath, cutoff=0, subsample_ref=100):
    df = df[(df["cutoff"] == cutoff) & (df["subsample_ref"] == subsample_ref)].copy()
    df = df[df["key"].isin(keys)]
    tbl = (
        df.groupby(["key", "study", "reference", "method"])["macro_f1"]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"macro_f1": "macro_f1_mean"})
    )
    tbl["cutoff"] = cutoff
    tbl["subsample_ref"] = subsample_ref
    tbl.to_csv(outpath, sep="\t", index=False)
    print(f"Saved: {outpath}")


def plot(df, keys, outpath, cutoff=0, subsample_ref=100):
    methods = ["scvi_knn", "scvi_rf", "seurat"]
    methods = [m for m in methods if m in df["method"].unique()]

    df = df[(df["cutoff"] == cutoff) & (df["subsample_ref"] == subsample_ref)].copy()
    df = df[df["key"].isin(keys)]

    df["ref_short"] = df["reference"]

    df["study_short"] = df["study"]

    refs = sorted(df["ref_short"].unique())
    study_order = sorted(df["study_short"].unique())

    nrows = len(keys)
    ncols = len(refs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 6 * nrows),
                             sharey=False, sharex=True)

    if nrows == 1:
        axes = [axes]

    import textwrap

    for ri, key in enumerate(keys):
        df_key = df[df["key"] == key]
        for ci, ref in enumerate(refs):
            ax = axes[ri][ci]
            df_sub = df_key[df_key["ref_short"] == ref]

            tbl = df_sub.pivot_table(
                index="study_short", columns="method", values="macro_f1", aggfunc="mean"
            ).reindex(study_order)

            x = np.arange(len(study_order))
            offsets = {"scvi_knn": -0.2, "scvi_rf": 0.0, "seurat": 0.2}

            for method in methods:
                if method not in tbl.columns:
                    continue
                vals = tbl[method].values
                ax.scatter(x + offsets[method], vals,
                           color=METHOD_COLORS[method], s=200, zorder=3,
                           label=METHOD_LABELS[method])

            # draw thin connecting lines across methods per study
            for si, study in enumerate(study_order):
                row = tbl.loc[study] if study in tbl.index else None
                if row is None:
                    continue
                pts = [(si + offsets[m], row[m]) for m in methods if m in row and not np.isnan(row[m])]
                if len(pts) > 1:
                    xs, ys = zip(*pts)
                    ax.plot(xs, ys, color="grey", lw=1.5, alpha=0.4, zorder=1)

            ax.set_xticks(x)
            ax.set_xticklabels(study_order, fontsize=16, rotation=45, ha="right")
            ax.tick_params(axis="y", labelsize=16)
            ax.set_ylim(0.3, 1.02)
            ax.axhline(0.5, color="lightgrey", lw=1.0, ls="--", zorder=0)
            ax.grid(axis="y", alpha=0.3)

            if ri == 0:
                ax.set_title("\n".join(textwrap.wrap(ref, width=30)), fontsize=18, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{key}\nmacro F1", fontsize=18)

    # shared legend
    handles = [mlines.Line2D([], [], color=METHOD_COLORS[m], marker="o",
                             linestyle="None", markersize=14, label=METHOD_LABELS[m])
               for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=18,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(f"Macro F1 by study × reference × method\n(cutoff={cutoff}, subsample_ref={subsample_ref})",
                 fontsize=20, y=1.01)
    plt.tight_layout()
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to sample_results.tsv.gz")
    parser.add_argument("--outpath", required=True, help="Output PNG path")
    parser.add_argument("--cutoff", type=float, default=0)
    parser.add_argument("--subsample_ref", type=int, default=100)
    parser.add_argument("--keys", nargs="+", default=["family", "class", "subclass"])
    args = parser.parse_args()

    df = load(args.results)
    plot(df, args.keys, args.outpath, args.cutoff, args.subsample_ref)
    tsv_out = os.path.join(os.path.dirname(args.outpath), "method_study_raw_scores.tsv")
    write_tsv(df, args.keys, tsv_out, args.cutoff, args.subsample_ref)
