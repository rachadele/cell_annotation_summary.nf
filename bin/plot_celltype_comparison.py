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

CENSUS_BY_ORGANISM = {
    "mus_musculus": "assets/census_map_mouse_author.tsv",
    "homo_sapiens": "assets/census_map_human.tsv",
}


def load(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(path, sep="\t")


def load_family_map(census_path, key):
    """Return {label_at_key: family} from the census map."""
    census = pd.read_csv(census_path, sep="\t")
    if key not in census.columns or "family" not in census.columns:
        return {}
    return (
        census.drop_duplicates(subset=key)
        .set_index(key)["family"]
        .to_dict()
    )


def plot_family(df, family, labels, key, outdir, cutoff, subsample_ref, methods, study_order, refs):
    """One figure: rows=references, cols=cell types within family."""
    nrows = len(refs)
    ncols = len(labels)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 5 * nrows),
        sharey=True,
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

    for ri, ref in enumerate(refs):
        for ci, label in enumerate(labels):
            ax = axes[ri][ci]
            df_sub = df[(df["label"] == label) & (df["reference"] == ref)]

            tbl = df_sub.pivot_table(
                index="study", columns="method", values="f1_score", aggfunc="mean"
            ).reindex(study_order)

            for method in methods:
                if method not in tbl.columns:
                    continue
                ax.scatter(
                    x + offsets[method], tbl[method].values,
                    color=METHOD_COLORS[method], s=120, zorder=3,
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
                    ax.plot(xs, ys, color="grey", lw=1.2, alpha=0.4, zorder=1)

            ax.set_xticks(x)
            ax.set_xticklabels(study_order, fontsize=12, rotation=45, ha="right")
            ax.tick_params(axis="y", labelsize=12)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="lightgrey", lw=1.0, ls="--", zorder=0)
            ax.grid(axis="y", alpha=0.3)

            if ri == 0:
                ax.set_title(label, fontsize=14, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(
                    "\n".join(textwrap.wrap(ref, width=25)) + "\nF1",
                    fontsize=12,
                )

    handles = [
        mlines.Line2D([], [], color=METHOD_COLORS[m], marker="o",
                      linestyle="None", markersize=10, label=METHOD_LABELS[m])
        for m in methods
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=14,
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        f"{family} — F1 by study × reference × method\n"
        f"(key={key}, cutoff={cutoff}, subsample_ref={subsample_ref})",
        fontsize=16, y=1.01,
    )
    plt.tight_layout()

    safe_family = family.replace(" ", "_").replace("/", "-")
    outpath = os.path.join(outdir, f"celltype_comparison_{key}_{safe_family}.png")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def plot(df, key, outdir, cutoff=0, subsample_ref=100, census_path=None):
    methods = [m for m in ["scvi_knn", "scvi_rf", "seurat"] if m in df["method"].unique()]

    df = df[
        (df["key"] == key)
        & (df["cutoff"] == cutoff)
        & (df["subsample_ref"] == subsample_ref)
    ].copy()

    if df.empty:
        print(f"No data for key={key}, cutoff={cutoff}, subsample_ref={subsample_ref} — skipping.")
        return

    refs = sorted(df["reference"].unique())
    study_order = sorted(df["study"].unique())
    labels = sorted(df["label"].unique())

    # Group labels by family; fall back to label itself if not in census
    if census_path and os.path.exists(census_path):
        family_map = load_family_map(census_path, key)
        family_groups = {}
        for lbl in labels:
            fam = family_map.get(lbl, lbl)
            family_groups.setdefault(fam, []).append(lbl)
    else:
        family_groups = {"All": labels}

    for family, fam_labels in sorted(family_groups.items()):
        plot_family(df, family, sorted(fam_labels), key, outdir,
                    cutoff, subsample_ref, methods, study_order, refs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results",       required=True, help="Path to label_results.tsv.gz")
    parser.add_argument("--outdir",        required=True, help="Output directory")
    parser.add_argument("--key",           default="class", help="Taxonomy level")
    parser.add_argument("--cutoff",        type=float, default=0)
    parser.add_argument("--subsample_ref", type=int, default=100)
    parser.add_argument("--census",        default=None, help="Census map TSV for family grouping")
    parser.add_argument("--organism",      default=None,
                        help="Organism (mus_musculus|homo_sapiens); auto-detects census if --census omitted")
    args = parser.parse_args()

    census = args.census
    if census is None and args.organism in CENSUS_BY_ORGANISM:
        census = CENSUS_BY_ORGANISM[args.organism]

    df = load(args.results)
    plot(df, args.key, args.outdir, args.cutoff, args.subsample_ref, census)
