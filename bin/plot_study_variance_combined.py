#!/usr/bin/env python3
"""
Cross-species F1 by cell type: strip + box, coloured by study.

Two-panel figure (Human / Mouse). X axis is cell type; each dot is one
study's mean F1 (averaged over method) for that cell type. Box
summarises across-study spread per cell type. Points are coloured by
study of origin.
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(__file__))
from plot_utils import set_pub_style

_BASE = (
    "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01"
)
DEFAULT_HS = os.path.join(
    _BASE,
    "homo_sapiens_main_branch/100/dataset_id/SCT/gap_false",
    "aggregated_results/files/label_results.tsv.gz",
)
DEFAULT_MM = os.path.join(
    _BASE,
    "mus_musculus_main_branch/100/dataset_id/SCT/gap_false",
    "aggregated_results/files/label_results.tsv.gz",
)

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
DEFAULT_HS_CENSUS = os.path.join(_ASSETS, "census_map_human.tsv")
DEFAULT_MM_CENSUS = os.path.join(_ASSETS, "census_map_mouse_author.tsv")

FAMILY_COLORS = {
    "GABAergic":          "#4e79a7",
    "Glutamatergic":      "#f28e2b",
    "Astrocyte":          "#59a14f",
    "Oligodendrocyte":    "#76b7b2",
    "OPC":                "#edc948",
    "Vascular":           "#e15759",
    "Microglia":          "#ff9da7",
    "CNS macrophage":     "#ff9da7",
    "Neural stem cell":   "#b07aa1",
    "Hippocampal neuron": "#9c755f",
    "Leukocyte":          "#bab0ac",
    "Neuron":             "#d4b483",
    "Non-neuron":         "#999999",
}


def load_family_map(census_path: str) -> dict:
    df = pd.read_csv(census_path, sep="\t")
    return (
        df.drop_duplicates(subset="subclass")
        .set_index("subclass")["family"]
        .to_dict()
    )


def load_data(path: str, key: str, cutoff: float,
              method: str, reference: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["study", "label", "f1_score", "key", "cutoff", "method", "reference"],
        na_values=["None", ""],
    )
    df = df[
        (df["key"] == key)
        & (df["cutoff"] == cutoff)
        & (df["method"] == method)
        & (df["reference"] == reference)
    ].copy()
    df["f1_score"] = pd.to_numeric(df["f1_score"], errors="coerce")
    df["study"] = df["study"].str.split().str[0]
    return df


def aggregate(df: pd.DataFrame, family_map: dict) -> pd.DataFrame:
    """Mean F1 per (label, study); attach family; drop NaNs."""
    agg = (
        df.groupby(["label", "study"])["f1_score"]
        .mean()
        .reset_index()
        .dropna(subset=["f1_score"])
    )
    agg["family"] = agg["label"].map(
        lambda lbl: family_map.get(lbl) or (lbl if lbl in FAMILY_COLORS else "Other")
    )
    return agg


def build_study_palette(studies) -> dict:
    """Assign a distinct colour to each study."""
    studies = sorted(studies)
    base = sns.color_palette("tab10", n_colors=10) + sns.color_palette("Set2", n_colors=8)
    return {s: base[i % len(base)] for i, s in enumerate(studies)}


def draw_strip_box(ax: plt.Axes, agg: pd.DataFrame, title: str,
                   study_palette: dict) -> None:
    """Box (cell-type-level spread across studies) + strip (coloured by study)."""
    label_order = (
        agg.groupby("label")["f1_score"].median().sort_values().index.tolist()
    )
    study_order = sorted(agg["study"].unique())

    sns.boxplot(
        data=agg,
        x="label",
        y="f1_score",
        order=label_order,
        ax=ax,
        width=0.6,
        color="white",
        linewidth=2.0,
        fliersize=0,
        showcaps=True,
        boxprops=dict(edgecolor="#333", facecolor="white"),
        medianprops=dict(color="#333", linewidth=3),
        whiskerprops=dict(color="#333", linewidth=2),
        capprops=dict(color="#333", linewidth=2),
    )
    sns.stripplot(
        data=agg,
        x="label",
        y="f1_score",
        order=label_order,
        hue="study",
        hue_order=study_order,
        palette=study_palette,
        ax=ax,
        size=18,
        alpha=0.9,
        jitter=0.22,
        dodge=False,
        edgecolor="black",
        linewidth=0.8,
        legend=False,
    )

    ax.set_title(title, fontweight="bold", loc="left", fontsize=60, pad=10)
    ax.set_xlabel("Cell type", fontsize=48, labelpad=14)
    ax.set_ylabel("Mean F1 score", fontsize=48, labelpad=14)
    ax.set_ylim(-0.02, 1.02)
    ax.tick_params(axis="x", labelsize=36, rotation=45)
    ax.tick_params(axis="y", labelsize=38)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def make_study_legend(fig: plt.Figure, palette_hs: dict, palette_mm: dict) -> None:
    """Two-column study legend (human + mouse)."""
    ax = fig.add_subplot(111)
    ax.axis("off")

    def handles(palette):
        return [mpatches.Patch(color=c, label=s) for s, c in palette.items()]

    leg_hs = ax.legend(
        handles=handles(palette_hs),
        title="Human studies",
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        frameon=False,
        fontsize=46,
        title_fontsize=52,
    )
    ax.add_artist(leg_hs)
    ax.legend(
        handles=handles(palette_mm),
        title="Mouse studies",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=46,
        title_fontsize=52,
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hs_label_results", default=DEFAULT_HS)
    p.add_argument("--mm_label_results", default=DEFAULT_MM)
    p.add_argument("--hs_census", default=DEFAULT_HS_CENSUS)
    p.add_argument("--mm_census", default=DEFAULT_MM_CENSUS)
    p.add_argument("--hs_key", default="subclass")
    p.add_argument("--mm_key", default="subclass")
    p.add_argument("--cutoff", default=0.0, type=float)
    p.add_argument("--method",    default="scvi")
    p.add_argument("--reference", default="whole cortex")
    p.add_argument("--outdir", default="combined_orgs")
    p.add_argument("--prefix", default="study_variance_combined")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    set_pub_style()

    family_map_hs = load_family_map(args.hs_census)
    family_map_mm = load_family_map(args.mm_census)

    print(f"Loading data (method={args.method}, reference={args.reference!r}, "
          f"hs_key={args.hs_key}, mm_key={args.mm_key})...")
    df_hs = load_data(args.hs_label_results, args.hs_key, args.cutoff,
                      args.method, args.reference)
    df_mm = load_data(args.mm_label_results, args.mm_key, args.cutoff,
                      args.method, args.reference)

    agg_hs = aggregate(df_hs, family_map_hs)
    agg_mm = aggregate(df_mm, family_map_mm)
    print(f"  Human: {agg_hs['label'].nunique()} cell types × "
          f"{agg_hs['study'].nunique()} studies ({len(agg_hs)} points)")
    print(f"  Mouse: {agg_mm['label'].nunique()} cell types × "
          f"{agg_mm['study'].nunique()} studies ({len(agg_mm)} points)")

    palette_hs = build_study_palette(agg_hs["study"].unique())
    palette_mm = build_study_palette(agg_mm["study"].unique())

    n_labels_hs = agg_hs["label"].nunique()
    n_labels_mm = agg_mm["label"].nunique()
    cell_w = 1.4
    fig_w = (n_labels_hs + n_labels_mm) * cell_w + 10
    fig_h = 18
    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        sharey=True,
        gridspec_kw={
            "width_ratios": [n_labels_hs, n_labels_mm],
            "wspace": 0.04,
        },
    )
    draw_strip_box(axes[0], agg_hs, "Human", palette_hs)
    draw_strip_box(axes[1], agg_mm, "Mouse", palette_mm)
    axes[1].set_ylabel("")

    out = os.path.join(args.outdir, f"{args.prefix}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

    fig_legend = plt.figure(figsize=(22, 12))
    make_study_legend(fig_legend, palette_hs, palette_mm)
    legend_out = os.path.join(args.outdir, f"{args.prefix}_legend.png")
    fig_legend.savefig(legend_out, dpi=300, bbox_inches="tight")
    print(f"Saved {legend_out}")
    plt.close(fig_legend)

    print("Done.")


if __name__ == "__main__":
    main()
