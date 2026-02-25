#!/usr/bin/env python3
"""
generate_results_summary.py

Reads evaluation outputs across all parameter combinations and writes a
structured markdown summary to docs/results_summary.md.

Usage:
    python bin/generate_results_summary.py \
        --base_dir 2024-07-01 \
        --outfile docs/results_summary.md
"""

import argparse
import glob
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY_ORDER = ["global", "class", "family", "subclass"]

# Short display names for known long reference strings
REF_SHORT = {
    "whole cortex": "whole cortex",
    "An integrated transcriptomic and epigenomic atlas of mouse primary motor cortex cell types": "motor cortex",
    "Single-cell RNA-seq for all cortical  hippocampal regions 10x": "hippocampal 10x",
    "Single-cell RNA-seq for all cortical  hippocampal regions SMART-Seq v4": "hippocampal SMART-Seq",
    "Human Multiple Cortical Areas SMART-seq": "Human MC SMART-seq",
    "Whole Taxonomy - MTG Seattle Alzheimers Disease Atlas SEA-AD": "SEA-AD MTG",
    "Whole Taxonomy - DLPFC Seattle Alzheimers Disease Atlas SEA-AD": "SEA-AD DLPFC",
    "Dissection Angular gyrus AnG": "Dissection AnG",
    "Dissection Anterior cingulate cortex ACC": "Dissection ACC",
    "Dissection Dorsolateral prefrontal cortex DFC": "Dissection DFC",
    "Dissection Primary auditory cortexA1": "Dissection A1",
    "Dissection Primary somatosensory cortex S1": "Dissection S1",
    "Dissection Primary visual cortexV1": "Dissection V1",
}


def shorten_ref(name):
    if name in REF_SHORT:
        return REF_SHORT[name]
    if len(name) > 30:
        return name[:29] + "…"
    return name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_optional(path):
    """Return DataFrame if path exists, else None."""
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t")
    return None


def find_models_dir(base):
    """Return path to the aggregated_models formula subdir files/, or None."""
    pattern = os.path.join(base, "aggregated_models", "*", "files")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def discover_datasets(base_dir):
    """Glob for dataset roots and return list of (name, base_path)."""
    pattern = os.path.join(base_dir, "*/100/dataset_id/SCT/gap_false")
    matches = sorted(glob.glob(pattern))
    datasets = []
    for m in matches:
        parts = m.split(os.sep)
        # The dataset name is the part right after base_dir
        base_parts = base_dir.rstrip(os.sep).split(os.sep)
        name = parts[len(base_parts)]
        datasets.append((name, m))
    return datasets


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_emmean(row):
    return f"{row['response']:.3f} [{row['asymp.LCL']:.3f}–{row['asymp.UCL']:.3f}]"


def df_to_md_table(df):
    """Convert a DataFrame to a GitHub-flavoured markdown table string."""
    cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join([header, sep] + rows)


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def section_method_comparison(models_dir):
    """Method comparison table from method_emmeans_summary.tsv."""
    if models_dir is None:
        return "> *Not available — rerun pipeline.*\n"

    path = os.path.join(models_dir, "method_emmeans_summary.tsv")
    df = load_optional(path)
    if df is None:
        return "> *Not available — rerun pipeline.*\n"

    lines = []
    keys_present = [k for k in KEY_ORDER if k in df["key"].values]

    # Build wide table: key | scvi [CI] | seurat [CI]
    records = []
    for key in keys_present:
        sub = df[df["key"] == key]
        row = {"Taxonomy level": key}
        for _, r in sub.iterrows():
            row[r["method"]] = fmt_emmean(r)
        records.append(row)

    tbl = pd.DataFrame(records).fillna("—")
    lines.append(df_to_md_table(tbl))
    lines.append("")

    # Prose summary: compare scvi vs seurat at subclass
    sub_df = df[df["key"] == "subclass"]
    if not sub_df.empty:
        scvi_row = sub_df[sub_df["method"] == "scvi"]
        seur_row = sub_df[sub_df["method"] == "seurat"]
        if not scvi_row.empty and not seur_row.empty:
            scvi_val = scvi_row.iloc[0]["response"]
            seur_val = seur_row.iloc[0]["response"]
            diff = scvi_val - seur_val
            direction = "outperforms" if diff > 0 else "underperforms relative to"
            lines.append(
                f"At subclass level, scvi {direction} seurat "
                f"(Δ = {abs(diff):.3f}, model-adjusted marginal means: "
                f"scvi {scvi_val:.3f}, seurat {seur_val:.3f})."
            )

    return "\n".join(lines) + "\n"


def section_cutoff_sensitivity(models_dir):
    """Cutoff sensitivity table for key=subclass."""
    if models_dir is None:
        return "> *Not available — rerun pipeline.*\n"

    path = os.path.join(models_dir, "method_cutoff_effects.tsv")
    df = load_optional(path)
    if df is None:
        return "> *Not available — rerun pipeline.*\n"

    sub = df[df["key"] == "subclass"].copy()
    if sub.empty:
        return "> *No subclass data in method_cutoff_effects.tsv.*\n"

    pivot = sub.pivot_table(index="cutoff", columns="method", values="fit").reset_index()
    pivot.columns.name = None
    pivot["cutoff"] = pivot["cutoff"].round(3)
    for col in pivot.columns[1:]:
        pivot[col] = pivot[col].round(3)
    pivot = pivot.rename(columns={"cutoff": "Cutoff"})

    lines = [df_to_md_table(pivot), ""]

    # Prose: note direction of change for each method
    methods_in = [c for c in pivot.columns if c != "Cutoff"]
    for method in methods_in:
        low = pivot.loc[pivot["Cutoff"] == pivot["Cutoff"].min(), method].values
        high = pivot.loc[pivot["Cutoff"] == pivot["Cutoff"].max(), method].values
        if len(low) and len(high):
            delta = float(high[0]) - float(low[0])
            direction = "decreases" if delta < 0 else "increases"
            lines.append(
                f"{method}: performance {direction} by {abs(delta):.3f} "
                f"from cutoff {pivot['Cutoff'].min()} to {pivot['Cutoff'].max()}."
            )

    return "\n".join(lines) + "\n"


def section_reference_comparison(models_dir):
    """Reference × method table for key=subclass."""
    if models_dir is None:
        return "> *Not available — rerun pipeline.*\n"

    path = os.path.join(models_dir, "reference_method_emmeans_summary.tsv")
    df = load_optional(path)
    if df is None:
        return "> *Not available — rerun pipeline.*\n"

    sub = df[df["key"] == "subclass"].copy()
    if sub.empty:
        return "> *No subclass data in reference_method_emmeans_summary.tsv.*\n"

    sub["ref_short"] = sub["reference"].apply(shorten_ref)

    pivot = sub.pivot_table(index="ref_short", columns="method", values="response").reset_index()
    pivot.columns.name = None
    for col in pivot.columns[1:]:
        pivot[col] = pivot[col].round(3)
    pivot = pivot.rename(columns={"ref_short": "Reference"})

    lines = [df_to_md_table(pivot), ""]

    # Prose: best and worst by mean across methods
    numeric_cols = [c for c in pivot.columns if c != "Reference"]
    if numeric_cols:
        pivot["_mean"] = pivot[numeric_cols].mean(axis=1)
        best_ref = pivot.loc[pivot["_mean"].idxmax(), "Reference"]
        worst_ref = pivot.loc[pivot["_mean"].idxmin(), "Reference"]
        lines.append(
            f"Best-performing reference (mean across methods): **{best_ref}**. "
            f"Lowest: **{worst_ref}**."
        )

    return "\n".join(lines) + "\n"


def section_celltype_performance(rankings_path):
    """Hard/easy cell type split from rankings_best.tsv, key=subclass."""
    df = load_optional(rankings_path)
    if df is None:
        return "> *Not available — rerun pipeline.*\n"

    sub = df[df["key"] == "subclass"].copy()
    if sub.empty:
        return "> *No subclass data in rankings_best.tsv.*\n"

    sub["ref_short"] = sub["reference"].apply(shorten_ref)

    lines = []

    # --- Well-classified ---
    well = sub[(sub["mean_f1_across_studies"] >= 0.90) & (sub["n_studies"] >= 3)].copy()
    well = well.sort_values("mean_f1_across_studies", ascending=False)
    lines.append("#### Consistently well-classified (mean_f1 ≥ 0.90, ≥ 3 studies)\n")
    if well.empty:
        lines.append("*None meeting criteria.*\n")
    else:
        tbl = well[["label", "method", "ref_short", "subsample_ref",
                    "mean_f1_across_studies", "n_studies"]].copy()
        tbl = tbl.rename(columns={
            "ref_short": "best_reference",
            "method": "best_method",
            "subsample_ref": "best_subsample",
            "mean_f1_across_studies": "mean_f1",
        })
        tbl["mean_f1"] = tbl["mean_f1"].round(3)
        lines.append(df_to_md_table(tbl))
        lines.append("")

    # --- Hard cell types ---
    hard = sub[sub["mean_f1_across_studies"] < 0.75].copy()
    hard = hard.sort_values("mean_f1_across_studies")
    lines.append("#### Hardest cell types (mean_f1 < 0.75)\n")
    if hard.empty:
        lines.append("*None meeting criteria.*\n")
    else:
        def make_notes(row):
            notes = []
            if row["n_studies"] < 3:
                notes.append(f"rare (<{row['n_studies']} studies)")
            return "; ".join(notes) if notes else "—"

        hard["notes"] = hard.apply(make_notes, axis=1)
        tbl = hard[["label", "mean_f1_across_studies", "std_f1_across_studies",
                    "n_studies", "notes"]].copy()
        tbl = tbl.rename(columns={
            "mean_f1_across_studies": "mean_f1",
            "std_f1_across_studies": "std_f1",
        })
        tbl["mean_f1"] = tbl["mean_f1"].round(3)
        tbl["std_f1"] = tbl["std_f1"].round(3)
        lines.append(df_to_md_table(tbl))
        lines.append("")

    return "\n".join(lines) + "\n"


def section_study_variance(sv_path):
    """Study variance table, top 10 by std_f1 at key=subclass, cutoff=0.0."""
    df = load_optional(sv_path)
    if df is None:
        return None  # caller will skip section

    sub = df[(df["key"] == "subclass") & (df["cutoff"] == 0.0)].copy()
    if sub.empty:
        return "> *No subclass/cutoff=0.0 data in study_variance_summary.tsv.*\n"

    top10 = sub.sort_values("std_f1", ascending=False).head(10)
    cols = ["label", "n_studies", "frac_studies", "mean_f1", "std_f1", "min_f1", "max_f1"]
    cols = [c for c in cols if c in top10.columns]
    tbl = top10[cols].copy()
    for col in ["frac_studies", "mean_f1", "std_f1", "min_f1", "max_f1"]:
        if col in tbl.columns:
            tbl[col] = tbl[col].round(3)

    lines = [df_to_md_table(tbl), ""]

    # Prose
    most_variable = top10.iloc[0]["label"] if not top10.empty else "N/A"
    lines.append(
        f"Cell type with highest study-to-study variability: **{most_variable}** "
        f"(std_f1 = {top10.iloc[0]['std_f1']:.3f})."
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Per-dataset section
# ---------------------------------------------------------------------------

def dataset_section(name, base):
    """Generate the full markdown section for one dataset."""
    lines = []
    lines.append(f"## Dataset: {name}\n")

    # --- Summary statistics from sample_results_summary ---
    sr_path = os.path.join(base, "aggregated_results", "files", "sample_results_summary.tsv")
    sr = load_optional(sr_path)

    if sr is not None:
        n_studies_path = os.path.join(base, "aggregated_results", "files", "study_factor_summary.tsv")
        sf = load_optional(n_studies_path)
        n_studies = len(sf) - 1 if sf is not None else "?"  # subtract header duplicates if any

        # Better: count unique studies from sample_results if available
        # study_factor_summary rows = studies
        if sf is not None:
            n_studies = len(sf)

        methods = sorted(sr["method"].dropna().unique().tolist())
        cutoffs = sorted(sr["cutoff"].dropna().unique().tolist())
        refs = sr["reference"].dropna().unique().tolist()
        subsamples = sorted(sr["subsample_ref"].dropna().unique().tolist())

        lines.append(
            f"**Studies:** {n_studies}  "
            f"**Methods:** {', '.join(methods)}  \n"
            f"**Cutoffs evaluated:** {min(cutoffs)} – {max(cutoffs)}  "
            f"**References:** {len(refs)}  "
            f"**Subsample sizes:** {', '.join(str(int(s)) for s in subsamples)}\n"
        )
    else:
        lines.append("> *aggregated_results not available — rerun pipeline.*\n")

    # --- Models dir ---
    models_dir = find_models_dir(base)

    # --- Method comparison ---
    lines.append("### Method Comparison (model-adjusted marginal means)\n")
    lines.append(section_method_comparison(models_dir))

    # --- Cutoff sensitivity ---
    lines.append("### Cutoff Sensitivity\n")
    lines.append(section_cutoff_sensitivity(models_dir))

    # --- Reference comparison ---
    lines.append("### Reference Comparison (subclass, model-adjusted)\n")
    lines.append(section_reference_comparison(models_dir))

    # --- Cell-type performance ---
    rankings_path = os.path.join(base, "celltype_rankings", "rankings", "rankings_best.tsv")
    lines.append("### Cell-Type Performance (subclass, best config per label)\n")
    lines.append(section_celltype_performance(rankings_path))

    # --- Study variance ---
    sv_path = os.path.join(base, "study_variance", "study_variance", "study_variance_summary.tsv")
    sv_section = section_study_variance(sv_path)
    if sv_section is not None:
        lines.append("### Study Variance\n")
        lines.append(sv_section)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reproducible results summary markdown"
    )
    parser.add_argument("--base_dir", default="2024-07-01",
                        help="Base directory containing dataset subdirectories")
    parser.add_argument("--outfile", default="docs/results_summary.md",
                        help="Output markdown file")
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = discover_datasets(args.base_dir)
    if not datasets:
        print(f"No datasets found under {args.base_dir}/*/100/dataset_id/SCT/gap_false/")
        return

    print(f"Found {len(datasets)} dataset(s): {[n for n, _ in datasets]}")

    lines = []
    lines.append("# Cell-Type Annotation Benchmarking: Results Summary\n")
    lines.append(
        "Automated summary generated by `bin/generate_results_summary.py`. "
        "Evaluates cell-type annotation accuracy across methods (scvi, seurat), "
        "confidence cutoffs (0.0–0.75), reference atlases, and subsample sizes "
        f"for datasets found in `{args.base_dir}/`.\n"
    )
    lines.append("---\n")

    for name, base in datasets:
        print(f"  Processing {name} ...")
        lines.append(dataset_section(name, base))
        lines.append("\n---\n")

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with open(args.outfile, "w") as fh:
        fh.write("\n".join(lines))

    print(f"\nWrote {args.outfile}")


if __name__ == "__main__":
    main()
