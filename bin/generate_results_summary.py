#!/usr/bin/env python3
"""
generate_results_summary.py

Reads evaluation outputs for old-pipeline results (scvi + seurat) and writes a
structured markdown summary.

Usage:
    python bin/generate_results_summary.py \
        --base_dir 2024-07-01 \
        --outfile docs/results_summary.md
"""

import argparse
import glob
import math
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KEY_ORDER = ["global", "family", "class", "subclass"]

REF_SHORT = {
    "whole cortex": "Whole cortex",
    "An integrated transcriptomic and epigenomic atlas of mouse primary motor cortex cell types": "Motor cortex",
    "Single-cell RNA-seq for all cortical  hippocampal regions 10x": "Cortical+Hipp. 10x",
    "Single-cell RNA-seq for all cortical  hippocampal regions SMART-Seq v4": "Cortical+Hipp. SSv4",
    "Single-cell RNA-seq for all cortical & hippocampal regions (10x)": "Cortical+Hipp. 10x",
    "Single-cell RNA-seq for all cortical & hippocampal regions (SMART-Seq v4)": "Cortical+Hipp. SSv4",
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

# Covariate column name → EMM file stem
COVARIATE_FILES = {
    "treatment_state": "treatment_emmeans_summary.tsv",
    "sex":             "sex_emmeans_summary.tsv",
    "disease_state":   "disease_state_emmeans_summary.tsv",
    "region_match":    "region_match_emmeans_summary.tsv",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def shorten_ref(name):
    if name in REF_SHORT:
        return REF_SHORT[name]
    if len(str(name)) > 35:
        return str(name)[:34] + "…"
    return str(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_optional(path):
    if os.path.exists(path):
        return pd.read_csv(path, sep="\t")
    gz = path + ".gz"
    if os.path.exists(gz):
        return pd.read_csv(gz, sep="\t")
    return None


def find_models_dir(base):
    pattern = os.path.join(base, "aggregated_models", "*", "files")
    matches = glob.glob(pattern)
    return matches[0] if matches else None


def find_model_formula(base):
    pattern = os.path.join(base, "aggregated_models", "*")
    matches = [m for m in glob.glob(pattern) if os.path.isdir(m)]
    if matches:
        return os.path.basename(matches[0]).replace("_", " ").replace("~", "~")
    return "unknown"


def detect_organism(path):
    if "mus_musculus" in path:
        return "mus_musculus"
    if "homo_sapiens" in path:
        return "homo_sapiens"
    return "unknown"


def detect_pipeline(base):
    """Return 'new' if scvi_rf or scvi_knn are present, else 'old'."""
    path = os.path.join(base, "aggregated_results", "files", "sample_results_summary.tsv")
    if not os.path.exists(path):
        path = path + ".gz"
    df = load_optional(path)
    if df is None or "method" not in df.columns:
        return "old"
    methods = set(df["method"].dropna().unique())
    if methods & {"scvi_rf", "scvi_knn"}:
        return "new"
    return "old"


def discover_datasets(base_dir):
    pattern = os.path.join(base_dir, "*/100/dataset_id/SCT/gap_false")
    matches = sorted(glob.glob(pattern))
    datasets = []
    for m in matches:
        parts = m.split(os.sep)
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
    cols = df.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    rows   = ["| " + " | ".join(str(v) for v in r) + " |" for _, r in df.iterrows()]
    return "\n".join([header, sep] + rows)


def classify_failure_mode(prec, rec, std_f1):
    modes = []
    if prec > 0.80 and rec < 0.50:
        modes.append("Label escape")
    elif prec < 0.50 and rec > 0.70:
        modes.append("Over-prediction")
    elif prec < 0.50 and rec < 0.50:
        modes.append("Coverage failure")
    if std_f1 > 0.20:
        modes.append("Study variance")
    return "; ".join(modes) if modes else "—"


# ---------------------------------------------------------------------------
# Section generators
# ---------------------------------------------------------------------------

def section_study_cohort(base, organism):
    path = os.path.join(base, "aggregated_results", "files", "study_factor_summary.tsv")
    df = load_optional(path)
    if df is None:
        return ""

    # Select display columns based on what's present
    priority = ["study", "treatment", "disease", "sex", "query_region",
                "number query samples", "number cells", "number unique subclasses"]
    cols = [c for c in priority if c in df.columns]
    tbl = df[cols].copy()
    tbl.columns = [c.replace("number query samples", "Samples")
                    .replace("number cells", "Cells")
                    .replace("number unique subclasses", "Subclasses")
                    .replace("query_region", "Region")
                   for c in tbl.columns]

    return df_to_md_table(tbl) + "\n"


def section_method_comparison(models_dir):
    if models_dir is None:
        return ""
    df = load_optional(os.path.join(models_dir, "method_emmeans_summary.tsv"))
    if df is None:
        return ""

    keys_present = [k for k in KEY_ORDER if k in df["key"].values]
    records = []
    for key in keys_present:
        sub = df[df["key"] == key]
        row = {"key": key}
        for _, r in sub.iterrows():
            row[r["method"]] = fmt_emmean(r)
        records.append(row)

    return df_to_md_table(pd.DataFrame(records).fillna("—")) + "\n"


def section_method_contrasts(models_dir):
    if models_dir is None:
        return ""
    df = load_optional(os.path.join(models_dir, "method_emmeans_estimates.tsv"))
    if df is None:
        return ""

    keys_present = [k for k in KEY_ORDER if k in df["key"].values]
    chunks = []
    for key in keys_present:
        sub = df[df["key"] == key][["contrast", "odds.ratio", "p.value"]].copy()
        sub.insert(0, "key", key)
        sub["odds.ratio"] = sub["odds.ratio"].round(3)
        def fmt_pval(p):
            if pd.isna(p):
                return "—"
            if p == 0:
                return "< 1e-300"
            if p < 0.001:
                return f"< 1e{int(math.floor(math.log10(p)))+1}"
            return f"{p:.3f}"
        sub["p.value"] = sub["p.value"].apply(fmt_pval)
        chunks.append(sub)

    return df_to_md_table(pd.concat(chunks, ignore_index=True)) + "\n"


def section_cutoff_sensitivity(models_dir):
    if models_dir is None:
        return ""
    df = load_optional(os.path.join(models_dir, "method_cutoff_effects.tsv"))
    if df is None:
        return ""

    # Detect F1 column name (fit or response)
    f1_col = "fit" if "fit" in df.columns else "response"
    keys_present = [k for k in KEY_ORDER if k in df["key"].values]
    chunks = []
    for key in keys_present:
        sub = df[df["key"] == key].copy()
        pivot = sub.pivot_table(index="cutoff", columns="method", values=f1_col).reset_index()
        pivot.columns.name = None
        pivot.insert(0, "key", key)
        pivot["cutoff"] = pivot["cutoff"].round(2)
        for col in pivot.columns[2:]:
            pivot[col] = pivot[col].round(3)
        chunks.append(pivot)

    return df_to_md_table(pd.concat(chunks, ignore_index=True)) + "\n"


def section_reference_comparison(models_dir):
    if models_dir is None:
        return ""
    df = load_optional(os.path.join(models_dir, "reference_method_emmeans_summary.tsv"))
    if df is None:
        return ""

    df["ref_short"] = df["reference"].apply(shorten_ref)
    keys_present = [k for k in KEY_ORDER if k in df["key"].values]
    chunks = []
    for key in keys_present:
        sub = df[df["key"] == key].copy()
        pivot = sub.pivot_table(index="ref_short", columns="method", values="response").reset_index()
        pivot.columns.name = None
        pivot.insert(0, "key", key)
        for col in pivot.columns[2:]:
            pivot[col] = pivot[col].round(3)
        chunks.append(pivot)

    return df_to_md_table(pd.concat(chunks, ignore_index=True)) + "\n"


def section_subsample_ref(models_dir):
    if models_dir is None:
        return ""
    df = load_optional(os.path.join(models_dir, "subsample_ref_emmeans_summary.tsv"))
    if df is None:
        return ""

    keys_present = [k for k in KEY_ORDER if k in df["key"].values]
    records = []
    for key in keys_present:
        sub = df[df["key"] == key]
        for _, r in sub.iterrows():
            records.append({
                "key": key,
                "subsample_ref": r["subsample_ref"],
                "EMM": f"{r['response']:.3f} [{r['asymp.LCL']:.3f}–{r['asymp.UCL']:.3f}]",
            })

    return df_to_md_table(pd.DataFrame(records)) + "\n"


def section_covariates(models_dir):
    if models_dir is None:
        return ""
    lines = []
    for covariate, fname in COVARIATE_FILES.items():
        path = os.path.join(models_dir, fname)
        df = load_optional(path)
        if df is None:
            continue

        # Covariate column is whatever is not key/SE/df/response/asymp.*
        fixed_cols = {"response", "SE", "df", "asymp.LCL", "asymp.UCL", "key"}
        cov_col = next((c for c in df.columns if c not in fixed_cols), None)
        if cov_col is None:
            continue

        keys_present = [k for k in KEY_ORDER if k in df["key"].values]
        records = []
        for key in keys_present:
            sub = df[df["key"] == key]
            for _, r in sub.iterrows():
                records.append({
                    "key": key,
                    covariate: r[cov_col] if cov_col in r.index else r.iloc[0],
                    "EMM": f"{r['response']:.3f} [{r['asymp.LCL']:.3f}–{r['asymp.UCL']:.3f}]",
                })

        if records:
            lines.append(f"**{covariate}**\n")
            lines.append(df_to_md_table(pd.DataFrame(records)) + "\n")

    return "\n".join(lines)


def section_study_variance(sv_path):
    df = load_optional(sv_path)
    if df is None:
        return None

    df_cut = df[df["cutoff"] == 0.0].copy()
    if df_cut.empty:
        return ""

    keys_present = [k for k in KEY_ORDER if k in df_cut["key"].values]
    has_prec_rec = "mean_precision" in df_cut.columns and "mean_recall" in df_cut.columns

    lines = []
    for key in keys_present:
        sub = df_cut[df_cut["key"] == key].copy()

        well = sub[sub["mean_f1"] >= 0.85].sort_values("mean_f1", ascending=False)
        hard = sub[(sub["mean_f1"] < 0.75) | (sub["std_f1"] > 0.20)].sort_values("mean_f1")

        for label, subset in [("Well-classified (mean F1 ≥ 0.85)", well),
                               ("Hard / high-variance (mean F1 < 0.75 or std > 0.20)", hard)]:
            if subset.empty:
                continue
            cols = ["label", "n_studies", "mean_f1", "std_f1"]
            if has_prec_rec:
                cols += ["mean_precision", "mean_recall"]
            tbl = subset[cols].copy()
            tbl.insert(0, "key", key)
            for c in ["mean_f1", "std_f1", "mean_precision", "mean_recall"]:
                if c in tbl.columns:
                    tbl[c] = tbl[c].round(3)
            if has_prec_rec and label.startswith("Hard"):
                tbl["failure_mode"] = [
                    classify_failure_mode(r["mean_precision"], r["mean_recall"], r["std_f1"])
                    for _, r in subset.iterrows()
                ]
            lines.append(f"**{key} — {label}**\n")
            lines.append(df_to_md_table(tbl) + "\n")

    return "\n".join(lines)


def section_celltype_rankings(rankings_path):
    df = load_optional(rankings_path)
    if df is None:
        return ""

    df["ref_short"] = df["reference"].apply(shorten_ref)
    keys_present = [k for k in KEY_ORDER if k in df["key"].values]
    chunks = []
    for key in keys_present:
        sub = df[df["key"] == key].copy()
        cols = ["label", "method", "ref_short", "subsample_ref",
                "mean_f1_across_studies", "win_fraction", "n_studies"]
        if "mean_support" in sub.columns:
            cols.append("mean_support")
        tbl = sub[cols].copy()
        tbl.insert(0, "key", key)
        tbl["mean_f1_across_studies"] = tbl["mean_f1_across_studies"].round(3)
        tbl["win_fraction"] = tbl["win_fraction"].round(3)
        chunks.append(tbl)

    return df_to_md_table(pd.concat(chunks, ignore_index=True)
                          .rename(columns={"ref_short": "reference"})) + "\n"


def section_pareto(pareto_path):
    df = load_optional(pareto_path)
    if df is None:
        return ""

    pareto = df[df["pareto"] == True].copy() if "pareto" in df.columns else df
    if "reference" in pareto.columns:
        pareto["reference"] = pareto["reference"].apply(shorten_ref)

    cols = [c for c in ["key", "method", "method_display", "reference", "subsample_ref",
                         "mean_f1", "total_duration_hrs", "total_memory_gb"] if c in pareto.columns]
    tbl = pareto[cols].copy()
    for c in ["mean_f1", "total_duration_hrs", "total_memory_gb"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].round(3)

    return df_to_md_table(tbl) + "\n"


def section_comptime(comptime_path):
    df = load_optional(comptime_path)
    if df is None:
        return ""

    cols = [c for c in ["method", "step", "subsample_ref", "mean_duration", "mean_memory"]
            if c in df.columns]
    tbl = df[cols].copy()
    for c in ["mean_duration", "mean_memory"]:
        if c in tbl.columns:
            tbl[c] = tbl[c].round(3)

    return df_to_md_table(tbl) + "\n"


def section_reference_coverage(organism, pipeline="old"):
    if organism == "mus_musculus":
        subdir = "tabulamuris-mus-musculus"
        org_prefix = "mus_musculus"
    elif pipeline == "old":
        subdir = "no-ma-et-al-homo-sapiens"
        org_prefix = "homo_sapiens"
    else:
        subdir = "ma-et-al-homo-sapiens"
        org_prefix = "homo_sapiens"

    coverage_base = os.path.join(PROJECT_DIR, "assets", "ref_coverage", subdir)
    lines = []
    for key in KEY_ORDER:
        path = os.path.join(coverage_base, f"{org_prefix}_{key}_ref_support.tsv")
        df = load_optional(path)
        if df is None:
            continue
        # First column is the label, rest are references
        df = df.copy()
        label_col = df.columns[0]
        ref_cols = df.columns[1:]
        short_cols = {c: shorten_ref(c) for c in ref_cols}
        df = df.rename(columns=short_cols)
        # Convert float to int where possible
        for c in df.columns[1:]:
            df[c] = df[c].apply(lambda x: int(x) if pd.notna(x) and float(x) == int(float(x)) else x)
        lines.append(f"**{key}**\n")
        lines.append(df_to_md_table(df) + "\n")

    return "\n".join(lines) if lines else ""



def section_reference_ranking(models_dir):
    """Rank references by mean EMM across all methods and taxonomy keys."""
    if models_dir is None:
        return ""
    ref_method = load_optional(os.path.join(models_dir, "reference_method_emmeans_summary.tsv"))
    if ref_method is None:
        return ""

    ref_method["ref_short"] = ref_method["reference"].apply(shorten_ref)
    mean_emm = (ref_method.groupby("ref_short")["response"]
                .mean().reset_index()
                .rename(columns={"response": "mean_emm", "ref_short": "reference"})
                .sort_values("mean_emm", ascending=False))
    mean_emm["mean_emm"] = mean_emm["mean_emm"].round(3)
    return df_to_md_table(mean_emm) + "\n"


def section_label_cutoff_sensitivity(base):
    path = os.path.join(base, "cutoff_plots", "label_f1_plots", "label_cutoff_summary.tsv")
    df = load_optional(path)
    if df is None:
        return ""

    cutoffs_show = [0.0, 0.25, 0.50, 0.75]
    lines = []
    for key in KEY_ORDER:
        sub = df[df["key"] == key].copy()
        if sub.empty:
            continue
        grp = sub.groupby(["label", "method", "cutoff"])["f1_score_mean"].mean().reset_index()
        pivot = grp.pivot_table(index=["label", "method"], columns="cutoff",
                                values="f1_score_mean").reset_index()
        pivot.columns.name = None
        available = [c for c in cutoffs_show if c in pivot.columns]
        col_names = {c: f"F1({c})" for c in available}
        pivot = pivot.rename(columns=col_names)
        display_cols = ["label", "method"] + [f"F1({c})" for c in available]
        pivot = pivot[[c for c in display_cols if c in pivot.columns]]
        for c in display_cols[2:]:
            if c in pivot.columns:
                pivot[c] = pivot[c].round(3)
        f1_0_col = "F1(0.0)" if "F1(0.0)" in pivot.columns else display_cols[2]
        pivot = pivot.sort_values(f1_0_col, ascending=False)
        lines.append(f"### {key.capitalize()}\n")
        lines.append(df_to_md_table(pivot.reset_index(drop=True)) + "\n")

        # Most cutoff-sensitive per method
        if "F1(0.0)" in pivot.columns and "F1(0.75)" in pivot.columns:
            pivot = pivot.copy()
            pivot["Drop"] = (pivot["F1(0.0)"] - pivot["F1(0.75)"]).round(3)
            top5 = (pivot.groupby("method", group_keys=False)
                    .apply(lambda g: g.nlargest(5, "Drop"))
                    .reset_index(drop=True))
            sens = top5[["method", "label", "F1(0.0)", "F1(0.75)", "Drop"]]
            lines.append("**Most cutoff-sensitive cell types (F1(0) → F1(0.75) drop):**\n")
            lines.append(df_to_md_table(sens) + "\n")

        # Low-F1 types (F1 < 0.5 at cutoff=0): full precision/recall across all cutoffs
        has_prec_rec = "precision_mean" in sub.columns and "recall_mean" in sub.columns
        if has_prec_rec and f1_0_col in pivot.columns:
            low_f1_labels = pivot.loc[pivot[f1_0_col] < 0.5, "label"].unique()
            if len(low_f1_labels) > 0:
                low_sub = sub[sub["label"].isin(low_f1_labels)].copy()
                grp_pr = (low_sub.groupby(["label", "method", "cutoff"])
                          [["f1_score_mean", "precision_mean", "recall_mean"]]
                          .mean().reset_index())
                grp_pr = grp_pr.sort_values(["label", "method", "cutoff"])
                for c in ["f1_score_mean", "precision_mean", "recall_mean"]:
                    grp_pr[c] = grp_pr[c].round(3)
                grp_pr = grp_pr.rename(columns={
                    "f1_score_mean": "F1", "precision_mean": "precision", "recall_mean": "recall"
                })
                lines.append("**Low-F1 cell types at cutoff=0 (F1 < 0.5) — precision/recall across all cutoffs:**\n")
                lines.append(df_to_md_table(grp_pr) + "\n")

    return "\n".join(lines)


def section_hippocampal_contamination(base):
    contam_path = os.path.join(base, "aggregated_results", "files", "contamination.tsv")
    cutoff_path = os.path.join(base, "cutoff_plots", "label_f1_plots", "label_cutoff_summary.tsv")
    contam = load_optional(contam_path)
    cutoff_df = load_optional(cutoff_path)
    if contam is None:
        return "Hippocampal contamination analysis not available — old pipeline does not include ref_support=0 filtering.\n"

    hippo_labels = ["Hippocampal neuron", "DG", "CA1-ProS", "CA3"]
    mask = contam["label"].isin(hippo_labels)
    if "key" in contam.columns:
        mask = mask & (contam["key"] == "class")
    hippo = contam[mask]

    if hippo.empty:
        return "No hippocampal contamination detected.\n"

    spurious = (hippo.groupby(["cutoff", "method"])["predicted_support"]
                .mean().reset_index()
                .rename(columns={"predicted_support": "mean_spurious_per_query"}))
    spurious["mean_spurious_per_query"] = spurious["mean_spurious_per_query"].round(3)

    if cutoff_df is not None:
        cls = cutoff_df[cutoff_df["key"] == "class"]
        non_hippo_recall = (cls[~cls["label"].isin(hippo_labels)]
                            .groupby(["cutoff", "method"])["recall_mean"]
                            .mean().reset_index()
                            .rename(columns={"recall_mean": "mean_recall_non_hippo"}))
        non_hippo_recall["mean_recall_non_hippo"] = non_hippo_recall["mean_recall_non_hippo"].round(3)
        tbl = spurious.merge(non_hippo_recall, on=["cutoff", "method"], how="left")
    else:
        tbl = spurious

    tbl = tbl.sort_values(["method", "cutoff"])
    return df_to_md_table(tbl) + "\n"


def section_assay_exploration(base):
    path = os.path.join(base, "assay_exploration", "assay_model", "files",
                        "assay_emmeans_summary.tsv")
    df = load_optional(path)
    if df is None:
        return ""

    keys_present = [k for k in KEY_ORDER if k in df["key"].values]
    records = []
    for key in keys_present:
        sub = df[df["key"] == key]
        for _, r in sub.iterrows():
            records.append({
                "key": key,
                "ref_type": r["ref_type"],
                "query_type": r["query_type"],
                "EMM": f"{r['response']:.3f} [{r['asymp.LCL']:.3f}–{r['asymp.UCL']:.3f}]",
            })

    return df_to_md_table(pd.DataFrame(records)) + "\n"


# ---------------------------------------------------------------------------
# Per-dataset section
# ---------------------------------------------------------------------------

def dataset_section(name, base, pipeline=None):
    organism = detect_organism(base)
    formula  = find_model_formula(base)
    models_dir = find_models_dir(base)
    if pipeline is None:
        pipeline = detect_pipeline(base)

    pipeline_label = "new (scvi_rf + scvi_knn + seurat)" if pipeline == "new" else "old (scvi + seurat)"

    lines = []
    lines.append(f"## {name}\n")
    lines.append(f"**Organism:** {organism}  ")
    lines.append(f"**Model formula:** `{formula}`  ")
    lines.append(f"**Pipeline:** {pipeline_label}\n")

    # Study cohort
    cohort = section_study_cohort(base, organism)
    if cohort:
        lines.append("### Study Cohort\n")
        lines.append(cohort)

    # Method comparison
    mc = section_method_comparison(models_dir)
    if mc:
        lines.append("### Method Performance (model-adjusted marginal means)\n")
        lines.append(mc)

    # Method contrasts
    contrasts = section_method_contrasts(models_dir)
    if contrasts:
        lines.append("### Method Pairwise Contrasts\n")
        lines.append(contrasts)

    # Cutoff sensitivity
    cutoff = section_cutoff_sensitivity(models_dir)
    if cutoff:
        lines.append("### Cutoff Sensitivity (method × cutoff EMMs)\n")
        lines.append(cutoff)

    # Reference comparison
    ref = section_reference_comparison(models_dir)
    if ref:
        lines.append("### Reference × Method Performance\n")
        lines.append(ref)

    # Reference ranking (mean EMM across methods and keys)
    ref_rank = section_reference_ranking(models_dir)
    if ref_rank:
        lines.append("### Reference Ranking (mean EMM across methods and keys)\n")
        lines.append(ref_rank)

    # Subsample
    sub = section_subsample_ref(models_dir)
    if sub:
        lines.append("### Reference Subsample Size\n")
        lines.append(sub)

    # Covariates
    cov = section_covariates(models_dir)
    if cov:
        lines.append("### Biological Covariates\n")
        lines.append(cov)

    # Study variance (all keys from combined TSV produced by plot_study_variance.py)
    sv_path = os.path.join(base, "study_variance", "study_variance", "study_variance_summary.tsv")
    sv = section_study_variance(sv_path)
    if sv:
        lines.append("### Between-Study Heterogeneity\n")
        lines.append(sv)

    # Cell-type rankings
    rankings_path = os.path.join(base, "celltype_rankings", "rankings", "rankings_best.tsv")
    rankings = section_celltype_rankings(rankings_path)
    if rankings:
        lines.append("### Cell-Type Rankings (best config per label)\n")
        lines.append(rankings)

    # Reference coverage
    coverage = section_reference_coverage(organism, pipeline=pipeline)
    if coverage:
        lines.append("### Reference Cell-Type Coverage\n")
        lines.append(coverage)

    # Per-cell-type cutoff sensitivity (new pipeline only)
    if pipeline == "new":
        label_cutoff = section_label_cutoff_sensitivity(base)
        if label_cutoff:
            lines.append("### Per-Cell-Type Cutoff Sensitivity\n")
            lines.append(label_cutoff)

    # Hippocampal contamination (mouse + new pipeline)
    if organism == "mus_musculus":
        if pipeline == "new":
            contam = section_hippocampal_contamination(base)
            if contam:
                lines.append("### Hippocampal Contamination\n")
                lines.append(contam)

    # Assay exploration (mouse only)
    if organism == "mus_musculus":
        assay = section_assay_exploration(base)
        if assay:
            lines.append("### Assay Exploration (mouse only)\n")
            lines.append(assay)

    # Pareto
    pareto_path = os.path.join(base, "celltype_rankings", "config_pareto", "config_pareto_table.tsv")
    pareto = section_pareto(pareto_path)
    if pareto:
        lines.append("### Pareto-Optimal Configurations\n")
        lines.append(pareto)

    # Compute time
    comptime_path = os.path.join(base, "comptime_plots", "comptime_summary.tsv")
    comptime = section_comptime(comptime_path)
    if comptime:
        lines.append("### Computational Time\n")
        lines.append(comptime)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reproducible results summary markdown (old pipeline)"
    )
    parser.add_argument("--base_dir", default=None,
                        help="Base directory containing dataset subdirectories "
                             "(globs for */100/dataset_id/SCT/gap_false)")
    parser.add_argument("--results_dir", default=None,
                        help="Path to a single dataset directory "
                             "(e.g. 2024-07-01/homo_sapiens_main_branch/100/dataset_id/SCT/gap_false). "
                             "Overrides --base_dir.")
    parser.add_argument("--outfile", default="summary.md",
                        help="Output markdown file path.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.results_dir:
        # Single-dataset mode: derive a display name from the path
        path = args.results_dir.rstrip("/")
        # Name = directory component right after the date prefix (e.g. homo_sapiens_main_branch)
        parts = path.split(os.sep)
        name = parts[-5] if len(parts) >= 5 else parts[0]
        datasets = [(name, path)]
    elif args.base_dir:
        datasets = discover_datasets(args.base_dir)
    else:
        datasets = discover_datasets("2024-07-01")

    if not datasets:
        src = args.results_dir or args.base_dir or "2024-07-01"
        print(f"No datasets found under {src}")
        return

    print(f"Found {len(datasets)} dataset(s): {[n for n, _ in datasets]}")

    # Detect pipeline from first dataset
    first_pipeline = detect_pipeline(datasets[0][1]) if datasets else "old"

    if first_pipeline == "old":
        title = "# Cell-Type Annotation Benchmarking: Results Summary (Old Pipeline)\n"
        warning = (
            "> WARNING: Old pipeline results (scVI monolithic + Seurat). No ref_support=0 filtering. "
            "Per-cell-type cutoff sensitivity tables unavailable. "
            "Compare with new pipeline results before drawing conclusions.\n"
        )
    else:
        title = "# Cell-Type Annotation Benchmarking: Results Summary\n"
        warning = ""

    lines = [
        title,
        warning,
        f"Generated from: `{args.results_dir or args.base_dir}/`\n",
        "---\n",
    ]

    for name, base in datasets:
        print(f"  Processing {name} ...")
        pipeline = detect_pipeline(base)
        lines.append(dataset_section(name, base, pipeline=pipeline))
        lines.append("\n---\n")

    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    with open(args.outfile, "w") as fh:
        fh.write("\n".join(lines))

    print(f"Wrote {args.outfile}")


if __name__ == "__main__":
    main()
