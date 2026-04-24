
from pathlib import Path
import os
import sys
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
from scipy.sparse import csr_matrix
import warnings
import json
import argparse
import ast
import re


# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--pipeline_results', type=str, nargs = "+",
                        help="files containing f1 results with params")
    parser.add_argument('--metadata_dir', type=str, default=None,
                        help="Directory containing per-study metadata TSVs for overriding sex/disease")
    parser.add_argument('--remove_outliers', type=str, nargs='*', default=None,
                        help="List of study names to exclude from all downstream analyses")
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args


# Column name aliases for sex and disease across studies
SEX_COLS    = ['sex']
DISEASE_COLS = ['disease', 'group', 'pathological diagnosis', 'treatment']

# Per-study disease value normalisation: map raw value -> 'control' or leave as disease label
DISEASE_CONTROL_VALUES = {
    'control', 'Control', 'reference subject role', 'Unaffected', 'no',
    'vehicle (PBS) for 12h', 'PN',   # GSE174332: PN = peripheral nerve (normal)
}

LABEL_ONLY_COLUMNS = [
    "label", "f1_score", "precision", "recall", "support",
    "accuracy", "predicted_support", "ref_support",
]

def _normalise_disease(val):
    """Return 'control' if val is a known control label, else the original value."""
    if pd.isna(val):
        return np.nan
    return 'control' if str(val) in DISEASE_CONTROL_VALUES else str(val)

def _normalise_sex(val, study=None):
    """Lowercase and expand single-letter codes; handle per-study numeric encodings."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    # Mathys-2023 (ROSMAP): 0 = male, 1 = female
    if study == 'Mathys-2023':
        numeric_map = {'0': 'male', '1': 'female'}
        if s in numeric_map:
            return numeric_map[s]
    mapping = {'m': 'male', 'f': 'female', 'fem': 'female'}
    return mapping.get(s, s)

def load_sample_metadata(metadata_dir):
    """
    Walk metadata_dir/<study>/<study>_sample_meta_std.tsv and build a dict
    mapping str(sample_id) -> {'sex': ..., 'disease': ...}.
    Returns empty dict if metadata_dir is None or missing.
    """
    if not metadata_dir or not os.path.isdir(metadata_dir):
        return {}

    sample_meta = {}
    for study in os.listdir(metadata_dir):
        study_path = os.path.join(metadata_dir, study)
        if not os.path.isdir(study_path):
            continue
        tsv_files = [f for f in os.listdir(study_path) if f.endswith('.tsv')]
        if not tsv_files:
            continue
        df = pd.read_csv(os.path.join(study_path, tsv_files[0]), sep='\t', dtype=str)

        # Resolve sex column
        sex_col = next((c for c in SEX_COLS if c in df.columns), None)
        # Resolve disease column
        disease_col = next((c for c in DISEASE_COLS if c in df.columns), None)

        for _, row in df.iterrows():
            sid = str(row['sample_id']) if 'sample_id' in df.columns else None
            if sid is None:
                continue
            sex_val     = _normalise_sex(row[sex_col], study=study) if sex_col     else np.nan
            disease_val = _normalise_disease(row[disease_col])       if disease_col else np.nan
            sample_meta[sid] = {'sex': sex_val, 'disease': disease_val}

    return sample_meta


def apply_sample_metadata(df, sample_meta):
    """
    For each row, look up sample_id (query.split('_')[1]) in sample_meta and
    fill in sex / disease where the existing value is NaN or empty.
    """
    if not sample_meta:
        return df

    # Ensure columns exist
    for col in ('sex', 'disease'):
        if col not in df.columns:
            df[col] = np.nan

    def _lookup(query, col):
        parts = str(query).split('_')
        if len(parts) < 2:
            return np.nan
        return sample_meta.get(parts[1], {}).get(col, np.nan)

    mask_sex     = df['sex'].isna()     | (df['sex'].astype(str).str.strip() == '')
    mask_disease = df['disease'].isna() | (df['disease'].astype(str).str.strip() == '')

    df.loc[mask_sex,     'sex']     = df.loc[mask_sex,     'query'].apply(lambda q: _lookup(q, 'sex'))
    df.loc[mask_disease, 'disease'] = df.loc[mask_disease, 'query'].apply(lambda q: _lookup(q, 'disease'))

    return df

def make_acronym(name):
    # Split on "_" and replace with spaces
    words = name.split("_")
    # Create acronym from the first letter of each word
    acronym = "".join(word[0].upper() for word in words if word)
    return acronym

def map_development_stage(stage):
    if pd.isna(stage):
        return None
    dev_stage_mapping_dict = {
        "HsapDv_0000083": "infant",
        "HsapDv_0000084": "toddler",
        "HsapDv_0000085": "child",
        "HsapDv_0000086": "adolescent",
        "HsapDv_0000088": "adult",
        "HsapDv_0000091": "late adult",
    }
    return dev_stage_mapping_dict[stage]

def write_factor_summary(df, factors):
    # 1. Summarize the number of unique levels for each factor
    unique_counts_df = df[factors].nunique().reset_index()
    unique_counts_df.to_csv("factor_unique_counts.tsv.gz", sep="\t", index=False, compression="gzip")

    cols = ['disease_state', 'treatment_state', 'sex']
    dfs = []

    for col in cols:
        if col in df.columns:
            # Group by factor column, then count unique sample_id
            unique_counts = (
                df.groupby(col)['query']
                .nunique()
                .reset_index()
                .rename(columns={col: 'level', 'sample': 'unique_sample_count'})
            )
            unique_counts['factor'] = col
            dfs.append(unique_counts)

    result_df = pd.concat(dfs, ignore_index=True)
    result_df.to_csv("factor_unique_sample_counts.tsv.gz", sep="\t", index=False, compression="gzip")

def update_metrics(df):
    # set metrics to nan when support is 0
    metrics = ['f1_score', 'precision',
               'recall', 'accuracy',
               'weighted_f1',
               'weighted_precision',
               'weighted_recall',
               'macro_f1',
               'macro_precision',
               'macro_recall',
               'nmi',
               'overall_accuracy',
               'ari']

    df.loc[df['support'] == 0, metrics] = None
    return df

# Function to extract and print unique factor values for each test study
def print_study_factor_table(label_results, organism):
    import pandas as pd
    if organism == "homo_sapiens":
        columns = ["disease", "sex", "dev_stage", "number query samples", "number cells", "query_region", "number unique subclasses"]
    else:
        columns = ["treatment", "genotype", "strain", "sex", "age", "query_region", "number query samples", "number cells", "number unique subclasses"]

    # Ensure DataFrames
    label_df = pd.DataFrame(label_results)
    # Determine study column name
    study_col = "study"
    if study_col not in label_df.columns:
        print("No study column found in results.")
        return

    # Prepare table
    table = []
    for study, group in label_df.groupby(study_col):
        row = {"study": study}
        for col in columns:
            if col == "number query samples":
                row[col] = group["query"].nunique()
            elif col == "number cells":
                row[col] = group.groupby("query")["total_cell_count"].first().sum()
            elif col == "number unique subclasses":
                # Find unique subclasses for this study in label_results
                n_subclasses = label_df[(label_df[study_col] == study) & (label_df["key"] == "subclass")]["label"].nunique()
                row[col] = n_subclasses
            elif col in group.columns:
                vals = group[col].dropna().unique()
                row[col] = ", ".join(map(str, vals))
        table.append(row)
    out_cols = [study_col] + columns
    # write to tsv
    table_df = pd.DataFrame(table)[out_cols]
    table_df.to_csv("study_factor_summary.tsv.gz", sep="\t", index=False, compression="gzip")


def transform_df(df, organism, sample_meta, remove_outliers):
    """
    Apply all row-level transforms (derive columns, standardize fields,
    organism-specific fixes) to a raw f1_results dataframe.
    Safe to call per-file before any cross-file concat.
    """
    df["study"] = df["query"].str.split("_").str[0]
    if remove_outliers:
        df = df[~df["study"].isin(remove_outliers)]
    if df.empty:
        return df

    if organism == "homo_sapiens" and sample_meta:
        df = apply_sample_metadata(df, sample_meta)

    df["query"] = df["query"].str.replace("_", " ")
    df["reference_acronym"] = df["reference"].apply(make_acronym)
    df["reference"] = df["reference"].str.replace("_", " ")

    qr = df["query_region"]
    rr = df["ref_region"]
    valid = qr.notna() & rr.notna()
    df["region_match"] = False
    df.loc[valid, "region_match"] = [
        q in r for q, r in zip(qr[valid].values, rr[valid].values)
    ]

    df["disease"] = np.where(df["disease"] == "Control", "control", df["disease"])
    df["disease"] = np.where(df["disease"].isnull(), "control", df["disease"])
    df["disease_state"] = np.where(df["disease"] == "control", "control", "disease")

    df["sex"] = df["sex"].str.replace(r"^M$", "male", regex=True, case=False)
    df["sex"] = df["sex"].str.replace(r"^F$", "female", regex=True, case=False)
    df["sex"] = df["sex"].str.replace(r"^feM$", "female", regex=True, case=False)

    if organism == "homo_sapiens":
        df["disease"] = np.where(df["study"] == "GSE211870", "control", df["disease"])
        df["dev_stage"] = df["dev_stage"].apply(map_development_stage)
        df["dev_stage"] = np.where(df["study"] == "rosmap", "late adult", df["dev_stage"])
        df["dev_stage"] = np.where(df["study"] == "pineda", "late adult", df["dev_stage"])
        df["sex"] = np.where(df["query"] == "lim C5382Cd", "male", df["sex"])
        df["dev_stage"] = np.where(df["query"] == "lim C5382Cd", "late adult", df["dev_stage"])

    if organism == "mus_musculus":
        df["treatment_state"] = np.where(df["treatment"].isnull(), "No treatment", "treatment")
        df["genotype"] = np.where(df["genotype"].isnull(), "wild type genotype", df["genotype"])
        df["treatment_state"] = df["treatment_state"].str.lower()

    df["disease_state"] = df["disease_state"].str.lower()
    df["sex"] = df["sex"].str.lower()
    return df


def main():
    args = parse_arguments()
    pipeline_results = args.pipeline_results

    # --- Load and concatenate input files ---
    # Only columns that are never mutated downstream are cast to category; mutated string
    # columns (disease, sex, reference, query, genotype, treatment, dev_stage) stay object.
    CATEGORICAL_COLS = {
        c: "category" for c in ["method", "key", "cutoff", "subsample_ref", "organism", "label"]
    }
    FLOAT32_COLS = {
        c: "float32" for c in [
            "f1_score", "accuracy", "precision", "recall",
            "weighted_f1", "weighted_precision", "weighted_recall",
            "macro_f1", "macro_precision", "macro_recall",
            "micro_f1", "micro_precision", "micro_recall",
            "nmi", "ari", "overall_accuracy",
        ]
    }
    READ_DTYPES = {**CATEGORICAL_COLS, **FLOAT32_COLS}

    # Load sample metadata once (cheap, disk only).
    sample_meta = {}  # populated lazily after organism is known

    sample_dfs = []
    label_dfs = []
    contamination_dfs = []
    organism = None

    # Stream per-file: transform → split into sample-slice / label-slice →
    # append. Never holds the full combined frame in memory.
    for filepath in pipeline_results:
        df = pd.read_csv(filepath, sep="\t", dtype=READ_DTYPES, low_memory=False)
        if df.empty:
            continue

        if organism is None:
            organism = df["organism"].unique()[0]
            if organism == "homo_sapiens" and args.metadata_dir:
                sample_meta = load_sample_metadata(args.metadata_dir)

        if organism == "mus_musculus":
            contam = df[(df["support"] == 0) & (df["predicted_support"] > 0)]
            if not contam.empty:
                contamination_dfs.append(contam)

        df = df[df["support"] > 0]
        if df.empty:
            continue

        df = transform_df(df, organism, sample_meta, args.remove_outliers)
        if df.empty:
            continue

        # Sample-level slice: drop label-specific columns and dedup within this file.
        sample_slice = df.drop(columns=LABEL_ONLY_COLUMNS).drop_duplicates()
        sample_slice = sample_slice[sample_slice["weighted_f1"].notnull()]
        if not sample_slice.empty:
            sample_dfs.append(sample_slice)

        # Label-level slice.
        label_slice = df[df["label"].notnull()]
        label_slice = label_slice[label_slice["f1_score"].notnull()]
        label_slice = label_slice[label_slice["label"] != "unkown"]
        if not label_slice.empty:
            label_dfs.append(label_slice)

        del df, sample_slice, label_slice

    if organism is None:
        raise RuntimeError("No non-empty input files provided to aggregate_results")

    # --- Contamination report (mouse only) ---
    if organism == "mus_musculus" and contamination_dfs:
        contamination = pd.concat(contamination_dfs, ignore_index=True)
        contamination.to_csv("contamination.tsv", sep="\t", index=False)
        del contamination
    del contamination_dfs

    # --- Sample-level results + summary ---
    sample_results = pd.concat(sample_dfs, ignore_index=True)
    del sample_dfs
    sample_results = sample_results.drop_duplicates().fillna("None")
    sample_results.to_csv("sample_results.tsv.gz", sep="\t", index=False, compression="gzip")

    weighted_metrics = [
        "weighted_f1", "weighted_precision", "weighted_recall",
        "macro_f1", "macro_precision", "macro_recall",
        "micro_f1", "micro_precision", "micro_recall",
        "nmi", "ari", "overall_accuracy"
    ]
    weighted_agg = {}
    for m in weighted_metrics:
        if m in sample_results.columns:
            sample_results[m] = pd.to_numeric(sample_results[m], errors='coerce')
            weighted_agg[f"{m}_mean"] = (m, "mean")
            weighted_agg[f"{m}_std"] = (m, "std")
    weighted_agg["count"] = ("weighted_f1", "count")
    weighted_summary = sample_results.groupby(
        ["method", "cutoff", "reference", "key", "subsample_ref"]
    ).agg(**weighted_agg).reset_index()
    weighted_summary.to_csv("sample_results_summary.tsv.gz", sep="\t", index=False, compression="gzip")
    del sample_results, weighted_summary

    # --- Label-level results + summary + factor summaries ---
    label_results = pd.concat(label_dfs, ignore_index=True)
    del label_dfs
    label_results = label_results.fillna("None")
    label_results.to_csv("label_results.tsv.gz", sep="\t", index=False, compression="gzip")

    label_metrics = ["f1_score", "precision", "recall"]
    for m in label_metrics:
        if m in label_results.columns:
            label_results[m] = pd.to_numeric(label_results[m], errors='coerce')
    label_agg = {}
    for m in label_metrics:
        if m in label_results.columns:
            label_agg[f"{m}_mean"] = (m, "mean")
            label_agg[f"{m}_std"] = (m, "std")
    label_agg["count"] = ("f1_score", "count")
    label_summary = label_results.groupby(
        ["label", "method", "cutoff", "reference", "key", "subsample_ref"]
    ).agg(**label_agg).reset_index()
    label_summary.to_csv("label_results_summary.tsv.gz", sep="\t", index=False, compression="gzip")
    del label_summary

    if organism == "homo_sapiens":
        columns_to_group = ["label", "method", "disease", "cutoff", "sex", "dev_stage", "reference", "study"]
    if organism == "mus_musculus":
        columns_to_group = ["label", "method", "treatment", "genotype", "strain", "cutoff", "sex", "age", "reference", "study"]
    factors = columns_to_group + ["query", "query_region", "ref_region"]
    write_factor_summary(label_results, factors)
    print_study_factor_table(label_results, organism)


if __name__ == "__main__":
    main()
