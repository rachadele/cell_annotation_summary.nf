
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
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

def make_acronym(name):
    # Split on "_" and replace with spaces
    words = name.split("_")
    # Create acronym from the first letter of each word
    acronym = "".join(word[0].upper() for word in words if word)
    return acronym

def map_development_stage(stage):
    # re write dict
    dev_stage_mapping_dict = {
        "HsapDv_0000083": "infant",
        "HsapDv_0000084": "toddler",
        "HsapDv_0000085": "child",
        "HsapDv_0000086": "adolescent",
        "HsapDv_0000088": "adult",
        "HsapDv_0000091": "late adult",
        np.nan: None
    }
    return dev_stage_mapping_dict[stage]
    
def write_factor_summary(df, factors):
    # 1. Summarize the number of unique levels for each factor
    unique_counts_df = df[factors].nunique().reset_index()
    unique_counts_df.to_csv("factor_unique_counts.tsv", sep="\t", index=False)

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
    result_df.to_csv("factor_unique_sample_counts.tsv", sep="\t", index=False)
    
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
def print_study_factor_table(label_f1_results, organism):
    import pandas as pd
    if organism == "homo_sapiens":
        columns = ["disease", "sex", "dev_stage", "number query samples", "number cells", "query_region", "number unique subclasses"]
    else:
        columns = ["treatment", "genotype", "strain", "sex", "age", "query_region", "number query samples", "number cells", "number unique subclasses"]

    # Ensure DataFrames
    label_df = pd.DataFrame(label_f1_results)
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
                # Find unique subclasses for this study in label_f1_results
                n_subclasses = label_df[(label_df[study_col] == study) & (label_df["key"] == "subclass")]["label"].nunique()
                row[col] = n_subclasses
            elif col in group.columns:
                vals = group[col].dropna().unique()
                row[col] = ", ".join(map(str, vals))
        table.append(row)
    out_cols = [study_col] + columns
    # write to tsv
    table_df = pd.DataFrame(table)[out_cols]
    table_df.to_csv("study_factor_summary.tsv", sep="\t", index=False)
 
def main():
    args = parse_arguments()
    pipeline_results = args.pipeline_results

    # --- Load and concatenate input files ---
    results_df = pd.DataFrame()
    for filepath in pipeline_results:
        temp_df = pd.read_csv(filepath, sep="\t")
        results_df = pd.concat([temp_df, results_df], ignore_index=True)

    organism = results_df["organism"].unique()[0]
    results_df = results_df[results_df['support'] > 0]

    # --- Derive columns ---
    results_df["study"] = results_df["query"].apply(lambda x: x.split("_")[0])
    results_df["query"] = results_df["query"].str.replace("_", " ")
    results_df["reference_acronym"] = results_df["reference"].apply(make_acronym)
    results_df["reference"] = results_df["reference"].str.replace("_", " ")
    results_df["region_match"] = results_df.apply(lambda row: row['query_region'] in row['ref_region'], axis=1)

    # --- Standardize disease ---
    results_df["disease"] = np.where(results_df["disease"] == "Control", "control", results_df["disease"])
    results_df["disease"] = np.where(results_df["disease"].isnull(), "control", results_df["disease"])
    results_df["disease_state"] = np.where(results_df["disease"] == "control", "control", "disease")

    # --- Standardize sex ---
    results_df["sex"] = results_df["sex"].str.replace(r"^M$", "male", regex=True, case=False)
    results_df["sex"] = results_df["sex"].str.replace(r"^F$", "female", regex=True, case=False)
    results_df["sex"] = results_df["sex"].str.replace(r"^feM$", "female", regex=True, case=False)

    # --- Organism-specific fixes ---
    if organism == "homo_sapiens":
        results_df["disease"] = np.where(results_df["study"] == "GSE211870", "control", results_df["disease"])
        results_df["dev_stage"] = results_df["dev_stage"].apply(map_development_stage)
        results_df["dev_stage"] = np.where(results_df["study"] == "rosmap", "late adult", results_df["dev_stage"])
        results_df["dev_stage"] = np.where(results_df["study"] == "pineda", "late adult", results_df["dev_stage"])
        results_df["sex"] = np.where(results_df["query"] == "lim C5382Cd", "male", results_df["sex"])
        results_df["dev_stage"] = np.where(results_df["query"] == "lim C5382Cd", "late adult", results_df["dev_stage"])

    if organism == "mus_musculus":
        results_df["treatment_state"] = np.where(results_df["treatment"].isnull(), "No treatment", "treatment")
        results_df["genotype"] = np.where(results_df["genotype"].isnull(), "wild type genotype", results_df["genotype"])
        results_df["treatment_state"] = results_df["treatment_state"].str.lower()

    # --- Lowercase normalization ---
    results_df["disease_state"] = results_df["disease_state"].str.lower()
    results_df["sex"] = results_df["sex"].str.lower()

    # --- Weighted F1 results ---
    label_columns = ["label", "f1_score", "precision", "recall", "support", "accuracy"]
    weighted_f1_results = results_df.drop(columns=label_columns)
    weighted_f1_results = weighted_f1_results.drop_duplicates()
    weighted_f1_results = weighted_f1_results[weighted_f1_results["weighted_f1"].notnull()]
    weighted_f1_results = weighted_f1_results.fillna("None")
    weighted_f1_results.to_csv("weighted_f1_results.tsv", sep="\t", index=False)

    weighted_summary = weighted_f1_results.groupby(["method", "cutoff", "reference", "key"]).agg(
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        weighted_f1_count=("weighted_f1", "count")
    ).reset_index()
    weighted_summary.to_csv("weighted_f1_summary.tsv", sep="\t", index=False)

    # --- Label F1 results ---
    label_results = results_df[results_df['label'].notnull()]
    label_results = label_results[label_results["f1_score"].notnull()]
    label_results = label_results.fillna("None")
    label_results = label_results[label_results["label"] != "unkown"]
    label_results.to_csv("label_f1_results.tsv", sep="\t", index=False)

    label_results['precision'] = pd.to_numeric(label_results['precision'], errors='coerce')
    label_results['recall'] = pd.to_numeric(label_results['recall'], errors='coerce')

    label_summary = label_results.groupby(["label", "method", "cutoff", "reference", "key"]).agg(
        label_f1_mean=("f1_score", "mean"),
        label_f1_std=("f1_score", "std"),
        label_f1_count=("f1_score", "count")
    ).reset_index()
    label_summary.to_csv("label_f1_summary.tsv", sep="\t", index=False)

    # --- Factor summaries ---
    if organism == "homo_sapiens":
        columns_to_group = ["label", "method", "disease", "cutoff", "sex", "dev_stage", "reference", "study"]
    if organism == "mus_musculus":
        columns_to_group = ["label", "method", "treatment", "genotype", "strain", "cutoff", "sex", "age", "reference", "study"]
    factors = columns_to_group + ["query", "query_region", "ref_region"]
    write_factor_summary(label_results, factors)
    print_study_factor_table(label_results, organism)


if __name__ == "__main__":
    main()
    
