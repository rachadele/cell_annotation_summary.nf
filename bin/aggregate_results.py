#!/user/bin/python3

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


 
def main():
    # Parse command line arguments
    args = parse_arguments()
    # Set organism and census_version from arguments
    pipeline_results = args.pipeline_results

    
    f1_df = pd.DataFrame() 
     
    #for file in os.listdir(pipeline_results):
    for filepath in pipeline_results:
    #filepath = os.path.join(pipeline_results, file)
    # method = filepath.split("/")[-3]
        temp_df = pd.read_csv(filepath,sep="\t")
        # temp_df["method"] = method
        f1_df = pd.concat([temp_df, f1_df], ignore_index=True)
     
    organism = f1_df["organism"].unique()[0]
    # replace "nan" with None
    # deal with 0 support
    # drop rows with 0 support
    f1_df = f1_df[f1_df['support'] > 0]
    #f1_df = update_metrics(f1_df)
    #f1_df = f1_df.replace("nan", None)
    
    
    # deal with control/disease state
        # catch controls that are lower or upper case
    f1_df["disease"] = np.where(f1_df["disease"] == "Control", "control", f1_df["disease"])
    # fill None with "control"
    f1_df["disease"] = np.where(f1_df["disease"].isnull(), "control", f1_df["disease"])    
    #----------weighted f1 results----------------
    # miscellaneous data wrangling
      
    f1_df["region_match"] = f1_df.apply(lambda row: row['query_region'] in row['ref_region'], axis=1)
    f1_df["reference_acronym"] = f1_df["reference"].apply(make_acronym)
    f1_df["reference"] = f1_df["reference"].str.replace("_", " ")
    f1_df["study"] = f1_df["query"].apply(lambda x: x.split("_")[0])
    f1_df["query"] = f1_df["query"].str.replace("_", " ")
    
        
    
    f1_df["disease_state"] = np.where(f1_df["disease"] == "control", "control", "disease")
 
    if organism == "homo_sapiens":
        # data wrangling for missing disease (all controls)
        f1_df["disease"] = np.where(f1_df["study"]=="GSE211870", "control", f1_df["disease"]) 
    
        # deal with annotation mismatch between gemma queries and curated queries
        f1_df["dev_stage"] = f1_df["dev_stage"].apply(map_development_stage) 
        
    # Data wrangling for Rosmap error (dev stage mistakely mapped as "infant")
        f1_df["dev_stage"] = np.where(f1_df["study"] == "rosmap" , "late adult", f1_df["dev_stage"])
        
    # data wrangling for missing Pineda dev stage   
        f1_df["dev_stage"] = np.where(f1_df["study"] == "pineda" , "late adult", f1_df["dev_stage"])

    # data wrangling for Lim sample missing from original metadata
        f1_df["sex"] = np.where(f1_df["query"]=="lim C5382Cd", "male", f1_df["sex"])
        f1_df["dev_stage"] = np.where(f1_df["query"] == "lim C5382Cd" , "late adult", f1_df["dev_stage"])


    # data wrangling for sex (Gemmma data uses male:female, conform to this naming scheme)
        f1_df["sex"] = f1_df["sex"].str.replace("M", "male")
        f1_df["sex"] = f1_df["sex"].str.replace("F", "female")
        # don't know why this is in the data
        f1_df["sex"] = f1_df["sex"].str.replace("feM","female")
         
    if organism == "mus_musculus":

        f1_df["treatment_state"] = np.where(f1_df["treatment"].isnull(), "No treatment", "treatment")
        f1_df["genotype"] = np.where(f1_df["genotype"].isnull(), "wild type genotype", f1_df["genotype"])
        f1_df["treatment_state"] = f1_df["treatment_state"].str.lower()


    # make everything lowercase
    f1_df["disease_state"] = f1_df["disease_state"].str.lower()
    f1_df["sex"] = f1_df["sex"].str.lower()
    #f1_df["dev_stage"] = f1_df["dev_stage"].str.lower()
    
        
#----------------drop label columns and save---------------
    outdir = "weighted_f1_distributions"
    label_columns = ["label", "f1_score","precision","recall","support", "accuracy"]
    os.makedirs(outdir, exist_ok=True)
    
    # Drop duplicates, but exclude 'ref_split' column (so duplicates in 'ref_split' are allowed)
    weighted_f1_results = f1_df.drop(columns=label_columns)
    # drop duplicates
    weighted_f1_results = weighted_f1_results.drop_duplicates()
    # Keep only rows where 'weighted_f1' is not null
    weighted_f1_results = weighted_f1_results[weighted_f1_results["weighted_f1"].notnull()] 
    # fill na with "None"
    weighted_f1_results = weighted_f1_results.fillna("None")
    weighted_f1_results.to_csv("weighted_f1_results.tsv", sep="\t", index=False)
    

   #------------summaries---------------- 

        # summarize by sample, key, method, mean, sd
    weighted_summary = weighted_f1_results.groupby(["method","cutoff","reference","key"]).agg(
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        weighted_f1_count=("weighted_f1", "count")
        #add precision and recall
        #weighted_precision_mean=("weighted_precision", "mean"),
        #weighted_precision_std=("weighted_precision", "std"),
        #weighted_recall_mean=("weighted_recall", "mean"),
        #weighted_recall_std=("weighted_recall", "std")
    ).reset_index()
    
    
    weighted_summary.to_csv("weighted_f1_summary.tsv", sep="\t", index=False)   
    

        # 

            

# -----------label f1 results----------------
    label_results = f1_df[f1_df['label'].notnull()]
    label_results = label_results[label_results["f1_score"].notnull()]

    # rename "support" to "intra-dataset support"
    #label_results = label_results.rename(columns={"support": "intra-dataset support"})

    label_results = label_results.fillna("None")
    label_results = label_results[label_results["label"] != "unkown"]
    
    label_results.to_csv("label_f1_results.tsv", sep="\t", index=False)
   # Ensure precision and recall are numeric and handle 'nan' strings (if needed)
    label_results['precision'] = pd.to_numeric(label_results['precision'], errors='coerce')
    label_results['recall'] = pd.to_numeric(label_results['recall'], errors='coerce')
 
    # plot distribution of label_f1 across different splits
    outdir = "label_distributions"
    os.makedirs(outdir, exist_ok=True)
    # make a count summary table for label_f1 by label, sample, disease_state, sex, dev_stage
    label_summary = label_results.groupby(["label","method","cutoff","reference","key"]).agg(
        label_f1_mean=("f1_score", "mean"),
        label_f1_std=("f1_score", "std"),
        label_f1_count=("f1_score", "count")
        # add precision and recall
       # label_precision_mean=("precision", "mean"),
        #label_precision_std=("precision", "std"),
      #  label_recall_mean=("recall", "mean"),
       # label_recall_std=("recall", "std")
    ).reset_index()
        
    
    label_summary.to_csv("label_f1_summary.tsv", sep="\t", index=False)
   
    if organism == "homo_sapiens":
        columns_to_group=["label","method", "disease", "cutoff", "sex", "dev_stage", "reference", "study"]
    if organism == "mus_musculus":
       columns_to_group=["label","method", "treatment", "genotype","strain","cutoff", "sex", "age", "reference", "study"] 
    factors=columns_to_group + ["query"] + ["query_region"] + ["ref_region"]
    write_factor_summary(label_results, factors) 

        
 
            
  
if __name__ == "__main__":
    main()
    
