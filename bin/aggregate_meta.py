    
#!/user/bin/python3

from pathlib import Path
import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import warnings
#import adata_functions
#from adata_functions import *
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
import json
import yaml

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--results_dirs', type=str, nargs = "+", help="Directories containing pipeline results", default=["/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/mus_musculus/sample/SCT/ref_50_query_null_cutoff_0_refsplit_dataset_id/scvi","/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/mus_musculus/sample/SCT/ref_50_query_null_cutoff_0_refsplit_dataset_id/seurat"])                                              
    parser.add_argument('--params_file', type=str, help="Path to the params file", default = "/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/mus_musculus/sample/SCT/ref_50_query_null_cutoff_0_refsplit_dataset_id/params.yaml")
    parser.add_argument('--run_name', type=str, help="Name of the original file", default = "ref_50_query_null_cutoff_0_refsplit_dataset_id")
    parser.add_argument('--ref_obs', type=str, default="/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/mus_musculus/sample/SCT/ref_50_query_null_cutoff_0_refsplit_dataset_id/refs/")
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args
  
def combine_ref_obs(ref_obs):
    combined_obs = pd.DataFrame()
    for root, dirs, files in os.walk(ref_obs):
        for file in files:
            temp = pd.read_csv(os.path.join(root, file), sep='\t')
            ref_name = file.split(".obs.tsv")[0]
            temp["reference"] = ref_name
            temp = temp[["subclass","reference"]] 
            value_counts = temp.value_counts().reset_index()
            combined_obs = pd.concat([value_counts, combined_obs], ignore_index=True)
    return combined_obs
    

def main():
    args = parse_arguments()
    params_file = args.params_file
    results_dirs = args.results_dirs
 #   f1_results_df = pd.DataFrame()
    run_name = args.run_name
    ref_obs = args.ref_obs
    
    # Process params_file
    # check if file exists
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"File '{params_file}' not found.")

    with open(params_file, "r") as file:
        parameters_dict = yaml.safe_load(file)  # Parse the YAML file into a Python dictionary

    keys_to_drop = ["ref_collections", "ref_keys", "outdir", 
                    "batch_keys", "relabel_r", "relabel_q", "tree_file","queries_adata"]

    # Use a loop to remove keys
    for key in keys_to_drop:
        parameters_dict.pop(key, None)
            

    combined_meta = pd.DataFrame()  # Initialize an empty DataFrame
    for result_path in results_dirs:
        method = os.path.basename(result_path)  # Method = last directory
        for root, dirs, files in os.walk(result_path):
            for file in files:
                if "predictions" in file and file.endswith(".tsv"):
                    print(file)
                    full_path = os.path.join(root, file)

                    # Extract directory hierarchy
                    ref_name = os.path.basename(os.path.dirname(full_path))         # Parent directory
                    query_name = os.path.basename(os.path.dirname(os.path.dirname(full_path)))  # Parent of parent

                    tempdf = pd.read_csv(os.path.join(root, file), sep="\t")  # Read the .tsv file
                    tempdf["method"] = method  # Add a method column
                    tempdf["reference"] = ref_name  # Add a reference column
                    tempdf["query"] = query_name
                    
                    for key, value in parameters_dict.items():
                        tempdf[key] = value
                    combined_meta = pd.concat([combined_meta, tempdf], ignore_index=True)  # Append to the DataFrame

    
    #add ref label support
    ref_support = combine_ref_obs(ref_obs) 
    
    # plot "correct" for each "subclass" label and facet by "outlier"
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_meta, x="label", y="correct", hue="outlier")
    plt.xticks(rotation=90)
    plt.title("Correct Predictions by Label and Outlier")
    plt.tight_layout()
    
    combined_meta.to_csv(f"{run_name}_combined_meta.tsv", sep="\t", index=False)
    
if __name__ == "__main__":
    main()