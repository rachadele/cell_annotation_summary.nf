    
#!/user/bin/python3

from pathlib import Path
import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import warnings
import scvi
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
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--f1_results', type=str, nargs = "+", help="Directories containing F1 results", default=["/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/mus_musculus/sample/SCT/ref_50_query_null_cutoff_0_refsplit_dataset_id/seurat"])                                              
    parser.add_argument('--params_file', type=str, help="Path to the params file", default = "/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/mus_musculus/sample/SCT/ref_50_query_null_cutoff_0_refsplit_dataset_id/params.yaml")
    parser.add_argument('--run_name', type=str, help="Name of the original file", default = "ref_50_query_null_cutoff_0_refsplit_dataset_id")
    parser.add_argument('--ref_obs', type=str, default="/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/mus_musculus/sample/SCT/ref_50_query_null_cutoff_0_refsplit_dataset_id/refs")
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
    f1_results = args.f1_results
    f1_results_df = pd.DataFrame()
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
            
    # Process f1_results
    for result_path in f1_results:  # Assuming f1_results is a list of paths
        method = result_path.split("/")[-1]  # Extract the method from the path
        for root, dirs, files in os.walk(result_path):
            for file in files:
                if file.endswith("f1.scores.tsv"):
                    print(f"Processing file: {root}")
                    # Check for .tsv files
                    tempdf = pd.read_csv(os.path.join(root, file), sep="\t")  # Read the .tsv file
                    tempdf["method"] = method  # Add a method column
                    for key, value in parameters_dict.items():
                        tempdf[key] = value
                    f1_results_df = pd.concat([f1_results_df, tempdf], ignore_index=True)  # Append to the DataFrame
    
    
    f1_results_df.to_csv(f"{run_name}_f1_results.tsv", sep="\t", index=False)
    
if __name__ == "__main__":
    main()