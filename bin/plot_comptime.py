#!/user/bin/python3

from pathlib import Path
import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import statsmodels as sm
from scipy import stats
import matplotlib.pyplot as plt
import re
import json
import yaml

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument("--all_runs", type=str, help="Path to trace results directory", default="/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/test_new_hierarchy/homo_sapiens")
       # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args


def convert_time(time_str):
    if pd.isna(time_str) or time_str == "-" or not isinstance(time_str, str):
        return np.nan  # Convert invalid values to NaN
    
    time_str = time_str.lower().replace(" ", "")  # Remove spaces
    
    # Extract hours, minutes, seconds, milliseconds
    match = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+\.?\d*)s)?(?:(\d+)ms)?", time_str)
    
    if not match:
        return np.nan  # Return NaN if no match
    
    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = float(match.group(3)) if match.group(3) else 0
    milliseconds = float(match.group(4)) / 1000 if match.group(4) else 0

    seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
    minutes = seconds / 60
    hours = minutes / 60
    return hours

# Function to convert percentage values
def convert_percent(value):
    return float(value.strip("%")) if isinstance(value, str) and value.endswith("%") else np.nan

def main():
        # set global fontsize
    sns.set(font_scale=1.5)
    plt.rcParams.update({'font.size': 25})
    
    args = parse_arguments()
    all_runs = args.all_runs
    
    reports = pd.DataFrame()
    
    for dir in next(os.walk(all_runs))[1]:  # Get top-level directories
        dir_path = os.path.join(all_runs, dir)
        files = next(os.walk(dir_path))[2]  # Get files inside the directory


        for file in files:
            if file == "trace.txt":
                trace_results = pd.read_csv(os.path.join(dir_path, file), sep="\t")
            #  reports = pd.concat([reports, trace_results], ignore_index=True)
        
                # read in params.yaml and add to trace dataframe
            if file == "params.yaml":
                with open(os.path.join(dir_path, file), "r") as f:
                    parameters_dict = yaml.safe_load(f)  # Parse the YAML file into a Python dictionary
                    # Convert the dictionary to a pandas DataFrame

        keys_to_drop = ["ref_collections", "ref_keys", "outdir", 
                        "batch_keys", "relabel_r", "relabel_q", "tree_file","queries_adata"]
        for key in keys_to_drop:
            parameters_dict.pop(key, None)
        for key, value in parameters_dict.items():
            trace_results[key] = value
                        
        reports = pd.concat([reports, trace_results], ignore_index=True)

    reports["%cpu"] = reports["%cpu"].apply(convert_percent)
    
    # Apply conversion functions
    reports["duration (hours)"] = reports["duration"].apply(convert_time)
    reports["realtime (hours)"] = reports["realtime"].apply(convert_time)

    reports["name"] = reports["name"].apply(lambda x: x.split(" ")[0])
    reports["peak virtual memory (GB)"] = reports["peak_vmem"].astype(str).str.split(" ", expand=True)[0].replace("-",np.nan).astype(float) / 1024  # Convert to GB
    # plot boxplots for duration, realtime and %cpu for rows wehre name==rfPredict vs name==predictSeurat
    # use a facetmap to plot the boxplots
    processes = ["rfPredict", "predictSeurat", "mapQuery","queryProcessSeurat","refProcessSeurat"]
    trace_subset = reports[reports["name"].isin(processes)]
    # replace rfPredict with SCVI Random Forest and predictSeurat with Seurat TransferData
    trace_subset["name"] = trace_subset["name"].replace({"rfPredict": "SCVI Random Forest", "predictSeurat": "Seurat TransferData", 
                                                         "mapQuery": "SCVI pre-process query", "queryProcessSeurat": "Seurat pre-process query", 
                                                         "refProcessSeurat": "Seurat pre-process reference"})
    #get median and mean for each method
   
    # turn this into a file
    with open("comptime_summary.txt", "w") as f:
        f.write("mean duration:\n")
        f.write(str(trace_subset.groupby("name")["duration (hours)"].mean()))
        f.write("\n")
        f.write("median duration:\n")
        f.write(str(trace_subset.groupby("name")["duration (hours)"].median()))
        f.write("\n")
        f.write("mean realtime:\n")
        f.write(str(trace_subset.groupby("name")["realtime (hours)"].mean()))
        f.write("\n")
        f.write("median realtime:\n")
        f.write(str(trace_subset.groupby("name")["realtime (hours)"].median()))
        f.write("\n")
        f.write("mean %cpu:\n")
        f.write(str(trace_subset.groupby("name")["%cpu"].mean()))
        f.write("\n")
        f.write("median %cpu:\n")
        f.write(str(trace_subset.groupby("name")["%cpu"].median()))
        f.write("\n")
        f.write("mean peak virtual memory:\n")
        f.write(str(trace_subset.groupby("name")["peak virtual memory (GB)"].mean()))
        f.write("\n")
        f.write("median peak virtual memory:\n")
        f.write(str(trace_subset.groupby("name")["peak virtual memory (GB)"].median()))
        f.write("\n")
        
       
    # Melt data for FacetGrid
    trace_melted = trace_subset.melt(id_vars=["name","subsample_ref","subsample_query","cutoff"], 
                                     value_vars=["duration (hours)","%cpu"], var_name="Metric", value_name="Value")

    # Create FacetGrid
    g = sns.FacetGrid(trace_melted, col="Metric", sharey=False, height=5, aspect=1)

    # Map the violin plot while keeping 'subsample_ref' and 'subsample_query' in the hue
    g.map_dataframe(sns.stripplot, x="name", y="Value", hue="subsample_ref",palette="Set3", dodge=True, jitter=True, legend=True)
    # set xlabels 90 degree rotation
    g.set_xticklabels(rotation=90)
    # Adjust legend
    g.add_legend(title="Number of cells subsampled from reference per cell type")
    plt.savefig("comptime.png",bbox_inches='tight')
    
if __name__ == "__main__":
    main()
    
    