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
from bs4 import BeautifulSoup

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument("--reports_dir", type=str, help="Path to trace results directory", default="/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/reports")
       # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args


def extract_nextflow_metrics(report_files, output_csv="aggregated_metrics.csv"):
    all_data = []

    for report_file in report_files:
        with open(report_file, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
        
        # Extract JSON data if present
        script_tags = soup.find_all("script")
        json_data = None
        for script in script_tags:
            try:
                json_data = json.loads(script.text.strip())
                break  # Assume the first valid JSON is the execution data
            except json.JSONDecodeError:
                continue

        if json_data:
            metrics = json_data.get("trace", [])  # Adjust based on structure
            df = pd.DataFrame(metrics)
        else:
            # Extract tables if JSON is not available
            tables = soup.find_all("table")
            if tables:
                df_list = pd.read_html(str(tables))
                df = pd.concat(df_list, ignore_index=True)
            else:
                continue  # Skip if no JSON or tables found

        df["source_file"] = os.path.basename(report_file)  # Track source
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    return final_df

    

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
    args = parse_arguments()
    reports_dir = args.reports_dir
    reports = pd.DataFrame()
    for root, dirs, files in os.walk(reports_dir):
        for file in files:
            if file.endswith(".txt"):
                trace_results = pd.read_csv(os.path.join(root, file), sep="\t")
                reports = pd.concat([reports, trace_results], ignore_index=True)
    
    reports["%cpu"] = reports["%cpu"].apply(convert_percent)
            
    # set global fontsize
    sns.set(font_scale=1.5)
    plt.rcParams.update({'font.size': 25})
    
    # Apply conversion functions
    reports["duration (hours)"] = reports["duration"].apply(convert_time)
    reports["realtime (hours)"] = reports["realtime"].apply(convert_time)

    reports["name"] = reports["name"].apply(lambda x: x.split(" ")[0])
    reports["peak virtual memory (GB)"] = reports["peak_vmem"].astype(str).str.split(" ", expand=True)[0].replace("-",np.nan).astype(float) / 1024  # Convert to GB
    # plot boxplots for duration, realtime and %cpu for rows wehre name==rfPredict vs name==predictSeurat
    # use a facetmap to plot the boxplots
    trace_subset = reports[reports["name"].isin(["rfPredict", "predictSeurat"])]
    # Melt data for FacetGrid
    trace_melted = trace_subset.melt(id_vars=["name"], value_vars=["duration (hours)","%cpu"], var_name="Metric", value_name="Value")

    # Create FacetGrid
    g = sns.FacetGrid(trace_melted, col="Metric", sharey=False, height=5, aspect=1)
    g.map(sns.violinplot, "name", "Value", palette="Set3", order=["rfPredict", "predictSeurat"])
    # remove y axis label
    g.set_ylabels("")
    plt.savefig("comptime.png")
    
if __name__ == "__main__":
    main()
    
    