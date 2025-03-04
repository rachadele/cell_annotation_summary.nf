#!/user/bin/python3

from pathlib import Path
import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import warnings
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import statsmodels as sm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")

    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_hsap/aggregated_results/label_f1_results.tsv")   
    parser.add_argument('--mapping_file', type=str, help="Mapping file", default = "/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/meta/census_map_human.tsv")
    
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args
    
    return parser.parse_args()


def plot_f1_score_distribution(label_f1_results, mapping_df, levels, level="global", methods=None, method_col="cutoff"):
    
    # Set global fontsize for matplotlib
    plt.rcParams.update({'font.size': 15})
    
    # If no methods are provided, use the unique values from the specified method column
    if methods is None:
        methods = label_f1_results[method_col].unique()
    
    # Ensure methods are sorted in increasing order
    methods = sorted(methods)
    
    # Get the order in levels["subclass"]
    all_subclasses = sorted(levels["subclass"])
    
    # Generate unique colors for each subclass
    color_palette = sns.color_palette("tab20", n_colors=len(all_subclasses))
    subclass_colors = dict(zip(all_subclasses, color_palette))
    
    # Set up the plot grid with shared x-axis
    fig, ax = plt.subplots(len(levels[level]), len(methods), figsize=(10 * len(methods), 5 * len(levels[level])), sharex=True)
    
    # Ensure ax is always a list for consistent indexing
    if len(levels[level]) == 1:
        ax = [ax]
    if len(methods) == 1:
        ax = [ax]
    
    # Loop over each cell type (higher-level grouping)
    for i, celltype in enumerate(levels[level]):
        # Get the subclasses for the current cell type
        group_subclasses = mapping_df[mapping_df[level] == celltype]["subclass"].unique()
        if len(group_subclasses) == 0:
            group_subclasses = [celltype]
        
        # Ensure subclasses are consistently ordered
        group_subclasses = [subclass for subclass in all_subclasses if subclass in group_subclasses]
        
        # Filter the data for the current group
        filtered_df = label_f1_results[label_f1_results["label"].isin(group_subclasses)]
        
        # Loop over each method
        for j, method in enumerate(methods):
            # Filter data by the specified method column
            method_df = filtered_df[filtered_df[method_col] == method] if method_col in filtered_df else filtered_df
            method_df["label"] = pd.Categorical(method_df["label"], categories=group_subclasses, ordered=True)
            
            # Create a boxplot with unique colors
            sns.boxplot(
                x="f1_score", 
                y="label", 
                data=method_df, 
                ax=ax[i][j], 
                orient="h", 
                showfliers=False,  # Remove outliers
                width=0.3,         # Make boxes thinner
                palette=[subclass_colors[label] for label in group_subclasses]  # Apply unique colors
            )
            
            # Titles and axis labels
            ax[i][j].set_title(f"{celltype} F1 Score - {method}")
            if j == 0:
                ax[i][j].set_ylabel('Subclass')
            else:
                ax[i][j].set_ylabel('')
                
            # Share x-axis labels: only show on the bottom row
            if i == len(levels[level]) - 1:
                ax[i][j].set_xlabel('F1 Score')
            else:
                ax[i][j].set_xlabel('')
                ax[i][j].set_xticklabels([])  # Hide tick labels for upper rows
    
    plt.tight_layout()
    plt.savefig(f"{level}_{method_col}_f1_score_distribution.png")


def main():
    # set global fontsize for matplotlib
    plt.rcParams.update({'font.size': 25})
    # Parse arguments
    args = parse_arguments()
    label_f1_results = args.label_f1_results
    
    # Read in data
    label_f1_results = pd.read_csv(args.label_f1_results, sep = "\t")
    mapping_df = pd.read_csv(args.mapping_file, sep = "\t")
    
    # filter for cutoff == 0
  #  label_f1_results = label_f1_results[label_f1_results["cutoff"] == 0]

    # Define the levels for each category
    subclasses = label_f1_results[label_f1_results["key"] == "subclass"]["label"].unique()
    classes = label_f1_results[label_f1_results["key"] == "class"]["label"].unique()
    families = label_f1_results[label_f1_results["key"] == "family"]["label"].unique()
    globals = label_f1_results[label_f1_results["key"] == "global"]["label"].unique()

    levels = {
        "subclass": subclasses,
        "class": classes,
        "family": families,
        "global": globals
    } 

    label_f1_results_filtered = label_f1_results[label_f1_results["cutoff"] == 0]

    # Example usage
    plot_f1_score_distribution(label_f1_results_filtered, mapping_df, levels, level="family", method_col="method")

    plot_f1_score_distribution(label_f1_results, mapping_df, levels, level="family", method_col="cutoff")
  

if __name__ == "__main__":
    main()