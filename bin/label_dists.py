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

    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_mmus/aggregated_results/label_f1_results.tsv")   
    parser.add_argument('--mapping_file', type=str, help="Mapping file", default = "/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/meta/census_map_mouse.tsv")
    
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args
    
    return parser.parse_args()

def plot_f1_score_distribution(label_f1_results, mapping_df, levels, level="global", methods=None):
    
    # Set global fontsize for matplotlib
    plt.rcParams.update({'font.size': 15})
    # If no methods are provided, use the unique cutoffs from the data
    if methods is None:
        methods = label_f1_results["cutoff"].unique()
    # Ensure methods are sorted in increasing order
    methods = sorted(methods)
    # Set up the plot (stack facets vertically, two columns for method comparison)
    fig, ax = plt.subplots(len(levels[level]), len(methods), figsize=(10 * len(methods), 5 * len(levels[level])))
    # If there's only one row or column, make sure ax is a list
    if len(levels[level]) == 1:
        ax = [ax]
    if len(methods) == 1:
        ax = [ax]
    # get the order in levels["subclass"]
    all_subclasses = sorted(levels["subclass"])
    # Loop over each celltype or level
    for i, celltype in enumerate(levels[level]):
        # Get the subclasses for the current celltype
        group_subclasses = mapping_df[mapping_df[level] == celltype]["subclass"].unique()
        if len(group_subclasses) == 0:
            group_subclasses = [celltype]
        # Ensure subclasses are consistently ordered across all methods
        group_subclasses = [subclass for subclass in all_subclasses if subclass in group_subclasses]
        # Filter the data for the current group
        filtered_df = label_f1_results[label_f1_results["label"].isin(group_subclasses)]
        # Loop over each method (cutoff or other method)
        for j, method in enumerate(methods):
            # Filter the data based on the method (cutoff or method)
            method_df = filtered_df[filtered_df["cutoff"] == method] if "cutoff" in filtered_df else filtered_df
            # sort by order of subclasses
            method_df["label"] = pd.Categorical(method_df["label"], categories=group_subclasses, ordered=True)
            # Create a sideways violin plot for the f1_score distribution
            sns.boxplot(x="f1_score", y="label", data=method_df, ax=ax[i][j], orient="h", hue="label", palette="Set2")
            # Add a title for each subplot
            ax[i][j].set_title(f"{celltype} F1 Score Distribution - {method}")
          #  ax[i][j].set_xlim(0, 1)  # Set the x-axis limits to 0 and 1
            # Only label the y-axis on the left-most column
            if j == 0:
                ax[i][j].set_ylabel('Subclass')
            else:
                ax[i][j].set_ylabel('')
                
            # Only label the x-axis on the bottom row
            if i == len(levels[level]) - 1:
                ax[i][j].set_xlabel('F1 Score')  # Only show once for the last row
            else:
                ax[i][j].set_xlabel('')
                
    plt.tight_layout()
    # Adjust layout for better visualization
    plt.savefig(f"{level}_f1_score_distribution.png")


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

    # Set the level you want to plot
    level = "class"


    # Example usage
    plot_f1_score_distribution(label_f1_results, mapping_df, levels, level="family")

  

if __name__ == "__main__":
    main()