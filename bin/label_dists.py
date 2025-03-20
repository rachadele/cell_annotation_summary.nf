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
set_seed = 42
import random
import numpy as np
random.seed(42)


# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")

    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_hsap/aggregated_results/label_f1_results.tsv")   
    parser.add_argument('--color_mapping_file', type=str, help="Mapping file", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/meta/color_mapping.tsv")
    parser.add_argument('--mapping_file', type=str, help="Mapping file", default = "/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/meta/census_map_human.tsv")
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args
    
    return parser.parse_args()

def make_stable_colors(color_mapping_df):
    
    all_subclasses = sorted(color_mapping_df["subclass"])
    # i need to hardcode a separate color palette based on the mmus mapping file
    # Generate unique colors for each subclass
    color_palette = sns.color_palette("husl", n_colors=len(all_subclasses))
    subclass_colors = dict(zip(all_subclasses, color_palette))
    return subclass_colors
    

def plot_score_distribution(label_f1_results, color_mapping_df, mapping_df, levels, level="global", method_col="cutoff", score_col="f1_score", subclass_col="subclass"):
    
    # Set global fontsize for matplotlib
    plt.rcParams['font.size'] = 25 
    methods = label_f1_results[method_col].unique()
    # Ensure methods are sorted in increasing order
    methods = sorted(methods)
    # Get the order in levels["subclass"]
    all_subclasses = sorted(levels[subclass_col])
    subclass_colors = make_stable_colors(color_mapping_df)
    # Set up the plot grid with shared x-axis
    fig, ax = plt.subplots(
            len(levels[level]), len(methods), figsize=(10 * len(methods), 5 * len(levels[level])),
            sharex=True  # Ensures all subplots have the same x-axis limits
        )
    # Ensure ax is always a list for consistent indexing
    if len(levels[level]) == 1:
        ax = [ax]
    if len(methods) == 1:
        ax = [ax]
    # Loop over each cell type (higher-level grouping)
    for i, celltype in enumerate(levels[level]):
        # Get the subclasses for the current cell type
        group_subclasses = mapping_df[mapping_df[level] == celltype][subclass_col].unique()
        if celltype == "Neuron":
            group_subclasses = "Ambiguous Neuron"
        else:
            # add option to change
            group_subclasses = mapping_df[mapping_df[level] == celltype][subclass_col].unique()
        subclasses_to_plot = [subclass for subclass in all_subclasses if subclass in group_subclasses]
        if len(subclasses_to_plot) == 0:
           subclasses_to_plot = [celltype]
     
        # Filter the data for the current group
        filtered_df = label_f1_results[(label_f1_results["label"].isin(subclasses_to_plot)) & (label_f1_results["key"] == subclass_col)]
        
        # Loop over each method
        for j, method in enumerate(methods):
            # Filter data by the specified method column
            method_df = filtered_df[filtered_df[method_col] == method] if method_col in filtered_df else filtered_df
            method_df["label"] = pd.Categorical(method_df["label"], categories=subclasses_to_plot, ordered=True)
                        # Calculate the SD for each group
            method_df['sd'] = method_df[score_col].std()

            # Clamp negative SD values to 0
            method_df['sd'] = method_df['sd'].apply(lambda x: max(x, 0))

            sns.boxplot(
                x=score_col, 
                y="label", 
                data=method_df, 
                ax=ax[i][j], 
                orient="h", 
                width=0.6,  # Controls the width of the boxes
                palette={label: subclass_colors[label] for label in subclasses_to_plot},
                showfliers=False,  # Hide outliers
                whis=[5, 95],  # Limits the whiskers to the 5th and 95th percentiles (effectively limiting the IQR)
                linewidth=2  # Optional: Makes the box edges thicker
            )
            
            # Titles and axis labels
            if i == 0:
                ax[i][j].set_title(f"{method}")
            else:
                ax[i][j].set_title('')
            if j == 0:
                ax[i][j].set_ylabel(f"{celltype}")
                ax[i][j].set_yticklabels(subclasses_to_plot)
            else:
                ax[i][j].set_ylabel('')
                ax[i][j].set_yticklabels([])  # Hide tick labels for right columns
                
            # Share x-axis labels: only show on the bottom row
            if i == len(levels[level]) - 1:
                ax[i][j].set_xticks(np.linspace(0, 1, 11))
                score_col_name = score_col.replace("_", " ").title()
                ax[i][j].set_xlabel(score_col_name)
            else:
                ax[i][j].set_xlabel('')
                ax[i][j].set_xticklabels([])  # Hide tick labels for upper rows
    
    plt.tight_layout()
    outdir = subclass_col
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir,f"{level}_{method_col}_{score_col}_distribution.png"))


def main():
    # set global fontsize for matplotlib
    plt.rcParams.update({'font.size': 25})
    # Parse arguments
    args = parse_arguments()
    label_f1_results = args.label_f1_results
    
    # Read in data
    label_f1_results = pd.read_csv(args.label_f1_results, sep = "\t")
    color_mapping_df = pd.read_csv(args.color_mapping_file, sep = "\t")
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

    label_f1_results_filtered = label_f1_results[label_f1_results["cutoff"].isin([0])]
    # add a column for support across the whole dataset
    
    #label_f1_results_filtered["inter_dataset_support"] = label_f1_results_filtered["label"].map(label_f1_results_filtered["label"].value_counts())
    
    
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="f1_score")
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="precision")
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="recall")
   
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="f1_score", subclass_col="class")
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="precision", subclass_col="class")
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="recall", subclass_col="class")
   
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="f1_score", subclass_col="family")
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="precision", subclass_col="family")
    plot_score_distribution(label_f1_results_filtered, color_mapping_df, mapping_df, levels, level="family", method_col="method",score_col="recall", subclass_col="family")
      
 
if __name__ == "__main__":
    main()