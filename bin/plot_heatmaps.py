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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
# make a figure for legend separately
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mmus/10/b7616d81259500d38e98a4c232f5b1/weighted_f1_results.tsv")
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args
    
def add_acronym_legend(acronym_mapping, figure=None, x=1.05, y=0.5, title=None):
    if acronym_mapping:
        legend_text = f"{title}\n" + "\n".join([f"{k}: {v}" for k, v in acronym_mapping.items()])        
        figure = figure or plt.gcf()
        figure.text(
            x, y, legend_text,
            fontsize=14,
            verticalalignment='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.1')
        )
        
def main():
    plt.rcParams.update({'font.size': 25}) 
    # Parse arguments
    args = parse_arguments()

    # Read in data
    #label_f1_results = pd.read_csv(label_f1_results, sep = "\t")
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep = "\t")
 
    
    # Pivot data to create a large heatmap
    heatmap_data = pd.pivot_table(
        weighted_f1_results, 
        values="weighted_f1", 
        index=["query", "study","sex", "dev_stage", "disease","key"],
        columns=["reference","cutoff","subsample_ref",'method',"ref_split"]
    )
    
    write_na_report(heatmap_data)
   # Create the column_metadata DataFrame based on heatmap data columns
    column_metadata = pd.DataFrame(heatmap_data.columns.tolist(), columns=["reference", "cutoff", "subsample_ref", "method","ref_split"])
    row_metadata = pd.DataFrame(heatmap_data.index.tolist(), columns=["query", "study","sex", "dev_stage", "disease","key"])

    # Define color palettes for each column category
    reference_colors = sns.color_palette("Set1", n_colors=len(column_metadata["reference"].unique()))
    method_colors = sns.color_palette("husl", n_colors=len(column_metadata["method"].unique()))
    ref_split_colors = sns.color_palette("Set2", n_colors=len(column_metadata["ref_split"].unique()))

    # For continuous variables, use a continuous colormap
    cutoff_cmap = sns.color_palette("coolwarm", as_cmap=True)  # Use a continuous colormap
    subsample_ref_cmap = sns.color_palette("viridis", as_cmap=True)  # Another continuous colormap

    # Define colors for the row categories
    study_colors = sns.color_palette("Set1", n_colors=len(row_metadata["query"].unique()))
    sex_colors = sns.color_palette("Set1", n_colors=len(row_metadata["sex"].unique()))
    dev_stage_colors = sns.color_palette("Set2", n_colors=len(row_metadata["dev_stage"].unique()))
    disease_colors = sns.color_palette("Set3", n_colors=len(row_metadata["disease"].unique()))
    key_colors = sns.color_palette("Set3", n_colors=len(row_metadata["key"].unique()))

    # Map color palettes to the rows (categorical)
    row_colors = pd.DataFrame({
        'Query': row_metadata['study'].map(dict(zip(row_metadata["study"].unique(), study_colors))),
        'Sex': row_metadata["sex"].map(dict(zip(row_metadata["sex"].unique(), sex_colors))),
        'Development stage': row_metadata["dev_stage"].map(dict(zip(row_metadata["dev_stage"].unique(), dev_stage_colors))),
        'Disease': row_metadata["disease"].map(dict(zip(row_metadata["disease"].unique(), disease_colors))),
        'Level': row_metadata["key"].map(dict(zip(row_metadata["key"].unique(), key_colors)))
    }, index=row_metadata.index)

    # Map color palettes to the columns
    col_colors = pd.DataFrame({
        'Reference': column_metadata['reference'].map(dict(zip(column_metadata["reference"].unique(), reference_colors))),
        'Method': column_metadata['method'].map(dict(zip(column_metadata["method"].unique(), method_colors))),
        'Split': column_metadata['ref_split'].map(dict(zip(column_metadata["ref_split"].unique(), ref_split_colors))),
        # Map continuous variables using a red colormap
        'Cutoff': [cutoff_cmap(value) for value in column_metadata['cutoff']],  # Apply red colormap to each value
        'Subsample': [subsample_ref_cmap(value) for value in column_metadata['subsample_ref']]  # Apply red colormap to each value
    }, index=column_metadata.index)



    heatmap_data.columns = range(heatmap_data.shape[1])
    heatmap_data.index = range(heatmap_data.shape[0])
    # Check for NaN or infinite values
    heatmap_data = heatmap_data.fillna(0)

    #handle NaN for now - have to fix this in the future
    n_rows, n_cols = heatmap_data.shape

    # Create the title string
    title = f"Classification of {n_rows} query samples across {n_cols} parameter combinations"

    g = sns.clustermap(
        heatmap_data, 
        cmap="Reds", 
        col_cluster=True, 
        row_cluster=True,  # Enable clustering
        figsize=(25, 18), 
        xticklabels=False, 
        yticklabels=False, 
        linewidths=0, 
        col_colors=col_colors, 
        row_colors=row_colors,  # Add parameter annotations
        annot=False,
        dendrogram_ratio=(0.1, 0.1),
        cbar_kws={"label": "Weighted F1"}
        # Adjust the size of the row and column dendrograms
        #cbar_pos=(0.02, 0.2, 0.03, 0.4)  # Adjust the colorbar position (optional)
    )
    g.fig.suptitle(title, y=1.05)  # Increase y value to move the title 
    plt.savefig("weighted_f1_heatmap.png", bbox_inches='tight')



    legend_dict = {
        "Method": zip(column_metadata["method"].unique(), method_colors),
        "Ref Split": zip(column_metadata["ref_split"].unique(), ref_split_colors),
        "Sex": zip(row_metadata["sex"].unique(), sex_colors),
        "Dev Stage": zip(row_metadata["dev_stage"].unique(), dev_stage_colors),
        "Study": zip(row_metadata["study"].unique(), study_colors),
        "Disease": zip(row_metadata["disease"].unique(), disease_colors),
        "Level": zip(row_metadata["key"].unique(), key_colors)
    }

    # For continuous variables, we use ScalarMappable to create a legend
    # For "Cutoff"
    cutoff_norm = Normalize(vmin=column_metadata["cutoff"].min(), vmax=column_metadata["cutoff"].max())
    cutoff_sm = ScalarMappable(norm=cutoff_norm, cmap=cutoff_cmap)
    legend_dict["Cutoff"] = [(f"{val:.2f}", cutoff_sm.to_rgba(val)) for val in column_metadata["cutoff"].unique()]

    # For "Subsample"
    subsample_ref_norm = Normalize(vmin=column_metadata["subsample_ref"].min(), vmax=column_metadata["subsample_ref"].max())
    subsample_ref_sm = ScalarMappable(norm=subsample_ref_norm, cmap=subsample_ref_cmap)
    legend_dict["Subsample"] = [(f"{val:.2f}", subsample_ref_sm.to_rgba(val)) for val in column_metadata["subsample_ref"].unique()]


    # Create a figure for the legends
    fig, axes = plt.subplots(len(legend_dict), 1, figsize=(6, len(legend_dict) * 1.5))

    # Loop through the legend dictionary and create the patches for each category
    for i, (title, elements) in enumerate(legend_dict.items()):
        patches = [Patch(facecolor=color, edgecolor="black", label=label) for label, color in elements]
        
        # Plot the legend on the individual axes
        axes[i].legend(handles=patches, title=title, loc="upper left", frameon=True, ncol=8)  # Change ncol for multiple columns
        
        # Hide the axes for the legends
        axes[i].set_axis_off()

    # Adjust the vertical spacing between the subplots to prevent overlap
    plt.subplots_adjust(hspace=0.6)  # Adjust this value if needed
    # Adjust the layout to ensure there is no overlap and display the figure
    plt.tight_layout()
    plt.savefig("legend.png", bbox_inches='tight')

def write_na_report(heatmap_data):
    # Check for NaN or infinite values.
    if heatmap_data.isnull().values.any():
        warnings.warn("NaN or infinite values detected in the data. Replacing with 0.")
       #subset for Nas and write to file
        na_subset = heatmap_data.loc[:, heatmap_data.isnull().any(axis=0)]
        # pick out only columns with NaN values
        na_subset.fillna("nan", inplace = True)
        na_subset.to_csv("na_values.tsv", sep = "\t")
    else:
        with open("na_values.tsv", "w") as f:
            f.write("No NaN or infinite values detected in the data.\n")
        

if __name__ == "__main__":
    main()