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
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_mmus/aggregated_results/weighted_f1_results.tsv")
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

def plot_heatmap_human(weighted_f1_results):
    weighted_f1_results = weighted_f1_results.fillna("None")     
    # Pivot data to create a large heatmap
    heatmap_data = pd.pivot_table(
        weighted_f1_results, 
        values="weighted_f1", 
        index=["query", "study","sex", "dev_stage", "disease","key"],
        columns=["reference","cutoff","subsample_ref",'method']
    )
    
    write_na_report(heatmap_data)  # Assuming this function is defined elsewhere
    
    # Create the column_metadata DataFrame based on heatmap data columns
    column_metadata = pd.DataFrame(heatmap_data.columns.tolist(), columns=["reference", "cutoff", "subsample_ref", "method"])
    row_metadata = pd.DataFrame(heatmap_data.index.tolist(), columns=["query", "study","sex", "dev_stage", "disease","key"])

# Select appropriate color palettes based on the organism
    reference_colors = sns.color_palette("Set1", n_colors=len(column_metadata["reference"].unique()))
    method_colors = sns.color_palette("husl", n_colors=len(column_metadata["method"].unique()))
    # ref_split_colors = sns.color_palette("Set2", n_colors=len(column_metadata["ref_split"].unique()))
    
    # For continuous variables, use a continuous colormap
  #  cutoff_cmap = sns.color_palette("coolwarm", as_cmap=True)
   # subsample_ref_cmap = sns.color_palette("viridis", as_cmap=True)

    # Define colors for the row categories
    study_colors = sns.color_palette("Set1", n_colors=len(row_metadata["study"].unique()))
    sex_colors = sns.color_palette("Set1", n_colors=len(row_metadata["sex"].unique()))
    dev_stage_colors = sns.color_palette("Set2", n_colors=len(row_metadata["dev_stage"].unique()))
    disease_colors = sns.color_palette("Set3", n_colors=len(row_metadata["disease"].unique()))
    key_colors = sns.color_palette("Set3", n_colors=len(row_metadata["key"].unique()))
    cutoff_colors = sns.color_palette("coolwarm", n_colors=len(column_metadata["cutoff"].unique()))
    subsample_ref_colors = sns.color_palette("viridis", n_colors=len(column_metadata["subsample_ref"].unique()))
    
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
        'Cutoff': column_metadata['cutoff'].map(dict(zip(column_metadata["cutoff"].unique(), cutoff_colors))),
        'Subsample': column_metadata['subsample_ref'].map(dict(zip(column_metadata["subsample_ref"].unique(), subsample_ref_colors)))
    }, index=column_metadata.index)
    
    heatmap_data.columns = range(heatmap_data.shape[1])
    heatmap_data.index = range(heatmap_data.shape[0])
    # Adjust the heatmap data (handle NaNs and create plot title)
    heatmap_data = heatmap_data.fillna(0)
    n_rows, n_cols = heatmap_data.shape
    n_samples = n_rows // 3
    title = f"Classification of {n_samples} query samples across {n_cols} parameter combinations"

    # Create and plot the heatmap with cluster map
    g = sns.clustermap(
        heatmap_data, 
        cmap="Reds", 
        col_cluster=True, 
        row_cluster=True,  
        figsize=(25, 18), 
        xticklabels=False, 
        yticklabels=False, 
        linewidths=0, 
        col_colors=col_colors, 
        row_colors=row_colors,  
        annot=False,
        dendrogram_ratio=(0.1, 0.1),
        cbar_kws={"label": "Weighted F1"}
    )
    g.fig.suptitle(title, y=1.05)  
    plt.savefig("weighted_f1_heatmap.png", bbox_inches='tight')

    # Prepare legend dictionary
    legend_dict = {
        "Method": zip(column_metadata["method"].unique(), method_colors),
       # "Ref Split": zip(column_metadata["ref_split"].unique(), ref_split_colors),
        "Sex": zip(row_metadata["sex"].unique(), sex_colors),
        "Dev Stage": zip(row_metadata["dev_stage"].unique(), dev_stage_colors),
        "Study": zip(row_metadata["study"].unique(), study_colors),
        "Disease": zip(row_metadata["disease"].unique(), disease_colors),
        "Level": zip(row_metadata["key"].unique(), key_colors),
        "Cutoff": zip(column_metadata["cutoff"].unique(), cutoff_colors),
        "Subsample": zip(column_metadata["subsample_ref"].unique(), subsample_ref_colors)
    }

    # Create a figure for the legends
    fig, axes = plt.subplots(len(legend_dict), 1, figsize=(6, len(legend_dict) * 1.5))

    # Loop through the legend dictionary and create the patches for each category
    for i, (title, elements) in enumerate(legend_dict.items()):
        patches = [Patch(facecolor=color, edgecolor="black", label=label) for label, color in elements]
        axes[i].legend(handles=patches, title=title, loc="upper left", frameon=True, ncol=8)  
        axes[i].set_axis_off()

    plt.subplots_adjust(hspace=0.6)  
    plt.tight_layout()
    plt.savefig("legend.png", bbox_inches='tight')
    


def plot_heatmap_mouse(weighted_f1_results):
    
    weighted_f1_results = weighted_f1_results.fillna("None")    
    # Pivot data to create a large heatmap
    heatmap_data = pd.pivot_table(
        weighted_f1_results, 
        values="weighted_f1", 
        index=["query", "study", "sex", "age", "genotype", "strain", "treatment", "key"],
        columns=["reference", "cutoff", "subsample_ref", "method"]
    )
    
    write_na_report(heatmap_data)
 

    # Create the column_metadata DataFrame based on heatmap data columns
    column_metadata = pd.DataFrame(heatmap_data.columns.tolist(), columns=["reference", "cutoff", "subsample_ref", "method"])
    row_metadata = pd.DataFrame(heatmap_data.index.tolist(), columns=["query", "study", "sex", "age", "genotype", "strain", "treatment", "key"])

    # Define color palettes for each column category
    reference_colors = sns.color_palette("Set1", n_colors=len(column_metadata["reference"].unique()))
    method_colors = sns.color_palette("husl", n_colors=len(column_metadata["method"].unique()))
   # ref_split_colors = sns.color_palette("Set2", n_colors=len(column_metadata["ref_split"].unique()))

    # For continuous variables, use a continuous colormap
   # cutoff_cmap = sns.color_palette("coolwarm", as_cmap=True)  # Use a continuous colormap
   # subsample_ref_cmap = sns.color_palette("viridis", as_cmap=True)  # Another continuous colormap

    # Define colors for the row categories
    study_colors = sns.color_palette("Set1", n_colors=len(row_metadata["study"].unique()))
    sex_colors = sns.color_palette("Set1", n_colors=len(row_metadata["sex"].unique()))
    age_colors = sns.color_palette("Set2", n_colors=len(row_metadata["age"].unique()))
    genotype_colors = sns.color_palette("Set3", n_colors=len(row_metadata["genotype"].unique()))
    strain_colors = sns.color_palette("Set3", n_colors=len(row_metadata["strain"].unique()))
    treatment_colors = sns.color_palette("Set1", n_colors=len(row_metadata["treatment"].unique()))
    key_colors = sns.color_palette("Set3", n_colors=len(row_metadata["key"].unique()))
    cutoff_colors = sns.color_palette("coolwarm", n_colors=len(column_metadata["cutoff"].unique()))
    subsample_ref_colors = sns.color_palette("viridis", n_colors=len(column_metadata["subsample_ref"].unique()))

    # Map color palettes to the rows (categorical)
    row_colors = pd.DataFrame({
        'Study': row_metadata['study'].map(dict(zip(row_metadata["study"].unique(), study_colors))),
        'Sex': row_metadata["sex"].map(dict(zip(row_metadata["sex"].unique(), sex_colors))),
        'Age': row_metadata["age"].map(dict(zip(row_metadata["age"].unique(), age_colors))),
        'Genotype': row_metadata["genotype"].map(dict(zip(row_metadata["genotype"].unique(), genotype_colors))),
        'Strain': row_metadata["strain"].map(dict(zip(row_metadata["strain"].unique(), strain_colors))),
        'Treatment': row_metadata["treatment"].map(dict(zip(row_metadata["treatment"].unique(), treatment_colors))),
        'Level': row_metadata["key"].map(dict(zip(row_metadata["key"].unique(), key_colors)))
    }, index=row_metadata.index)

    # Map color palettes to the columns
    col_colors = pd.DataFrame({
        'Reference': column_metadata['reference'].map(dict(zip(column_metadata["reference"].unique(), reference_colors))),
        'Method': column_metadata['method'].map(dict(zip(column_metadata["method"].unique(), method_colors))),
       # 'Split': column_metadata['ref_split'].map(dict(zip(column_metadata["ref_split"].unique(), ref_split_colors))),
        'Cutoff': column_metadata['cutoff'].map(dict(zip(column_metadata["cutoff"].unique(), cutoff_colors))),
        'Subsample': column_metadata['subsample_ref'].map(dict(zip(column_metadata["subsample_ref"].unique(), subsample_ref_colors)))
    }, index=column_metadata.index)

    heatmap_data.columns = range(heatmap_data.shape[1])
    heatmap_data.index = range(heatmap_data.shape[0])

    # Handle NaN or infinite values
    heatmap_data = heatmap_data.fillna(0)
    n_rows, n_cols = heatmap_data.shape
    n_samples = n_rows // 4
    # Create the title string
    title = f"Classification of {n_samples} query samples across {n_cols} parameter combinations"

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
    )
    g.fig.suptitle(title, y=1.05)  # Increase y value to move the title 
    plt.savefig("mouse_weighted_f1_heatmap.png", bbox_inches='tight')

    legend_dict = {
        "Method": zip(column_metadata["method"].unique(), method_colors),
      #  "Ref Split": zip(column_metadata["ref_split"].unique(), ref_split_colors),
        "Sex": zip(row_metadata["sex"].unique(), sex_colors),
        "Age": zip(row_metadata["age"].unique(), age_colors),
        "Genotype": zip(row_metadata["genotype"].unique(), genotype_colors),
        "Strain": zip(row_metadata["strain"].unique(), strain_colors),
        "Treatment": zip(row_metadata["treatment"].unique(), treatment_colors),
        "Level": zip(row_metadata["key"].unique(), key_colors),
        "Cutoff": zip(column_metadata["cutoff"].unique(), cutoff_colors),
        "Subsample": zip(column_metadata["subsample_ref"].unique(), subsample_ref_colors)
    }

    # For continuous variables, we use ScalarMappable 
    
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
    plt.savefig("mouse_legend.png", bbox_inches='tight')

def main():
    plt.rcParams.update({'font.size': 25}) 
    # Parse arguments
    args = parse_arguments()

    # Read in data
    #label_f1_results = pd.read_csv(label_f1_results, sep = "\t")
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep = "\t")
 
    organism = weighted_f1_results["organism"].unique()[0]
    
    if organism == "homo_sapiens":
        plot_heatmap_human(weighted_f1_results)
        
    if organism == "mus_musculus":
        plot_heatmap_mouse(weighted_f1_results)
    
    
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