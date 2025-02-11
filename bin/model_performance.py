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
#import adata_functions
#from adata_functions import *
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import os
import json
import ast
import sys
import matplotlib.lines as mlines

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--weighted_f1_results', type=str, default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/full_query/weighted_f1_results.tsv", help="Aggregated weighted results")
    parser.add_argument('--label_f1_results', type=str, default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/full_query/label_f1_results.tsv", help="Label level f1 results")                                            
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

    
def setup_plot(var, split, facet=None):
    """Set up the basic plot structure."""
    
    plt.figure(figsize=(17, 10))
    plt.xlabel(split.replace("_"," ").capitalize(), fontsize=25)
    var = var.replace("_", " ")
    #plt.ylabel(f"{var}", fontsize=25)
    plt.ylabel("Performance (weighted F1)", fontsize=25)
    #plt.title(f'Distribution of {var} across {split}', fontsize=25)


def add_violin_plot(df, var, split, facet):
    """Add a violin plot to the figure."""
    # remove extra weighted f1 values
    
    #df = df.drop_duplicates(subset=[split, facet, var])
    sns.violinplot(
        data=df, 
        y=var, 
        x=split, 
        palette="Set2", 
        hue=facet, 
        split=False, 
        dodge=True
    )

def add_strip_plot(df, var, split, facet, add_region_match=True):
    """Add a strip plot to the figure."""
    # remove extra weighted f1 values
    # doesn't change overall values
    #df = df.drop_duplicates(subset=[split, facet, var])
    df['match_region'] = df.apply(lambda row: row['query_region'] in row['ref_region'], axis=1)
    # Map match_region to colors before plotting
    df['color'] = df['match_region'].map({True: 'red', False: 'grey'})
    
    # Separate data into two groups based on 'match_region'
    mask = df['match_region']
    match_region_df = df[mask]
    non_match_region_df = df[~mask]
    
    # Create the strip plot for non-matching region data
    ax = sns.stripplot(
        data=non_match_region_df,
        y=var,
        x=split,
        hue=facet,
        dodge=True,          
        palette="Set2",      
        size=2,
        alpha=0.8,           
        jitter=True,
        marker="o",
        edgecolor='grey',    # Grey edge color for non-match region
        linewidth=0.5
    )
    # Create the strip plot for matching region data with customized edgecolor
    sns.stripplot(
        data=match_region_df,
        y=var,
        x=split,
        hue=facet,
        dodge=True,          
        palette="Set2",      
        size=2,
        alpha=0.8,           
        jitter=True,
        marker="o",
        edgecolor='r',       # Red edge color for match region
        linewidth=0.5,         
        legend=None,         # Disable legend for second plot
        ax=ax                # Add to same axis
    )
    # Create custom legend handles for edge color
          

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

def save_plot(var, split, facet, outdir):
    """Save the plot to the specified directory."""
    os.makedirs(outdir, exist_ok=True)
    var = var.replace(" ", "_")
    save_path = os.path.join(outdir, f"{var}_{split}_{facet}_distribution.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_distribution(df, var, outdir, split=None, facet=None, acronym_mapping=None, add_region_match=True):
    """
    Create a violin and strip plot for the given variable across groups.
    
    Parameters:
        df (pd.DataFrame): Data to plot.
        var (str): Column name for the variable to plot.
        outdir (str): Directory to save the plot.
        split (str): Column name to split the x-axis.
        facet (str): Column name for facet grouping (optional).
        acronym_mapping (dict): Mapping for acronyms to add as a legend (optional).
    """
    setup_plot(var, split)
    add_violin_plot(df, var, split, facet)
    if add_region_match:
        add_strip_plot(df, var, split, facet)
    #plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0)
    plt.xticks(rotation=90, ha="right", fontsize=25)
    plt.yticks(fontsize=25)

    # Add the custom legend to the plot
    #plt.legend(handles=[red_patch, grey_patch], title="Match region", loc='upper left', bbox_to_anchor=(1, 1.02))
    
    handles, labels = plt.gca().get_legend_handles_labels()
    facet_legend = plt.legend(
    handles=handles,  # Use all the handles
    labels=labels,    # Ensure that we provide all the labels too
    title=facet.replace("_", " ").capitalize() if facet else "Group",  # Title formatting
    loc='upper left',
    bbox_to_anchor=(1.05, 1),
    borderaxespad=0,
    fontsize=15
    )


    if add_region_match:
        red_patch = mlines.Line2D([], [], color='red', marker='o', markersize=7, label='Matching region')
        grey_patch = mlines.Line2D([], [], color='grey', marker='o', markersize=7, label='Non-Matching region')
        # Add custom legend for match_region
        plt.gca().add_artist(facet_legend)  # Add facet legend separately
        plt.legend(
            handles=[red_patch, grey_patch],
            #title="Match region",
            loc='upper left',
            bbox_to_anchor=(1.05, 0.3),
            fontsize=15
        )

    # Move the legend to the desired location
    #sns.move_legend(plt, bbox_to_anchor=(1, 1.02), loc='upper left')

    add_acronym_legend(acronym_mapping, title=split.split("_")[0].capitalize())
    plt.tight_layout()
    save_plot(var, split, facet, outdir)

 
        
def make_acronym(ref_name):
    # Split on "_" and replace with spaces
    words = ref_name.split("_")
    # Create acronym from the first letter of each word
    acronym = "".join(word[0].upper() for word in words if word)
    return acronym
    
def main():
    
    args = parse_arguments()
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
    label_results = pd.read_csv(args.label_f1_results, sep="\t")

    # Boxplots: Show the effect of categorical parameters
    categorical_columns = ['query', 'method', 'reference']
    outdir = "weighted_f1_distributions"
    label_columns = ["label", "f1_score"]
    os.makedirs(outdir, exist_ok=True)


    for key in weighted_f1_results["key"].unique():
        df_subset = weighted_f1_results[weighted_f1_results["key"] == key]
        outdir = f"weighted_f1_distributions/{key}"
        os.makedirs(outdir, exist_ok=True)
        for col in categorical_columns:
            if col not in ["method"]:   
                plot_distribution(df_subset, var="weighted_f1", outdir=outdir, split=col, facet="method", 
                                acronym_mapping=None)
                
    for key in label_results["key"].unique():
        df_subset = label_results[label_results["key"] == key]
        outdir = f"label_distributions/{key}"
        os.makedirs(outdir, exist_ok=True)
        for col in categorical_columns:
          # if col != "method":
            plot_distribution(df_subset, var="f1_score",outdir=outdir, split=col, facet="label", 
                        acronym_mapping = None, add_region_match=False)
    
            
    