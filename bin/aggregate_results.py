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
    parser.add_argument('--pipeline_results', type=str, nargs = "+", 
                        help="files containing f1 results with params")                                            
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

 
        
def make_acronym(name):
    # Split on "_" and replace with spaces
    words = name.split("_")
    # Create acronym from the first letter of each word
    acronym = "".join(word[0].upper() for word in words if word)
    return acronym

def map_development_stage(stage):
    # re write dict
    dev_stage_mapping_dict = {
        "HsapDv_0000083": "infant",
        "HsapDv_0000084": "toddler",
        "HsapDv_0000085": "child",
        "HsapDv_0000086": "adolescent",
        "HsapDv_0000088": "adult",
        "HsapDv_0000091": "late adult",
        np.nan: "late adult"
    }
    return dev_stage_mapping_dict[stage]
    
        
 
def main():
    # Parse command line arguments
    args = parse_arguments()
    # Set organism and census_version from arguments
    pipeline_results = args.pipeline_results

    
    f1_df = pd.DataFrame() 
     
    #for file in os.listdir(pipeline_results):
    for filepath in pipeline_results:
    #filepath = os.path.join(pipeline_results, file)
    # method = filepath.split("/")[-3]
        temp_df = pd.read_csv(filepath,sep="\t")
        # temp_df["method"] = method
        f1_df = pd.concat([temp_df, f1_df], ignore_index=True)
        
    f1_df["region_match"] = f1_df.apply(lambda row: row['query_region'] in row['ref_region'], axis=1)
    f1_df["reference_acronym"] = f1_df["reference"].apply(make_acronym)
    #f1_df["query_acronym"] = f1_df["query"].apply(make_acronym)
    f1_df["reference"] = f1_df["reference"].str.replace("_", " ")
    f1_df["study"] = f1_df["query"].apply(lambda x: x.split("_")[0])
    f1_df["query"] = f1_df["query"].str.replace("_", " ")
    f1_df["disease_state"] = np.where(f1_df["disease"] == "Control", "Control", "Disease")
    f1_df["dev_stage"] = f1_df["dev_stage"].apply(map_development_stage)
    
    # replace rosmap infant with rosmap late adult
    f1_df["dev_stage"] = np.where(f1_df["study"] == "rosmap" , "late adult", f1_df["dev_stage"])
    f1_df["dev_stage"] = np.where(f1_df["query"] == "lim C5382Cd" , "late adult", f1_df["dev_stage"])
    f1_df["sex"] = np.where(f1_df["query"]=="lim C5382Cd", "M", f1_df["sex"])
    f1_df["sex"] = f1_df["sex"].str.replace("male", "M")
    
    # Boxplots: Show the effect of categorical parameters
    categorical_columns = ['query', 'reference','method','ref_split', 'region_match',"subsample_ref","sex","disease_state","dev_stage","cutoff"] #organism, other categoricals
    outdir = "weighted_f1_distributions"
    label_columns = ["label", "f1_score","ref_support","label_accuracy"]
    os.makedirs(outdir, exist_ok=True)
    
    # Drop duplicates, but exclude 'ref_split' column (so duplicates in 'ref_split' are allowed)
    weighted_f1_results = f1_df.drop(columns=label_columns)
    # drop duplicates
    weighted_f1_results = weighted_f1_results.drop_duplicates()
    # Keep only rows where 'weighted_f1' is not null
    weighted_f1_results = weighted_f1_results[weighted_f1_results["weighted_f1"].notnull()] 
    weighted_f1_results.to_csv("weighted_f1_results.tsv", sep="\t", index=False)
    
    
        # Example: adding hue and faceting to the weighted F1 scores distribution plot
    sns.histplot(weighted_f1_results, x='weighted_f1', hue='key', multiple="fill", palette="Set1")
    plt.xlabel("Weighted F1")
    plt.ylabel("Frequency")
    plt.title("Distribution of Weighted F1 Scores by Key")
    plt.savefig("weighted_f1_distribution.png")
    #plt.show()
    
    
    # summarize by sample, key, method, mean, sd
    weighted_summary = weighted_f1_results.groupby(["key", "query", "disease_state","sex","dev_stage","method","cutoff"]).agg(
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        weighted_f1_count=("weighted_f1", "count")
    ).reset_index()
    weighted_summary.to_csv("weighted_f1_summary.tsv", sep="\t", index=False) 
    # plot boxplots of the raw data
    # Loop through columns and plot a boxplot for each
    for column in ["method", "disease", "cutoff", "sex", "dev_stage", "reference", "study"]:
        plt.figure(figsize=(10, 6))  # Set the size for each plot
        sns.boxplot(data=weighted_f1_results, x=column, y="weighted_f1", hue="key")
        plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
        plt.title(f"Boxplot of Weighted F1 Score by {column.capitalize()} and Key")
        plt.tight_layout()  # Ensure everything fits
        plt.show()
    
        

    order = ["subclass", "class", "family"]  # Desired order  


# -----------label f1 results----------------
    label_results = f1_df[f1_df['label'].notnull()]
    label_results = label_results[label_results["f1_score"].notnull()]
    label_results = label_results.drop_duplicates(subset=label_results.columns.difference(['ref_support']))
    #label_results = label_results.drop_duplicates(subset=label_results.columns.difference(['ref_split']))
    label_results.to_csv("label_f1_results.tsv", sep="\t", index=False)
    
    # plot distribution of label_f1 across different splits
    outdir = "label_distributions"
    os.makedirs(outdir, exist_ok=True)

    # make a count summary table for label_f1 by label, sample, disease_state, sex, dev_stage
    label_summary = label_results.groupby(["key","label", "query", "disease_state","sex","dev_stage"]).agg(
        label_f1_mean=("f1_score", "mean"),
        label_f1_std=("f1_score", "std"),
        label_f1_count=("f1_score", "count")
    ).reset_index()
    label_summary.to_csv("label_f1_summary.tsv", sep="\t", index=False)
   

        
    # Create the FacetGrid
    g = sns.FacetGrid(label_results, col="key", hue="label", height=4, aspect=1.5)
    # Map the KDE plot to the FacetGrid
    g.map(sns.histplot, 'f1_score', multiple="layer", bins=40, binrange=(0, 1))
    # Set axis labels and titles
    g.set_axis_labels("F1 Scores", "Frequency")
    g.set_titles("F1 Score Distribution by {col_name}")
    # Add a legend
    g.add_legend()
    # Save and display the plot
    g.savefig("label_f1_distribution_facet.png")
    #plt.show()

  
if __name__ == "__main__":
    main()
    
