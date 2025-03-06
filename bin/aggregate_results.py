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
        np.nan: None
    }
    return dev_stage_mapping_dict[stage]
    
def write_factor_summary(df, factors): 
    # summarize the number of unique levels for each item in factors
    # make a value_counts table for each factor

    factor_summary = df[factors].nunique().reset_index()
    factor_summary.columns = ["factor", "levels"]
    factor_summary.to_csv("factor_summary.tsv", sep="\t", index=False) 
    value_counts = df[["study","query","query_region"]].value_counts().reset_index()
    value_counts.to_csv(f"region_study_value_counts.tsv", sep="\t", index=False)
 
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
     
    organism = f1_df["organism"].unique()[0]
    # replace "nan" with None
    f1_df = f1_df.replace("nan", None)
    #----------weighted f1 results----------------
    # fix errors and add factors
    
       
    f1_df["region_match"] = f1_df.apply(lambda row: row['query_region'] in row['ref_region'], axis=1)
    f1_df["reference_acronym"] = f1_df["reference"].apply(make_acronym)
    #f1_df["query_acronym"] = f1_df["query"].apply(make_acronym)
    f1_df["reference"] = f1_df["reference"].str.replace("_", " ")
    f1_df["study"] = f1_df["query"].apply(lambda x: x.split("_")[0])
    f1_df["query"] = f1_df["query"].str.replace("_", " ")
    
    
    f1_df["disease"] = np.where(f1_df["study"]== "GSE211870", "Control", f1_df["disease"])
    # eric queries
    f1_df["disease_state"] = np.where(f1_df["disease"] == "Control", "Control", "Disease")
    # handle Gemma
    
    if organism == "homo_sapiens":
        # deal with annotation mismatch between gemma queries and curated queries
        f1_df["dev_stage"] = f1_df["dev_stage"].apply(map_development_stage)
    
        # replace rosmap infant with rosmap late adult
        f1_df["dev_stage"] = np.where(f1_df["study"] == "rosmap" , "late adult", f1_df["dev_stage"])
        f1_df["dev_stage"] = np.where(f1_df["query"] == "lim C5382Cd" , "late adult", f1_df["dev_stage"])
        f1_df["dev_stage"] = np.where(f1_df["study"] == "pineda" , "late adult", f1_df["dev_stage"])
        #f1_df["dev_stage"] = np.where(f1_df["dev_stage"] == np.nan , "late adult", f1_df["dev_stage"])
       # f1_df["dev_sage"] = np.where(f1_df["dev_stage"] == None , "late adult", f1_df["dev_stage"])

        f1_df["sex"] = np.where(f1_df["query"]=="lim C5382Cd", "M", f1_df["sex"])
        f1_df["sex"] = f1_df["sex"].str.replace("male", "M")
        
    if organism == "mus_musculus":
        f1_df["disease_state"] = np.where(f1_df["disease"] == "No disease", "Control", "Disease")

#----------------drop label columns and save---------------
    outdir = "weighted_f1_distributions"
    label_columns = ["label", "f1_score","precision","recall","support", "accuracy"]
    os.makedirs(outdir, exist_ok=True)
    
    # Drop duplicates, but exclude 'ref_split' column (so duplicates in 'ref_split' are allowed)
    weighted_f1_results = f1_df.drop(columns=label_columns)
    # drop duplicates
    weighted_f1_results = weighted_f1_results.drop_duplicates()
    # Keep only rows where 'weighted_f1' is not null
    weighted_f1_results = weighted_f1_results[weighted_f1_results["weighted_f1"].notnull()] 
    # fill na with "None"
    weighted_f1_results = weighted_f1_results.fillna("None")
    weighted_f1_results.to_csv("weighted_f1_results.tsv", sep="\t", index=False)
    
#-----------------plotting distribution ---------------- 
    g = sns.FacetGrid(weighted_f1_results, col="key", hue="study", height=4, aspect=1.5)
    # Map the KDE plot to the FacetGrid
    g.map(sns.histplot, 'weighted_f1', multiple="layer", bins=40, binrange=(0, 1))
    # Set axis labels and titles
    g.set_axis_labels("F1 Scores", "Frequency")
    g.set_titles("F1 Score Distribution by {col_name}")
    # Add a legend
    g.add_legend()
    # Save and display the plot
    g.savefig("weighted_f1_distribution.png")
    #plt.show()
    
   #------------summaries---------------- 

        # summarize by sample, key, method, mean, sd
    weighted_summary = weighted_f1_results.groupby(["method","cutoff","reference","key"]).agg(
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        weighted_f1_count=("weighted_f1", "count"),
        #add precision and recall
        weighted_precision_mean=("weighted_precision", "mean"),
        weighted_precision_std=("weighted_precision", "std"),
        weighted_recall_mean=("weighted_recall", "mean"),
        weighted_recall_std=("weighted_recall", "std")
    ).reset_index()
    
    
    weighted_summary.to_csv("weighted_f1_summary.tsv", sep="\t", index=False)   
    
    #-------------boxplots--------------
    df_list = [group for _, group in weighted_f1_results.groupby('key')]

    for df in df_list:
        key = df["key"].values[0]
        #remove cutoff from columns_to_group
        columns_to_group = ["cutoff"] 
        df["cutoff"] = pd.Categorical(df["cutoff"])
        for column in columns_to_group:
            plt.figure(figsize=(10, 6))  # Set the figure size for the plot
            sns.boxplot(data=df, x="method", y="weighted_f1", hue=column, dodge=True)

            # Rotate x-axis labels for better visibility
            plt.xticks(rotation=90)

            # Add titles and labels
            plt.title(f"{key}: Weighted F1 Score by Method and {column}")
            plt.xlabel("Method")
            plt.ylabel("Weighted F1 Score")

            # Show the plot
            plt.tight_layout()
            plt.show()  # To display the plot

            # Save the plot
            plt.savefig(f"{key}_weighted_f1_boxplot_{column}.png", bbox_inches="tight")

        # 

            



# -----------label f1 results----------------
    label_results = f1_df[f1_df['label'].notnull()]
    label_results = label_results[label_results["f1_score"].notnull()]
    print(label_results)


    label_results = label_results.fillna("None")
    # fill "Neuron" ,"Glutamatergic", and "GABAergic" in "subclass" column with "ambigious {label}"
    # eventually add all ambiguous labels?
    # This way we can represent ambiguous author labels
    label_results["label"] = label_results.apply(
        lambda row: f"Ambiguous {row['label']}" if row["key"] == "subclass" and row["label"] in ["Neuron", "Glutamatergic", "GABAergic"] else row["label"], 
        axis=1
    )

    
    
    label_results.to_csv("label_f1_results.tsv", sep="\t", index=False)
   # Ensure precision and recall are numeric and handle 'nan' strings (if needed)
    label_results['precision'] = pd.to_numeric(label_results['precision'], errors='coerce')
    label_results['recall'] = pd.to_numeric(label_results['recall'], errors='coerce')
 
    # plot distribution of label_f1 across different splits
    outdir = "label_distributions"
    os.makedirs(outdir, exist_ok=True)
    # make a count summary table for label_f1 by label, sample, disease_state, sex, dev_stage
    label_summary = label_results.groupby(["label","method","cutoff","reference","key"]).agg(
        label_f1_mean=("f1_score", "mean"),
        label_f1_std=("f1_score", "std"),
        label_f1_count=("f1_score", "count")
        # add precision and recall
       # label_precision_mean=("precision", "mean"),
        #label_precision_std=("precision", "std"),
      #  label_recall_mean=("recall", "mean"),
       # label_recall_std=("recall", "std")
    ).reset_index()
        
    
    label_summary.to_csv("label_f1_summary.tsv", sep="\t", index=False)
   

    if organism == "homo_sapiens":
        columns_to_group=["label","method", "disease", "cutoff", "sex", "dev_stage", "reference", "study"]
    if organism == "mus_musculus":
       columns_to_group=["label","method", "treatment", "genotype","strain","cutoff", "sex", "age", "reference", "study"] 
    factors=columns_to_group + ["query"] + ["query_region"] + ["ref_region"]
    write_factor_summary(label_results, factors) 
   
        
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
    
