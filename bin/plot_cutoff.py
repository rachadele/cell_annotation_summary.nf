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
import random
random.seed(42)

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_hsap/aggregated_results/weighted_f1_results.tsv")
    parser.add_argument('--vars', type=str, nargs = "+", help="Names of factor columns")
    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_hsap/aggregated_results/label_f1_results.tsv")   
    parser.add_argument('--color_mapping_file', type=str, help="Mapping file", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/meta/color_mapping.tsv")
    parser.add_argument('--mapping_file', type=str, help="Mapping file", default = "/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/meta/census_map_human.tsv")
    
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

def make_stable_colors(color_mapping_df):
    
    all_subclasses = sorted(color_mapping_df["subclass"])
    # i need to hardcode a separate color palette based on the hsap mapping file
    # Generate unique colors for each subclass
    color_palette = sns.color_palette("husl", n_colors=len(all_subclasses))
    subclass_colors = dict(zip(all_subclasses, color_palette))
    return subclass_colors
    
    
def plot_line(df, x, y, hue, col, style, title, xlabel, ylabel, save_path):
    # set global fontsize
    plt.rcParams.update({'font.size': 14})  # Set default font size for all plot elements
    
    all_levels = df[hue].unique()
    
    colors = {subclass: color for subclass, color in zip(all_levels, sns.hls_palette(len(all_levels)))}

    # change figure size
    plt.figure(figsize=(22, 10))
    g = sns.relplot(
        data=df, x=x, y=y,
        col=col, hue=hue, style=style, palette=colors,
        kind="line", height=4, aspect=0.75, legend="full"  # Adjust figure size
    )
    title = title.replace("_", " ").title()  # Capitalize and substitute "_" with " " 
    g.figure.suptitle(title, y=1, x = 0.5)  # Add title above plots
    g.set_axis_labels(xlabel, ylabel)  # Set axis labels

    g.set(xticks=[0,0.25,0.5,0.75])
    # set xtick fontsize
    # Rotate x-tick labels for better visibility
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize =12)  # Rotate 45 degrees and align
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # make legend fontzie smaller
    plt.setp(g._legend.get_texts(), fontsize=12)  # Adjust legend font size
    
   # plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")




def plot_f1_by_celltype(label_f1_results, levels, color_mapping_df, mapping_df, outdir="label_f1_plots", level="family"):
    os.makedirs(outdir, exist_ok=True)
    new_outdir = os.path.join(outdir, level)
    os.makedirs(new_outdir, exist_ok=True)
    plt.rcParams.update({'font.size': 25})

    all_subclasses = sorted(levels["subclass"])
    
    # Create a consistent color palette based on all possible subclasses
    subclass_colors = make_stable_colors(color_mapping_df)

    for i, celltype in enumerate(levels[level]):
        # Get the subclasses for the current celltype
        group_subclasses = mapping_df[mapping_df[level] == celltype]["subclass"].unique()
        if len(group_subclasses) == 0:
            group_subclasses = [celltype]
        
        # Ensure subclasses are consistently ordered across all methods
        subclasses_to_plot = [subclass for subclass in all_subclasses if subclass in group_subclasses]
        
        # Filter the data for the current group
        filtered_df = label_f1_results[label_f1_results["label"].isin(subclasses_to_plot)]
        
        # Initialize the FacetGrid
        g = sns.FacetGrid(filtered_df, col="method", sharey=True, col_wrap=3, height=5)
        
        # Map the lineplot with consistent colors across all facets
        g.map_dataframe(sns.lineplot, x="cutoff", 
                        y="f1_score",
                        hue="label", 
                        marker="o", 
                        palette={label: subclass_colors[label] for label in subclasses_to_plot}
)
        
        g.set_axis_labels("Cutoff", "F1 Score")
        g.set_titles("{col_name}")
        g.add_legend(title="Label")
        plt.suptitle(f"{celltype} F1 Score vs. Cutoff", y=1.05)

        # Save plot
        celltype = celltype.replace(" ", "_").replace("/", "_")
        save_path = os.path.join(new_outdir, f"{celltype}_f1_score.png")
        g.savefig(save_path, bbox_inches="tight")
        plt.close()
   

 
def main():

    args = parse_arguments()
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
    weighted_f1_results["study"] = weighted_f1_results["query"].str.split(" ").str[0]
    label_f1_results = pd.read_csv(args.label_f1_results, sep="\t")
    label_f1_results["study"] = label_f1_results["query"].str.split("_").str[0]
    color_mapping_df = pd.read_csv(args.color_mapping_file, sep="\t")
    mapping_df = pd.read_csv(args.mapping_file, sep="\t")
    organism = weighted_f1_results["organism"].unique()[0]
    
    if organism == "homo_sapiens":
        categoricals = ['study','reference','method','ref_split','region_match',"sex","disease_state","dev_stage"]
    
    if organism == "mus_musculus":
        categoricals = ['study','reference','method','region_match','strain',"sex","genotype","age","treatment"]

    # Convert factor columns to categorical
    for factor in categoricals:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('category')
        label_f1_results[factor] =label_f1_results[factor].astype('category')
    
    for factor in ["cutoff", "subsample_ref"]:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('float')
        label_f1_results[factor] =label_f1_results[factor].astype('float')
   
    if organism == "homo_sapiens":
        categories = ['reference', 'ref_split', 'study', 'region_match', 'disease_state', 
                    'dev_stage', 'sex', 'query_region', 'ref_region', 'subsample_ref']
    if organism == "mus_musculus":
       categories = ['reference', 'ref_split', 'study', 'region_match', 'treatment', 
                    'age', 'sex', 'query_region', 'ref_region', 'subsample_ref','genotype','strain'] 


#--------plot label vs cutoff-------------
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
    
    plot_f1_by_celltype(label_f1_results, levels, color_mapping_df, mapping_df, outdir="label_f1_plots", level="family")
    plot_f1_by_celltype(label_f1_results, levels, color_mapping_df, mapping_df, outdir="label_f1_plots", level="class") 
    
    
    
    #-----------------plot weighted f1 score-------------------
    parent = "weighted_f1_plots"
    os.makedirs(parent, exist_ok=True)
    for key_value in weighted_f1_results["key"].unique():
        
        outdir = os.path.join(parent, key_value)
        os.makedirs(outdir, exist_ok=True)
        weighted_subset = weighted_f1_results[weighted_f1_results["key"] == key_value]
        # Convert the relevant categories into a long format for faceting

        
        # replace acce "_" with " " for all category
        # capitalize all strings
        for category in categories:
            if weighted_subset[category].dtype == "object":
                weighted_subset[category] = weighted_subset[category].str.replace("_", " ")

        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="reference", xlabel="Cutoff", style=None,
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_ref.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="study", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_query.png"))

        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="subsample_ref", xlabel="Cutoff", style=None,
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_subsample_ref.png"))

        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="query_region", style=None, xlabel="Cutoff",
                    ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_query_region.png")) 
        
        if organism == "homo_sapiens":
            plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="region_match", style=None, xlabel="Cutoff",
                        ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_region_match.png"))
            
            plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="disease_state", style=None, xlabel="Cutoff",
                    ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_disease_state.png"))
            
          
        if organism == "mus_musculus":
            plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="genotype", style=None, xlabel="Cutoff",
                        ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_genotype.png"))
            
            plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="strain", style=None, xlabel="Cutoff",
                    ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_strain.png"))
            
            plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="age", style=None, xlabel="Cutoff",
                    ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_age.png"))
            
            plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="treatment", style=None, xlabel="Cutoff",
                    ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_treatment.png"))
           
            
                                              
if __name__ == "__main__":
    main()