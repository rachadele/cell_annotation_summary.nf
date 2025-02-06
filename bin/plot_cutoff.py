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
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_eval/full_query/weighted_f1_results.tsv")
    parser.add_argument('--vars', type=str, nargs = "+", help="Names of factor columns")
    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_eval/full_query/label_f1_results.tsv")   
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args


def plot_line(df, x, y, hue, col, style, title, xlabel, ylabel, save_path):
    plt.rcParams.update({'font.size': 12})  # Set default font size for all plot elements

    g = sns.relplot(
        data=df, x=x, y=y,
        col=col, hue=hue, style=style,
        kind="line", height=4, aspect=0.75, legend="full"  # Adjust figure size
    )
    
    g.figure.suptitle(title, y=1.5, x = 0.5)  # Add title above plots
    g.set_axis_labels(xlabel, ylabel)  # Set axis labels
    g.set(xticks=[0,0.25,0.75])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
   # plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

 
def main():
    
    args = parse_arguments()
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
    weighted_f1_results["manuscript"] = weighted_f1_results["query"].str.split(" ").str[0]
    label_f1_results = pd.read_csv(args.label_f1_results, sep="\t")
    label_f1_results["manuscript"] = label_f1_results["query"].str.split("_").str[0]
    categoricals = ['manuscript','reference','method','ref_split', 'region_match',"sex","disease_state","dev_stage"]


    # Convert factor columns to categorical
    for factor in categoricals:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('category')
        label_f1_results[factor] =label_f1_results[factor].astype('category')
    
    for factor in ["cutoff", "subsample_ref"]:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('float')
        label_f1_results[factor] =label_f1_results[factor].astype('float')
    

    # plot cutoff vs f1
    # color by label and facet by method 

    outdir="label_f1_plots"
    os.makedirs(outdir,exist_ok=True) 
    for key_value in label_f1_results["key"].unique():
        subset = label_f1_results[label_f1_results["key"] == key_value]
        weighted_subset = weighted_f1_results[weighted_f1_results["key"] == key_value]
        weighted_subset["label"] = "Weighted F1"

        # Merge the data
        subset["Metric"] = "Label F1 Score"
        weighted_subset["Metric"] = "Weighted F1 Score"
        combined_df = pd.concat([subset.rename(columns={"f1_score": "score"}), 
                                weighted_subset.rename(columns={"weighted_f1": "score"})])

        plot_line(combined_df, x="cutoff", 
                  y="score", col="method", hue="label",
                  style="Metric", title=f"{key_value}", 
                  xlabel="Cutoff", ylabel="F1 Score", 
                  save_path=os.path.join(outdir,f"{key_value}_f1_score.png"))
        
    
    
    parent = "weighted_f1_plots"
    os.makedirs(parent, exist_ok=True)
    for key_value in weighted_f1_results["key"].unique():
        outdir = os.path.join(parent, key_value)
        os.makedirs(outdir, exist_ok=True)
        weighted_subset = weighted_f1_results[weighted_f1_results["key"] == key_value]
        # Convert the relevant categories into a long format for faceting
        #categories = ['reference', 'ref_split', 'manuscript', 'region_match', 'disease_state', 
                      #'dev_stage', 'sex', 'query_region', 'ref_region', 'subsample_ref']
        #weighted_subset_long = pd.melt(weighted_subset, id_vars=['cutoff', 'weighted_f1', 'method'], value_vars=categories, 
                                    #var_name='category', value_name='category_value')

        ## Create a FacetGrid with different categories
        #g = sns.FacetGrid(weighted_subset_long, col='category_value', col_wrap=4, height=3, sharey=False)

        ## Map the lineplot to the grid
        #g.map(sns.lineplot, 'cutoff', 'weighted_f1', 'method')

        ## Customize the plot
        #g.set_axis_labels('Cutoff', 'F1 Score')
        #g.set_titles("{col_name}")
        #g.set(ylim=(0, 1))  # Adjust as needed for your data

        ## Save the combined plot
        #g.fig.suptitle(f"{key_value} - Weighted F1 Score", fontsize=16)
        #g.fig.tight_layout()
        #g.fig.subplots_adjust(top=0.9)  # Adjust the title
        #g.savefig(os.path.join(outdir, f"{key_value}_combined_weighted_f1_score.png"))

    

        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="reference", xlabel="Cutoff", style="ref_split",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_ref.png"))
   
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method",hue="ref_split", xlabel="Cutoff", style=None,
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_ref_split.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="manuscript", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_query.png"))

        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="region_match", style=None, xlabel="Cutoff",
                    ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_region_match.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="disease_state", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_disease_state.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="dev_stage", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_dev_stage.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="disease_state", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_disease_state.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="sex", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_sex.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="query_region", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_query_region.png"))
        
        #plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="ref_region", style=None, xlabel="Cutoff",  
         #         ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_reference_region.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="subsample_ref", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_subsample_ref.png"))
        
     
                                                                                  
                                                                                  
if __name__ == "__main__":
    main()