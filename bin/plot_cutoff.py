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
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/query_500_sctransform/weighted_f1_results.tsv")
    parser.add_argument('--vars', type=str, nargs = "+", help="Names of factor columns")
    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/query_500_sctransform/label_f1_results.tsv")   
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args


def plot_line(df, x, y, hue, col, style, title, xlabel, ylabel, save_path):
    # change figure size
    plt.figure(figsize=(20, 10))
    g = sns.relplot(
        data=df, x=x, y=y,
        col=col, hue=hue, style=style,
        kind="line", height=4, aspect=0.75, legend="full"  # Adjust figure size
    )
    title = title.replace("_", " ").title()  # Capitalize and substitute "_" with " " 
    g.figure.suptitle(title, y=1, x = 0.5)  # Add title above plots
    g.set_axis_labels(xlabel, ylabel)  # Set axis labels
    # set titles with capitalization and substitute "_" with " "
    g.set(xticks=[0,0.25,0.5,0.75])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # make legend fontzie smaller
    plt.setp(g._legend.get_texts(), fontsize=12)  # Adjust legend font size
   # plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")

 
def main():
    plt.rcParams.update({'font.size': 15})  # Set default font size for all plot elements

    args = parse_arguments()
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
    weighted_f1_results["study"] = weighted_f1_results["query"].str.split(" ").str[0]
    label_f1_results = pd.read_csv(args.label_f1_results, sep="\t")
    label_f1_results["study"] = label_f1_results["query"].str.split("_").str[0]
    categoricals = ['study','reference','method','ref_split', 'region_match',"sex","disease_state","dev_stage"]


    # Convert factor columns to categorical
    for factor in categoricals:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('category')
        label_f1_results[factor] =label_f1_results[factor].astype('category')
    
    for factor in ["cutoff", "subsample_ref"]:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('float')
        label_f1_results[factor] =label_f1_results[factor].astype('float')
    
    categories = ['reference', 'ref_split', 'study', 'region_match', 'disease_state', 
                'dev_stage', 'sex', 'query_region', 'ref_region', 'subsample_ref']

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
        
        label_long = pd.melt(subset, id_vars=['cutoff', 'f1_score', 'method','label','reference',"subsample_ref"], value_vars=categories, 
                                    var_name='category', value_name='category_value')
        
        g = sns.FacetGrid(label_long, col="method", row="subsample_ref", height=6, sharey=False)
        
            # Use map_dataframe to allow hue
        g.map_dataframe(sns.lineplot, x="cutoff", y="f1_score", hue="label")
        # Set axis labels and titles
        g.set_axis_labels("Cutoff", "F1 Score")
        # capitalize titled

        # Adjust layout
        g.fig.subplots_adjust(top=0.85)
        # Extract legend handles and labels
        handles, labels = g.axes[0, 0].get_legend_handles_labels()
        # Create a separate legend figure
        fig_legend, ax_legend = plt.subplots(figsize=(25, 10))
        ax_legend.legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=12)
        ax_legend.axis("off")
        plt.show()
        g.savefig(os.path.join(outdir, f"{key_value}_label_f1_score_ref.png"))
        fig_legend.savefig(os.path.join(outdir, f"{key_value}_label_ref_legend.png"))

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
   
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method",hue="reference", xlabel="Cutoff", style=None,
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_ref_split.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="study", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_query.png"))

        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="region_match", style=None, xlabel="Cutoff",
                    ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_region_match.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="disease_state", style=None, xlabel="Cutoff",
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_disease_state.png"))
        
        #plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="dev_stage", style=None, xlabel="Cutoff",
                  #ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_dev_stage.png"))
        
        #plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="disease_state", style=None, xlabel="Cutoff",
                  #ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_disease_state.png"))
        
        #plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="sex", style=None, xlabel="Cutoff",
                  #ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_sex.png"))
        
        ##plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="query_region", style=None, xlabel="Cutoff",
         ##         ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_query_region.png"))
        
        plot_line(weighted_subset, x="cutoff", y="weighted_f1", col="method", hue="subsample_ref", xlabel="Cutoff", style=None,
                  ylabel="F1 Score", title=f"{key_value}", save_path=os.path.join(outdir,f"{key_value}_weighted_f1_score_subsample_ref.png"))
        

        # Facet by both method (columns) and ref_split (rows)
        g = sns.FacetGrid(weighted_subset, col="method", row="ref_split", height=6, sharey=False)
        # Use map_dataframe to allow hue
        g.map_dataframe(sns.lineplot, x="cutoff", y="weighted_f1", hue="reference", style="ref_split")
        # Set axis labels and titles
        g.set_axis_labels("Cutoff", "F1 Score")
        # capitalize titled

        # Adjust layout
        g.fig.subplots_adjust(top=0.95)
        # Extract legend handles and labels
        handles, labels = g.axes[0, 0].get_legend_handles_labels()
        # Create a separate legend figure
        fig_legend, ax_legend = plt.subplots(figsize=(15, 10))
        ax_legend.legend(handles, labels, loc="center", ncol=2, frameon=False, fontsize=12)
        ax_legend.axis("off")

        g.savefig(os.path.join(outdir, f"{key_value}_weighted_f1_score_ref.png"))
        fig_legend.savefig(os.path.join(outdir, f"{key_value}_ref_legend.png"))

        plt.show()
                                              
if __name__ == "__main__":
    main()