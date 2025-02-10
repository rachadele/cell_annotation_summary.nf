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
    parser.add_argument('--vars', type=str, nargs = "+", help="Names of factor columns")
    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/full_query/label_f1_results.tsv")   
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args
    
    return parser.parse_args()

def main():
    
    # Parse arguments
    args = parse_arguments()
    vars = args.vars
    label_f1_results = args.label_f1_results
    
    # Read in data
    label_f1_results = pd.read_csv(label_f1_results, sep = "\t")
    
    # For each label at the "subclass" level, calculate the correlation between the label's f1 score and the support
    
    # Create a dictionary to store the results
    label_support_corr = {}
    subclass_subset = label_f1_results[label_f1_results["key"] == "subclass"]
    
    # For each label, calculate the correlation between the f1 score and the support
    for label in subclass_subset["label"].unique():
        label_subset = subclass_subset[subclass_subset["label"] == label]
        # if ref_support is not NA, calculate the correlation
        if label_subset["ref_support"].isna().sum() == 0:
            corr = label_subset["f1_score"].corr(label_subset["ref_support"])
            label_support_corr[label] = corr
        
    # plot correlations between f1 and support as heatmap
    label_support_corr_df = pd.DataFrame(label_support_corr, index = ["correlation"]).T
    label_support_corr_df = label_support_corr_df.reset_index()
    label_support_corr_df.columns = ["label", "correlation"]
    label_support_corr_df = label_support_corr_df.sort_values(by = "correlation", ascending = False)
    
    # plot bar plot and color by correlation
    #add legend
    plt.figure(figsize=(10, 8)) 
    sns.barplot(x = "label", y = "correlation", data = label_support_corr_df, hue="correlation", palette="coolwarm")
    
    plt.xticks(rotation=90)
    #move legend outside of plot
    plt.legend(title='Correlation', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title("Correlation between F1 score and reference support for each label")
    plt.savefig("label_support_corr.png")


if __name__ == "__main__":
    main()