    
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


# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_eval/weighted_f1_results.tsv")
    parser.add_argument('--vars', type=str, nargs = "+", help="Names of factor columns")
    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_eval/label_f1_results.tsv")   
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

def main():
    args = parse_arguments()
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
    label_f1_results = pd.read_csv(args.label_f1_results, sep="\t")
   
    if not args.vars:
    # Assuming f1_results is your pandas DataFrame and factor_names is a list of column names
        factor_names = ["ref_tissue","query_tissue","tissue_match","subsample_ref","method"]  # replace with actual factor column names
    else:
        factor_names = args.vars
    # Convert factor columns to categorical
    for factor in factor_names:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('category')
        label_f1_results[factor] =label_f1_results[factor].astype('category')
    # Convert weighted_f1 to numeric
    #f1_results['weighted_f1'] = pd.to_numeric(f1_results['weighted_f1'], errors='coerce')
    #weighted_f1_results = f1_results[factor_names + ["key"] + ["weighted_f1"]].drop_duplicates()
    df_list = [group for _, group in weighted_f1_results.groupby('key')]
    # run anova for each df in df_list
    for df in df_list:
    # Run anova
        formula = 'weighted_f1 ~ ' + ' + '.join(factor_names)
        model = ols(formula, data=df).fit()
        aov_table = anova_lm(model, typ=2)
        
        # FDR correction
        aov_table['FDR'] = np.append(stats.false_discovery_control(aov_table['PR(>F)'].dropna()), np.nan)
        
        print(aov_table)
        # Apply -log10 transformation to the FDR values for the plot
        aov_table['-log10(FDR)'] = -np.log10(aov_table['FDR'])
        aov_table = aov_table.drop('Residual', axis=0)
        
        fig, ax = plt.subplots(figsize=(10, 5))
 
        # Plot with the transformed y-values
        sns.barplot(x='index', y='-log10(FDR)', data=aov_table.reset_index(), ax=ax, hue='index')
        # Set labels and title
        ax.set_ylabel('-log10(FDR)', fontsize=20)
        ax.set_xlabel('Factor', fontsize=20)
        ax.set_title(f"{df['key'].values[0]}", fontsize=20)
        # Rotate x-axis labels
        plt.xticks(rotation=90, fontsize=20)
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"{df['key'].values[0]}_anova_global_plot.png")
        
    # do this for individual label f1 scores

    #label_results = f1_results[f1_results['label'].notnull()]
    df_list = [group for _, group in label_f1_results.groupby('key')]
    for df in df_list:
        df = df[df['label'].notnull()]
        # Run anova
        formula = 'f1_score ~ ' + ' + '.join(factor_names)  + ' + label'
        model = ols(formula, data=df).fit()
        aov_table = anova_lm(model, typ=2)
    
        # FDR correction
        aov_table['FDR'] = np.append(stats.false_discovery_control(aov_table['PR(>F)'].dropna()), np.nan)
    
        print(aov_table)
        # Apply -log10 transformation to the FDR values for the plot
        aov_table['-log10(FDR)'] = -np.log10(aov_table['FDR'])
        aov_table = aov_table.drop('Residual', axis=0)
    
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot with the transformed y-values
        sns.barplot(x='index', y='-log10(FDR)', data=aov_table.reset_index(), ax=ax, hue='index')
        # Set labels and title
        ax.set_ylabel('-log10(FDR)', fontsize=20)
        ax.set_xlabel('Factor', fontsize=20)
        ax.set_title(f"{df['key'].values[0]}", fontsize=20)
        # Rotate x-axis labels
        plt.xticks(rotation=90, fontsize=20)
        plt.yticks(fontsize=20)
        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"{df['key'].values[0]}_anova_label_plot.png")


if __name__ == "__main__":
    main()
  