    
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
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/full_query/weighted_f1_results.tsv")
    parser.add_argument('--vars', type=str, nargs = "+", help="Names of factor columns")
    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/full_query/label_f1_results.tsv")   
    parser.add_argument('--factor_names', type=str, nargs = "+", help="Names of factor columns", default=["study","reference","method","ref_split","region_match","subsample_ref","disease_State"])
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

    
def run_anova(df, factor_names, f1_column):
    # Select factors with more than one unique value
    factor_names_uniq = [col for col in factor_names if df[col].nunique() > 1]
    # Construct formula
    formula = f'{f1_column} ~ ' + ' + '.join(factor_names_uniq)
    # Fit model
    # find r2 from model
    model = ols(formula, data=df).fit()
    r2 = model.rsquared
    # Run ANOVA with error handling for warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        aov_table = anova_lm(model, typ=2)
    
    # Apply FDR correction
    aov_table['FDR'] = np.append(stats.false_discovery_control(aov_table['PR(>F)'].dropna()), np.nan)
    aov_table['Rsquared'] = r2
    return aov_table

def plot_anova_results(aov_combined_df, title_prefix):
    
    plt.rcParams.update({'font.size': 20})  # Set default font size for all plot elements 
    custom_order = ['subclass', 'class', 'family']
    
    # rename factors
    
    # Set 'key' column as categorical with a custom order
    aov_combined_df['key'] = pd.Categorical(aov_combined_df['key'], categories=custom_order, ordered=True)
   
    aov_combined_df["factor"] = aov_combined_df.index
    aov_combined_df['factor'] = aov_combined_df['factor'].str.replace("_"," ").str.capitalize().replace("Label", "Cell type")

    aov_combined_df.reset_index(drop=True, inplace=True)
     # Plot with faceting by 'key'
    sns.set(style="whitegrid")
    g = sns.FacetGrid(aov_combined_df, col="key", col_wrap=3, height=5)
    g.map(sns.barplot, 'F', 'factor', hue='factor', data=aov_combined_df, dodge=False, palette="Set1")

    # Set axis labels and titles
    g.set_axis_labels('F-statistic', 'Factor')
    g.set_titles(col_template="{col_name}")
   # g.set(xticks=[], yticks=[])

    g.set_xlabels(fontsize=10)
    g.set_xticklabels(rotation=90)
    g.set_ylabels(fontsize=12)
        # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{title_prefix}_anova_plot.png")
    
def main():
    
    args = parse_arguments()
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
    label_f1_results = pd.read_csv(args.label_f1_results, sep="\t")
    categoricals = ['study','reference','method','ref_split', 'region_match',"sex","disease_state"]

    if not args.vars:
    # Assuming f1_results is your pandas DataFrame and factor_names is a list of column names
        factor_names = ['study','reference','method','ref_split',
                        'region_match',"subsample_ref","sex","disease_state","cutoff"] # replace with actual factor column names

    else:
        factor_names = args.vars

    # Convert factor columns to categorical
    for factor in categoricals:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('category')
        label_f1_results[factor] =label_f1_results[factor].astype('category')
    for factor in ["cutoff", "subsample_ref","age"]:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('float')
        label_f1_results[factor] =label_f1_results[factor].astype('float')
   
   # change names and capitalize categories
   

    aov_combined = []
    df_list = [group for _, group in weighted_f1_results.groupby('key')]
    factor_names = factor_names
    for df in df_list:
        aov_table = run_anova(df, factor_names, f1_column="weighted_f1")
        aov_table.to_csv(df['key'].values[0] + "_anova_global_table.tsv", sep="\t")
        aov_table = aov_table.drop('Residual', axis=0)
        aov_table['key'] = df['key'].values[0] 
        aov_combined.append(aov_table)

    aov_combined_df = pd.concat(aov_combined)
    plot_anova_results(aov_combined_df, "weighted_f1")
    
    
    aov_table_label = []
    df_list = [group for _, group in label_f1_results.groupby('key')]
    factor_names_label = factor_names + ["label"]
    for df in df_list:
        aov_table = run_anova(df, factor_names_label, f1_column="f1_score")
        aov_table.to_csv(df['key'].values[0] + "_anova_label_table.tsv", sep="\t")
        aov_table = aov_table.drop('Residual', axis=0)
        aov_table['key'] = df['key'].values[0] 
        aov_table_label.append(aov_table)

    aov_combined_df = pd.concat(aov_table_label)
    plot_anova_results(aov_combined_df, "label_f1") 
    

    
if __name__ == "__main__":
    main()
  