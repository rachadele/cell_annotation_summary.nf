    
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
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_eval/weighted_f1_results.tsv")
    parser.add_argument('--vars', type=str, nargs = "+", help="Names of factor columns")
    parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_eval/label_f1_results.tsv")   
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args


def compute_pca(df, f1_column, factor_names):
    """
    Parameters:
    - df (pd.DataFrame): DataFrame containing the F1 scores and metadata factors.
    - f1_column (str): Either "weighted_F1" or "label_F1".
    - factor_cols (list): List of categorical metadata factors.

    Returns:
    - df_pca (pd.DataFrame): DataFrame with PC1, PC2, and factor columns.
    - explained_variance (array): Explained variance ratio of the components.
    """
    # Identify categorical columns
    factor_cols= [col for col in factor_names if df[col].nunique() > 1]
    categorical_cols = [col for col in factor_cols if df[col].dtype != "numeric"]

    # scale F1 column
    X_f1 = df[[f1_column]].values
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X_f1)

    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop="first")
    X_categorical = encoder.fit_transform(df[categorical_cols])

    # Combine standardized F1 scores with encoded categorical data
    X_combined = np.hstack([X_numeric, X_categorical])
    n_samples, n_features = X_combined.shape
    n_components = min(n_samples, n_features, 10)  # Adjust to valid range

    # Run PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_combined)

    # Convert PCA output to DataFrame
    df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
    df_pca = pd.concat([df_pca, df[categorical_cols].reset_index(drop=True)], axis=1)

    return df_pca, pca.explained_variance_ratio_, factor_cols

def plot_pca(df_pca, explained_variance, factor_cols, title_prefix, outdir):
    """
    Plots PCA results colored by each factor and saves the plots.

    Parameters:
    - df_pca (pd.DataFrame): DataFrame with PC1, PC2, and factors.
    - explained_variance (array): Explained variance ratio from PCA.
    - factor_cols (list): List of factors to color plots.
    - title_prefix (str): Prefix for plot titles.
    - outdir (str): Directory to save plots (optional).

    Returns:
    - None (displays and saves plots)
    """
    if outdir:
        os.makedirs(outdir, exist_ok=True)  # Create directory if it doesn't exist

    for factor in factor_cols:
            # Create a figure with two subplots: one for PC1 vs PC2, and one for PC3 vs PC4
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Plot for PC1 vs PC2
            sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue=factor, palette="tab10", alpha=0.7, ax=axes[0])
            axes[0].set_title(f"{title_prefix} - Colored by {factor} (PC1 vs PC2)")
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Plot for PC3 vs PC4
            sns.scatterplot(data=df_pca, x="PC3", y="PC4", hue=factor, palette="tab10", alpha=0.7, ax=axes[1])
            axes[1].set_title(f"{title_prefix} - Colored by {factor} (PC3 vs PC4)")
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Adjust layout to prevent overlapping of titles/labels
            plt.tight_layout()

            # Save plot if outdir is specified
            if outdir:
                plt.savefig(os.path.join(outdir, f"{title_prefix}_{factor}_PCA.png"), bbox_inches="tight", dpi=300)

            plt.show()

    # Explained variance plot
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100, alpha=0.7, color="blue")
    plt.xticks(range(1, len(explained_variance) + 1))
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance (%)")
    plt.title(f"{title_prefix} - Explained Variance")

    # Save variance plot
    if outdir:
        plt.savefig(os.path.join(outdir, f"{title_prefix}_explained_variance.png"), bbox_inches="tight", dpi=300)

    plt.show()

    

def run_anova(df, factor_names, f1_column):
    # Select factors with more than one unique value
    factor_names_uniq = [col for col in factor_names if df[col].nunique() > 1]
    # Construct formula
    formula = f'{f1_column} ~ ' + ' + '.join(factor_names_uniq)
    # Fit model
    model = ols(formula, data=df).fit()
    # Run ANOVA with error handling for warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        aov_table = anova_lm(model, typ=2)
    # Apply FDR correction
    aov_table['FDR'] = np.append(stats.false_discovery_control(aov_table['PR(>F)'].dropna()), np.nan)
    return aov_table

def plot_anova_results(aov_combined_df, title_prefix):
    
    custom_order = ['subclass', 'class', 'family']
    
    # Set 'key' column as categorical with a custom order
    aov_combined_df['key'] = pd.Categorical(aov_combined_df['key'], categories=custom_order, ordered=True)
   
    aov_combined_df["factor"] = aov_combined_df.index
    aov_combined_df.reset_index(drop=True, inplace=True)
     # Plot with faceting by 'key'
    sns.set(style="whitegrid")
    g = sns.FacetGrid(aov_combined_df, col="key", col_wrap=3, height=5)
    g.map(sns.barplot, 'F', 'factor', hue='factor', data=aov_combined_df, dodge=False, palette="Set1")

    # Set axis labels and titles
    g.set_axis_labels('F-statistic', 'Factor')
    g.set_titles(col_template="{col_name}", fontsize=20)
   # g.set(xticks=[], yticks=[])

    g.set_xlabels(fontsize=20)
    g.set_ylabels(fontsize=20)
        # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{title_prefix}_anova_plot.png")
    
def main():
    
    args = parse_arguments()
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
    label_f1_results = pd.read_csv(args.label_f1_results, sep="\t")
    categoricals = ['manuscript','reference','method','ref_split', 'region_match',"sex","disease","dev_stage"]

    if not args.vars:
    # Assuming f1_results is your pandas DataFrame and factor_names is a list of column names
        factor_names = ['manuscript','reference','method','ref_split', 
                        'region_match',"subsample_ref","sex","disease","dev_stage","cutoff"] # replace with actual factor column names

    else:
        factor_names = args.vars

    # Convert factor columns to categorical
    for factor in categoricals:
        weighted_f1_results[factor] =weighted_f1_results[factor].astype('category')
        label_f1_results[factor] =label_f1_results[factor].astype('category')
    

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
    factor_names = factor_names + ["label"]
    for df in df_list:
        aov_table = run_anova(df, factor_names, f1_column="f1_score")
        aov_table.to_csv(df['key'].values[0] + "_anova_label_table.tsv", sep="\t")
        aov_table = aov_table.drop('Residual', axis=0)
        aov_table['key'] = df['key'].values[0] 
        aov_table_label.append(aov_table)

    aov_combined_df = pd.concat(aov_table_label)
    plot_anova_results(aov_combined_df, "label_f1")    
    
if __name__ == "__main__":
    main()
  