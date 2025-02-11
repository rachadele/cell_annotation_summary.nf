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
from statsmodels.stats.multitest import multipletests
from matplotlib import cm
from matplotlib.colors import Normalize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import glm

# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--weighted_f1_results', type=str, default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/full_query/weighted_f1_results.tsv", help="Aggregated weighted results")
    parser.add_argument('--label_f1_results', type=str, default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/full_query/label_f1_results.tsv", help="Label level f1 results")                                            
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args


def run_lm(df, formula):

    model = ols(formula, data=df).fit() 
    # Get model summary
    model_summary = model.summary()
    # Extract R-squared value
    r2 = model.rsquared
    aic = model.aic
    # Extract coefficients and p-values table (this will typically be in model_summary.tables[1])
    model_summary_df = model.summary2().tables[1]
    # Apply FDR correction to p-values
    pvals = model_summary_df['P>|t|']
    _, fdr_corrected_pvals, _, _ = multipletests(pvals, method='fdr_bh')
    model_summary_df['FDR'] = fdr_corrected_pvals
    # Add R-squared value
    model_summary_df['Rsquared'] = r2
    model_summary_df['AIC'] = aic
    return model.summary2().tables[0], model_summary_df


def run_lm_regressed(df, formula, control_vars=['study']):
    
    # Step 1: Regress out the effects of "Study" and "Cutoff"
    # Create a formula that includes both control variables
    control_formula = f"{formula.split('~')[0]} ~ " + ' + '.join(control_vars)
    control_model = ols(control_formula, data=df).fit()
    
    # Get residuals (this is the outcome with the "Study" effects removed)
    df['residualized_outcome'] = control_model.resid

    # Step 2: Fit the main model using the residualized outcome
    # Replace the original outcome with the residualized outcome in the formula
    main_formula = formula.replace(formula.split('~')[0], 'residualized_outcome')
    model = ols(main_formula, data=df).fit()

    # Get model summary
    model_summary = model.summary()
    # Extract R-squared and AIC values
    r2 = model.rsquared
    aic = model.aic
    # Extract coefficients and p-values table
    model_summary_df = model.summary2().tables[1]
    # Apply FDR correction to p-values
    pvals = model_summary_df['P>|t|']
    _, fdr_corrected_pvals, _, _ = multipletests(pvals, method='fdr_bh')
    model_summary_df['FDR'] = fdr_corrected_pvals
    # Add R-squared and AIC values to the summary dataframe
    model_summary_df['Rsquared'] = r2
    model_summary_df['AIC'] = aic

    return model.summary2().tables[0], model_summary_df


def run_glm(df, formula):
    # Fit the GLM model with Gaussian family (similar to OLS)
    model = glm(formula, data=df, family=sm.families.Gaussian()).fit() 

    # Get model summary
    model_summary = model.summary()

    # Extract coefficients and p-values table
    model_summary_df = model.summary2().tables[1]

    return model_summary, model_summary_df


 
        
def make_acronym(ref_name):
    # Split on "_" and replace with spaces
    words = ref_name.split("_")
    # Create acronym from the first letter of each word
    acronym = "".join(word[0].upper() for word in words if word)
    return acronym



def plot_model_summary(model_summary):
    plt.rcParams.update({'font.size': 20})
    model_summary["FDR<0.05"] = model_summary["FDR"] < 0.05
    model_summary["Term"] = model_summary["Term"].str.replace("T.", "")
    formula = model_summary["formula"].unique()[0]

    for key, subset in model_summary.groupby("key"):
        plt.figure(figsize=(30, max(10, len(subset) * 0.5)))  # Scale height dynamically
        ax = sns.barplot(
            data=subset, y="Term", x="Coef.", hue="FDR<0.05", dodge=False, errorbar=None, palette="coolwarm"
        )

        # Set y-axis limits to remove extra space
        ax.set_ylim(-0.5, len(subset) - 0.5)        
        for patch, (_, row) in zip(ax.patches, subset.iterrows()):
            x_center = patch.get_x() + patch.get_width() / 2  # Get center of the bar horizontally
            y_center = patch.get_y() + patch.get_height() / 2  # Get center of the bar vertically
            plt.errorbar(x=x_center, y=y_center, xerr=2 * row["Std.Err."], fmt="none", color="black", capsize=5, capthick=2)

        plt.title(f"Model Coefficients for {key} ({formula})", fontsize=24)
        plt.xlabel("Coefficient", fontsize=18)
        plt.ylabel("Term", fontsize=18)
        plt.tight_layout()
        plt.savefig(f"{key}_{formula}_lm_coefficients.png")
        plt.show()


def plot_model_metrics(df_list, formulas):
    plt.rcParams.update({'font.size': 15})
    results = []
    
    for df in df_list:
        for formula in formulas:
            model_summary, model_summary_df = run_lm(df, formula)
            for key in df['key'].unique():
                key_df = model_summary_df.copy()
                key_df['Dataset'] = key
                key_df['Formula'] = formula
                results.append(key_df[['Dataset', 'Formula', 'Rsquared', 'AIC']])

    # Combine results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    # Plot Rsquared vs. AIC
    plt.figure(figsize=(15, 6))
    scatter = sns.scatterplot(data=results_df, x="AIC", y="Rsquared", hue="Dataset", style="Formula", palette="tab10", edgecolor="black", s=100)
    
    # Add labels
    plt.title("Rsquared vs AIC Across Models")
    plt.xlabel("AIC")
    plt.ylabel("R-squared")
    
    # Improve legend
    plt.legend(title="Dataset", bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plt.savefig("model_metrics.png")
    plt.show()
    
def main():
    
    args = parse_arguments()
    weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
    label_results = pd.read_csv(args.label_f1_results, sep="\t")

 
    # Example: adding hue and faceting to the weighted F1 scores distribution plot
    sns.histplot(weighted_f1_results, x='weighted_f1', hue='key', multiple="fill", palette="Set1")
    plt.xlabel("Weighted F1")
    plt.ylabel("Frequency")
    plt.title("Distribution of Weighted F1 Scores by Key")
    plt.savefig("weighted_f1_distribution.png")
    plt.show()

    # Create the FacetGrid
    g = sns.FacetGrid(label_results, col="key", hue="label", height=4, aspect=1.5)
    # Map the KDE plot to the FacetGrid
    g.map(sns.histplot, 'f1_score', multiple="layer")
    # Set axis labels and titles
    g.set_axis_labels("F1 Scores", "Frequency")
    g.set_titles("F1 Score Distribution by {col_name}")
    # Add a legend
    g.add_legend()
    # Save and display the plot
    g.savefig("label_f1_distribution_facet.png")
    plt.show()

    # restrict to cutoff=0 for now
    weighted_f1_results = weighted_f1_results[weighted_f1_results["cutoff"] == 0]
    
    factor_names = ["study", "reference", "method"]

    # compare models
    formulas = ["weighted_f1 ~ " + " + ".join(factor_names),
                "weighted_f1 ~ " + " * ".join(factor_names),
                "weighted_f1 ~ " + " + ".join(factor_names) + " + reference:method"]

    # set base level of method to "seurat"
    df_list = [group for _, group in weighted_f1_results.groupby('key')]
    plot_model_metrics(df_list, formulas)
    

    #formulas = ["weighted_f1 ~ " + " + ".join(factor_names),
     #       "weighted_f1 ~ " + " + ".join(factor_names) + " + reference:method"]
    lm_combined = pd.DataFrame()
    for df in df_list:
        df["method"] = pd.Categorical(df["method"], categories=["seurat","scvi"], ordered=True)
        for formula in formulas:
            
            model_summary, model_summary_df = run_lm_regressed(df, formula)
            print(df['key'].unique())
            print(formula)
            print(model_summary_df["Rsquared"].unique())
            print(model_summary_df["AIC"].unique())

            model_summary_df['key'] = df['key'].values[0] 
            model_summary_df['formula'] = formula
            
            lm_combined = pd.concat([lm_combined, model_summary_df], ignore_index=True)
            model_summary_df["Term"] = model_summary_df.index
            
            plot_model_summary(model_summary_df)
            
            
    lm_combined.to_csv("model_summary.tsv", sep="\t")    
        
if __name__ == "__main__":
    main()