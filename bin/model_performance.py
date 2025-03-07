#!/user/bin/python3


import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

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
import itertools
import textwrap
from matplotlib.patches import Patch  
from matplotlib.lines import Line2D

# Function to parse command line arguments
def parse_arguments():
  parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
  parser.add_argument('--weighted_f1_results', type=str, default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_hsap/aggregated_results/weighted_f1_results.tsv", help="Aggregated weighted results")
  parser.add_argument('--label_f1_results', type=str, default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_hsap/aggregated_results/label_f1_results.tsv", help="Label level f1 results")                                            
  if __name__ == "__main__":
      known_args, _ = parser.parse_known_args()
      return known_args


def run_lmm_random(df, formula, group_var='study'):
    """
    Runs a linear mixed-effects model treating `group_var` as a random effect.
    
    Parameters:
        df (pd.DataFrame): The dataset.
        formula (str): The model formula (e.g., 'f1_score ~ cutoff + variable').
        group_var (str): The grouping variable for the random effect (default: 'study').
    
    Returns:
        pd.DataFrame: Model summary with FDR-corrected p-values.
    """
    # Extract the outcome variable
    outcome_var = formula.split('~')[0].strip()
    
    # Log-transform the outcome (avoid log(0))
    df[outcome_var] = np.log(df[outcome_var] + 1e-6)

    # Fit the model: fixed effects + study as a random effect
    model = smf.mixedlm(formula, df, groups=df[group_var], re_formula="~1").fit()

    # Extract coefficients and p-values
    model_summary_df = model.summary().tables[1]
    
    # Convert to DataFrame
    model_summary_df = pd.DataFrame(model_summary_df.data[1:], columns=model_summary_df.data[0])
    model_summary_df = model_summary_df.set_index(model_summary_df.columns[0])
    model_summary_df["SE"] = model.bse
    
    # Ensure numeric p-values for FDR correction
    model_summary_df["P>|z|"] = pd.to_numeric(model_summary_df["P>|z|"], errors="coerce")
    
    # Apply FDR correction
    pvals = model_summary_df["P>|z|"].dropna()
    _, fdr_corrected_pvals, _, _ = multipletests(pvals, method="fdr_bh")
    model_summary_df.loc[pvals.index, "FDR"] = fdr_corrected_pvals
    
    # Add model fit statistics
    model_summary_df["LogLik"] = model.llf
    model_summary_df["AIC"] = model.aic
    model_summary_df["BIC"] = model.bic
    
    return model, model.summary2().tables[0], model_summary_df


def plot_model_summary(model_summary, outdir, key):
    plt.rcParams.update({'font.size': 30})
    model_summary["FDR<0.05"] = model_summary["FDR"] < 0.05
    model_summary["Term"] = model_summary["Term"].str.replace("T.", "")
    formula = model_summary["formula"].unique()[0]
    
    # Use the subset directly (no need for grouping)
    plt.figure(figsize=(30, max(10, len(model_summary) * 0.5)))  # Scale height dynamically

    ax = sns.barplot(
        data=model_summary, 
        y="Term", 
        x="Coef.", 
        hue="Term",  # Color bars by term
        dodge=False, 
        errorbar=None, 
        palette="tab20"  # Ensures enough colors for multiple terms
    )

    # Highlight significant terms (FDR < 0.05) with black edges or transparency
    for patch, (_, row) in zip(ax.patches, model_summary.iterrows()):
        if row["FDR<0.05"]:
            patch.set_edgecolor("red")  # Add red border for significant terms
            patch.set_linewidth(2)
        else:
            patch.set_alpha(0.5)  # Reduce opacity for non-significant terms

        term_color = patch.get_facecolor()  # Get color of the bar
        y_tick_label = ax.get_yticklabels()[int(patch.get_y() + patch.get_height() / 2)]  # Match tick label to bar
        y_tick_label.set_color(term_color)

    # Set y-axis limits to remove extra space
    ax.set_ylim(-0.5, len(model_summary) - 0.5)        
    for patch, (_, row) in zip(ax.patches, model_summary.iterrows()):
        x_center = patch.get_x() + patch.get_width() / 2  # Get center of the bar horizontally
        y_center = patch.get_y() + patch.get_height() / 2  # Get center of the bar vertically
        plt.errorbar(x=x_center, y=y_center, xerr=2 * row["Std.Err."], fmt="none", color="black", capsize=5, capthick=2)

    # Add a custom legend for FDR < 0.05
    handles, labels = ax.get_legend_handles_labels()

    # Add custom legend for FDR < 0.05
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='FDR < 0.05')
    ]
    ax.legend(handles + legend_elements, labels + ['FDR < 0.05'], loc='upper left', bbox_to_anchor=(1, 1))

    title = f"{key} ({formula})"
    wrapped_title = "\n".join(textwrap.wrap(title, width=40))  # Adjust width as needed

    # Plot
    plt.title(wrapped_title, fontsize=30)      
    plt.xlabel("Coefficient", fontsize=30)
    plt.ylabel("Term", fontsize=30)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,f"{key}_{formula}_lm_coefficients.png"))


def plot_model_metrics(df_list, formulas):
  plt.rcParams.update({'font.size': 15})
  results = []
    
  for df in df_list:
      for formula in formulas:
          model, model_summary, model_summary_df = run_lmm_random(df, formula)
          for key in df['key'].unique():
              key_df = model_summary_df.copy()
              key_df['Dataset'] = key
              key_df['Formula'] = formula
              results.append(key_df[['Dataset', 'Formula', 'LogLik', 'AIC']])

  # Combine results into a single DataFrame
  results_df = pd.concat(results, ignore_index=True)

  # Plot Rsquared vs. AIC
  plt.figure(figsize=(20, 6))
  scatter = sns.scatterplot(data=results_df, x="AIC", y="Rsquared", hue="Dataset", style="Formula", palette="tab10", edgecolor="black", s=100)
    
  # Add labels
  plt.title("LogLik vs AIC Across Models")
  plt.xlabel("AIC")
  plt.ylabel("LogLik")
    
  # Improve legend
  plt.legend(title="Dataset", bbox_to_anchor=(1, 1))
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()
    
  plt.savefig("model_metrics.png", bbox_inches="tight")
  plt.show()



def run_and_store_model(df, formula, formula_dir, key):
  """Runs the linear model, saves results."""
  model, model_summary, model_summary_coefs = run_lmm_random(df, formula)
  model_summary_coefs["formula"] = formula
  model_summary_coefs["key"] = key
  model_summary_coefs["Term"] = model_summary_coefs.index
  model_summary_coefs.to_csv(os.path.join(formula_dir, "model_summary_coefs.tsv"), sep="\t", index=False)
  model_summary.to_csv(os.path.join(formula_dir, "model_summary.tsv"), sep="\t", index=False)
  plot_model_summary(model_summary_coefs, outdir=formula_dir, key=key)
  return model_summary_coefs
 

# Convert a pandas DataFrame to an R DataFrame
def pandas_to_r(df):
    r_df = ro.pandas2ri.py2rpy(df)
    return r_df
 
 
def main():
    
  args = parse_arguments()
  weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
  #weighted_f1_results = weighted_f1_results[weighted_f1_results["cutoff"] == 0]
  organism = weighted_f1_results["organism"].unique()[0]
  #label_results = pd.read_csv(args.label_f1_results, sep="\t")
  #label_results = label_results[label_results["cutoff"] == 0]
  
  
  factor_names = ["study", "reference", "method", "cutoff"]
  #if organism 
  formulas = [
      "weighted_f1 ~ " + " + ".join(factor_names) + " + reference:method + method:cutoff",
      "weighted_f1 ~ " + " + ".join(factor_names) + " + reference:method + method:cutoff +  disease_state",
      "weighted_f1 ~ " + " + ".join(factor_names) + " + reference:method + method:cutoff + disease_state + sex"]
   
  df_list = [group for _, group in weighted_f1_results.groupby('key')]
  # use only "subclass" df
  df_list = [df for df in df_list if df["key"].values[0] == "subclass"]
  plot_model_metrics(df_list, formulas)
  
  model_summary_coefs_combined = [] 
  for df in df_list:
      df["method"] = pd.Categorical(df["method"], categories=["seurat", "scvi"], ordered=True)
      key = df["key"].values[0]
      for formula in formulas:
          formula_dir = os.path.join("weighted_f1", formula.replace(" ",""), key)
          os.makedirs(formula_dir, exist_ok=True)
          model_summary_coefs = run_and_store_model(df, formula, formula_dir, key)
          model_summary_coefs_combined.append(model_summary_coefs)
          
  model_summary_coefs_combined = pd.concat(model_summary_coefs_combined)
  f1_type = "weighted"
  model_summary_coefs_combined.to_csv(f"{f1_type}_model_summary_coefs_combined.tsv", sep="\t", index=False)

      
    
    
      
if __name__ == "__main__":
    main()