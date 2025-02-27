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
import itertools
import textwrap

# Function to parse command line arguments
def parse_arguments():
  parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
  parser.add_argument('--weighted_f1_results', type=str, default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_dataset_id/weighted_f1_results.tsv", help="Aggregated weighted results")
  parser.add_argument('--label_f1_results', type=str, default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_dataset_id/label_f1_results.tsv", help="Label level f1 results")                                            
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
  # f1_score ~ study
  control_model = ols(control_formula, data=df).fit()
    
  # Get residuals (this is the outcome with the "Study" effects removed)
  df['residualized_outcome'] = control_model.resid

  # Step 2: Fit the main model using the residualized outcome
  # Replace the original outcome with the residualized outcome in the formula
  main_formula = formula.replace(formula.split('~')[0], 'residualized_outcome')
  # f1 
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

  return model, model.summary2().tables[0], model_summary_df


def plot_model_summary(model_summary, outdir, key):
  plt.rcParams.update({'font.size': 30})
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
          model_summary, model_summary_df = run_lm(df, formula)
          for key in df['key'].unique():
              key_df = model_summary_df.copy()
              key_df['Dataset'] = key
              key_df['Formula'] = formula
              results.append(key_df[['Dataset', 'Formula', 'Rsquared', 'AIC']])

  # Combine results into a single DataFrame
  results_df = pd.concat(results, ignore_index=True)

  # Plot Rsquared vs. AIC
  plt.figure(figsize=(20, 6))
  scatter = sns.scatterplot(data=results_df, x="AIC", y="Rsquared", hue="Dataset", style="Formula", palette="tab10", edgecolor="black", s=100)
    
  # Add labels
  plt.title("Rsquared vs AIC Across Models")
  plt.xlabel("AIC")
  plt.ylabel("R-squared")
    
  # Improve legend
  plt.legend(title="Dataset", bbox_to_anchor=(1, 1))
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.tight_layout()
    
  plt.savefig("model_metrics.png", bbox_inches="tight")
  plt.show()



def run_and_store_model(df, formula, formula_dir, key):
  """Runs the linear model, saves results."""
  model, model_summary, model_summary_coefs = run_lm_regressed(df, formula)
  model_summary_coefs["formula"] = formula
  model_summary_coefs["key"] = key
  model_summary_coefs["Term"] = model_summary_coefs.index
  model_summary_coefs.to_csv(os.path.join(formula_dir, "model_summary_coefs.tsv"), sep="\t", index=False)
  model_summary.to_csv(os.path.join(formula_dir, "model_summary.tsv"), sep="\t", index=False)
  plot_model_summary(model_summary_coefs, outdir=formula_dir, key=key)
  return model_summary_coefs
 
def main():
    
  args = parse_arguments()
  weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep="\t")
  #weighted_f1_results = weighted_f1_results[weighted_f1_results["cutoff"] == 0]
  
  label_results = pd.read_csv(args.label_f1_results, sep="\t")
  #label_results = label_results[label_results["cutoff"] == 0]
  
  
  factor_names = ["study", "reference", "method", "cutoff", "disease_state", "sex"]

  formulas = [
      "weighted_f1 ~ " + " + ".join(factor_names) + " + reference:method + method:cutoff"] 
      #"weighted_f1 ~ " + " + ".join(factor_names) + " + reference:method + method:cutoff + study:sex"]
   
  df_list = [group for _, group in weighted_f1_results.groupby('key')]
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