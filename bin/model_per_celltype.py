import numpy as np
import pandas as pd
from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import defaultdict
from pymer4.models import Lmer

def parse_arguments():
  parser = argparse.ArgumentParser(description="Correct model predictions based on reference keys and mapping file.")
  parser.add_argument('--label_f1_results', type=str, help="Path to the predicted metadata file", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mus_musculus/old_refs/label_results/class/Cajal-Retzius_cell/f1_results.tsv")
  parser.add_argument('--mapping_file', type=str, help="Path to the cell type hierarchy file", default = "/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/meta/census_map_mouse_author.tsv")
  parser.add_argument('--ref_keys', type=str, nargs="+", help="Reference keys to map", default=["subclass", "class", "family", "global"])
  parser.add_argument('--cell_type', type=str, help="Cell type to analyze", default="Cajal-Retzius_cell")
  # deal with jupyter kernel arguments
  if __name__ == "__main__":
      known_args, _ = parser.parse_known_args()
      return known_args

def run_llmer(label_f1_results, formula, family="gaussian"):
  print(f"Fitting model with formula: {formula}")
  model = Lmer(formula, data=label_f1_results, family=family)
  model.fit()
  return(model)

def plot_model_coefficients(model, cell_type):
  # plot the model coefficients
  # get random effec  
  
  model_df = model.coefs.reset_index().rename(columns={"index": "term"})
  # extract random effects
  
  model_df["coef"] = model_df["term"].replace("C\\(", "", regex=True).replace("\\)", "", regex=True)
  model_df["Significance"] = model_df["P-val"].apply(lambda x: "P<0.05" if x < 0.05 else "P>=0.05")
 
  # Compute error bars
  model_df["ci_lower"] = model_df["Estimate"] - model_df["2.5_ci"]
  model_df["ci_upper"] = model_df["97.5_ci"] - model_df["Estimate"]
 
  plt.figure(figsize=(10, 6))
  sns.barplot(data=model_df, y="coef", x="Estimate", 
              hue="Significance", dodge=False, palette=["#1f77b4", "#ff7f0e"])
  
  # add lines for CI
  for i, row in model_df.iterrows():
      plt.errorbar(row["Estimate"], i, 
                   xerr=[[row["ci_lower"]], [row["ci_upper"]]], 
                   fmt='none', color='black', capsize=5)
      
  plt.title(f"Model Coefficients for {cell_type}")
  plt.xlabel("Coefficient Estimate")
  plt.ylabel("Coefficient")
  plt.savefig(f"{cell_type}_model_coefficients.png")

  
def plot_random_slopes(model, cell_type):
 # Get all fixed effect names (from .coefs index)
  all_terms = model.coefs.index.tolist()
  for term in all_terms:
      try:
          print(f"Plotting random slopes for: {term}")
          model.plot(param=term, figsize=(8, 6))
          plt.title(f"Effect of '{term}' by Study")
          plt.tight_layout()
          plt.savefig(f"{cell_type}_random_slope_{term}.png")
          plt.close()
      except Exception as e:
          print(f"Skipping {term}: {e}")


def main():
  args = parse_arguments()
  label_f1_results_path = args.label_f1_results
  mapping_file = args.mapping_file
  ref_keys = args.ref_keys
  cell_type = args.cell_type
  # Load predicted metadata
  label_f1_results = pd.read_csv(label_f1_results_path, sep="\t")
  # logistic regression of correct vs incorrect predictions
  feature_cols = ["reference", 
                  "method", 
                  "cutoff"]

  valid_feature_cols = []

  for col in feature_cols:
    if len(label_f1_results[col].unique()) <= 1:
        print(f"Warning: {col} has only one unique value, skipping.")
    else:
        valid_feature_cols.append(col)

  # Replace original list if needed
  feature_cols = valid_feature_cols

  if len(feature_cols) == 0:
      raise ValueError("No valid feature columns found. Exiting.")
      
  outcome = "f1_score"
  random_effect_col = "study"
  formula = f"{outcome} ~ {' + '.join(feature_cols)} + cutoff:method + reference:method + (1 | {random_effect_col})"
  # Fit the model
  model = run_llmer(label_f1_results, formula, family="gaussian")
  
  plot_model_coefficients(model, cell_type)
  plot_random_slopes(model, cell_type)
  

  
    
if __name__ == "__main__":
  main() 