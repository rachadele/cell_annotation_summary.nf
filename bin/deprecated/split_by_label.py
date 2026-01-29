#!/user/bin/python3

from pathlib import Path
import os
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import statsmodels as sm
from scipy import stats
import matplotlib.pyplot as plt
set_seed = 42
import random
import numpy as np
random.seed(42)


# Function to parse command line arguments
def parse_arguments():
  parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")

  parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mus_musculus/new_refs/aggregated_results/label_f1_results.tsv")   
  # deal with jupyter kernel arguments
  if __name__ == "__main__":
      known_args, _ = parser.parse_known_args()
      return known_args
  
  return parser.parse_args()

def plot_f1_distribution(df, key, label_name):

  df_long = df.melt(id_vars=["key", "label","study"], value_vars=["f1_score"], var_name="metric", value_name="score")

  plt.figure(figsize=(10, 6))
  sns.histplot(df_long, x="score", bins=30, kde=False, hue="study")
  plt.title(f'F1 Score Distribution for {label_name}')
  plt.xlabel('F1 Score')
  plt.ylabel('Frequency')

  # Save the plot
  output_file = f"{label_name}/f1_distribution.png"
  plt.savefig(output_file)
  plt.close()

def main():
  args = parse_arguments()
  label_f1_results = args.label_f1_results

	# Read the label f1 results
  df = pd.read_csv(label_f1_results, sep="\t")
	# split by label (pool all granularity levels together)
  for label, label_df in df.groupby("label"):
    label_name = label.replace(" ", "_").replace("/", "_")
    os.makedirs(label_name, exist_ok=True)
    key = label_df["key"].iloc[0]
    plot_f1_distribution(label_df, key, label_name)
    output_file = f"{label_name}/f1_results.tsv"
    label_df.to_csv(output_file, sep="\t", index=False)

if __name__ == "__main__":
	main()