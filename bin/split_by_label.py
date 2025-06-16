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

  parser.add_argument('--label_f1_results', type=str, help="Label level f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mus_musculus/aggregated_results/label_f1_results.tsv")   
  # deal with jupyter kernel arguments
  if __name__ == "__main__":
      known_args, _ = parser.parse_known_args()
      return known_args
  
  return parser.parse_args()

def main():
  args = parse_arguments()
  label_f1_results = args.label_f1_results
	
	# Read the label f1 results
	df = pd.read_csv(label_f1_results, sep="\t")
	# split by key and label and save to individual files
	for key, group in df.groupby("key"):
		for label, sub_group in group.groupby("label"):
			label_df = sub_group
			label_name = label.replace(" ", "_").replace("\\/", "_")
      output_file = f"{key}_{label_name}_f1_results.tsv"
      label_df.to_csv(output_file, sep="\t", index=False)
  
if __name__ == "__main__":
	main()