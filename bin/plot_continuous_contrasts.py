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
import random
random.seed(42)
from collections import defaultdict
import matplotlib.ticker as ticker
import textwrap
import re


SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 35

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--key', type = str, help = "key of factor to plot", default = "subclass")
    parser.add_argument("--contrast", type=str, help="List of continuous contrasts to plot", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/homo_sapiens/model_eval/label/f1_score_~_label_+_support_+_cutoff_+_method_+_method:cutoff_+_method:support/subclass/files/label_support_effects.tsv")
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args
    

def plot_fit_with_ci(df, x_col="cutoff", group_col="method",
                     fit_col="fit", lower_col="lower", upper_col="upper", outdir=None):
    """
    Generic plotting function to visualize fit with confidence intervals.
    """
    #sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))

    for facet, group in df.groupby(group_col):
        # Sort for consistent plotting
        group = group.sort_values(by=x_col)
        plt.plot(group[x_col], group[fit_col], marker="o", label=facet)
        plt.fill_between(group[x_col], group[lower_col], group[upper_col], alpha=0.2)

    plt.xlabel(x_col.capitalize())
    plt.ylabel("Fitted F1 Score")
   # plt.title("Fit with Confidence Interval by Method")
    plt.legend(title=group_col.capitalize())
    plt.tight_layout()
    plt.savefig(os.path.join(outdir,f"{x_col}_{group_col}_fit_with_ci.png"))

def main():
  args = parse_arguments()
  contrast = pd.read_csv(args.contrast, sep = "\t")
  if "cutoff" in contrast.columns:
    x_col = "cutoff"
  elif "support" in contrast.columns:    
    x_col = "support"
  outdir = x_col
  if not os.path.exists(outdir):
      os.makedirs(outdir)
  group_col = "method"
  plot_fit_with_ci(df=contrast, x_col=x_col, group_col=group_col,
                    fit_col="fit", lower_col="lower", upper_col="upper", outdir=outdir)


if __name__ == "__main__":
    main()