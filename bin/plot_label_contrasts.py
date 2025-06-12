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
MEDIUM_SIZE = 35
BIGGER_SIZE = 40

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
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mus_musculus/aggregated_results/weighted_f1_results.tsv")
    parser.add_argument('--emmeans_estimates', type=str, help = "OR and pvalues from emmeans", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mus_musculus/model_eval/label/f1_score_~_label_+_support_+_cutoff_+_method_+_method:cutoff_+_method:support/subclass/files/label_emmeans_estimates.tsv" )
    parser.add_argument('--emmeans_summary', type = str, help = "emmeans summary", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mus_musculus/model_eval/label/f1_score_~_label_+_support_+_cutoff_+_method_+_method:cutoff_+_method:support/subclass/files/label_emmeans_summary.tsv")
    parser.add_argument('--key', type = str, help = "key of factor to plot", default = "subclass")
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args

def main():
     
  args = parse_arguments()

  weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep = "\t")
  # label_f1_results = pd.read_csv(args.label_f1_results, sep = "\t")
  contrast_results = pd.read_csv(args.emmeans_estimates, sep = "\t")
  emmeans_summary = pd.read_csv(args.emmeans_summary, sep = "\t")
  
  
  factors = emmeans_summary.loc[:, :"response"].iloc[:, :-1].columns.tolist()
  key = args.key
    
if __name__=="__main__":
    main() 