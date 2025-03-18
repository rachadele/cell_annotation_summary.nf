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



# Function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
    parser.add_argument('--weighted_f1_results', type=str, help="Aggregated weighted results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_mmus/aggregated_results/weighted_f1_results.tsv")
  #  parser.add_argument('--label_f1_results', type=str, help="Label key f1 results", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_mmus/aggregated_results/label_f1_results.tsv")   
    parser.add_argument('--estimates', type=str, help = "OR and pvalues from emmeans", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_mmus/model_eval/weighted/weighted_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_treatment_state_+_sex_+_method:cutoff_+_reference:method/subclass/files/reference_method_emmeans_estimates.tsv")
    parser.add_argument('--emmeans_summary', type = str, help = "emmeans summary", default= "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/SCT_integrated_mmus/model_eval/weighted/weighted_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_treatment_state_+_sex_+_method:cutoff_+_reference:method/subclass/files/reference_method_emmeans_summary.tsv")
    parser.add_argument('--key', type = str, help = "key of factor to plot", default = "subclass")
    # deal with jupyter kernel arguments
    if __name__ == "__main__":
        known_args, _ = parser.parse_known_args()
        return known_args
      
def wrap_labels(ax, width=10):
  """Wrap x-axis labels to a fixed width."""
  plt.draw()  # Ensure labels are drawn

  # Get current tick positions and labels
  tick_labels = [textwrap.fill(label.get_text(), 10) for label in ax.get_xticklabels()]
  tick_positions = ax.get_xticks()
  # Apply fixed locator and wrapped labels
  ax.xaxis.set_major_locator(ticker.FixedLocator(tick_positions))
  ax.set_xticklabels(tick_labels, rotation=0, ha="center")


def plot_contrast_twofactors(weighted_f1_results, factor1, factor2, outdir):

    # Set the aesthetic style and context
    sns.set(style="whitegrid", palette="colorblind", context="talk")

    # Calculate the positions for the point plot
    unique_factor1 = weighted_f1_results[factor1].unique()
    unique_factor2 = weighted_f1_results[factor2].unique()
    offsets = {factor: i * 0.4 - 0.2 for i, factor in enumerate(unique_factor2)}
        # Define color palettes
    boxplot_palette = sns.color_palette("Set2", n_colors=len(unique_factor2))
    errorbar_palette = sns.color_palette("Set1", n_colors=len(unique_factor2))
    color_map = dict(zip(unique_factor2, errorbar_palette))
    
    for factor in [factor1, factor2]:
      weighted_f1_results[factor] = weighted_f1_results[factor].astype(str)
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(30, 10))

    # Plot the boxplot
    sns.boxplot(
        y=factor1,
        x="weighted_f1",
        hue=factor2,
        data=weighted_f1_results,
        ax=ax,
        linewidth=2.5,
        width=0.8,
        showfliers=False,
        whis=[5, 95],
        showcaps=False,
        palette=boxplot_palette
    )


    # Plot the point plot with error bars
    for i, (f1, f2) in enumerate(weighted_f1_results[[factor1, factor2]].drop_duplicates().values):
        subset = weighted_f1_results[(weighted_f1_results[factor1] == f1) & (weighted_f1_results[factor2] == f2)]
        y = unique_factor1.tolist().index(f1) + offsets[f2]
        x = subset['response'].values[0]
        lower_err = x - subset['asymp.LCL'].values[0]  # Lower bound
        upper_err = subset['asymp.UCL'].values[0] - x  # Upper bound
    

        ax.errorbar(
            x=x,
            y=y,
            xerr=[[lower_err], [upper_err]],
            fmt='o',
            color=color_map[f2],
            capsize=5,
            capthick=3,
            elinewidth=3,
            label=f'estimated marginal mean (95% CI) for {f2}'
        )
    wrap_labels(ax, width=2)
    # Customize the legend to avoid duplication
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.01))
    plt.grid(False)  # Disables gridlines completely

    # Set the title for the plot
    ax.set_title(f'{factor1.capitalize()} vs. {factor2.capitalize()} at Cutoff = 0')

    # Adjust layout to accommodate the title
    plt.tight_layout()

    # Show the plot
    plt.savefig(os.path.join(outdir,f"{factor1}_{factor2}_boxplot.png"))

 

def plot_contrast_onefactor(weighted_f1_results, factor1, outdir, key):
    # Set the aesthetic style and context
    sns.set(style="whitegrid", palette="colorblind", context="talk")

    # Calculate the positions for the point plot
    unique_factor1 = weighted_f1_results[factor1].unique()
    
    # Define color palettes
    boxplot_palette = sns.color_palette("Set2", n_colors=len(unique_factor1))
    errorbar_palette = sns.color_palette("Set1", n_colors=len(unique_factor1))
    color_map = dict(zip(unique_factor1, errorbar_palette))
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(30, 10))

    # Plot the boxplot
    sns.boxplot(
        y=factor1,
        x="weighted_f1",
        data=weighted_f1_results,
        ax=ax,
        linewidth=2.5,
        width=0.8,
        showfliers=False,
        whis=[5, 95],
        showcaps=False,
        palette=boxplot_palette
    )

    # Plot the point plot with error bars
    for i, f1 in enumerate(unique_factor1):
        subset = weighted_f1_results[weighted_f1_results[factor1] == f1]
        y = i
        x = subset['response'].values[0]
        lower_err = x - subset['asymp.LCL'].values[0]
        upper_err = subset['asymp.UCL'].values[0] - x

        ax.errorbar(
            x=x,
            y=y,
            xerr=[[lower_err], [upper_err]],
            fmt='o',
            color=color_map[f1],
            capsize=5,
            capthick=3,
            elinewidth=3,
            label=f'estimated marginal mean (95% CI) for {f1}'
        )
    wrap_labels(ax, width=2)
    # Customize the legend to avoid duplication
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.01))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)

    # Set the title for the plot
    ax.set_title(f'{factor1.capitalize()} at {key} level, Cutoff = 0')

    # Adjust layout to accommodate the title
    plt.tight_layout()

    # Show the plot
    plt.savefig(os.path.join(outdir,f"{factor1}_boxplot.png"))



def make_contrast_dict(estimates, n_factors=2):
  contrast_dict = defaultdict(lambda: defaultdict(dict))
  for index,row in estimates.iterrows():
      contrast = row["contrast"]
      contrast = contrast.replace("(","").replace(")","")
      group1, group2 = contrast.split(" / ")
      
      if n_factors == 2:
         level1 = group1.split(" ")[-1]
         level2 =  group2.split(" ")[-1]
         group1 = group1.replace(f" {level1}", "")
         group2 = group2.replace(f" {level2}", "")
      else:
          level1 = None
          level2 = None
    # Store the odds ratio, SE, and p-value in the dictionary
      contrast_dict[contrast] = {
          'odds_ratio': float(row['odds.ratio']),
          'SE': float(row['SE']),
          'p_value': float(row['p.value']),
          'group1': group1,
          'group2': group2,
          'level1': level1,
          'level2': level2
      }
  return contrast_dict
      
def main():
  args = parse_arguments()
  weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep = "\t")
  # label_f1_results = pd.read_csv(args.label_f1_results, sep = "\t")
  estimates = pd.read_csv(args.estimates, sep = "\t")
  emmeans_summary = pd.read_csv(args.emmeans_summary, sep = "\t")
  factors = emmeans_summary.loc[:, :"response"].iloc[:, :-1].columns.tolist()
  key = args.key
 # make_contrast_dict(estimates, n_factors = len(factors))
  
  # filter to cutoff=0 
  # replace np.nan with "None"
  emmeans_summary = emmeans_summary.replace({np.nan: "None"})
  weighted_f1_results = weighted_f1_results.replace({np.nan: "None"})
  weighted_f1_results = weighted_f1_results[(weighted_f1_results["cutoff"] == 0 ) & (weighted_f1_results["key"] == key)]
  outdir = key
  os.makedirs(outdir, exist_ok=True)
  # merge means with weighted_f1_results
  
  if len(factors) == 1:
    factor1 = factors[0]
    factor2=None
  if len(factors) == 2:
    factor1, factor2 = factors

  for factor in factors:
    weighted_f1_results[factor] = weighted_f1_results[factor].astype(str)
    emmeans_summary[factor] = emmeans_summary[factor].astype(str)
    
  if factor2:
    weighted_f1_results = weighted_f1_results.merge(emmeans_summary, on = [factor1, factor2])
    plot_contrast_twofactors(weighted_f1_results, factor1, factor2, outdir)
  else:
    weighted_f1_results = weighted_f1_results.merge(emmeans_summary, on = factor1)
    plot_contrast_onefactor(weighted_f1_results, factor1, outdir, key)
    

  

  ## Plot all at once
  #plt.figure(figsize=(8, 20))
  #plt.errorbar(y="contrast", x="odds.ratio", xerr="SE", fmt='o', color='blue',
              #capsize=5, capthick=3, elinewidth=3, data=estimates)

  #plt.tight_layout()
  #plt.show()

  
if __name__ == "__main__":  
    main()