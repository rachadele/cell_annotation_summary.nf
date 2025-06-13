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
    parser.add_argument('--emmeans_estimates', type=str, help = "OR and pvalues from emmeans", default = "/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mus_musculus/model_eval/weighted/weighted_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_treatment_state_+_sex_+_method:cutoff_+_reference:method/subclass/files/subsample_ref_emmeans_estimates.tsv" )
    parser.add_argument('--emmeans_summary', type = str, help = "emmeans summary", default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/mus_musculus/model_eval/weighted/weighted_f1_~_reference_+_method_+_cutoff_+_subsample_ref_+_treatment_state_+_sex_+_method:cutoff_+_reference:method/subclass/files/subsample_ref_emmeans_summary.tsv")
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


def plot_contrast_twofactors(weighted_f1_results, factor1, factor2, outdir, contrast_results):
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
    fig, ax = plt.subplots(figsize=(30, 15))
    # Axis label size
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
        
         # Add ORs and p-values only for same `factor1`
    for f1 in unique_factor1:
        contrasts = contrast_results[(contrast_results["group1"] == f1) & (contrast_results["group2"] == f1)]
        #if len(contrasts) > 1:
        for idx, contrast in enumerate(contrasts.index):
          print(idx)
          f2a = contrasts["level1"].iloc[idx]
          f2b = contrasts["level2"].iloc[idx]
          odds_ratio = contrasts['odds.ratio'].iloc[idx]
          p_value = contrasts['p.value'].iloc[idx]

        
          y1 = unique_factor1.tolist().index(f1) + offsets[f2a] +0.15
          y2 = unique_factor1.tolist().index(f1) + offsets[f2b] +0.15

          y_low, y_high = sorted([y1, y2])

          # Bracket dimensions
          x_base = max(x, subset['response'].values[0]) + 0.1
          arm_length = 0.02  # horizontal length of the arms
          arm_height = 0.05  # vertical height of the arms

          # Draw short horizontal arms
          plt.plot([x_base, x_base + arm_length], [y_low, y_low], color='black')   # bottom
          plt.plot([x_base, x_base + arm_length], [y_high, y_high], color='black') # top

          # Draw vertical connector between them
          plt.plot([x_base + arm_length, x_base + arm_length], [y_low, y_high], color='black')

          star_y = (y1 + y2) / 2
          star_x = x_base + arm_length + 0.01
        
          if p_value < 1e-4:
              stars = '***'
          elif p_value < 1e-3:
              stars = '**'
          elif p_value < 0.01:
              stars = '*'
          else:
              stars = None
          if stars:
                ax.text(star_x, star_y, '*', ha='left', va='center',
                        fontsize=20, fontweight='bold', color='red')

    ax.set_title(f'{factor1.capitalize()}, Cutoff = 0', fontsize=40)
    ax.set_ylabel(factor1, fontsize=40)
    ax.set_xlabel('Weighted F1', fontsize=40)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)

    # Customize the legend to avoid duplication
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="lower left", bbox_to_anchor=(-1, -0.25), fontsize=35)
    plt.grid(False)  # Disables gridlines completely

    # Set the title for the plot
    #ax.set_title(f'{factor1.capitalize()} vs. {factor2.capitalize()} at Cutoff = 0', fontsize=40)

    # Adjust layout to accommodate the title
   # plt.tight_layout()

    # Show the plot
    plt.savefig(os.path.join(outdir,f"{factor1}_{factor2}_boxplot.png"), bbox_inches='tight')

 


def plot_contrast_onefactor(weighted_f1_results, factor1, outdir, key, contrast_results):
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
        x=factor1,
        y="weighted_f1",
        data=weighted_f1_results,
        ax=ax,
        linewidth=2.5,
        width=0.8,
        showfliers=False,
        whis=[5, 95],
        showcaps=False,
        palette=boxplot_palette
    )

    # Point plot with error bars
    x_positions = []
    y_positions = []
    for i, f1 in enumerate(unique_factor1):
        subset = weighted_f1_results[weighted_f1_results[factor1] == f1]
        x = i
        y = subset['response'].values[0]
        lower_err = y - subset['asymp.LCL'].values[0]
        upper_err = subset['asymp.UCL'].values[0] - y
        x_positions.append(x)
        y_positions.append(y)
        ax.errorbar(
            x=x,
            y=y,
            yerr=[[lower_err], [upper_err]],
            fmt='o',
            color=color_map[f1],
            capsize=5,
            capthick=5,
            elinewidth=5,
            label=f'estimated marginal mean (95% CI) for {f1}'
        )

    # Add contrast bar and odds ratio for pairwise contrasts
    for idx, contrast in contrast_results.iterrows():
        f2a = contrast['group1'].replace("subsample_ref", "")
        f2b = contrast['group2'].replace("subsample_ref", "")
    
        p_value = contrast['p.value']

        # Get x positions for the two levels
        x1 = unique_factor1.tolist().index(f2a)
        x2 = unique_factor1.tolist().index(f2b)

        y1 = weighted_f1_results[weighted_f1_results[factor1] == f2a]['response'].values[0]
        y2 = weighted_f1_results[weighted_f1_results[factor1] == f2b]['response'].values[0]
        
        # Set top y position for the bracket
        y_top = max(y1, y2) + 0.1 + (idx * 0.15)

        # Height of the short vertical arms
        arm_height = 0.05

        # Draw short vertical arms
        ax.plot([x1, x1], [y_top - arm_height, y_top], color='black', linewidth=1)
        ax.plot([x2, x2], [y_top - arm_height, y_top], color='black', linewidth=1)

        # Draw horizontal connector
        ax.plot([x1, x2], [y_top, y_top], color='black', linewidth=1)

        # Add stars with colors based on p-value
        # don't add stars for p-values > 0.05
        
        if p_value < 1e-4:
            stars = '***'
        elif p_value < 1e-3:
            stars = '**'
        elif p_value < 0.01:
            stars = '*'
        else:
            stars = None
        if stars:
            ax.text((x1 + x2) / 2, y_top + 0.02, stars, ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color="red")


    # Manually set font sizes
   # ax.set_title(f'{factor1.capitalize()} at {key} level, Cutoff = 0', fontsize=40)
    ax.set_xlabel(factor1.replace("_"," "), fontsize=40)
    ax.set_ylabel('Weighted F1', fontsize=40)
    ax.tick_params(axis='x', labelsize=35)
    ax.tick_params(axis='y', labelsize=35)

    wrap_labels(ax, width=2)

    # Customize the legend to avoid duplication
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.01), fontsize=35)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    plt.grid(False)  # Disables gridlines completely

    # Adjust layout to accommodate the title
    plt.tight_layout()

    # Save the plot
    plt.savefig(os.path.join(outdir, f"{factor1}_boxplot.png"))
    

def split_contrast(contrast):
    # Replace slashes inside parentheses with dashes
    contrast = re.sub(r'(?<=\()[^)]*?/', lambda match: match.group(0).replace('/', '-'), contrast)
    
    # Now split by slashes outside of parentheses
    groups = re.split(r'\s*/\s*(?![^\(]*\))', contrast)
    
    # Remove parentheses from the groups
    groups = [group.replace('(', '').replace(')', '') for group in groups]
    
    return groups

  
def reformat_contrast_df(contrast_results, n_factors=2):
  rows = []
  
  for _, row in contrast_results.iterrows():
      contrast = row["contrast"]
      # Use the split_contrast function to properly split the contrast
      groups = split_contrast(contrast)
      # Ensure that the contrast splits correctly
      if len(groups) == 2:
          group1, group2 = groups

      if n_factors == 2:
          level1 = group1.split(" ")[-1]
          level2 = group2.split(" ")[-1]
          group1 = group1.replace(f" {level1}", "")
          group2 = group2.replace(f" {level2}", "")
      else:
          level1 = None
          level2 = None
      
      rows.append({
          'contrast': contrast,
          'group1': group1,
          'group2': group2,
          'level1': level1,
          'level2': level2,
          'odds.ratio': float(row['odds.ratio']),
          'SE': float(row['SE']),
          'p.value': float(row['p.value'])
      })
  
  return pd.DataFrame(rows)



def main():
  
  args = parse_arguments()

  weighted_f1_results = pd.read_csv(args.weighted_f1_results, sep = "\t")
  # label_f1_results = pd.read_csv(args.label_f1_results, sep = "\t")
  contrast_results = pd.read_csv(args.emmeans_estimates, sep = "\t")
  emmeans_summary = pd.read_csv(args.emmeans_summary, sep = "\t")
  
  
  factors = emmeans_summary.loc[:, :"response"].iloc[:, :-1].columns.tolist()
  key = args.key

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
  
  contrast_results=reformat_contrast_df(contrast_results, n_factors = len(factors))
  print(contrast_results)
  if factor2:

    weighted_f1_results = weighted_f1_results.merge(emmeans_summary, on = [factor1, factor2])
    plot_contrast_twofactors(weighted_f1_results, factor1, factor2, outdir, contrast_results)
  else:
    weighted_f1_results = weighted_f1_results.merge(emmeans_summary, on = factor1)
    plot_contrast_onefactor(weighted_f1_results, factor1, outdir, key, contrast_results)
    
  
if __name__ == "__main__":  
    main()