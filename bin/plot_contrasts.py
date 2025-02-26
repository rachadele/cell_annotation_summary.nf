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
from collections import defaultdict


# Function to parse command line arguments
def parse_arguments():
  parser = argparse.ArgumentParser(description="Plot contrasts for referend:method and save mean and SD of F1 scores for each contrast.")
  parser.add_argument('--f1_results', type=str, default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/query_500_sctransform/weighted_f1_results.tsv", help="Aggregated weighted results")
  parser.add_argument('--model_summary_coefs', type=str, default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/aggregated_results/query_500_sctransform/model_eval/weighted_model_summary_coefs_combined.tsv", help="Model summary coefficients")
  parser.add_argument('--type', type=str, default="weighted", help="Type of f1 results")
  if __name__ == "__main__":
      known_args, _ = parser.parse_known_args()
      return known_args



def plot_contrasts(model_contrasts, f1_results, output_prefix="f1_scores"):
    """
    Generates violin plots of F1 scores for each key in model_contrasts.

    Parameters:
    - model_contrasts: dict containing contrast information per key and model.
    - f1_results: DataFrame containing F1 scores with a 'key' column.
    - output_prefix: str, prefix for saved plot filenames.
    """
    # set global fontsize
    plt.rcParams.update({'font.size': 25})
    for key, contrast_data in model_contrasts.items():
        plt.figure(figsize=(16, 15))  # Create a new figure for each key
        ax = plt.gca()  # Get current axis
        legend_handles = {}  # Dictionary to store unique legend handles

        for model, all_contrasts in contrast_data.items():
            for contrast_dict in all_contrasts:
                val = contrast_dict['value'] # reference value
                group = contrast_dict['group'] # "reference" string
                facet = contrast_dict['facet'] # "method" string
                FDR = contrast_dict['FDR']

                if f1_results.empty:
                    raise ValueError("No f1 results")
                # Subset F1 data
                f1_data_subset = f1_results[(f1_results['key'] == key) & (f1_results[group] == val)]

                # Plot violin plot for F1 scores
                sns.boxplot(x=group, y='weighted_f1', 
                            data=f1_data_subset, hue=facet, showmeans=True, 
                            meanprops={'marker':'o', 'markerfacecolor':'red', 'markeredgecolor':'black'},
                            palette='Set3')
                
                        # Collect legend handles
                handles, labels = ax.get_legend_handles_labels()
                for h, l in zip(handles, labels):
                    legend_handles[l] = h  # Store unique labels
                    
                # If FDR < 0.05, add a star above the highest violin plot
                if FDR.astype(float)< 0.05:
                    max_y = f1_data_subset['weighted_f1'].max()
                    plt.text(x=val, y=max_y, s='*', fontsize=14, ha='center', color='red')

          
          
        # Customize plot appearance
        plt.title(f"F1 Scores for {key} - {model}")
        plt.xlabel(group)
        plt.ylabel('F1 Score')
        plt.xticks(fontsize=10, rotation=90)
        # Set unique legend
        # make legend font
        if legend_handles:
            ax.legend(legend_handles.values(), 
                      legend_handles.keys(), title=facet.capitalize(), bbox_to_anchor=(1, 1))
        else:
            ax.get_legend().remove()

        # Save and show the plot
       # plt.tight_layout()
        filename = f"{output_prefix}_{key}.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.show()

        # Close the figure after plotting
        plt.close()


def get_contrast_stats(f1_results, model_summary_coefs, model_contrasts, metric='weighted_f1'):
    stats_list = []

    for key in model_summary_coefs['key'].unique():
        for model in model_summary_coefs['formula'].unique():
          if model in model_contrasts[key]:
            for contrast in model_contrasts[key][model]:
                group = contrast['group']
                facet = contrast['facet']
                FDR = contrast['FDR']
                formula = contrast['contrast']
                val = contrast['value']
                # Group by `facet` and `group`, then compute mean & std
                f1_data_subset = f1_results[(f1_results['key'] == key) & (f1_results[group] == val)]
                grouped_stats = (
                    f1_data_subset.groupby([facet])[metric]
                    .agg(['mean', 'std'])
                    .reset_index()
                )
                grouped_stats[group] = val
                # Store key, model, contrast info
                grouped_stats['key'] = key
                grouped_stats['model'] = model
                grouped_stats['contrast'] = formula
                grouped_stats['FDR'] = FDR
                stats_list.append(grouped_stats)

    pd.concat(stats_list, ignore_index=True).to_csv('contrast_stats.tsv', sep='\t', index=False)
       

def main():
  args=parse_arguments()
  f1_results = pd.read_csv(args.f1_results, sep="\t")
  model_summary_coefs = pd.read_csv(args.model_summary_coefs, sep="\t")
  type_f1 = args.type
  # create a dict to store model and its contrasts
  # get contrasts
  models = model_summary_coefs['formula'].unique()
  # for each model, get the contrasts
  # split the model summary coefficients by
  model_summary_coefs_list = [group for _, group in model_summary_coefs.groupby('key')]
  model_contrasts = defaultdict(dict)

  f1_results = f1_results[f1_results['cutoff'] == 0]
  # drop duplicates without ref_split
  
  for df in model_summary_coefs_list:
    key = df['key'].unique()[0]
    model_contrasts[key] = {}
    for model in models:
      model_summary_coefs_subset = df[df['formula'] == model]
      # this doesn't apply to some models
      # we're just gonna ignore the additive model for now
      contrasts=model_summary_coefs_subset['Term'].unique()
      method_interaction_contrasts = [contrast for contrast in contrasts if ':' in contrast]
      if len(method_interaction_contrasts) == 0:
        continue
      structured_data = []
      facet = model.split('~')[1].strip().split(":")[1]
      group = model.split('~')[1].strip().split(":")[0].split(" ")[-1]
      for item in method_interaction_contrasts:
        reference = item.split(':')[0].split('[')[1][:-1]  # Extracts the text inside 'reference[...]'
        method = item.split(':')[1].split('[')[1][:-1]  # Extracts the text inside 'method[...]'
        # Create a dictionary for each item
        structured_data.append({
            'group': group,
            'value': reference,
            'facet': facet,
            'contrast': item,
            'FDR': model_summary_coefs_subset[model_summary_coefs_subset['Term'] == item]['FDR'].values[0]
        })
        model_contrasts[key][model] = structured_data

  plot_contrasts(model_contrasts, f1_results, output_prefix=type_f1)
  get_contrast_stats(f1_results, model_summary_coefs, model_contrasts)


 
    
    
if __name__=="__main__":
    main()