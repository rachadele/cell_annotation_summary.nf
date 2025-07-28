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
from matplotlib.patches import Patch
import re
random.seed(42)


def parse_arguments():
  parser = argparse.ArgumentParser(description="Download model file based on organism, census version, and tree file.")
  parser.add_argument('--query_paths', type=str, help="path to query file", default = "/space/grp/rschwartz/rschwartz/evaluation_data_wrangling/plotting_tests_mmus")
  parser.add_argument('--new_meta_paths', type=str, help="path to relabeled query metadata file", default = "/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/2024-07-30/mus_musculus/sample/SCT/ref_500_query_null_cutoff_0_refsplit_dataset_id/scvi/GSE124952/whole_cortex")
  parser.add_argument('--organism', type=str, help="organism", default = "mus_musculus")
  parser.add_argument('--ref_keys', type=str, help="keys to add", default = ["subclass","class","family","global"])
  parser.add_argument('--gene_mapping', type=str, default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/meta/gemma_genes.tsv")
  # deal with jupyter kernel arguments
  if __name__ == "__main__":
      known_args, _ = parser.parse_known_args()
      return known_args


def assemble_meta(new_meta_paths, pattern = "predictions"):
  # restrict only to "whole cortex"
  new_meta = {}
  for root, dirs, files in os.walk(new_meta_paths):
    for file in files:
      if re.search(pattern, file):  # Use regex search for flexibilit
        filepath=os.path.join(root,file)
        query_name = "_".join(os.path.basename(file).split("_")[:2])
        new_meta[query_name] = pd.read_csv(filepath, sep="\t")
      
  return(new_meta)
    # join 0 and 1
  

    
def read_adata(query_paths, gene_mapping, new_meta, ref_keys):
    
  queries = []
  for root,dir,files in os.walk(query_paths):
    for file in files:
      filepath = os.path.join(root,file)
      study =file.split(".h5ad")[0]
      if len(study.split("_")) > 1:
        query_name = "_".join(study.split("_")[:2])
      else:
          query_name=study
      query = sc.read_h5ad(filepath)
      # filter query for cells and genes
      sc.pp.filter_cells(query, min_counts=3)
      sc.pp.filter_cells(query, min_genes =200)
      if "feature_name" not in query.var.columns:
        query.var = query.var.merge(gene_mapping["OFFICIAL_SYMBOL"], left_index=True, right_index=True, how="left")
        # make symbol the index
        query.var.set_index("OFFICIAL_SYMBOL", inplace=True)
        #drop nan values
      else:
        query.var.set_index("feature_name", inplace=True)
      if new_meta.get(query_name) is None:
        continue
      meta = new_meta[query_name]
      query.obs=query.obs.reset_index()
      for key in ref_keys:
        query.obs[key] = meta[key]
        query.obs["predicted_"+ key] = meta["predicted_"+key]
      sc.pp.normalize_total(query)
      sc.pp.log1p(query)
      query.obs["study"]=query_name.split("_")[0]
      query.var_names = query.var_names.astype(str)
      query.var_names_make_unique()  # Make var_names unique

      query.obs_names_make_unique()
      queries.append(query)
      
  return(queries)

def combine_adata(queries):
  # turn  values into a list
  combined = ad.concat(queries, join="outer")  # Adjust join as needed

  sc.pp.highly_variable_genes(combined, n_top_genes=2000)
  
  #sc.pp.combat(combined, key="study")
  sc.tl.pca(combined, svd_solver='arpack')
  sc.pp.neighbors(combined, n_neighbors=40, n_pcs=50)
  sc.tl.umap(combined)
    # Remove var_names containing 'nan'
  combined.var_names = combined.var_names.astype(str)
  combined = combined[:, ~combined.var_names.str.contains("nan", case=True)]
 # sc.pp.filter_cells(combined, min_counts=500)
 # sc.pp.filter_cells(combined, min_genes =300)
  return(combined)


def subsample_cells(adata, n_cells=5):
  adata.obs_names_make_unique()
  # Store sampled indices
  sampled_indices = []

  for subclass, group in adata.obs.groupby("predicted_subclass"):
      if len(group) >= n_cells:
          sampled_indices.extend(np.random.choice(group.index, n_cells, replace=False))
      else:
          sampled_indices.extend(group.index)  # Keep all available if < n_cells

  # Subset adata
  adata = adata[sampled_indices]

  return adata


def plot_annotated_heatmap(adata, markers, groupby=["subclass",
                                                    "predicted_subclass","study"], 
                           figsize=(5, 10), prefix=""):
    # sort adata by groupby columns
    adata = adata[adata.obs.sort_values(groupby).index] 
    valid_markers = [gene for gene in markers if gene in adata.var_names]

    # Extract expression data
    expr_matrix =adata[:, valid_markers].X.toarray()
    expr_matrix = pd.DataFrame(expr_matrix, index=adata.obs.index, columns=valid_markers)
        # Extract categorical annotation
        
        
    annotations = adata.obs[groupby].astype(str).copy()

    # Flatten unique categories from all groupby columns
    unique_categories = sorted(set(annotations.values.ravel()))

    # Generate colors
    num_colors = len(unique_categories)
    palette = sns.color_palette("tab20", min(num_colors, 20)) + sns.color_palette("Set3", max(0, num_colors - 20))
    color_map = dict(zip(unique_categories, palette))

    # Create row_colors DataFrame
    row_colors = pd.DataFrame(index=annotations.index)

    legend_dict = {}
    for col in groupby:
        row_colors[col] = annotations[col].map(color_map)  # Process each column separately
        legend_dict[col] = [(label, color_map[label]) for label in sorted(annotations[col].unique())]


    # Map colors to annotations
    row_colors = row_colors.reset_index(drop=True)
    expr_matrix = expr_matrix.reset_index(drop=True)
    # Plot heatmap with annotations
    g = sns.clustermap(expr_matrix, 
                       row_colors=row_colors, 
                       row_cluster=False, 
                       col_cluster=False,
                       z_score=None,
                       standard_scale=1,
                      # cbar_pos=(0.02, 0.2, 0.03, 0.5),
                      # dendrogram_ratio=(0.1, 0.1),
                       figsize=figsize)
  #  g.ax_row_colors.set_position([0.1, 0.12, 0.2, 0.5])  # Adjust last value (height)

    g.ax_heatmap.set_yticks([])
    g.ax_heatmap.set_yticklabels([])
    g.ax_heatmap.set_xlabel("")
    for label in g.ax_heatmap.get_xticklabels():
      label.set_rotation(90)

  #  plt.tight_layout()
    plt.savefig(f"{prefix}_heatmap.png", bbox_inches="tight")
        # Create a figure for the legends
    fig, axes = plt.subplots(len(legend_dict), 1, figsize=(6, len(legend_dict) * 1.5))
    # Loop through the legend dictionary and create the patches for each category
    for i, (title, elements) in enumerate(legend_dict.items()):
        patches = [Patch(facecolor=color, edgecolor="black", label=label) for label, color in elements]
        axes[i].legend(handles=patches, title=title, loc="upper left", frameon=True, ncol=3)  
        axes[i].set_axis_off()
    plt.subplots_adjust(hspace=0.6)  
    plt.tight_layout()
    plt.savefig(f"{prefix}_legend.png")


def plot_umap(adata, groupby=["subclass", "predicted_subclass"], markers=None, prefix=""):
    # Plot umap with annotations
    sc.pl.umap(adata, color=groupby + markers, ncols=2, use_raw=False)
    plt.savefig(f"{prefix}_umap.png")
  
def plot_glutamatergic_analysis(adata, markers, groupby, prefix="glutamatergic"):
    
    glut = adata[(adata.obs["subclass"] == "Glutamatergic") & 
                 (adata.obs["predicted_family"] == "Glutamatergic")].copy()
    glut.obs_names_make_unique()
    

    # Annotated heatmap
    plot_annotated_heatmap(glut, markers, prefix=prefix, groupby=groupby)
 #   sc.pl.heatmap(glut, markers, dendrogram=True,
  #                groupby="predicted_subclass", use_raw=False)

    # UMAP
    plot_umap(glut, groupby=groupby, markers=markers, prefix=prefix)

def plot_GABAergic_analysis(adata, markers, groupby, prefix="GABAergic"):
    gaba = adata[(adata.obs["family"] == "GABAergic") & 
                 (adata.obs["predicted_family"] == "GABAergic")].copy()
    gaba.obs_names_make_unique()

    # Annotated heatmap
    plot_annotated_heatmap(gaba, markers, groupby, prefix=prefix)
    
    # UMAP
    plot_umap(gaba, groupby=groupby, markers=markers, prefix=prefix)

def main():
  args=parse_arguments()
  query_paths = args.query_paths
  new_meta_paths=args.new_meta_paths
  ref_keys = args.ref_keys
  organism=args.organism
  gene_mapping=pd.read_csv(args.gene_mapping, sep="\t")
  
  # Drop rows with missing values in the relevant columns
  gene_mapping = gene_mapping.dropna(subset=["ENSEMBL_ID", "OFFICIAL_SYMBOL"])

  # Set the index of gene_mapping to "ENSEMBL_ID" and ensure it's unique
  gene_mapping = gene_mapping.drop_duplicates(subset="ENSEMBL_ID")
  gene_mapping.set_index("ENSEMBL_ID", inplace=True)
  
  new_meta = assemble_meta(new_meta_paths)
  queries = read_adata(query_paths, gene_mapping, new_meta, ref_keys)
  combined = combine_adata(queries)


  
  #combined = combined[~combined.obs["predicted_subclass"].isin(["Ambiguous Glutamatergic neuron","Hippocampal neuron","Ambiguous GABAergic neuron"])]

# find NaN in predicted_subclass
  combined = combined[combined.obs["predicted_subclass"].notna()]
  combined.obs["author cell type"] = combined.obs["subclass"]
  combined.obs["predicted cell type"] = combined.obs["predicted_subclass"]
# 
  combined_subsample = subsample_cells(combined, n_cells=2000)
  
  #groupby = ["predicted_subclass", "subclass"]
  groupby = ["author cell type", "predicted cell type"]
  markers = ["SLC30A3","CUX2","RORB","FEZF2","BCL11B","TLE4","FOXP2"]
  if organism == "mus_musculus":
    markers = [marker.lower().capitalize() for marker in markers]
  #make markers lowercase then capitalize
  plot_glutamatergic_analysis(combined, markers, groupby, f"{organism}_glutamatergic")
  
  
  markers = ["LAMP5","PVALB","SNCG","SST","VIP"]
  if organism == "mus_musculus":
    markers = [marker.lower().capitalize() for marker in markers]
  #make markers lowercase then capitalize
  plot_GABAergic_analysis(combined, markers, groupby, prefix=f"{organism}_GABAergic")
  

  

  
if __name__ == "__main__":
  main()