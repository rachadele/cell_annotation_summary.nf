#!/usr/bin/env python3
"""
Cell Type Trends Figure

Creates a detailed visualization of performance metrics across specific cell types.
Focuses on identifying "easy" vs. "hard" cell types and method disagreement.

Panels:
A: Performance Landscape (Ranked dot plot of F1 scores per cell type)
B: Method Disagreement (Lollipop chart showing F1 difference: scVI - Seurat)

Usage:
    python plot_celltype_trends.py \
        --label_f1_results path/to/label_f1_results.tsv \
        --key subclass \
        --cutoff 0 \
        --outdir figures
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure, add_panel_label,
    METHOD_COLORS, METHOD_NAMES,
    FULL_WIDTH, STANDARD_HEIGHT
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate cell type trends figure"
    )
    parser.add_argument(
        '--label_f1_results', type=str, default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/homo_sapiens/100/dataset_id/SCT/gap_false/aggregated_results/files/label_f1_results.tsv",
        help="Path to label_f1_results.tsv"
    )
    parser.add_argument(
        '--key', type=str, default='class',
        help="Taxonomy level to visualize (e.g., subclass, class)"
    )
    parser.add_argument(
        '--cutoff', type=float, default=0,
        help="Confidence cutoff to filter data (default: 0)"
    )
    parser.add_argument(
        '--reference', type=str, default=None,
        help="Reference to filter by (e.g., 'whole_cortex')"
    )
    parser.add_argument(
        '--top_n', type=int, default=40,
        help="Number of top cell types to display (to avoid overcrowding)"
    )
    parser.add_argument(
        '--mapping_file', type=str, default="/space/grp/rschwartz/rschwartz/nextflow_eval_pipeline/meta/census_map_mouse_author.tsv",
        help="Path to cell type mapping file (tsv) for lineage stratification"
    )
    parser.add_argument(
        '--lineage_key', type=str, default='family',
        help="Column in mapping file to group/stratify by (e.g., class, family, global)"
    )
    parser.add_argument(
        '--outdir', type=str, default='figures',
        help="Output directory for figures"
    )
    parser.add_argument(
        '--output_prefix', type=str, default='celltype_trends',
        help="Prefix for output filenames"
    )

    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]

def clean_label_names(label_series):
    """Clean up cell type names for display."""
    return label_series.str.replace('_', ' ').str.capitalize()

def calculate_stats(df):
    """
    Calculate mean and SEM F1 score per label per method.
    """
    stats_df = df.groupby(['label', 'method'])['f1_score'].agg(['mean', 'sem', 'count']).reset_index()
    return stats_df

def load_and_merge_mapping(stats_df, mapping_file, join_key='subclass', lineage_key='class'):
    """
    Load mapping file and merge lineage info into stats_df.
    Returns merged df and list of unique lineages.
    """
    if not mapping_file or not os.path.exists(mapping_file):
        print(f"Warning: Mapping file not found at {mapping_file}. Skipping stratification.")
        return stats_df, None

    print(f"Loading mapping file: {mapping_file}")
    map_df = pd.read_csv(mapping_file, sep='\t')
    
    # Check if keys exist
    if lineage_key not in map_df.columns:
        print(f"Warning: Lineage key '{lineage_key}' not in mapping file. Available: {map_df.columns.tolist()}")
        return stats_df, None
        
    # Standardize join keys
    if join_key not in map_df.columns:
        possible_keys = [k for k in map_df.columns if 'label' in k or 'type' in k or 'subclass' in k]
        if possible_keys:
            join_key = possible_keys[0]
            print(f"Assuming join key is '{join_key}'")
        else:
             print(f"Warning: Join key '{join_key}' not found in mapping file.")
             return stats_df, None

    # Merge
    map_subset = map_df[[join_key, lineage_key]].drop_duplicates()
    merged = stats_df.merge(map_subset, left_on='label', right_on=join_key, how='left')
    
    # Drop rows where lineage is missing or explicitly 'Unknown'
    merged = merged.dropna(subset=[lineage_key])
    merged = merged[merged[lineage_key] != 'Unknown']
    
    print(f"Retained {len(merged['label'].unique())} labels after filtering for valid {lineage_key}")
    
    return merged, lineage_key

def rank_celltypes(stats_df, lineage_key=None, top_n=None):
    """
    Rank cell types.
    If lineage_key is provided: Sort by Lineage, then by Mean F1.
    Else: Sort by Mean F1.
    """
    # Calculate overall mean per label (across methods)
    label_means = stats_df.groupby('label')['mean'].mean().reset_index()
    
    if lineage_key and lineage_key in stats_df.columns:
        # Get lineage for each label
        label_lineage = stats_df[['label', lineage_key]].drop_duplicates().set_index('label')
        label_means = label_means.merge(label_lineage, left_on='label', right_index=True)
        
        # Sort by Lineage then Mean
        # We want to group by lineage, so primary sort is lineage.
        # Within lineage, sort by performance (mean F1)
        label_means = label_means.sort_values([lineage_key, 'mean'], ascending=[True, True])
    else:
        label_means = label_means.sort_values('mean', ascending=True)
        
    sorted_labels = label_means['label'].tolist()
    
    if top_n and len(sorted_labels) > top_n:
        # If we filter by top_n, we should probably take the best performing ones overall?
        # Or should we respect the lineage structure?
        # If we just slice the end, we might cut off half a lineage.
        # Let's take the top N best performing labels, then re-sort them by lineage.
        
        # 1. Identify top N labels by score
        top_labels = stats_df.groupby('label')['mean'].mean().nlargest(top_n).index.tolist()
        
        # 2. Filter our sorted list to only include these
        sorted_labels = [l for l in sorted_labels if l in top_labels]
        
    return sorted_labels

def create_landscape_plot(ax, stats_df, sorted_labels, lineage_key=None):
    """
    Panel A: Dot plot of F1 scores.
    """
    plot_data = stats_df[stats_df['label'].isin(sorted_labels)].copy()
    plot_data['label'] = pd.Categorical(plot_data['label'], categories=sorted_labels, ordered=True)
    plot_data = plot_data.sort_values('label')
    
    lineage_map = {}
    if lineage_key and lineage_key in plot_data.columns:
        unique_lineages = sorted(plot_data[lineage_key].unique())
        palette = sns.color_palette("husl", len(unique_lineages))
        lineage_map = dict(zip(unique_lineages, palette))

    # Plot points
    for method, color in METHOD_COLORS.items():
        method_data = plot_data[plot_data['method'] == method]
        if method_data.empty:
            continue
            
        label_map = {l: i for i, l in enumerate(sorted_labels)}
        y_indices = method_data['label'].map(label_map)
        
        ax.errorbar(
            x=method_data['mean'],
            y=y_indices,
            xerr=method_data['sem'].fillna(0),
            fmt='o',
            color=color,
            label=METHOD_NAMES.get(method, method),
            alpha=0.8,
            markersize=5,
            capsize=3,
            linewidth=1,
            elinewidth=1.5
        )

    ax.set_yticks(np.arange(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels, fontsize=8)
    ax.set_xlabel('Mean F1 Score (±SEM)', fontsize=10)
    ax.set_ylim(-0.5, len(sorted_labels) - 0.5)
    ax.set_xlim(0, 1.05)
    ax.tick_params(axis='x', labelsize=8)
    
    # Add separator lines and coloring for lineages
    if lineage_key and lineage_key in plot_data.columns:
        l_map = plot_data[['label', lineage_key]].drop_duplicates().set_index('label')[lineage_key]
        ordered_lineages = [l_map.get(l) for l in sorted_labels]
        
        for i, tick in enumerate(ax.get_yticklabels()):
            label_text = tick.get_text()
            if label_text in l_map:
                lin = l_map[label_text]
                tick.set_color(lineage_map.get(lin, 'black'))

        # Find transition points for separator lines
        for i in range(len(ordered_lineages) - 1):
            if ordered_lineages[i] != ordered_lineages[i+1]:
                ax.axhline(i + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                
    ax.grid(axis='y', linestyle=':', alpha=0.3)
    return lineage_map

def create_difference_plot(ax, stats_df, sorted_labels, lineage_key=None):
    """
    Panel B: Lollipop chart of (scVI - Seurat) difference.
    """
    pivot_df = stats_df.pivot_table(index='label', columns='method', values='mean')
    
    if 'scvi' not in pivot_df.columns or 'seurat' not in pivot_df.columns:
        ax.text(0.5, 0.5, "Insufficient method data", ha='center', va='center')
        return ax
        
    pivot_df['diff'] = pivot_df['scvi'] - pivot_df['seurat']
    pivot_df = pivot_df.loc[pivot_df.index.intersection(sorted_labels)]
    pivot_df = pivot_df.reindex(sorted_labels)
    
    colors = [METHOD_COLORS['scvi'] if x > 0 else METHOD_COLORS['seurat'] for x in pivot_df['diff']]
    y_pos = np.arange(len(pivot_df))
    
    ax.hlines(y=y_pos, xmin=0, xmax=pivot_df['diff'], color=colors, alpha=0.6, linewidth=1.5)
    ax.scatter(pivot_df['diff'], y_pos, color=colors, s=20, alpha=1)
    
    ax.axvline(0, color='gray', linestyle='-', linewidth=0.8)
    
    # Add separator lines
    if lineage_key and lineage_key in stats_df.columns:
         l_map = stats_df[['label', lineage_key]].drop_duplicates().set_index('label')[lineage_key]
         ordered_lineages = [l_map.get(l) for l in sorted_labels]
         for i in range(len(ordered_lineages) - 1):
            if ordered_lineages[i] != ordered_lineages[i+1]:
                ax.axhline(i + 0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_yticks(y_pos)

def main():
    args = parse_arguments()
    set_pub_style()
    # Override some styles for this specific plot
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 10
    })
    
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.label_f1_results, sep='\t')
    
    # Filter data
    mask = (df['key'] == args.key) & (df['cutoff'] == args.cutoff)
    if args.reference:
        print(f"Filtering for reference='{args.reference}'...")
        mask &= (df['reference'] == args.reference)
    
    df = df[mask]
    
    if df.empty:
        print("Error: No data found after filtering!")
        return
    
    stats_df = calculate_stats(df)
    stats_df, active_lineage = load_and_merge_mapping(stats_df, args.mapping_file, 
                                                    join_key=args.key, 
                                                    lineage_key=args.lineage_key)
    
    sorted_labels = rank_celltypes(stats_df, lineage_key=active_lineage, top_n=args.top_n)
    n_labels = len(sorted_labels)
    fig_height = max(STANDARD_HEIGHT, n_labels * 0.18) # Slightly tighter packing
    
    # Single Panel Figure
    fig, ax1 = plt.subplots(1, 1, figsize=(FULL_WIDTH, fig_height))
    
    # Panel A
    lineage_map = create_landscape_plot(ax1, stats_df, sorted_labels, lineage_key=active_lineage)
    # add_panel_label(ax1, 'A', x=-0.2)
    
    # Shared Legends
    from matplotlib.lines import Line2D
    
    # Method Legend
    method_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS['scvi'], label='scVI', markersize=8),
                      Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS['seurat'], label='Seurat', markersize=8)]
    leg1 = fig.legend(handles=method_handles, title='Method', loc='upper left', bbox_to_anchor=(0.1, 1.02), ncol=2, frameon=False)
    
    # Lineage Legend (if active)
    if lineage_map:
        lineage_handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=c, label=l, markersize=8) 
                          for l, c in lineage_map.items()]
        n_cols = min(6, len(lineage_handles))
        fig.legend(handles=lineage_handles, title=active_lineage.capitalize(), 
                   loc='upper right', bbox_to_anchor=(0.9, 1.02), ncol=n_cols, frameon=False, fontsize=8, title_fontsize=10)
    
    output_path = os.path.join(args.outdir, args.output_prefix)
    save_figure(fig, output_path, formats=['png', 'pdf'])
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

