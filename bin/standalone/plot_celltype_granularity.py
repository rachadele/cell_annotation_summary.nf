#!/usr/bin/env python3
"""
Cell Type Granularity Comparison Figure

Creates a multi-panel visualization of F1 score distributions across different
taxonomy levels (subclass, class, family, global), aligned horizontally so that
cell types of the same lineage are in the same order on each plot.

Usage:
    python plot_celltype_granularity.py \
        --label_results path/to/label_results.tsv \
        --mapping_file path/to/census_map.tsv \
        --outdir figures
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure,
    METHOD_COLORS, METHOD_NAMES,
    FULL_WIDTH, STANDARD_HEIGHT
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate multi-granularity cell type trends figure"
    )
    parser.add_argument(
        '--label_results', type=str,
        default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mmus_new_tabulamuris/100/dataset_id/SCT/gap_false/aggregated_results/files/label_results.tsv",
        help="Path to label_results.tsv"
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
        '--mapping_file', type=str,
        default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/assets/census_map_mouse_author.tsv",
        help="Path to cell type mapping file for lineage colors"
    )
    parser.add_argument(
        '--lineage_key', type=str, default='global',
        help="Column in mapping file to color by (e.g., class, family)"
    )
    parser.add_argument(
        '--organism', type=str, default='mus_musculus',
        help="Organism name (e.g., mus_musculus, homo_sapiens)"
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help="Output directory for figures"
    )
    parser.add_argument(
        '--output_prefix', type=str, default='celltype_granularity',
        help="Prefix for output filenames"
    )
    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]


def load_lineage_mapping(mapping_file):
    """Load mapping file and return the full dataframe plus a label->global dict.

    Returns:
        map_df: Full mapping dataframe
        label_to_global: Dict mapping any label to its global lineage
    """
    if not mapping_file or not os.path.exists(mapping_file):
        return None, {}

    try:
        map_df = pd.read_csv(mapping_file, sep='\t')

        # Build mapping from any label to global lineage
        label_to_global = {}
        for col in map_df.columns:
            if col == 'global':
                continue
            for _, row in map_df.iterrows():
                val = row[col]
                global_val = row['global']
                if pd.notna(val) and pd.notna(global_val) and val not in label_to_global:
                    label_to_global[val] = global_val

        # Self-mappings for global values
        for v in map_df['global'].dropna().unique():
            if v not in label_to_global:
                label_to_global[v] = v

        return map_df, label_to_global
    except Exception as e:
        print(f"Warning: Could not load mapping file: {e}")
        return None, {}


def main():
    args = parse_arguments()

    # Set publication style
    set_pub_style()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.label_results}...")
    df = pd.read_csv(args.label_results, sep='\t', low_memory=False)

    # Filter by cutoff and reference
    mask = (df['cutoff'] == args.cutoff)
    if args.reference:
        mask &= (df['reference'] == args.reference)
    df = df[mask].copy()

    if df.empty:
        print("Error: No data found after filtering!")
        return

    # Load mapping file
    map_df, label_to_global = load_lineage_mapping(args.mapping_file)
    if map_df is None:
        print("Error: Could not load mapping file!")
        return

    # Define granularity levels (from finest to coarsest)
    granularity_levels = ['subclass', 'class', 'family', 'global']

    # Filter to only levels that exist in the data
    available_levels = [level for level in granularity_levels if level in df['key'].unique()]
    print(f"Available granularity levels: {available_levels}")

    if not available_levels:
        print("Error: No matching granularity levels found in data!")
        return

    # Get unique global lineages and create color palette
    unique_globals = sorted(map_df['global'].dropna().unique())
    global_palette = dict(zip(unique_globals, sns.color_palette("husl", len(unique_globals))))

    # Create figure with subplots side by side
    n_levels = len(available_levels)

    # Calculate figure dimensions
    # Find max number of labels across all levels for height
    max_labels = 0
    level_data = {}
    for level in available_levels:
        level_df = df[df['key'] == level].copy()
        n_labels = level_df['label'].nunique()
        max_labels = max(max_labels, n_labels)
        level_data[level] = level_df

    fig_height = max(8, max_labels * 0.4)
    fig_width = FULL_WIDTH * 1.0 * n_levels  # Wider to accommodate legend

    fig, axes = plt.subplots(1, n_levels, figsize=(fig_width, fig_height), sharey=False)
    if n_levels == 1:
        axes = [axes]

    # Determine methods present in data
    methods = sorted(df['method'].unique())
    method_palette = {m: METHOD_COLORS.get(m, '#333333') for m in methods}

    # Process each granularity level
    for idx, level in enumerate(available_levels):
        ax = axes[idx]
        level_df = level_data[level]

        # Get labels and their global lineages
        labels = level_df['label'].unique()
        label_globals = {label: label_to_global.get(label, 'Unknown') for label in labels}

        # Calculate median F1 per label for sorting
        label_medians = level_df.groupby('label')['f1_score'].median()

        # Create sorting dataframe
        label_info = pd.DataFrame({
            'label': label_medians.index,
            'median_f1': label_medians.values
        })
        label_info['global_lineage'] = label_info['label'].map(label_globals).fillna('Unknown')

        # Sort by global lineage first, then by median F1 within lineage
        label_info = label_info.sort_values(['global_lineage', 'median_f1'], ascending=[True, True])
        sorted_labels = label_info['label'].tolist()

        # Set categorical order
        level_df['label'] = pd.Categorical(level_df['label'], categories=sorted_labels, ordered=True)

        # Create boxplot
        sns.boxplot(
            data=level_df,
            y='label',
            x='f1_score',
            hue='method',
            order=sorted_labels,
            hue_order=methods,
            palette=method_palette,
            orient='h',
            ax=ax,
            linewidth=1.5,
            fliersize=0,
            width=0.6 if len(methods) > 1 else 0.4
        )

        # Remove seaborn's auto-generated legend
        if ax.get_legend():
            ax.get_legend().remove()

        # Axis formatting
        ax.set_title(level.capitalize())
        ax.set_xlabel('F1 Score')
        ax.set_ylabel('')
        ax.set_xlim(-0.05, 1.1)

        # Color y-axis labels by global lineage
        for tick in ax.get_yticklabels():
            label_text = tick.get_text()
            global_lineage = label_globals.get(label_text, 'Unknown')
            if global_lineage in global_palette:
                tick.set_color(global_palette[global_lineage])

    # Add shared legend on the right
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Lineage legend
    lineage_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
               markersize=12, label=lineage)
        for lineage, color in global_palette.items()
    ]

    # Method legend
    if len(methods) > 1:
        method_handles = [
            Patch(facecolor=METHOD_COLORS.get(m, '#333333'), edgecolor='black', linewidth=1, label=METHOD_NAMES.get(m, m))
            for m in methods
        ]
        fig.legend(
            handles=method_handles,
            title='Method',
            loc='upper right',
            bbox_to_anchor=(0.99, 0.99),
            frameon=False
        )

    fig.legend(
        handles=lineage_handles,
        title='Lineage',
        loc='lower right',
        bbox_to_anchor=(0.99, 0.01),
        frameon=False
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)  # Make more room for legend

    # Save
    output_path = os.path.join(args.outdir, f"{args.output_prefix}_{args.organism}")
    print(f"Saving to {output_path}...")
    save_figure(fig, output_path, formats=['png'], dpi=300)
    plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
