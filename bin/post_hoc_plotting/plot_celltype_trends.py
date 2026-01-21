#!/usr/bin/env python3
"""
Cell Type Trends Figure

Creates a visualization of F1 score distributions across cell types,
showing boxplots with individual points to reveal the spread of performance.

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

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure,
    METHOD_COLORS, METHOD_NAMES,
    FULL_WIDTH, STANDARD_HEIGHT
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate cell type trends figure"
    )
    parser.add_argument(
        '--label_f1_results', type=str,
        default="/space/grp/rschwartz/rschwartz/evaluation_summary.nf/2024-07-01/mmus_new_tabulamuris/100/dataset_id/SCT/gap_false/aggregated_results/files/label_f1_results.tsv",
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


def load_lineage_mapping(mapping_file, lineage_key):
    """Load mapping file and return a function that looks up lineage for any label.

    For a given label, searches all columns in the mapping file to find a match,
    then returns the corresponding value from the lineage_key column.
    """
    if not mapping_file or not os.path.exists(mapping_file):
        return {}

    try:
        map_df = pd.read_csv(mapping_file, sep='\t')
        if lineage_key not in map_df.columns:
            return {}

        mapping = {}

        # For each column, find all unique values and map them to the lineage_key value
        for col in map_df.columns:
            if col == lineage_key:
                continue
            # For each unique value in this column, get the corresponding lineage_key value
            for _, row in map_df.iterrows():
                val = row[col]
                lineage_val = row[lineage_key]
                if pd.notna(val) and pd.notna(lineage_val) and val not in mapping:
                    mapping[val] = lineage_val

        # Also add self-mappings for lineage_key values (e.g., "Neuron" -> "Neuron")
        for v in map_df[lineage_key].dropna().unique():
            if v not in mapping:
                mapping[v] = v

        return mapping
    except Exception as e:
        print(f"Warning: Could not load mapping file: {e}")
        return {}


def create_celltype_plot(plot_df, labels, label_lineages, lineage_palette, methods, args,
                         output_suffix="", show_method_legend=True, markersize=8):
    """Create and save a cell type trends plot.

    Args:
        plot_df: DataFrame with plot data
        labels: List of cell type labels in order
        label_lineages: Dict mapping labels to lineages
        lineage_palette: Dict mapping lineages to colors
        methods: List of methods to include in plot
        args: Command line arguments
        output_suffix: Suffix to add to output filename (e.g., "_scvi_only")
        show_method_legend: Whether to show the method legend
        markersize: Size of the marker points
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Filter plot_df to only include specified methods
    plot_df = plot_df[plot_df['method'].isin(methods)].copy()

    # Figure size based on number of labels
    n_labels = len(labels)
    fig_height = max(10, n_labels * 0.6)
    fig_width = FULL_WIDTH * 2.2

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Create horizontal box plot colored by method
    method_palette = {m: METHOD_COLORS[m] for m in methods}

    sns.boxplot(
        data=plot_df,
        y='label',
        x='f1_score',
        hue='method',
        order=labels,
        hue_order=methods,
        palette=method_palette,
        orient='h',
        ax=ax,
        linewidth=1.5,
        fliersize=0,
        width=0.6 if len(methods) > 1 else 0.4
    )

    # Remove seaborn's auto-generated legend (we'll add our own)
    if ax.get_legend():
        ax.get_legend().remove()

    # Axis formatting
    ax.set_xlabel('Agreement with Author Labels (F1 Score)')
    ax.set_ylabel('Cell Type')
    ax.set_xlim(-0.05, 1.1)

    # Color y-axis labels by lineage (must be after set_yticklabels)
    if lineage_palette:
        for tick in ax.get_yticklabels():
            label_text = tick.get_text()
            lineage = label_lineages.get(label_text)
            if lineage and lineage in lineage_palette:
                tick.set_color(lineage_palette[lineage])

    # Create side legend for methods (only if show_method_legend is True)
    if show_method_legend:
        method_handles = [
            Patch(facecolor=METHOD_COLORS[m], edgecolor='black', linewidth=1, label=METHOD_NAMES[m])
            for m in methods
        ]

        # Method legend on right side
        method_legend = ax.legend(
            handles=method_handles,
            title='Method',
            loc='upper left',
            bbox_to_anchor=(1.02, 1),
            frameon=False
        )
        ax.add_artist(method_legend)

    # Lineage legend (if applicable)
    if lineage_palette:
        lineage_handles = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor=color,
                   markersize=12, label=lineage)
            for lineage, color in lineage_palette.items()
        ]
        ax.legend(
            handles=lineage_handles,
            title='Lineage',
            loc='lower left' if show_method_legend else 'upper left',
            bbox_to_anchor=(1.02, 0 if show_method_legend else 1),
            frameon=False
        )

    plt.tight_layout()

    # Save
    output_path = os.path.join(args.outdir, f"{args.output_prefix}_{args.organism}{output_suffix}")
    print(f"Saving to {output_path}...")
    save_figure(fig, output_path, formats=['png'], dpi=300)
    plt.close(fig)


def main():
    args = parse_arguments()

    # Set publication style (uses 20pt fonts from plot_utils)
    set_pub_style()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.label_f1_results}...")
    df = pd.read_csv(args.label_f1_results, sep='\t')

    # Filter data
    mask = (df['key'] == args.key) & (df['cutoff'] == args.cutoff)
    if args.reference:
        mask &= (df['reference'] == args.reference)

    df = df[mask].copy()

    if df.empty:
        print("Error: No data found after filtering!")
        return

    print(f"Found {len(df)} observations across {df['label'].nunique()} cell types")
    print(f"DEBUG: All unique labels in data: {sorted(df['label'].unique())}")
    print(f"DEBUG: Methods in data: {df['method'].unique()}")

    # Load lineage mapping for colors (searches all columns for label matches)
    lineage_map = load_lineage_mapping(args.mapping_file, args.lineage_key)
    print(f"DEBUG: Lineage map has {len(lineage_map)} entries")
    print(f"DEBUG: Lineage map keys: {list(lineage_map.keys())}")

    # Check which labels are missing from lineage map
    all_data_labels = df['label'].unique()
    missing_labels = [l for l in all_data_labels if l not in lineage_map]
    if missing_labels:
        print(f"DEBUG: Labels NOT in lineage map: {missing_labels}")

    # Calculate median F1 per label
    label_medians = df.groupby('label')['f1_score'].median()

    # Create dataframe with label, lineage, and median F1 for sorting
    label_info = pd.DataFrame({
        'label': label_medians.index,
        'median_f1': label_medians.values
    })
    label_info['lineage'] = label_info['label'].map(lineage_map).fillna('Unknown')

    # Sort by lineage first, then by median F1 within lineage
    label_info = label_info.sort_values(['lineage', 'median_f1'], ascending=[True, True])

    labels = label_info['label'].tolist()
    print(f"DEBUG: Labels after sorting: {labels}")

    # Filter to selected labels and set order
    plot_df = df[df['label'].isin(labels)].copy()
    print(f"DEBUG: plot_df has {len(plot_df)} rows and {plot_df['label'].nunique()} unique labels")
    plot_df['label'] = pd.Categorical(plot_df['label'], categories=labels, ordered=True)

    # Get lineage for each label (for y-axis coloring)
    label_lineages = {label: lineage_map.get(label, label) for label in labels}
    unique_lineages = sorted([v for v in set(label_lineages.values()) if v is not None])

    # Create lineage color palette
    if unique_lineages:
        lineage_palette = dict(zip(unique_lineages, sns.color_palette("husl", len(unique_lineages))))
    else:
        lineage_palette = {}

    # Create plot with both methods
    print("Creating plot with both methods...")
    create_celltype_plot(plot_df, labels, label_lineages, lineage_palette,
                         methods=['scvi', 'seurat'], args=args, output_suffix="")

    # Create plot with scvi only (no method legend)
    print("Creating plot with scvi only...")
    create_celltype_plot(plot_df, labels, label_lineages, lineage_palette,
                         methods=['scvi'], args=args, output_suffix="_scvi_only",
                         show_method_legend=False)

    print("Done!")


if __name__ == "__main__":
    main()
