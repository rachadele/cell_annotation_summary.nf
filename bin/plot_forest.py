#!/usr/bin/env python3
"""
Forest Plot for Method Comparison

Creates detailed statistical comparison forest plots showing:
- Point estimates with 95% CI from emmeans
- Multiple factors (method, reference, subsample_ref, treatment/disease)
- Faceted by taxonomy level

Usage:
    python plot_forest.py \
        --emmeans_files factor1_emmeans_summary.tsv factor2_emmeans_summary.tsv \
        --factor_names method reference \
        --key subclass \
        --outdir figures
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Dict, Optional

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure, add_panel_label,
    METHOD_COLORS, METHOD_NAMES, KEY_ORDER,
    forest_plot, FULL_WIDTH, SINGLE_COL, HALF_HEIGHT, STANDARD_HEIGHT
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate forest plots for factor comparisons"
    )
    parser.add_argument(
        '--emmeans_files', type=str, nargs='+', required=True,
        help="Paths to emmeans_summary.tsv files"
    )
    parser.add_argument(
        '--factor_names', type=str, nargs='+', default=None,
        help="Names for each factor (inferred from filename if not provided)"
    )
    parser.add_argument(
        '--key', type=str, default='subclass',
        help="Taxonomy level label for title"
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help="Output directory for figures"
    )
    parser.add_argument(
        '--output_prefix', type=str, default='forest_plot',
        help="Prefix for output filenames"
    )
    parser.add_argument(
        '--orientation', type=str, choices=['horizontal', 'vertical'],
        default='horizontal',
        help="Orientation of CI bars"
    )
    parser.add_argument(
        '--facet_by_key', action='store_true',
        help="If multiple taxonomy levels, facet by key"
    )

    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]


def infer_factor_name(filepath: str) -> str:
    """Infer factor name from emmeans filename."""
    basename = os.path.basename(filepath)
    # Pattern: {factor}_emmeans_summary.tsv
    if '_emmeans_summary' in basename:
        return basename.split('_emmeans_summary')[0]
    return basename.replace('.tsv', '')


def get_factor_column(df: pd.DataFrame) -> str:
    """Get the factor column from emmeans dataframe."""
    # Standard emmeans columns
    standard_cols = {'response', 'SE', 'df', 'asymp.LCL', 'asymp.UCL', 'key'}
    factor_cols = [c for c in df.columns if c not in standard_cols]
    if factor_cols:
        return factor_cols[0]
    return None


def format_factor_labels(values: pd.Series, factor_name: str) -> List[str]:
    """Format factor values for display."""
    labels = []
    for val in values:
        if factor_name == 'method':
            labels.append(METHOD_NAMES.get(val, str(val)))
        elif factor_name == 'reference':
            # Truncate long reference names
            s = str(val)
            labels.append(s[:40] + '...' if len(s) > 40 else s)
        elif factor_name == 'subsample_ref':
            labels.append(f'n={val}')
        else:
            labels.append(str(val))
    return labels


def get_color_for_factor(factor_name: str, value: str) -> str:
    """Get color for a factor value."""
    if factor_name == 'method':
        return METHOD_COLORS.get(value, '#333333')
    # Default colors for other factors
    return '#333333'


def create_single_forest(
    ax: plt.Axes,
    data: pd.DataFrame,
    factor_col: str,
    factor_name: str,
    vertical: bool = True,
    title: Optional[str] = None
) -> plt.Axes:
    """Create a single forest plot for one factor."""

    # Sort by estimate
    data = data.sort_values('response', ascending=True).reset_index(drop=True)

    n_groups = len(data)
    positions = np.arange(n_groups)

    # Get colors
    colors = [get_color_for_factor(factor_name, val) for val in data[factor_col]]

    # Calculate reference line (grand mean)
    reference = data['response'].mean()

    if vertical:
        # Horizontal CIs (groups on y-axis)
        for i, (idx, row) in enumerate(data.iterrows()):
            ax.hlines(i, row['asymp.LCL'], row['asymp.UCL'],
                     colors=colors[i], linewidth=2, zorder=1)
            ax.scatter(row['response'], i, s=80,
                      c=[colors[i]], zorder=2, edgecolors='white', linewidth=0.5)

        ax.axvline(reference, color='gray', linestyle='--',
                  linewidth=0.8, alpha=0.7, zorder=0)

        ax.set_yticks(positions)
        labels = format_factor_labels(data[factor_col], factor_name)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Estimated Marginal Mean F1')
        ax.set_ylabel(factor_name.replace('_', ' ').title())

    else:
        # Vertical CIs (groups on x-axis)
        for i, (idx, row) in enumerate(data.iterrows()):
            ax.vlines(i, row['asymp.LCL'], row['asymp.UCL'],
                     colors=colors[i], linewidth=2, zorder=1)
            ax.scatter(i, row['response'], s=80,
                      c=[colors[i]], zorder=2, edgecolors='white', linewidth=0.5)

        ax.axhline(reference, color='gray', linestyle='--',
                  linewidth=0.8, alpha=0.7, zorder=0)

        ax.set_xticks(positions)
        labels = format_factor_labels(data[factor_col], factor_name)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Estimated Marginal Mean F1')
        ax.set_xlabel(factor_name.replace('_', ' ').title())

    if title:
        ax.set_title(title, fontweight='bold', fontsize=11)

    return ax


def create_faceted_forest(
    data: pd.DataFrame,
    factor_col: str,
    factor_name: str,
    key_col: str = 'key',
    key_order: List[str] = None,
    vertical: bool = True
) -> plt.Figure:
    """Create faceted forest plot across taxonomy levels."""

    if key_order is None:
        key_order = [k for k in KEY_ORDER if k in data[key_col].unique()]

    n_keys = len(key_order)
    if n_keys == 0:
        n_keys = 1
        key_order = [data[key_col].iloc[0]] if key_col in data.columns else ['all']

    # Create figure
    fig, axes = plt.subplots(1, n_keys, figsize=(SINGLE_COL * n_keys, HALF_HEIGHT),
                             sharey=True if vertical else False,
                             sharex=True if not vertical else False)

    if n_keys == 1:
        axes = [axes]

    for ax, key in zip(axes, key_order):
        if key_col in data.columns:
            key_data = data[data[key_col] == key].copy()
        else:
            key_data = data.copy()

        if len(key_data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(key.title(), fontweight='bold')
            continue

        create_single_forest(
            ax=ax,
            data=key_data,
            factor_col=factor_col,
            factor_name=factor_name,
            vertical=vertical,
            title=key.title()
        )

        # Only show y-axis label on first plot
        if ax != axes[0] and vertical:
            ax.set_ylabel('')

    fig.tight_layout()
    return fig


def main():
    args = parse_arguments()

    # Set publication style
    set_pub_style()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Determine factor names
    if args.factor_names:
        factor_names = args.factor_names
    else:
        factor_names = [infer_factor_name(f) for f in args.emmeans_files]

    # Ensure same length
    if len(factor_names) < len(args.emmeans_files):
        factor_names.extend([f'factor_{i}' for i in range(len(factor_names), len(args.emmeans_files))])

    # Load and process each factor
    for filepath, factor_name in zip(args.emmeans_files, factor_names):
        print(f"Processing {factor_name}...")

        # Load data
        data = pd.read_csv(filepath, sep='\t')

        # Get factor column
        factor_col = get_factor_column(data)
        if factor_col is None:
            print(f"  Warning: Could not determine factor column for {filepath}")
            continue

        # Check if multiple taxonomy levels
        vertical = args.orientation == 'horizontal'

        if 'key' in data.columns and data['key'].nunique() > 1 and args.facet_by_key:
            # Faceted plot
            fig = create_faceted_forest(
                data=data,
                factor_col=factor_col,
                factor_name=factor_name,
                key_col='key',
                vertical=vertical
            )
        else:
            # Single plot
            n_groups = len(data[factor_col].unique())
            fig_height = max(HALF_HEIGHT, n_groups * 0.4)
            fig, ax = plt.subplots(figsize=(SINGLE_COL, fig_height))

            create_single_forest(
                ax=ax,
                data=data,
                factor_col=factor_col,
                factor_name=factor_name,
                vertical=vertical,
                title=f'{factor_name.replace("_", " ").title()} ({args.key})'
            )

            fig.tight_layout()

        # Save
        output_path = os.path.join(args.outdir, f'{args.output_prefix}_{factor_name}')
        print(f"  Saving to {output_path}...")
        save_figure(fig, output_path, formats=['pdf', 'png'], dpi=300)
        plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
