#!/usr/bin/env python3
"""
Factor Contrast Plots

Creates forest-style plots for factor comparisons showing:
- Point estimates with 95% CI from emmeans
- Reference line at grand mean
- Optional significance indicators

Replaces the old bracket-based visualization with a cleaner forest plot approach.

Usage:
    python plot_contrasts.py \
        --weighted_f1_results path/to/weighted_f1_results.tsv \
        --emmeans_estimates path/to/factor_emmeans_estimates.tsv \
        --emmeans_summary path/to/factor_emmeans_summary.tsv \
        --key subclass \
        --outdir figures
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure,
    METHOD_COLORS, METHOD_NAMES,
    forest_plot,
    SINGLE_COL, HALF_HEIGHT, FULL_WIDTH
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot factor contrasts with forest plots"
    )
    parser.add_argument(
        '--weighted_f1_results', type=str, required=True,
        help="Path to aggregated weighted results TSV"
    )
    parser.add_argument(
        '--emmeans_estimates', type=str, required=True,
        help="Path to emmeans estimates TSV (odds ratios and p-values)"
    )
    parser.add_argument(
        '--emmeans_summary', type=str, required=True,
        help="Path to emmeans summary TSV (marginal means)"
    )
    parser.add_argument(
        '--key', type=str, default='subclass',
        help="Taxonomy level to filter by"
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help="Output directory for figures"
    )
    parser.add_argument(
        '--show_boxplot', action='store_true',
        help="Include boxplot of raw data behind forest plot"
    )
    parser.add_argument(
        '--significance_threshold', type=float, default=0.05,
        help="P-value threshold for significance marking"
    )

    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]


def get_factors_from_emmeans(df: pd.DataFrame) -> List[str]:
    """Extract factor column names from emmeans dataframe."""
    standard_cols = {'response', 'SE', 'df', 'asymp.LCL', 'asymp.UCL', 'key'}
    return [c for c in df.columns if c not in standard_cols]


def format_factor_value(value, factor_name: str) -> str:
    """Format factor values for display."""
    if factor_name == 'method':
        return METHOD_NAMES.get(value, str(value))
    elif factor_name == 'subsample_ref':
        return f'n={value}'
    elif factor_name == 'reference':
        s = str(value)
        return s[:30] + '...' if len(s) > 30 else s
    return str(value)


def get_color_for_value(value, factor_name: str) -> str:
    """Get color for factor value."""
    if factor_name == 'method':
        return METHOD_COLORS.get(value, '#333333')
    return '#666666'


def parse_contrast(contrast_str: str, n_factors: int = 1) -> dict:
    """Parse contrast string into components."""
    # Handle parentheses and slashes
    contrast_str = re.sub(r'(?<=\()[^)]*?/', lambda m: m.group(0).replace('/', '-'), contrast_str)
    groups = re.split(r'\s*/\s*(?![^\(]*\))', contrast_str)
    groups = [g.replace('(', '').replace(')', '') for g in groups]

    result = {'raw': contrast_str}
    if len(groups) == 2:
        result['group1'] = groups[0]
        result['group2'] = groups[1]

        if n_factors == 2:
            # Extract level from end of each group
            parts1 = groups[0].rsplit(' ', 1)
            parts2 = groups[1].rsplit(' ', 1)
            if len(parts1) == 2:
                result['factor1_val1'] = parts1[0]
                result['level1'] = parts1[1]
            if len(parts2) == 2:
                result['factor1_val2'] = parts2[0]
                result['level2'] = parts2[1]

    return result


def create_single_factor_forest(
    ax: plt.Axes,
    emmeans_summary: pd.DataFrame,
    emmeans_estimates: pd.DataFrame,
    factor_col: str,
    weighted_f1: Optional[pd.DataFrame] = None,
    show_boxplot: bool = False,
    sig_threshold: float = 0.05
) -> plt.Axes:
    """
    Create forest plot for single factor.
    """
    # Sort by response value
    plot_data = emmeans_summary.sort_values('response', ascending=True).reset_index(drop=True)

    n_groups = len(plot_data)
    positions = np.arange(n_groups)

    # Optional: show boxplot of raw data
    if show_boxplot and weighted_f1 is not None:
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            factor_val = row[factor_col]
            raw_data = weighted_f1[weighted_f1[factor_col] == factor_val]['weighted_f1']
            if len(raw_data) > 0:
                bp = ax.boxplot(
                    raw_data, positions=[i], vert=False,
                    widths=0.5, patch_artist=True,
                    showfliers=False, showcaps=False
                )
                bp['boxes'][0].set_facecolor('#f0f0f0')
                bp['boxes'][0].set_alpha(0.5)

    # Draw CI bars and points
    for i, (idx, row) in enumerate(plot_data.iterrows()):
        color = get_color_for_value(row[factor_col], factor_col)

        # CI bar
        ax.hlines(i, row['asymp.LCL'], row['asymp.UCL'],
                 colors=color, linewidth=2.5, zorder=2)

        # Point estimate
        ax.scatter(row['response'], i, s=100, c=[color],
                  zorder=3, edgecolors='white', linewidth=0.5)

    # Reference line at grand mean
    grand_mean = plot_data['response'].mean()
    ax.axvline(grand_mean, color='gray', linestyle='--',
              linewidth=0.8, alpha=0.6, zorder=0)

    # Add significance markers if contrast estimates provided
    if emmeans_estimates is not None and len(emmeans_estimates) > 0:
        # Parse contrasts to find significant pairs
        for _, contrast_row in emmeans_estimates.iterrows():
            if contrast_row.get('p.value', 1) < sig_threshold:
                # Could add significance markers here
                pass

    # Axis formatting
    ax.set_yticks(positions)
    labels = [format_factor_value(row[factor_col], factor_col)
              for _, row in plot_data.iterrows()]
    ax.set_yticklabels(labels)
    ax.set_xlabel('Estimated Marginal Mean F1')
    ax.set_ylabel(factor_col.replace('_', ' ').title())

    return ax


def create_two_factor_forest(
    ax: plt.Axes,
    emmeans_summary: pd.DataFrame,
    factor1: str,
    factor2: str,
    emmeans_estimates: Optional[pd.DataFrame] = None,
    weighted_f1: Optional[pd.DataFrame] = None
) -> plt.Axes:
    """
    Create forest plot for two-factor interaction.
    Groups by factor1, colors by factor2.
    """
    # Get unique values
    factor1_vals = emmeans_summary[factor1].unique()
    factor2_vals = emmeans_summary[factor2].unique()

    n_factor1 = len(factor1_vals)
    n_factor2 = len(factor2_vals)

    # Calculate positions with dodging
    positions = []
    labels = []
    colors = []

    for i, f1_val in enumerate(factor1_vals):
        for j, f2_val in enumerate(factor2_vals):
            offset = (j - (n_factor2 - 1) / 2) * 0.25
            positions.append(i + offset)
            labels.append(format_factor_value(f1_val, factor1))
            colors.append(get_color_for_value(f2_val, factor2))

    # Plot each point
    plot_idx = 0
    for f1_val in factor1_vals:
        for f2_val in factor2_vals:
            row_data = emmeans_summary[
                (emmeans_summary[factor1] == f1_val) &
                (emmeans_summary[factor2] == f2_val)
            ]
            if len(row_data) == 0:
                plot_idx += 1
                continue

            row = row_data.iloc[0]
            pos = positions[plot_idx]
            color = colors[plot_idx]

            # CI bar
            ax.hlines(pos, row['asymp.LCL'], row['asymp.UCL'],
                     colors=color, linewidth=2, zorder=2)

            # Point
            ax.scatter(row['response'], pos, s=80, c=[color],
                      zorder=3, edgecolors='white', linewidth=0.5)

            plot_idx += 1

    # Reference line
    grand_mean = emmeans_summary['response'].mean()
    ax.axvline(grand_mean, color='gray', linestyle='--',
              linewidth=0.8, alpha=0.6, zorder=0)

    # Axis formatting
    ax.set_yticks(range(n_factor1))
    ax.set_yticklabels([format_factor_value(v, factor1) for v in factor1_vals])
    ax.set_xlabel('Estimated Marginal Mean F1')
    ax.set_ylabel(factor1.replace('_', ' ').title())

    # Legend for factor2
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=get_color_for_value(v, factor2),
                         markersize=8, label=format_factor_value(v, factor2))
              for v in factor2_vals]
    ax.legend(handles=handles, title=factor2.replace('_', ' ').title(),
             loc='lower right', frameon=False)

    return ax


def main():
    args = parse_arguments()

    # Set publication style
    set_pub_style()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print("Loading data...")
    weighted_f1 = pd.read_csv(args.weighted_f1_results, sep='\t')
    emmeans_estimates = pd.read_csv(args.emmeans_estimates, sep='\t')
    emmeans_summary = pd.read_csv(args.emmeans_summary, sep='\t')

    # Get factors
    factors = get_factors_from_emmeans(emmeans_summary)
    print(f"Detected factors: {factors}")

    # Filter weighted_f1 to key and cutoff=0
    weighted_f1 = weighted_f1.replace({np.nan: 'None'})
    emmeans_summary = emmeans_summary.replace({np.nan: 'None'})

    weighted_f1_filtered = weighted_f1[
        (weighted_f1['cutoff'] == 0) &
        (weighted_f1['key'] == args.key)
    ].copy()

    # Convert factor columns to string for matching
    for factor in factors:
        if factor in weighted_f1_filtered.columns:
            weighted_f1_filtered[factor] = weighted_f1_filtered[factor].astype(str)
        if factor in emmeans_summary.columns:
            emmeans_summary[factor] = emmeans_summary[factor].astype(str)

    # Create figure
    if len(factors) == 1:
        factor1 = factors[0]
        n_levels = emmeans_summary[factor1].nunique()
        fig_height = max(HALF_HEIGHT, n_levels * 0.4)

        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, fig_height))

        create_single_factor_forest(
            ax=ax,
            emmeans_summary=emmeans_summary,
            emmeans_estimates=emmeans_estimates,
            factor_col=factor1,
            weighted_f1=weighted_f1_filtered if args.show_boxplot else None,
            show_boxplot=args.show_boxplot,
            sig_threshold=args.significance_threshold
        )

        ax.set_title(f'{factor1.replace("_", " ").title()} ({args.key})',
                    fontweight='bold', fontsize=11)

    elif len(factors) == 2:
        factor1, factor2 = factors
        n_levels = emmeans_summary[factor1].nunique()
        fig_height = max(HALF_HEIGHT, n_levels * 0.5)

        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, fig_height))

        create_two_factor_forest(
            ax=ax,
            emmeans_summary=emmeans_summary,
            factor1=factor1,
            factor2=factor2,
            emmeans_estimates=emmeans_estimates,
            weighted_f1=weighted_f1_filtered
        )

        ax.set_title(f'{factor1.replace("_", " ").title()} x {factor2.replace("_", " ").title()} ({args.key})',
                    fontweight='bold', fontsize=11)

    else:
        print(f"Warning: {len(factors)} factors detected. Only 1 or 2 factors supported.")
        return

    fig.tight_layout()

    # Save
    output_path = os.path.join(args.outdir, f'{"_".join(factors)}_forest')
    print(f"Saving figure to {output_path}...")
    save_figure(fig, output_path, formats=['pdf', 'png'], dpi=300)

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
