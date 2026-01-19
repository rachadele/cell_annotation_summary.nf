#!/usr/bin/env python3
"""
Main Overview Panel Figure for Publication

Creates an information-dense 2x2 panel figure showing:
A: Cutoff sensitivity curves (F1 vs cutoff by method)
B: Method effect forest plot (emmeans comparison)
C: Taxonomy level slope chart (subclass → global)
D: Cross-study variability (strip plot by study)

Usage:
    python plot_main_figure.py \
        --cutoff_effects path/to/method_cutoff_effects.tsv \
        --emmeans_summary path/to/method_emmeans_summary.tsv \
        --weighted_f1 path/to/weighted_f1_results.tsv \
        --emmeans_dir path/to/models_dir \
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

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure, add_panel_label,
    METHOD_COLORS, METHOD_NAMES, KEY_ORDER,
    cutoff_sensitivity_plot, forest_plot, slope_chart, study_swarm_plot,
    FULL_WIDTH, STANDARD_HEIGHT, combine_taxonomy_levels
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate main publication figure with 2x2 panel layout"
    )
    parser.add_argument(
        '--cutoff_effects', type=str, required=True,
        help="Path to method_cutoff_effects.tsv"
    )
    parser.add_argument(
        '--emmeans_summary', type=str, required=True,
        help="Path to method_emmeans_summary.tsv for primary taxonomy level"
    )
    parser.add_argument(
        '--weighted_f1', type=str, required=True,
        help="Path to weighted_f1_results.tsv"
    )
    parser.add_argument(
        '--emmeans_dir', type=str, default=None,
        help="Base directory containing emmeans files for all taxonomy levels (for slope chart)"
    )
    parser.add_argument(
        '--key', type=str, default='subclass',
        help="Primary taxonomy level for filtering"
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help="Output directory for figures"
    )
    parser.add_argument(
        '--output_prefix', type=str, default='main_figure',
        help="Prefix for output filenames"
    )

    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]


def create_panel_a(ax, cutoff_data):
    """
    Panel A: Cutoff sensitivity curves.
    Shows F1 vs cutoff with CI ribbons, colored by method.
    """
    cutoff_sensitivity_plot(
        ax=ax,
        data=cutoff_data,
        x_col='cutoff',
        y_col='fit',
        lower_col='lower',
        upper_col='upper',
        group_col='method',
        colors=METHOD_COLORS,
        show_ci=True,
        ci_alpha=0.2,
        line_width=2,
        marker='o',
        marker_size=5
    )

    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence Cutoff')
    ax.set_ylabel('Estimated F1 Score')

    # Move legend to avoid overlap
    ax.legend(loc='lower left', frameon=False)

    return ax


def create_panel_b(ax, emmeans_data):
    """
    Panel B: Method effect forest plot.
    Horizontal forest plot showing scvi vs seurat with 95% CI.
    """
    forest_plot(
        ax=ax,
        data=emmeans_data,
        estimate_col='response',
        lower_col='asymp.LCL',
        upper_col='asymp.UCL',
        group_col='method',
        color_col='method',
        colors=METHOD_COLORS,
        vertical=True,
        show_reference_line=True,
        reference_value=None,  # Uses grand mean
        marker_size=120,
        line_width=3
    )

    ax.set_xlabel('Estimated Marginal Mean F1')
    ax.set_ylabel('')

    # Format y-axis labels
    yticks = ax.get_yticks()
    ylabels = [METHOD_NAMES.get(l.get_text(), l.get_text())
               for l in ax.get_yticklabels()]
    ax.set_yticklabels(ylabels)

    return ax


def create_panel_c(ax, taxonomy_emmeans):
    """
    Panel C: Taxonomy level slope chart.
    Connected dot plot showing method performance across subclass → global.
    """
    # Aggregate by method and key if needed
    if 'key' in taxonomy_emmeans.columns:
        plot_data = taxonomy_emmeans.copy()
    else:
        # If no key column, this might be single-level data
        print("Warning: No 'key' column found for slope chart. Using single level.")
        return ax

    slope_chart(
        ax=ax,
        data=plot_data,
        x_col='key',
        y_col='response',
        group_col='method',
        colors=METHOD_COLORS,
        x_order=KEY_ORDER,
        marker_size=80,
        line_width=2,
        line_alpha=0.8,
        show_legend=True
    )

    ax.set_xlabel('Taxonomy Level')
    ax.set_ylabel('Estimated Marginal Mean F1')
    ax.set_ylim(0.5, 1)

    return ax


def create_panel_d(ax, weighted_f1_data, key='subclass', max_studies=15):
    """
    Panel D: Cross-study variability.
    Strip/swarm plot showing weighted F1 by study, colored by method.
    """
    # Filter to specific taxonomy level and cutoff=0
    plot_data = weighted_f1_data[
        (weighted_f1_data['key'] == key) &
        (weighted_f1_data['cutoff'] == 0)
    ].copy()

    if len(plot_data) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
               transform=ax.transAxes)
        return ax

    # Get studies sorted by median F1
    medians = plot_data.groupby('study')['weighted_f1'].median().sort_values(ascending=False)
    top_studies = medians.head(max_studies).index.tolist()

    # Filter to top studies
    plot_data = plot_data[plot_data['study'].isin(top_studies)]

    # Truncate long study names
    plot_data['study_short'] = plot_data['study'].apply(
        lambda x: x[:35] + '...' if len(str(x)) > 35 else x
    )

    study_swarm_plot(
        ax=ax,
        data=plot_data,
        study_col='study_short',
        value_col='weighted_f1',
        hue_col='method',
        colors=METHOD_COLORS,
        order=[plot_data[plot_data['study'] == s]['study_short'].iloc[0]
               for s in top_studies if s in plot_data['study'].values],
        dodge=True,
        size=2,
        alpha=0.6
    )

    ax.set_xlabel('Weighted F1 Score')
    ax.set_ylabel('')
    ax.legend(loc='lower right', frameon=False)

    return ax


def main():
    args = parse_arguments()

    # Set publication style
    set_pub_style()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print("Loading data...")
    cutoff_data = pd.read_csv(args.cutoff_effects, sep='\t')
    emmeans_data = pd.read_csv(args.emmeans_summary, sep='\t')
    weighted_f1_data = pd.read_csv(args.weighted_f1, sep='\t')

    # Load taxonomy-level emmeans for slope chart
    if args.emmeans_dir and os.path.exists(args.emmeans_dir):
        taxonomy_emmeans = combine_taxonomy_levels(
            args.emmeans_dir,
            levels=KEY_ORDER,
            file_pattern='method_emmeans_summary.tsv'
        )
    else:
        # Create from single file with key column if available
        taxonomy_emmeans = emmeans_data.copy()
        if 'key' not in taxonomy_emmeans.columns:
            taxonomy_emmeans['key'] = args.key

    # Create figure with 2x2 layout
    print("Creating figure...")
    fig = plt.figure(figsize=(FULL_WIDTH, STANDARD_HEIGHT))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # Panel A: Cutoff sensitivity
    ax_a = fig.add_subplot(gs[0, 0])
    create_panel_a(ax_a, cutoff_data)
    add_panel_label(ax_a, 'A', x=-0.15, y=1.05)

    # Panel B: Method forest plot
    ax_b = fig.add_subplot(gs[0, 1])
    create_panel_b(ax_b, emmeans_data)
    add_panel_label(ax_b, 'B', x=-0.15, y=1.05)

    # Panel C: Taxonomy slope chart
    ax_c = fig.add_subplot(gs[1, 0])
    if len(taxonomy_emmeans) > 0 and taxonomy_emmeans['key'].nunique() > 1:
        create_panel_c(ax_c, taxonomy_emmeans)
    else:
        # Fallback: show single-level data as bar
        ax_c.text(0.5, 0.5, 'Taxonomy data\nnot available',
                 ha='center', va='center', transform=ax_c.transAxes,
                 fontsize=10, color='gray')
        ax_c.set_xlabel('Taxonomy Level')
        ax_c.set_ylabel('Estimated F1')
    add_panel_label(ax_c, 'C', x=-0.15, y=1.05)

    # Panel D: Cross-study swarm
    ax_d = fig.add_subplot(gs[1, 1])
    create_panel_d(ax_d, weighted_f1_data, key=args.key)
    add_panel_label(ax_d, 'D', x=-0.15, y=1.05)

    # Save figure
    output_path = os.path.join(args.outdir, args.output_prefix)
    print(f"Saving figure to {output_path}...")
    save_figure(fig, output_path, formats=['pdf', 'png'], dpi=300)

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
