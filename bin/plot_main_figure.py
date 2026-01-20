#!/usr/bin/env python3
"""
Main Overview Panel Figure for Publication

Creates an information-dense 2x2 panel figure showing:
A: Cutoff sensitivity curves (F1 vs cutoff by method)
B: Reference atlas comparison (emmeans by reference, colored by method)
C: Taxonomy level slope chart (subclass → global)
D: Cross-study variability (strip plot by study)

Usage:
    python plot_main_figure.py \
        --cutoff_effects path/to/method_cutoff_effects.tsv \
        --reference_emmeans path/to/reference_method_emmeans_summary.tsv \
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
        '--reference_emmeans', type=str, required=True,
        help="Path to reference_method_emmeans_summary.tsv"
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

    # Remove legend (will use shared figure legend)
    ax.get_legend().remove() if ax.get_legend() else None

    return ax


def create_panel_b(ax, reference_emmeans):
    """
    Panel B: Reference atlas comparison.
    Forest plot showing emmeans by reference, colored by method.
    """
    # Get unique references and methods
    references = reference_emmeans['reference'].unique()
    methods = sorted(reference_emmeans['method'].unique())
    n_methods = len(methods)

    # Sort references by mean F1 across methods
    ref_means = reference_emmeans.groupby('reference')['response'].mean().sort_values(ascending=True)
    references = ref_means.index.tolist()

    # Truncate long reference names
    def truncate_ref(name, max_len=60):
        if len(str(name)) > max_len:
            return str(name)[:max_len-3] + '...'
        return str(name)

    # Plot each reference with methods dodged
    for ref_idx, ref in enumerate(references):
        ref_data = reference_emmeans[reference_emmeans['reference'] == ref]

        for method_idx, method in enumerate(methods):
            method_data = ref_data[ref_data['method'] == method]
            if len(method_data) == 0:
                continue

            row = method_data.iloc[0]
            color = METHOD_COLORS.get(method, '#333333')

            # Dodge position
            offset = (method_idx - (n_methods - 1) / 2) * 0.25
            y_pos = ref_idx + offset

            # CI bar
            ax.hlines(y_pos, row['asymp.LCL'], row['asymp.UCL'],
                     colors=color, linewidth=2, zorder=1)

            # Point estimate
            ax.scatter(row['response'], y_pos, s=60, c=[color],
                      zorder=2, edgecolors='white', linewidth=0.5)

    # Reference line at grand mean
    grand_mean = reference_emmeans['response'].mean()
    ax.axvline(grand_mean, color='gray', linestyle='--',
              linewidth=0.8, alpha=0.6, zorder=0)

    # Axis formatting
    ax.set_yticks(range(len(references)))
    # fix for setting yticklabels
    ax.set_yticklabels([truncate_ref(r) for r in references])
    ax.set_xlabel('Estimated Marginal Mean F1')
    ax.set_ylabel('')

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
        show_legend=False
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

    # Remove legend (will use shared figure legend)
    ax.get_legend().remove() if ax.get_legend() else None

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
    reference_emmeans = pd.read_csv(args.reference_emmeans, sep='\t')
    weighted_f1_data = pd.read_csv(args.weighted_f1, sep='\t')

    # Load taxonomy-level emmeans for slope chart
    if args.emmeans_dir and os.path.exists(args.emmeans_dir):
        taxonomy_emmeans = combine_taxonomy_levels(
            args.emmeans_dir,
            levels=KEY_ORDER,
            file_pattern='method_emmeans_summary.tsv'
        )
    else:
        # Create empty dataframe if no emmeans_dir
        taxonomy_emmeans = pd.DataFrame()


    # --- Top row: Panels A, B, C ---
    print("Creating top row (A, B, C)...")
    fig_top, axs_top = plt.subplots(1, 3, figsize=(FULL_WIDTH * 3.7, STANDARD_HEIGHT * 1.1))
    fig_top.subplots_adjust(wspace=0.55)
    # Panel A: Cutoff sensitivity (left)
    create_panel_a(axs_top[0], cutoff_data)
    add_panel_label(axs_top[0], 'A', x=-0.18, y=1.05, fontsize=20)
    # Panel B: Taxonomy slope chart (center)
    if len(taxonomy_emmeans) > 0 and 'key' in taxonomy_emmeans.columns and taxonomy_emmeans['key'].nunique() > 1:
        create_panel_c(axs_top[1], taxonomy_emmeans)
    else:
        axs_top[1].text(0.5, 0.5, 'Taxonomy data\nnot available',
                       ha='center', va='center', transform=axs_top[1].transAxes,
                       fontsize=10, color='gray')
        axs_top[1].set_xlabel('Taxonomy Level', fontsize=10)
        axs_top[1].set_ylabel('Estimated F1', fontsize=10)
    add_panel_label(axs_top[1], 'B', x=-0.18, y=1.05, fontsize=20)
    # Panel C: Cross-study swarm (right)
    create_panel_d(axs_top[2], weighted_f1_data, key=args.key)
    axs_top[2].set_ylabel('Query Datasets', fontsize=20)
    add_panel_label(axs_top[2], 'C', x=-0.18, y=1.05, fontsize=20)
    # Shared legend for top row
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color=METHOD_COLORS['scvi'],
               linestyle='-', linewidth=2, markersize=8,
               label=METHOD_NAMES['scvi']),
        Line2D([0], [0], marker='o', color=METHOD_COLORS['seurat'],
               linestyle='-', linewidth=2, markersize=8,
               label=METHOD_NAMES['seurat'])
    ]
    fig_top.legend(handles=legend_handles, loc='upper center',
                  ncol=2, frameon=False, fontsize=20,
                  bbox_to_anchor=(0.5, 1.02))
    # Save top row
    output_path_top = os.path.join(args.outdir, args.output_prefix + '_top')
    print(f"Saving top row to {output_path_top}...")
    save_figure(fig_top, output_path_top, formats=['png'], dpi=300)
    plt.close(fig_top)

    # --- Bottom row: Panel D ---
    print("Creating bottom row (D)...")
    fig_bottom, ax_bottom = plt.subplots(1, 1, figsize=(FULL_WIDTH * 2.2, STANDARD_HEIGHT * 1.1))
    create_panel_b(ax_bottom, reference_emmeans)
    ax_bottom.set_ylabel('Reference Datasets', fontsize=20)
    add_panel_label(ax_bottom, 'D', x=-0.04, y=1.05, fontsize=20)
    # Shared legend for bottom row
    fig_bottom.legend(handles=legend_handles, loc='upper center',
                     ncol=2, frameon=False, fontsize=20,
                     bbox_to_anchor=(0.5, 1.02))
    output_path_bottom = os.path.join(args.outdir, args.output_prefix + '_bottom')
    print(f"Saving bottom row to {output_path_bottom}...")
    save_figure(fig_bottom, output_path_bottom, formats=['png'], dpi=300)
    plt.close(fig_bottom)

    # --- Combine PNGs vertically ---
    print("Combining top and bottom PNGs into final figure...")
    from PIL import Image
    top_img = Image.open(output_path_top + '.png')
    bottom_img = Image.open(output_path_bottom + '.png')
    total_width = max(top_img.width, bottom_img.width)
    total_height = top_img.height + bottom_img.height
    combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    combined_img.paste(top_img, (0, 0))
    combined_img.paste(bottom_img, (0, top_img.height))
    final_output_path = os.path.join(args.outdir, args.output_prefix + '_combined.png')
    combined_img.save(final_output_path)
    print(f"Saved combined figure to {final_output_path}")


if __name__ == "__main__":
    main()
