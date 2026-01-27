#!/usr/bin/env python3
"""
Publication Figures for Cell Type Annotation Benchmarking

Creates publication-quality figures showing:
A: Cutoff sensitivity curves (F1 vs cutoff by method)
B: Taxonomy level slope chart (subclass -> global)
C: Reference atlas comparison (emmeans by reference, colored by method)
D: Experimental factor contrasts (disease, sex, treatment, region)

Usage:
    python plot_pub_figures.py \
        --cutoff_effects path/to/method_cutoff_effects.tsv \
        --reference_emmeans path/to/reference_method_emmeans_summary.tsv \
        --method_emmeans subclass/method_emmeans.tsv class/method_emmeans.tsv ... \
        --factor_emmeans disease_state_emmeans.tsv sex_emmeans.tsv ... \
        --organism homo_sapiens \
        --outdir figures
"""

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure, add_panel_label,
    METHOD_COLORS, METHOD_NAMES, KEY_ORDER,
    cutoff_sensitivity_plot, slope_chart,
    FULL_WIDTH, STANDARD_HEIGHT
)


# Experimental factor display names and ordering
FACTOR_DISPLAY_NAMES = {
    'disease_state': 'Disease State',
    'disease': 'Disease State',
    'sex': 'Sex',
    'region_match': 'Region Match',
    'treatment_state': 'Treatment',
    'treatment': 'Treatment',
    'subsample_ref': 'Reference Size'
}

# Factors to plot for each organism
HUMAN_FACTORS = ['disease_state', 'disease', 'sex', 'region_match']
MOUSE_FACTORS = ['treatment_state', 'treatment', 'sex']


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate publication figures for benchmarking results"
    )
    parser.add_argument(
        '--cutoff_effects', type=str, default=None,
        help="Path to method_cutoff_effects.tsv for primary key"
    )
    parser.add_argument(
        '--reference_emmeans', type=str, default=None,
        help="Path to reference_method_emmeans_summary.tsv for primary key"
    )
    parser.add_argument(
        '--method_emmeans', type=str, nargs='+', default=[],
        help="Paths to method_emmeans_summary.tsv files (one per taxonomy level)"
    )
    parser.add_argument(
        '--factor_emmeans', type=str, nargs='+', default=[],
        help="Paths to factor emmeans summary files (disease_state, sex, etc.)"
    )
    parser.add_argument(
        '--organism', type=str, default='homo_sapiens',
        choices=['homo_sapiens', 'mus_musculus'],
        help="Organism (affects which experimental factors to show)"
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help="Output directory for figures"
    )
    parser.add_argument(
        '--output_prefix', type=str, default='pub_figure',
        help="Prefix for output filenames"
    )

    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]


def extract_key_from_path(filepath: str) -> str:
    """Extract taxonomy key from file path."""
    # Look for key in path components
    for key in KEY_ORDER:
        if f'/{key}/' in filepath or filepath.startswith(f'{key}/'):
            return key
    # Try to extract from filename pattern like subclass_method_emmeans_summary.tsv
    basename = os.path.basename(filepath)
    for key in KEY_ORDER:
        if basename.startswith(f'{key}_'):
            return key
    return None


def extract_factor_from_path(filepath: str) -> str:
    """Extract factor name from emmeans file path."""
    basename = os.path.basename(filepath)
    # Pattern: factor_emmeans_summary.tsv
    match = re.match(r'(.+)_emmeans_summary\.tsv', basename)
    if match:
        return match.group(1)
    return None


def load_method_emmeans_from_files(filepaths: list) -> pd.DataFrame:
    """
    Load and combine method emmeans from multiple files.

    Parameters
    ----------
    filepaths : list
        List of paths to method_emmeans_summary.tsv files

    Returns
    -------
    pd.DataFrame
        Combined data with 'key' column for taxonomy level
    """
    dfs = []
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue

        key = extract_key_from_path(filepath)
        if key is None:
            print(f"  Warning: Could not extract key from: {filepath}")
            continue

        df = pd.read_csv(filepath, sep='\t')
        df['key'] = key
        dfs.append(df)
        print(f"  Loaded {key}: {len(df)} rows")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_factor_emmeans_from_files(filepaths: list, organism: str) -> dict:
    """
    Load factor emmeans from file paths.

    Parameters
    ----------
    filepaths : list
        List of paths to factor emmeans summary files
    organism : str
        Organism name to determine which factors to load

    Returns
    -------
    dict
        Dictionary mapping factor name to DataFrame
    """
    factor_data = {}
    target_factors = HUMAN_FACTORS if organism == 'homo_sapiens' else MOUSE_FACTORS

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"  Warning: File not found: {filepath}")
            continue

        factor = extract_factor_from_path(filepath)
        if factor is None:
            print(f"  Warning: Could not extract factor from: {filepath}")
            continue

        # Check if this is a factor we want
        if factor not in target_factors:
            continue

        df = pd.read_csv(filepath, sep='\t')
        factor_data[factor] = df
        print(f"  Loaded {factor}: {len(df)} levels")

    return factor_data


def create_panel_a(ax, cutoff_data):
    """Panel A: Cutoff sensitivity curves."""
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

    if ax.get_legend():
        ax.get_legend().remove()

    return ax


def create_panel_b(ax, taxonomy_emmeans):
    """Panel B: Taxonomy level slope chart."""
    if len(taxonomy_emmeans) == 0 or 'key' not in taxonomy_emmeans.columns:
        ax.text(0.5, 0.5, 'Taxonomy data\nnot available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        return ax

    slope_chart(
        ax=ax,
        data=taxonomy_emmeans,
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


def create_panel_c(ax, reference_emmeans):
    """Panel C: Reference atlas comparison."""
    references = reference_emmeans['reference'].unique()
    methods = sorted(reference_emmeans['method'].unique())
    n_methods = len(methods)

    ref_means = reference_emmeans.groupby('reference')['response'].mean().sort_values(ascending=True)
    references = ref_means.index.tolist()

    def truncate_ref(name, max_len=60):
        if len(str(name)) > max_len:
            return str(name)[:max_len-3] + '...'
        return str(name)

    for ref_idx, ref in enumerate(references):
        ref_data = reference_emmeans[reference_emmeans['reference'] == ref]

        for method_idx, method in enumerate(methods):
            method_data = ref_data[ref_data['method'] == method]
            if len(method_data) == 0:
                continue

            row = method_data.iloc[0]
            color = METHOD_COLORS.get(method, '#333333')
            offset = (method_idx - (n_methods - 1) / 2) * 0.25
            y_pos = ref_idx + offset

            ax.hlines(y_pos, row['asymp.LCL'], row['asymp.UCL'],
                     colors=color, linewidth=2, zorder=1)
            ax.scatter(row['response'], y_pos, s=60, c=[color],
                      zorder=2, edgecolors='white', linewidth=0.5)

    grand_mean = reference_emmeans['response'].mean()
    ax.axvline(grand_mean, color='gray', linestyle='--',
              linewidth=0.8, alpha=0.6, zorder=0)

    ax.set_yticks(range(len(references)))
    ax.set_yticklabels([truncate_ref(r) for r in references])
    ax.set_xlabel('Estimated Marginal Mean F1')
    ax.set_ylabel('')

    return ax


def create_panel_d(ax, factor_emmeans: dict, organism: str):
    """Panel D: Experimental factor contrasts."""
    if not factor_emmeans:
        ax.text(0.5, 0.5, 'No factor data available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        return ax

    # Determine factor order based on organism
    if organism == 'homo_sapiens':
        factor_order = [f for f in HUMAN_FACTORS if f in factor_emmeans]
    else:
        factor_order = [f for f in MOUSE_FACTORS if f in factor_emmeans]

    if not factor_order:
        ax.text(0.5, 0.5, 'No factor data available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        return ax

    plot_rows = []
    y_labels = []
    y_positions = []
    separator_positions = []
    current_pos = 0

    for factor_idx, factor in enumerate(factor_order):
        df = factor_emmeans[factor]
        standard_cols = {'response', 'SE', 'df', 'asymp.LCL', 'asymp.UCL'}
        factor_cols = [c for c in df.columns if c not in standard_cols]

        if not factor_cols:
            continue

        primary_factor_col = factor_cols[0]
        df_sorted = df.sort_values('response', ascending=False)

        for _, row in df_sorted.iterrows():
            level = str(row[primary_factor_col])
            if level.lower() in ['none', 'nan', '']:
                continue

            level_display = level.replace('_', ' ').title()
            plot_rows.append({
                'factor': factor,
                'level': level_display,
                'response': row['response'],
                'lower': row['asymp.LCL'],
                'upper': row['asymp.UCL']
            })
            y_labels.append(f"{level_display}")
            y_positions.append(current_pos)
            current_pos += 1

        if factor_idx < len(factor_order) - 1:
            separator_positions.append(current_pos - 0.5)
            current_pos += 0.5

    if not plot_rows:
        ax.text(0.5, 0.5, 'No factor data available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='gray')
        return ax

    factor_colors = {
        'disease_state': '#2ecc71',
        'disease': '#2ecc71',
        'sex': '#9b59b6',
        'region_match': '#e74c3c',
        'treatment_state': '#3498db',
        'treatment': '#3498db',
        'subsample_ref': '#f39c12'
    }

    for i, (row_data, y_pos) in enumerate(zip(plot_rows, y_positions)):
        color = factor_colors.get(row_data['factor'], '#333333')
        ax.hlines(y_pos, row_data['lower'], row_data['upper'],
                 colors=color, linewidth=2.5, zorder=2)
        ax.scatter(row_data['response'], y_pos, s=100, c=[color],
                  zorder=3, edgecolors='white', linewidth=0.5)

    for sep_pos in separator_positions:
        ax.axhline(sep_pos, color='lightgray', linestyle='-',
                  linewidth=0.5, alpha=0.7, zorder=0)

    all_responses = [r['response'] for r in plot_rows]
    grand_mean = np.mean(all_responses)
    ax.axvline(grand_mean, color='gray', linestyle='--',
              linewidth=0.8, alpha=0.6, zorder=0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Estimated Marginal Mean F1')
    ax.set_ylabel('')

    factor_label_positions = {}
    for row_data, y_pos in zip(plot_rows, y_positions):
        factor = row_data['factor']
        if factor not in factor_label_positions:
            factor_label_positions[factor] = []
        factor_label_positions[factor].append(y_pos)

    for factor, positions in factor_label_positions.items():
        mid_pos = np.mean(positions)
        display_name = FACTOR_DISPLAY_NAMES.get(factor, factor)
        ax.text(1.02, mid_pos, display_name, transform=ax.get_yaxis_transform(),
               fontsize=9, fontweight='bold', va='center', ha='left',
               color=factor_colors.get(factor, '#333333'))

    return ax


def main():
    args = parse_arguments()

    set_pub_style()
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print("Loading data...")
    cutoff_data = pd.read_csv(args.cutoff_effects, sep='\t')
    reference_emmeans = pd.read_csv(args.reference_emmeans, sep='\t')

    # Load taxonomy-level emmeans for slope chart
    print("Loading taxonomy-level emmeans...")
    if args.method_emmeans:
        taxonomy_emmeans = load_method_emmeans_from_files(args.method_emmeans)
    else:
        taxonomy_emmeans = pd.DataFrame()

    # Load experimental factor emmeans
    print("Loading experimental factor emmeans...")
    if args.factor_emmeans:
        factor_emmeans = load_factor_emmeans_from_files(args.factor_emmeans, args.organism)
    else:
        factor_emmeans = {}

    # --- Create Figure 1: Top row (A, B, C) ---
    print("\nCreating top row (A, B, C)...")
    fig_top, axs_top = plt.subplots(1, 3, figsize=(FULL_WIDTH * 3.7, STANDARD_HEIGHT * 1.1))
    fig_top.subplots_adjust(wspace=0.55)

    create_panel_a(axs_top[0], cutoff_data)
    add_panel_label(axs_top[0], 'A', x=-0.18, y=1.05, fontsize=20)

    if len(taxonomy_emmeans) > 0 and 'key' in taxonomy_emmeans.columns and taxonomy_emmeans['key'].nunique() > 1:
        create_panel_b(axs_top[1], taxonomy_emmeans)
    else:
        axs_top[1].text(0.5, 0.5, 'Taxonomy data\nnot available',
                       ha='center', va='center', transform=axs_top[1].transAxes,
                       fontsize=10, color='gray')
        axs_top[1].set_xlabel('Taxonomy Level', fontsize=10)
        axs_top[1].set_ylabel('Estimated F1', fontsize=10)
    add_panel_label(axs_top[1], 'B', x=-0.18, y=1.05, fontsize=20)

    create_panel_d(axs_top[2], factor_emmeans, args.organism)
    axs_top[2].set_ylabel('Factor Levels', fontsize=16)
    add_panel_label(axs_top[2], 'C', x=-0.18, y=1.05, fontsize=20)

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

    output_path_top = os.path.join(args.outdir, args.output_prefix + '_top')
    print(f"Saving top row to {output_path_top}...")
    save_figure(fig_top, output_path_top, formats=['png'], dpi=300)
    plt.close(fig_top)

    # --- Create Figure 2: Bottom row (D) ---
    print("\nCreating bottom row (D)...")
    fig_bottom, ax_bottom = plt.subplots(1, 1, figsize=(FULL_WIDTH * 2.2, STANDARD_HEIGHT * 1.1))
    create_panel_c(ax_bottom, reference_emmeans)
    ax_bottom.set_ylabel('Reference Datasets', fontsize=20)
    add_panel_label(ax_bottom, 'D', x=-0.04, y=1.05, fontsize=20)

    fig_bottom.legend(handles=legend_handles, loc='upper center',
                     ncol=2, frameon=False, fontsize=20,
                     bbox_to_anchor=(0.5, 1.02))

    output_path_bottom = os.path.join(args.outdir, args.output_prefix + '_bottom')
    print(f"Saving bottom row to {output_path_bottom}...")
    save_figure(fig_bottom, output_path_bottom, formats=['png'], dpi=300)
    plt.close(fig_bottom)

    # --- Combine PNGs vertically ---
    print("\nCombining top and bottom PNGs into final figure...")
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

    print("\nDone!")


if __name__ == "__main__":
    main()
