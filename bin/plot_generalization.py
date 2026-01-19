#!/usr/bin/env python3
"""
Cross-Study Generalization Figure

Creates visualizations showing variability across studies/datasets:
- Option A: Raincloud plots (half-violin + jittered points + box)
- Option B: Caterpillar/forest plot by study

Usage:
    python plot_generalization.py \
        --weighted_f1 path/to/weighted_f1_results.tsv \
        --key subclass \
        --plot_type raincloud \
        --outdir figures
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Optional

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure,
    METHOD_COLORS, METHOD_NAMES, KEY_ORDER,
    FULL_WIDTH, SINGLE_COL, STANDARD_HEIGHT, HALF_HEIGHT
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate cross-study generalization figures"
    )
    parser.add_argument(
        '--weighted_f1', type=str, required=True,
        help="Path to weighted_f1_results.tsv"
    )
    parser.add_argument(
        '--key', type=str, default='subclass',
        help="Taxonomy level to filter by"
    )
    parser.add_argument(
        '--cutoff', type=float, default=0,
        help="Cutoff value to filter by"
    )
    parser.add_argument(
        '--plot_type', type=str, choices=['raincloud', 'caterpillar', 'both'],
        default='raincloud',
        help="Type of plot to generate"
    )
    parser.add_argument(
        '--max_studies', type=int, default=20,
        help="Maximum number of studies to show"
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help="Output directory for figures"
    )
    parser.add_argument(
        '--output_prefix', type=str, default='generalization',
        help="Prefix for output filenames"
    )

    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]


def truncate_study_name(name: str, max_len: int = 40) -> str:
    """Truncate long study names."""
    if pd.isna(name):
        return 'Unknown'
    name = str(name)
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + '...'


def create_raincloud_plot(
    data: pd.DataFrame,
    study_col: str = 'study',
    value_col: str = 'weighted_f1',
    hue_col: str = 'method',
    max_studies: int = 20
) -> plt.Figure:
    """
    Create raincloud plot for cross-study comparison.
    Half-violin (distribution) + jittered points + box.
    """
    # Calculate median per study and sort
    medians = data.groupby(study_col)[value_col].median().sort_values(ascending=False)
    top_studies = medians.head(max_studies).index.tolist()

    # Filter and prepare data
    plot_data = data[data[study_col].isin(top_studies)].copy()
    plot_data['study_short'] = plot_data[study_col].apply(
        lambda x: truncate_study_name(x, 40)
    )

    # Create ordered mapping
    study_order = [truncate_study_name(s, 40) for s in top_studies]

    n_studies = len(study_order)
    fig_height = max(HALF_HEIGHT, n_studies * 0.35)
    fig, ax = plt.subplots(figsize=(FULL_WIDTH * 0.8, fig_height))

    methods = sorted(plot_data[hue_col].unique())
    n_methods = len(methods)

    for study_idx, study in enumerate(study_order):
        study_data = plot_data[plot_data['study_short'] == study]

        for method_idx, method in enumerate(methods):
            method_data = study_data[study_data[hue_col] == method][value_col].dropna()

            if len(method_data) < 2:
                continue

            color = METHOD_COLORS.get(method, '#333333')
            y_offset = (method_idx - (n_methods - 1) / 2) * 0.3
            y_pos = study_idx + y_offset

            # Half violin (top)
            try:
                kernel = stats.gaussian_kde(method_data)
                x_range = np.linspace(method_data.min(), method_data.max(), 100)
                density = kernel(x_range)
                density = density / density.max() * 0.25  # Scale

                ax.fill_betweenx(
                    x_range, y_pos, y_pos + density,
                    alpha=0.3, color=color
                )
            except Exception:
                pass

            # Jittered points
            jitter = np.random.uniform(-0.1, 0, len(method_data))
            ax.scatter(
                method_data, y_pos + jitter,
                s=3, alpha=0.5, color=color, edgecolors='none'
            )

            # Box (simplified - just median and quartiles)
            q1 = method_data.quantile(0.25)
            q3 = method_data.quantile(0.75)
            median = method_data.median()

            box_height = 0.08
            rect = plt.Rectangle(
                (q1, y_pos - 0.15 - box_height/2),
                q3 - q1, box_height,
                facecolor=color, alpha=0.8, edgecolor='white', linewidth=0.5
            )
            ax.add_patch(rect)

            # Median line
            ax.vlines(median, y_pos - 0.15 - box_height/2, y_pos - 0.15 + box_height/2,
                     colors='white', linewidth=1.5)

    # Formatting
    ax.set_yticks(range(len(study_order)))
    ax.set_yticklabels(study_order, fontsize=8)
    ax.set_xlabel('Weighted F1 Score')
    ax.set_ylabel('')
    ax.set_xlim(0, 1.05)

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS.get(m, '#333'),
                         markersize=8, label=METHOD_NAMES.get(m, m)) for m in methods]
    ax.legend(handles=handles, loc='lower right', frameon=False)

    ax.invert_yaxis()  # Highest median at top
    fig.tight_layout()

    return fig


def create_caterpillar_plot(
    data: pd.DataFrame,
    study_col: str = 'study',
    value_col: str = 'weighted_f1',
    hue_col: str = 'method',
    max_studies: int = 20
) -> plt.Figure:
    """
    Create caterpillar/forest plot by study.
    Shows mean + CI per study, sorted by mean F1.
    """
    # Calculate stats per study and method
    stats_df = data.groupby([study_col, hue_col])[value_col].agg(
        ['mean', 'std', 'count']
    ).reset_index()

    # Calculate 95% CI
    stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
    stats_df['ci_lower'] = stats_df['mean'] - 1.96 * stats_df['se']
    stats_df['ci_upper'] = stats_df['mean'] + 1.96 * stats_df['se']

    # Get overall mean per study for sorting
    study_means = stats_df.groupby(study_col)['mean'].mean().sort_values(ascending=False)
    top_studies = study_means.head(max_studies).index.tolist()

    # Filter
    stats_df = stats_df[stats_df[study_col].isin(top_studies)].copy()
    stats_df['study_short'] = stats_df[study_col].apply(lambda x: truncate_study_name(x, 40))

    study_order = [truncate_study_name(s, 40) for s in top_studies]

    n_studies = len(study_order)
    fig_height = max(HALF_HEIGHT, n_studies * 0.35)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.5, fig_height))

    methods = sorted(stats_df[hue_col].unique())
    n_methods = len(methods)

    for study_idx, study in enumerate(study_order):
        study_data = stats_df[stats_df['study_short'] == study]

        for method_idx, method in enumerate(methods):
            method_row = study_data[study_data[hue_col] == method]

            if len(method_row) == 0:
                continue

            row = method_row.iloc[0]
            color = METHOD_COLORS.get(method, '#333333')
            y_offset = (method_idx - (n_methods - 1) / 2) * 0.25
            y_pos = study_idx + y_offset

            # CI bar
            ax.hlines(y_pos, row['ci_lower'], row['ci_upper'],
                     colors=color, linewidth=1.5, alpha=0.8)

            # Mean point
            ax.scatter(row['mean'], y_pos, s=40, c=[color],
                      edgecolors='white', linewidth=0.5, zorder=3)

    # Reference line at overall mean
    overall_mean = data[value_col].mean()
    ax.axvline(overall_mean, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    # Formatting
    ax.set_yticks(range(len(study_order)))
    ax.set_yticklabels(study_order, fontsize=8)
    ax.set_xlabel('Mean Weighted F1 Score')
    ax.set_ylabel('')
    ax.set_xlim(0, 1.05)

    # Legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=METHOD_COLORS.get(m, '#333'),
                         markersize=8, label=METHOD_NAMES.get(m, m)) for m in methods]
    ax.legend(handles=handles, loc='lower right', frameon=False)

    ax.invert_yaxis()
    fig.tight_layout()

    return fig


def create_summary_strip_plot(
    data: pd.DataFrame,
    study_col: str = 'study',
    value_col: str = 'weighted_f1',
    hue_col: str = 'method',
    max_studies: int = 20
) -> plt.Figure:
    """
    Create a simple strip plot with median markers.
    """
    # Sort studies by median
    medians = data.groupby(study_col)[value_col].median().sort_values(ascending=False)
    top_studies = medians.head(max_studies).index.tolist()

    plot_data = data[data[study_col].isin(top_studies)].copy()
    plot_data['study_short'] = plot_data[study_col].apply(lambda x: truncate_study_name(x, 40))

    study_order = [truncate_study_name(s, 40) for s in top_studies]

    n_studies = len(study_order)
    fig_height = max(HALF_HEIGHT, n_studies * 0.3)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, fig_height))

    methods = sorted(plot_data[hue_col].unique())
    palette = [METHOD_COLORS.get(m, '#333333') for m in methods]

    # Strip plot
    sns.stripplot(
        data=plot_data,
        y='study_short',
        x=value_col,
        hue=hue_col,
        order=study_order,
        hue_order=methods,
        palette=palette,
        dodge=True,
        size=3,
        alpha=0.6,
        jitter=True,
        ax=ax
    )

    # Add median markers
    for study_idx, study in enumerate(study_order):
        study_data = plot_data[plot_data['study_short'] == study]
        for method_idx, method in enumerate(methods):
            method_data = study_data[study_data[hue_col] == method][value_col]
            if len(method_data) > 0:
                median = method_data.median()
                y_offset = (method_idx - (len(methods) - 1) / 2) * 0.2
                ax.scatter(median, study_idx + y_offset, marker='|',
                          s=100, c='black', zorder=5, linewidths=2)

    ax.set_xlabel('Weighted F1 Score')
    ax.set_ylabel('')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim(0, 1.05)

    fig.tight_layout()
    return fig


def main():
    args = parse_arguments()

    # Set publication style
    set_pub_style()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print("Loading data...")
    data = pd.read_csv(args.weighted_f1, sep='\t')

    # Filter by key and cutoff
    print(f"Filtering to key={args.key}, cutoff={args.cutoff}...")
    filtered_data = data[
        (data['key'] == args.key) &
        (data['cutoff'] == args.cutoff)
    ].copy()

    if len(filtered_data) == 0:
        print("Error: No data after filtering. Check key and cutoff values.")
        sys.exit(1)

    print(f"  {len(filtered_data)} observations across {filtered_data['study'].nunique()} studies")

    # Generate plots
    if args.plot_type in ['raincloud', 'both']:
        print("Creating raincloud plot...")
        fig = create_raincloud_plot(
            filtered_data,
            max_studies=args.max_studies
        )
        output_path = os.path.join(args.outdir, f'{args.output_prefix}_raincloud')
        save_figure(fig, output_path, formats=['pdf', 'png'], dpi=300)
        plt.close(fig)

    if args.plot_type in ['caterpillar', 'both']:
        print("Creating caterpillar plot...")
        fig = create_caterpillar_plot(
            filtered_data,
            max_studies=args.max_studies
        )
        output_path = os.path.join(args.outdir, f'{args.output_prefix}_caterpillar')
        save_figure(fig, output_path, formats=['pdf', 'png'], dpi=300)
        plt.close(fig)

    print("Done!")


if __name__ == "__main__":
    main()
