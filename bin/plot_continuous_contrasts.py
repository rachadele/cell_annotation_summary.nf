#!/usr/bin/env python3
"""
Parameter Sensitivity Plot (Continuous Contrasts)

Creates clean line plots showing how continuous parameters (cutoff, support)
affect model performance, with CI ribbons.

Key insight: scvi degrades faster at high cutoffs than seurat

Usage:
    python plot_continuous_contrasts.py \
        --contrast path/to/method_cutoff_effects.tsv \
        --key subclass \
        --outdir figures
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure,
    METHOD_COLORS, METHOD_NAMES,
    cutoff_sensitivity_plot,
    SINGLE_COL, HALF_HEIGHT
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot continuous contrast effects (cutoff/support sensitivity)"
    )
    parser.add_argument(
        '--contrast', type=str, required=True,
        help="Path to effects TSV file (e.g., method_cutoff_effects.tsv)"
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
        '--output_prefix', type=str, default=None,
        help="Output filename prefix (auto-generated if not provided)"
    )

    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]


def main():
    args = parse_arguments()

    # Set publication style
    set_pub_style()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print(f"Loading data from {args.contrast}...")
    contrast_data = pd.read_csv(args.contrast, sep='\t')

    # Determine x column (cutoff or support)
    if 'cutoff' in contrast_data.columns:
        x_col = 'cutoff'
        x_label = 'Confidence Cutoff'
    elif 'support' in contrast_data.columns:
        x_col = 'support'
        x_label = 'Support (Cell Count)'
    else:
        print("Error: Could not find 'cutoff' or 'support' column")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, HALF_HEIGHT))

    # Use the cutoff sensitivity plot function
    cutoff_sensitivity_plot(
        ax=ax,
        data=contrast_data,
        x_col=x_col,
        y_col='fit',
        lower_col='lower',
        upper_col='upper',
        group_col='method',
        colors=METHOD_COLORS,
        show_ci=True,
        ci_alpha=0.2,
        line_width=2,
        marker='o',
        marker_size=6
    )

    # Customize labels
    ax.set_xlabel(x_label)
    ax.set_ylabel('Estimated F1 Score')
    ax.set_ylim(0, 1)

    # Add title if desired
    # ax.set_title(f'{x_col.title()} Sensitivity ({args.key})', fontweight='bold')

    # Position legend to avoid overlap with data
    ax.legend(loc='lower left', frameon=False)

    # Tight layout
    fig.tight_layout()

    # Generate output filename
    if args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = f'{x_col}_sensitivity'

    output_path = os.path.join(args.outdir, output_prefix)
    print(f"Saving figure to {output_path}...")
    save_figure(fig, output_path, formats=['pdf', 'png'], dpi=300)

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
