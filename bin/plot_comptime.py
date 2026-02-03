#!/usr/bin/env python3
"""
Computational Performance Figure (Supplementary S1)

Creates horizontal bar charts showing:
- Mean runtime by process
- Memory usage as secondary axis or separate panel

Cleaner than previous violin plot approach.

Usage:
    python plot_comptime.py \
        --all_runs path/to/runs_directory \
        --outdir figures
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Import shared utilities
from plot_utils import (
    set_pub_style, save_figure,
    METHOD_COLORS,
    SINGLE_COL, HALF_HEIGHT, FULL_WIDTH, STANDARD_HEIGHT
)


# Process display names and colors
# Keys are the final component of the Nextflow process name (after last ':')
PROCESS_NAMES = {
    'RF_PREDICT': 'scVI Predict',
    'PREDICT_SEURAT': 'Seurat Predict',
    'MAP_QUERY': 'scVI Query',
    'QUERY_PROCESS_SEURAT': 'Seurat Query',
    'REF_PROCESS_SEURAT': 'Seurat Ref'
}

PROCESS_COLORS = {
    'RF_PREDICT': '#1f77b4',
    'PREDICT_SEURAT': '#ff7f0e',
    'MAP_QUERY': '#2ca02c',
    'QUERY_PROCESS_SEURAT': '#d62728',
    'REF_PROCESS_SEURAT': '#9467bd'
}

# Process ordering (prediction methods first, then preprocessing)
PROCESS_ORDER = [
    'RF_PREDICT', 'PREDICT_SEURAT',
    'MAP_QUERY', 'QUERY_PROCESS_SEURAT', 'REF_PROCESS_SEURAT'
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate computational performance figures"
    )
    parser.add_argument(
        '--all_runs', type=str, required=True,
        help="Path to directory containing trace results"
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help="Output directory for figures"
    )
    parser.add_argument(
        '--output_prefix', type=str, default='comptime',
        help="Prefix for output filenames"
    )

    if __name__ == "__main__":
        return parser.parse_args()
    return parser.parse_known_args()[0]


def convert_time(time_str) -> float:
    """Convert time string to hours."""
    if pd.isna(time_str) or time_str == "-" or not isinstance(time_str, str):
        return np.nan

    time_str = time_str.lower().replace(" ", "")

    # Match hours, minutes, seconds, milliseconds
    match = re.match(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+\.?\d*)s)?(?:(\d+)ms)?", time_str)

    if not match:
        return np.nan

    hours = int(match.group(1)) if match.group(1) else 0
    minutes = int(match.group(2)) if match.group(2) else 0
    seconds = float(match.group(3)) if match.group(3) else 0
    milliseconds = float(match.group(4)) / 1000 if match.group(4) else 0

    total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds
    return total_seconds / 3600  # Return hours


def convert_percent(value) -> float:
    """Convert percentage string to float."""
    if isinstance(value, str) and value.endswith("%"):
        return float(value.strip("%"))
    return np.nan


def convert_memory(value_str) -> float:
    """Convert memory string to GB."""
    if pd.isna(value_str) or value_str == "-":
        return np.nan

    if isinstance(value_str, str):
        parts = value_str.strip().split()
        if len(parts) >= 1:
            try:
                val = float(parts[0])
                # Assume MB if no unit, convert to GB
                return val / 1024
            except ValueError:
                return np.nan
    return np.nan


def load_trace_data(all_runs_dir: str) -> pd.DataFrame:
    """Load and combine all trace files found recursively."""
    reports = pd.DataFrame()

    for root, dirs, files in os.walk(all_runs_dir):
        if "trace.txt" not in files:
            continue

        trace_path = os.path.join(root, "trace.txt")
        params_path = os.path.join(root, "params.yaml")

        try:
            trace = pd.read_csv(trace_path, sep="\t")

            # Load params if available
            if os.path.exists(params_path):
                with open(params_path, "r") as f:
                    params = yaml.safe_load(f)

                # Remove non-scalar params
                keys_to_drop = ["ref_collections", "ref_keys", "outdir",
                               "batch_keys", "relabel_r", "relabel_q",
                               "tree_file", "queries_adata"]
                for key in keys_to_drop:
                    params.pop(key, None)

                for key, value in params.items():
                    trace[key] = value

            reports = pd.concat([reports, trace], ignore_index=True)

        except Exception as e:
            print(f"Warning: Could not load {trace_path}: {e}")

    return reports


def create_line_plot(
    ax: plt.Axes,
    stats: pd.DataFrame,
    mean_col: str,
    std_col: str,
    ylabel: str,
    process_order: list = None
) -> plt.Axes:
    """
    Create a line/point plot with error bars for process steps on the x-axis.
    """
    if process_order is None:
        process_order = stats.index.tolist()

    available = [p for p in process_order if p in stats.index]
    plot_data = stats.loc[available]

    positions = np.arange(len(available))
    colors = [PROCESS_COLORS.get(p, '#666666') for p in available]
    labels = [PROCESS_NAMES.get(p, p) for p in available]

    # Plot connecting line
    ax.plot(positions, plot_data[mean_col], color='grey', linewidth=1, zorder=1)

    # Plot points with error bars per process
    for i, (proc, pos) in enumerate(zip(available, positions)):
        ax.errorbar(
            pos, plot_data.loc[proc, mean_col],
            yerr=plot_data.loc[proc, std_col],
            fmt='o', color=colors[i], markersize=8,
            capsize=4, capthick=1.5, elinewidth=1.5,
            markeredgecolor='white', markeredgewidth=0.5,
            zorder=2
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, None)

    return ax


def main():
    args = parse_arguments()

    # Set publication style
    set_pub_style()

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    print("Loading trace data...")
    reports = load_trace_data(args.all_runs)

    if len(reports) == 0:
        print("Error: No trace data found")
        return

    # Process data
    print("Processing data...")

    # Convert columns
    if '%cpu' in reports.columns:
        reports['%cpu'] = reports['%cpu'].apply(convert_percent)
    if 'duration' in reports.columns:
        reports['duration_hours'] = reports['duration'].apply(convert_time)
    if 'realtime' in reports.columns:
        reports['realtime_hours'] = reports['realtime'].apply(convert_time)
    if 'peak_vmem' in reports.columns:
        reports['memory_gb'] = reports['peak_vmem'].apply(convert_memory)

    # Extract process name: "SEURAT_PIPELINE:PREDICT_SEURAT (5)" -> "PREDICT_SEURAT"
    reports['process'] = reports['name'].apply(
        lambda x: str(x).split('(')[0].strip().split(':')[-1] if pd.notna(x) else None
    )

    # Filter to relevant processes
    processes = [p for p in PROCESS_ORDER if p in reports['process'].values]
    trace_subset = reports[reports['process'].isin(processes)].copy()

    if len(trace_subset) == 0:
        print("Error: No relevant processes found in trace data")
        return

    # Calculate statistics
    stats = trace_subset.groupby('process').agg({
        'duration_hours': ['mean', 'std', 'count'],
        'realtime_hours': ['mean', 'std'],
        'memory_gb': ['mean', 'std']
    })

    # Flatten column names
    stats.columns = ['mean_duration', 'std_duration', 'count',
                     'mean_realtime', 'std_realtime',
                     'mean_memory', 'std_memory']

    # Fill NaN std with 0
    stats = stats.fillna(0)

    print(f"Found {len(stats)} processes with data")
    print(stats)

    # Save summary table
    stats_out = stats.copy()
    stats_out.index = stats_out.index.map(lambda x: PROCESS_NAMES.get(x, x))
    stats_out.to_csv(os.path.join(args.outdir, f"{args.output_prefix}_summary.tsv"), sep='\t')

    # Create figure with two stacked panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FULL_WIDTH, STANDARD_HEIGHT))

    # Panel A: Runtime
    create_line_plot(ax1, stats, 'mean_duration', 'std_duration',
                     'Duration (hours)', PROCESS_ORDER)
    ax1.set_title('A. Runtime', fontweight='bold', loc='left')

    # Panel B: Memory
    create_line_plot(ax2, stats, 'mean_memory', 'std_memory',
                     'Peak Memory (GB)', PROCESS_ORDER)
    ax2.set_title('B. Peak Memory', fontweight='bold', loc='left')

    fig.tight_layout(h_pad=3)

    # Save
    output_path = os.path.join(args.outdir, args.output_prefix)
    print(f"Saving figure to {output_path}...")
    save_figure(fig, output_path, formats=['pdf', 'png'], dpi=300)

    plt.close(fig)
    print("Done!")


if __name__ == "__main__":
    main()
