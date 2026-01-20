#!/usr/bin/env python3
"""
Shared plotting utilities for publication figures.
Provides consistent styling, color palettes, and helper functions for:
- Forest plots
- Raincloud plots
- Slope charts
- Publication-quality figure export
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Optional, Dict, Tuple, Union


# =============================================================================
# Color Palettes
# =============================================================================

METHOD_COLORS = {
    'scvi': '#1f77b4',    # Blue
    'seurat': '#ff7f0e'   # Orange
}

# Colorblind-safe backup palette
COLORBLIND_PALETTE = sns.color_palette("colorblind")

# Taxonomy level ordering (finest to coarsest)
KEY_ORDER = ['subclass', 'class', 'family', 'global']

# Method display names
METHOD_NAMES = {
    'scvi': 'scVI',
    'seurat': 'Seurat'
}


# =============================================================================
# Publication Style Settings
# =============================================================================

def set_pub_style():
    """
    Set matplotlib parameters for publication-quality figures.
    Uses Arial/Helvetica font, appropriate sizing, and minimal chartjunk.
    """
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 20,

        # Axes settings
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',

        # Tick settings
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,

        # Legend settings
        'legend.fontsize': 20,
        'legend.frameon': False,

        # Line settings
        'lines.linewidth': 1.5,

        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,

        # PDF/SVG export
        'pdf.fonttype': 42,  # TrueType fonts
        'ps.fonttype': 42,
    })


def reset_style():
    """Reset matplotlib to default style."""
    plt.rcdefaults()


# =============================================================================
# Figure Size Constants (in mm, converted to inches)
# =============================================================================

MM_TO_INCH = 1 / 25.4

# Full page width
FULL_WIDTH = 180 * MM_TO_INCH  # ~7.1 inches

# Single column width
SINGLE_COL = 85 * MM_TO_INCH  # ~3.3 inches

# Standard heights
STANDARD_HEIGHT = 150 * MM_TO_INCH  # ~5.9 inches
HALF_HEIGHT = 75 * MM_TO_INCH  # ~3.0 inches


def get_figure_size(width: str = 'full', aspect: float = 0.8) -> Tuple[float, float]:
    """
    Get figure dimensions for publication.

    Parameters
    ----------
    width : str
        'full' for full page width, 'single' for single column
    aspect : float
        Height/width ratio

    Returns
    -------
    tuple
        (width, height) in inches
    """
    w = FULL_WIDTH if width == 'full' else SINGLE_COL
    return (w, w * aspect)


# =============================================================================
# Forest Plot
# =============================================================================

def forest_plot(
    ax: plt.Axes,
    data: pd.DataFrame,
    estimate_col: str = 'response',
    lower_col: str = 'asymp.LCL',
    upper_col: str = 'asymp.UCL',
    group_col: str = 'method',
    color_col: Optional[str] = None,
    colors: Optional[Dict] = None,
    vertical: bool = True,
    show_reference_line: bool = True,
    reference_value: Optional[float] = None,
    marker_size: float = 80,
    line_width: float = 2,
    sort_by: Optional[str] = None,
    ascending: bool = True
) -> plt.Axes:
    """
    Create a forest plot showing point estimates with confidence intervals.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : pd.DataFrame
        DataFrame with estimates and CI bounds
    estimate_col : str
        Column name for point estimates
    lower_col : str
        Column name for lower CI bound
    upper_col : str
        Column name for upper CI bound
    group_col : str
        Column name for grouping variable (y-axis labels)
    color_col : str, optional
        Column to use for coloring points
    colors : dict, optional
        Color mapping dictionary
    vertical : bool
        If True, groups on y-axis (horizontal CIs). If False, groups on x-axis.
    show_reference_line : bool
        Whether to show a reference line
    reference_value : float, optional
        Value for reference line. If None, uses grand mean.
    marker_size : float
        Size of point markers
    line_width : float
        Width of CI lines
    sort_by : str, optional
        Column to sort by. If None, uses estimate_col.
    ascending : bool
        Sort order

    Returns
    -------
    matplotlib.axes.Axes
    """
    if colors is None:
        colors = METHOD_COLORS

    # Sort data
    sort_col = sort_by if sort_by else estimate_col
    data = data.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    n_groups = len(data)
    positions = np.arange(n_groups)

    # Get colors for each point
    if color_col and color_col in data.columns:
        point_colors = [colors.get(val, '#333333') for val in data[color_col]]
    else:
        point_colors = ['#333333'] * n_groups

    # Calculate reference line value
    if reference_value is None and show_reference_line:
        reference_value = data[estimate_col].mean()

    if vertical:
        # Horizontal CI bars (groups on y-axis)
        for i, (idx, row) in enumerate(data.iterrows()):
            ax.hlines(i, row[lower_col], row[upper_col],
                     colors=point_colors[i], linewidth=line_width, zorder=1)
            ax.scatter(row[estimate_col], i, s=marker_size,
                      c=[point_colors[i]], zorder=2, edgecolors='white', linewidth=0.5)

        ax.set_yticks(positions)
        ax.set_yticklabels(data[group_col])
        ax.set_xlabel('Estimated F1 Score')

        if show_reference_line:
            ax.axvline(reference_value, color='gray', linestyle='--',
                      linewidth=0.8, alpha=0.7, zorder=0)
    else:
        # Vertical CI bars (groups on x-axis)
        for i, (idx, row) in enumerate(data.iterrows()):
            ax.vlines(i, row[lower_col], row[upper_col],
                     colors=point_colors[i], linewidth=line_width, zorder=1)
            ax.scatter(i, row[estimate_col], s=marker_size,
                      c=[point_colors[i]], zorder=2, edgecolors='white', linewidth=0.5)

        ax.set_xticks(positions)
        ax.set_xticklabels(data[group_col], rotation=45, ha='right')
        ax.set_ylabel('Estimated F1 Score')

        if show_reference_line:
            ax.axhline(reference_value, color='gray', linestyle='--',
                      linewidth=0.8, alpha=0.7, zorder=0)

    return ax


# =============================================================================
# Raincloud Plot
# =============================================================================

def raincloud_plot(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    colors: Optional[Dict] = None,
    orient: str = 'h',
    width_viol: float = 0.5,
    width_box: float = 0.15,
    jitter: float = 0.04,
    alpha_viol: float = 0.3,
    alpha_points: float = 0.6,
    point_size: float = 3,
    order: Optional[List] = None,
    hue_order: Optional[List] = None
) -> plt.Axes:
    """
    Create a raincloud plot (half-violin + jittered points + boxplot).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : pd.DataFrame
        Data to plot
    x_col : str
        Column for x-axis (categorical)
    y_col : str
        Column for y-axis (continuous)
    hue_col : str, optional
        Column for color grouping
    colors : dict, optional
        Color mapping dictionary
    orient : str
        'h' for horizontal, 'v' for vertical
    width_viol : float
        Width of violin plot
    width_box : float
        Width of boxplot
    jitter : float
        Amount of jitter for points
    alpha_viol : float
        Transparency of violin
    alpha_points : float
        Transparency of points
    point_size : float
        Size of scatter points
    order : list, optional
        Order of categories
    hue_order : list, optional
        Order of hue levels

    Returns
    -------
    matplotlib.axes.Axes
    """
    if colors is None:
        colors = METHOD_COLORS

    # Get categories
    if order is None:
        order = sorted(data[x_col].unique())

    if hue_col:
        if hue_order is None:
            hue_order = sorted(data[hue_col].unique())
        n_hue = len(hue_order)
    else:
        n_hue = 1

    # Color palette
    if hue_col:
        palette = [colors.get(h, COLORBLIND_PALETTE[i]) for i, h in enumerate(hue_order)]
    else:
        palette = [colors.get(order[0], '#1f77b4')]

    for cat_idx, cat in enumerate(order):
        cat_data = data[data[x_col] == cat]

        if hue_col:
            for hue_idx, hue in enumerate(hue_order):
                hue_data = cat_data[cat_data[hue_col] == hue][y_col].dropna()
                if len(hue_data) == 0:
                    continue

                color = palette[hue_idx]
                offset = (hue_idx - (n_hue - 1) / 2) * 0.25
                pos = cat_idx + offset

                _draw_raincloud_component(
                    ax, hue_data, pos, color, orient,
                    width_viol, width_box, jitter, alpha_viol, alpha_points, point_size
                )
        else:
            values = cat_data[y_col].dropna()
            if len(values) == 0:
                continue
            color = palette[0]
            _draw_raincloud_component(
                ax, values, cat_idx, color, orient,
                width_viol, width_box, jitter, alpha_viol, alpha_points, point_size
            )

    # Set axis labels and ticks
    if orient == 'h':
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlabel(y_col.replace('_', ' ').title())
    else:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right')
        ax.set_ylabel(y_col.replace('_', ' ').title())

    return ax


def _draw_raincloud_component(
    ax, values, position, color, orient,
    width_viol, width_box, jitter, alpha_viol, alpha_points, point_size
):
    """Helper function to draw one raincloud component."""
    from scipy import stats

    # Calculate kernel density
    try:
        kernel = stats.gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 100)
        density = kernel(x_range)
        density = density / density.max() * width_viol  # Normalize
    except Exception:
        return

    if orient == 'h':
        # Half violin (top)
        ax.fill_betweenx(x_range, position, position + density,
                        alpha=alpha_viol, color=color)

        # Jittered points (bottom)
        jittered = position - np.random.uniform(0, width_viol * 0.5, len(values))
        ax.scatter(values, jittered, s=point_size, alpha=alpha_points,
                  color=color, edgecolors='none')

        # Boxplot
        bp = ax.boxplot(values, positions=[position - width_viol * 0.3],
                       widths=width_box, vert=False, patch_artist=True,
                       showfliers=False, showcaps=False)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.8)
        bp['medians'][0].set_color('white')
    else:
        # Half violin (right)
        ax.fill_between(x_range, position, position + density,
                       alpha=alpha_viol, color=color)

        # Jittered points (left)
        jittered = position - np.random.uniform(0, width_viol * 0.5, len(values))
        ax.scatter(jittered, values, s=point_size, alpha=alpha_points,
                  color=color, edgecolors='none')

        # Boxplot
        bp = ax.boxplot(values, positions=[position - width_viol * 0.3],
                       widths=width_box, vert=True, patch_artist=True,
                       showfliers=False, showcaps=False)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.8)
        bp['medians'][0].set_color('white')


# =============================================================================
# Slope Chart / Connected Dot Plot
# =============================================================================

def slope_chart(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    colors: Optional[Dict] = None,
    x_order: Optional[List] = None,
    marker_size: float = 80,
    line_width: float = 2,
    line_alpha: float = 0.8,
    show_legend: bool = True
) -> plt.Axes:
    """
    Create a slope chart / connected dot plot.
    Lines connect the same group across x-axis categories.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : pd.DataFrame
        Data with columns for x, y, and group
    x_col : str
        Column for x-axis categories
    y_col : str
        Column for y-axis values
    group_col : str
        Column for grouping (lines connect same group)
    colors : dict, optional
        Color mapping for groups
    x_order : list, optional
        Order of x-axis categories
    marker_size : float
        Size of markers
    line_width : float
        Width of connecting lines
    line_alpha : float
        Transparency of lines
    show_legend : bool
        Whether to show legend

    Returns
    -------
    matplotlib.axes.Axes
    """
    if colors is None:
        colors = METHOD_COLORS

    if x_order is None:
        x_order = sorted(data[x_col].unique())

    x_positions = {cat: i for i, cat in enumerate(x_order)}

    # Plot each group
    groups = data[group_col].unique()
    for group in groups:
        group_data = data[data[group_col] == group].copy()
        group_data['x_pos'] = group_data[x_col].map(x_positions)
        group_data = group_data.sort_values('x_pos')

        color = colors.get(group, '#333333')
        label = METHOD_NAMES.get(group, group)

        # Plot line
        ax.plot(group_data['x_pos'], group_data[y_col],
               color=color, linewidth=line_width, alpha=line_alpha,
               marker='o', markersize=np.sqrt(marker_size),
               markerfacecolor=color, markeredgecolor='white',
               markeredgewidth=0.5, label=label, zorder=2)

    # Set axis labels
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels([x.title() for x in x_order])
    ax.set_ylabel(y_col.replace('_', ' ').title())

    if show_legend:
        ax.legend(loc='best', frameon=False)

    return ax


# =============================================================================
# Cutoff Sensitivity Plot
# =============================================================================

def cutoff_sensitivity_plot(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str = 'cutoff',
    y_col: str = 'fit',
    lower_col: str = 'lower',
    upper_col: str = 'upper',
    group_col: str = 'method',
    colors: Optional[Dict] = None,
    show_ci: bool = True,
    ci_alpha: float = 0.2,
    line_width: float = 2,
    marker: str = 'o',
    marker_size: float = 6
) -> plt.Axes:
    """
    Create a cutoff sensitivity plot with CI ribbons.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : pd.DataFrame
        Data with cutoff, fit, and CI columns
    x_col : str
        Column for x-axis (cutoff values)
    y_col : str
        Column for y-axis (fitted values)
    lower_col : str
        Column for lower CI bound
    upper_col : str
        Column for upper CI bound
    group_col : str
        Column for method grouping
    colors : dict, optional
        Color mapping
    show_ci : bool
        Whether to show CI ribbons
    ci_alpha : float
        Transparency of CI ribbons
    line_width : float
        Width of lines
    marker : str
        Marker style
    marker_size : float
        Marker size

    Returns
    -------
    matplotlib.axes.Axes
    """
    if colors is None:
        colors = METHOD_COLORS

    for method, group in data.groupby(group_col):
        group = group.sort_values(x_col)
        color = colors.get(method, '#333333')
        label = METHOD_NAMES.get(method, method)

        # Plot line with markers
        ax.plot(group[x_col], group[y_col],
               color=color, linewidth=line_width,
               marker=marker, markersize=marker_size,
               label=label, zorder=2)

        # Plot CI ribbon
        if show_ci and lower_col in group.columns and upper_col in group.columns:
            ax.fill_between(group[x_col], group[lower_col], group[upper_col],
                          color=color, alpha=ci_alpha, zorder=1)

    ax.set_xlabel('Confidence Cutoff')
    ax.set_ylabel('Estimated F1 Score')
    ax.legend(loc='best', frameon=False)

    return ax


# =============================================================================
# Swarm/Strip Plot for Cross-Study
# =============================================================================

def study_swarm_plot(
    ax: plt.Axes,
    data: pd.DataFrame,
    study_col: str = 'study',
    value_col: str = 'weighted_f1',
    hue_col: str = 'method',
    colors: Optional[Dict] = None,
    order: Optional[List] = None,
    dodge: bool = True,
    size: float = 3,
    alpha: float = 0.7
) -> plt.Axes:
    """
    Create a swarm/strip plot for cross-study comparison.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    data : pd.DataFrame
        Data with study, value, and hue columns
    study_col : str
        Column for study names
    value_col : str
        Column for values
    hue_col : str
        Column for method/color grouping
    colors : dict, optional
        Color mapping
    order : list, optional
        Order of studies (sorted by median if None)
    dodge : bool
        Whether to dodge points by hue
    size : float
        Point size
    alpha : float
        Point transparency

    Returns
    -------
    matplotlib.axes.Axes
    """
    if colors is None:
        colors = METHOD_COLORS

    # Order by median value if not specified
    if order is None:
        medians = data.groupby(study_col)[value_col].median().sort_values(ascending=False)
        order = medians.index.tolist()

    # Create palette
    hue_order = list(colors.keys())
    palette = [colors.get(h, '#333333') for h in hue_order]

    # Use stripplot (swarmplot can be slow with many points)
    sns.stripplot(
        data=data,
        y=study_col,
        x=value_col,
        hue=hue_col,
        order=order,
        hue_order=hue_order,
        palette=palette,
        dodge=dodge,
        size=size,
        alpha=alpha,
        ax=ax,
        jitter=True
    )

    ax.set_xlabel('Weighted F1 Score')
    ax.set_ylabel('Study')
    ax.legend(loc='lower right', frameon=False)

    return ax


# =============================================================================
# Helper Functions
# =============================================================================

def add_panel_label(ax: plt.Axes, label: str, x: float = -0.1, y: float = 1.05,
                   fontsize: int = 20, fontweight: str = 'bold'):
    """Add a panel label (A, B, C, etc.) to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
           fontsize=fontsize, fontweight=fontweight, va='bottom', ha='right')


def create_method_legend(colors: Optional[Dict] = None,
                         names: Optional[Dict] = None) -> List[mpatches.Patch]:
    """Create legend patches for methods."""
    if colors is None:
        colors = METHOD_COLORS
    if names is None:
        names = METHOD_NAMES

    patches = []
    for method, color in colors.items():
        label = names.get(method, method)
        patches.append(mpatches.Patch(color=color, label=label))

    return patches


def truncate_study_name(name: str, max_len: int = 30) -> str:
    """Truncate long study names for display."""
    if len(name) <= max_len:
        return name
    return name[:max_len-3] + '...'


def save_figure(fig: plt.Figure, filename: str, formats: List[str] = ['pdf', 'png'],
               dpi: int = 300):
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    formats : list
        List of formats to save ('pdf', 'png', 'svg')
    dpi : int
        DPI for raster formats
    """
    for fmt in formats:
        fig.savefig(f"{filename}.{fmt}", format=fmt, dpi=dpi, bbox_inches='tight')


# =============================================================================
# Data Loading Helpers
# =============================================================================

def load_emmeans_summary(filepath: str) -> pd.DataFrame:
    """Load emmeans summary TSV file."""
    return pd.read_csv(filepath, sep='\t')


def load_cutoff_effects(filepath: str) -> pd.DataFrame:
    """Load method_cutoff_effects TSV file."""
    return pd.read_csv(filepath, sep='\t')


def load_weighted_f1_results(filepath: str) -> pd.DataFrame:
    """Load weighted_f1_results TSV file."""
    return pd.read_csv(filepath, sep='\t')


def combine_taxonomy_levels(
    base_path: str,
    levels: List[str] = None,
    file_pattern: str = 'method_emmeans_summary.tsv'
) -> pd.DataFrame:
    """
    Combine emmeans data across taxonomy levels.

    Parameters
    ----------
    base_path : str
        Base directory containing level subdirectories
    levels : list
        List of taxonomy levels to combine
    file_pattern : str
        Filename to load from each level directory

    Returns
    -------
    pd.DataFrame
        Combined data with 'key' column for taxonomy level
    """
    import os

    if levels is None:
        levels = KEY_ORDER

    dfs = []
    for level in levels:
        filepath = os.path.join(base_path, level, 'files', file_pattern)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, sep='\t')
            df['key'] = level
            dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()
