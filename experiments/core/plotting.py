import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


def set_paper_style() -> None:
    """Apply a paper-oriented matplotlib style shared by all experiments.

    Parameters:
        None: This function does not accept runtime parameters.
    """

    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 14,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "lines.linewidth": 2.5,
        "axes.grid": False,
        "mathtext.fontset": "cm",
        "font.family": "DejaVu Serif",
    })


def save_figure(fig, output_dir: Path, basename: str, dpi: int = 600) -> Dict[str, str]:
    """Save a figure to both PDF and PNG formats.

    Parameters:
        fig: Matplotlib figure instance.
        output_dir: Target figure directory.
        basename: Filename without extension.
        dpi: DPI used for the PNG export.
    """

    output_dir.mkdir(parents = True, exist_ok = True)
    pdf_path = output_dir / f"{basename}.pdf"
    png_path = output_dir / f"{basename}.png"
    fig.tight_layout()
    fig.savefig(pdf_path, dpi = dpi, bbox_inches = "tight")
    fig.savefig(png_path, dpi = dpi, bbox_inches = "tight")
    plt.close(fig)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def plot_error_band(
    x_values: List[float],
    mean_values: List[float],
    std_values: List[float],
    color: str,
    label: str,
    x_label: str,
    y_label: str,
    basename: str,
    output_dir: Path,
    dpi: int = 600,
    figsize = (7, 6),
    log_y: bool = False
) -> Dict[str, str]:
    """Create a single-curve line plot with a shaded standard-deviation band.

    Parameters:
        x_values: X-axis values.
        mean_values: Mean curve values.
        std_values: Standard deviation values.
        color: Line color.
        label: Legend label.
        x_label: X-axis label.
        y_label: Y-axis label.
        basename: Output filename without extension.
        output_dir: Target figure directory.
        dpi: DPI used for the PNG export.
        figsize: Figure size tuple.
        log_y: Whether to use logarithmic y scaling.
    """

    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    x_array = np.array(x_values)
    mean_array = np.array(mean_values)
    std_array = np.array(std_values)
    ax.plot(x_array, mean_array, color = color, label = label)
    ax.fill_between(x_array, mean_array - std_array, mean_array + std_array, color = color, alpha = 0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis = "both", which = "major", labelsize = 14, width = 2, length = 6)
    ax.tick_params(axis = "both", which = "minor", labelsize = 12, width = 1, length = 3)
    if label:
        ax.legend()
    if log_y:
        ax.set_yscale("log")
    return save_figure(fig, output_dir, basename, dpi)


def plot_multi_curve(
    x_values: List[float],
    curves: List[Dict[str, object]],
    x_label: str,
    y_label: str,
    basename: str,
    output_dir: Path,
    dpi: int = 600,
    figsize = (9, 9),
    log_y: bool = False
) -> Dict[str, str]:
    """Create a multi-curve comparison plot with optional error bands.

    Parameters:
        x_values: Shared X-axis values.
        curves: Curve descriptors with `mean`, `std`, `color`, and `label`.
        x_label: X-axis label.
        y_label: Y-axis label.
        basename: Output filename without extension.
        output_dir: Target figure directory.
        dpi: DPI used for the PNG export.
        figsize: Figure size tuple.
        log_y: Whether to use logarithmic y scaling.
    """

    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    x_array = np.array(x_values)
    for curve in curves:
        mean_array = np.array(curve["mean"])
        std_array = np.array(curve["std"])
        ax.plot(x_array, mean_array, color = curve["color"], label = curve["label"])
        ax.fill_between(x_array, mean_array - std_array, mean_array + std_array, color = curve["color"], alpha = 0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis = "both", which = "major", labelsize = 14, width = 2, length = 6)
    ax.tick_params(axis = "both", which = "minor", labelsize = 12, width = 1, length = 3)
    ax.legend()
    if log_y:
        ax.set_yscale("log")
    return save_figure(fig, output_dir, basename, dpi)


def plot_matrix_heatmap(
    matrix,
    title: str,
    basename: str,
    output_dir: Path,
    dpi: int = 900,
    cmap: str = "gray_r"
) -> Dict[str, str]:
    """Create a matrix heatmap with overlaid numeric annotations.

    Parameters:
        matrix: Two-dimensional tensor or array to visualize.
        title: Figure title.
        basename: Output filename without extension.
        output_dir: Target figure directory.
        dpi: DPI used for the PNG export.
        cmap: Matplotlib colormap name.
    """

    set_paper_style()
    matrix_np = matrix.detach().cpu().numpy() if hasattr(matrix, "detach") else np.array(matrix)
    fig, ax = plt.subplots(1, 1, figsize = (6, 6))
    im = ax.imshow(matrix_np, cmap = cmap)
    for row_index in range(matrix_np.shape[0]):
        for col_index in range(matrix_np.shape[1]):
            ax.text(col_index, row_index, f"{matrix_np[row_index, col_index]:.2f}", ha = "center", va = "center", color = "#A00000")
    fig.colorbar(im)
    ax.set_title(title)
    return save_figure(fig, output_dir, basename, dpi)


def plot_distance_panels(
    x_values: List[int],
    panel_curves: List[Dict[str, object]],
    basename: str,
    output_dir: Path,
    dpi: int = 600,
    figsize = (14, 18)
) -> Dict[str, str]:
    """Create a grid of distance-to-identity panels for multiple matrices.

    Parameters:
        x_values: Shared iteration axis.
        panel_curves: Panel descriptors containing labels, mean/std values, and styling.
        basename: Output filename without extension.
        output_dir: Target figure directory.
        dpi: DPI used for the PNG export.
        figsize: Figure size tuple.
    """

    set_paper_style()
    fig, axes = plt.subplots(3, 2, figsize = figsize)
    for panel_index, panel in enumerate(panel_curves):
        row_index = panel_index // 2
        col_index = panel_index % 2
        ax = axes[row_index, col_index]
        x_array = np.array(x_values)
        mean_array = np.array(panel["mean"])
        std_array = np.array(panel["std"])
        ax.plot(x_array, mean_array, color = panel["color"], label = panel["label"])
        ax.fill_between(x_array, mean_array - std_array, mean_array + std_array, color = panel["color"], alpha = 0.2)
        ax.set_ylim([0.0, 1.0])
        ax.legend()
        ax.tick_params(axis = "both", which = "major", labelsize = 12, width = 2, length = 5)
    return save_figure(fig, output_dir, basename, dpi)


def write_figure_index(path: Path, entries: List[Dict[str, object]]) -> None:
    """Write a JSON index describing generated figures.

    Parameters:
        path: Target JSON file path.
        entries: List of figure metadata records.
    """

    with path.open("w", encoding = "utf-8") as handle:
        json.dump(entries, handle, indent = 2, ensure_ascii = False)
