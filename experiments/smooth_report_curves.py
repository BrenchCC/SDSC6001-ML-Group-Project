import os
import sys
import math
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt

sys.path.append(os.getcwd())

from experiments.core.plotting import save_figure, set_paper_style

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for smoothed report curve generation.

    Parameters:
        None: Arguments are read from `sys.argv`.
    """

    parser = argparse.ArgumentParser(description = "Generate smoothed report curves from saved experiment metrics.")
    parser.add_argument(
        "--result-dir",
        default = "results_cpu",
        help = "Root directory that contains experiment outputs."
    )
    parser.add_argument(
        "--preset",
        default = "cpu",
        help = "Preset directory name used by the experiment outputs."
    )
    parser.add_argument(
        "--dpi",
        type = int,
        default = 240,
        help = "DPI used when exporting smoothed PNG figures."
    )
    return parser.parse_args()


def smooth(scalars: List[float]) -> List[float]:
    """Smooth a scalar sequence with TensorBoard-style EMA.

    Parameters:
        scalars: Scalar sequence to smooth.
    """

    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def main():
    """Generate smoothed plots for all report-facing curve artifacts.

    Parameters:
        None: Runtime state is read from parsed CLI arguments.
    """

    args = parse_args()
    result_root = Path(args.result_dir)
    artifacts = []
    artifacts.extend(render_theorem3_curves(result_root = result_root, preset = args.preset, dpi = args.dpi))
    artifacts.extend(render_theorem4_curves(result_root = result_root, preset = args.preset, dpi = args.dpi))
    artifacts.extend(render_extension_curves(result_root = result_root, preset = args.preset, dpi = args.dpi))
    index_path = result_root / "smoothed_curve_artifacts.json"
    result_root.mkdir(parents = True, exist_ok = True)
    with index_path.open("w", encoding = "utf-8") as handle:
        json.dump(artifacts, handle, indent = 2, ensure_ascii = False)
    logger.info("Smoothed curve artifact index written to %s", index_path)
    print(index_path)


def render_theorem3_curves(result_root: Path, preset: str, dpi: int) -> List[Dict[str, Any]]:
    """Render smoothed Theorem 3 curves.

    Parameters:
        result_root: Root directory that contains experiment outputs.
        preset: Preset directory name.
        dpi: DPI used when exporting smoothed PNG figures.
    """

    experiment_root = result_root / "rotation_adam_p0" / preset
    metrics = load_metrics(experiment_root)
    if not metrics:
        return []
    output_dir = experiment_root / "figures_smoothed"
    artifacts = []
    artifacts.append({
        "experiment": "rotation_adam_p0",
        "basename": "rotation_demonstration_adam_pnull_loss_plot_smooth",
        "paths": plot_single_smoothed_curve(
            x_values = metrics["iterations"],
            mean_values = metrics["loss_summary_log"]["mean"],
            std_values = metrics["loss_summary_log"]["std"],
            color = "blue",
            label = "Adam with $P = 0$",
            x_label = "Iteration",
            y_label = "Smoothed log(Loss)",
            basename = "rotation_demonstration_adam_pnull_loss_plot_smooth",
            output_dir = output_dir,
            dpi = dpi
        ),
    })
    for component_name in ["A0", "A1", "A2"]:
        artifacts.append({
            "experiment": "rotation_adam_p0",
            "basename": f"rotation_demonstration_dist_to_id_adam_pnull_{component_name}_smooth",
            "paths": plot_multi_smoothed_curve(
                x_values = metrics["iterations"],
                curves = [
                    {
                        "mean": metrics["raw_distance_summary"][component_name]["mean"],
                        "std": metrics["raw_distance_summary"][component_name]["std"],
                        "color": "red",
                        "label": f"${component_name}$",
                    },
                    {
                        "mean": metrics["rotated_distance_summary"][component_name]["mean"],
                        "std": metrics["rotated_distance_summary"][component_name]["std"],
                        "color": "blue",
                        "label": f"$\\Sigma^{{1/2}} {component_name} \\Sigma^{{1/2}}$",
                    },
                ],
                x_label = "Iteration",
                y_label = "Smoothed Distance to Id",
                basename = f"rotation_demonstration_dist_to_id_adam_pnull_{component_name}_smooth",
                output_dir = output_dir,
                dpi = dpi
            ),
        })
    write_local_index(output_dir / "smoothed_figure_index.json", artifacts)
    return artifacts


def render_theorem4_curves(result_root: Path, preset: str, dpi: int) -> List[Dict[str, Any]]:
    """Render smoothed Theorem 4 curves.

    Parameters:
        result_root: Root directory that contains experiment outputs.
        preset: Preset directory name.
        dpi: DPI used when exporting smoothed PNG figures.
    """

    experiment_root = result_root / "rotation_adam" / preset
    metrics = load_metrics(experiment_root)
    if not metrics:
        return []
    output_dir = experiment_root / "figures_smoothed"
    artifacts = []
    artifacts.append({
        "experiment": "rotation_adam",
        "basename": "rotation_demonstration_adam_loss_plot_smooth",
        "paths": plot_single_smoothed_curve(
            x_values = metrics["iterations"],
            mean_values = metrics["loss_summary_log"]["mean"],
            std_values = metrics["loss_summary_log"]["std"],
            color = "red",
            label = "Adam",
            x_label = "Iteration",
            y_label = "Smoothed log(ICL Test Loss)",
            basename = "rotation_demonstration_adam_loss_plot_smooth",
            output_dir = output_dir,
            dpi = dpi
        ),
    })
    panel_curves = [
        ("B0", "red", "$B_0$"),
        ("B1", "orange", "$B_1$"),
        ("A0", "green", "$\\Sigma^{1/2} A_0 \\Sigma^{1/2}$"),
        ("A1", "blue", "$\\Sigma^{1/2} A_1 \\Sigma^{1/2}$"),
        ("A2", "black", "$\\Sigma^{1/2} A_2 \\Sigma^{1/2}$"),
    ]
    artifacts.append({
        "experiment": "rotation_adam",
        "basename": "rotation_demonstration_dist_to_id_adam_smooth",
        "paths": plot_panel_smoothed_curves(
            x_values = metrics["iterations"],
            panel_curves = [
                {
                    "mean": metrics["distance_summary"][component_name]["mean"],
                    "std": metrics["distance_summary"][component_name]["std"],
                    "color": color,
                    "label": label,
                }
                for component_name, color, label in panel_curves
            ],
            basename = "rotation_demonstration_dist_to_id_adam_smooth",
            output_dir = output_dir,
            dpi = dpi
        ),
    })
    for component_name, color, label in panel_curves:
        artifacts.append({
            "experiment": "rotation_adam",
            "basename": f"rotation_demonstration_dist_to_id_adam_{component_name}_smooth",
            "paths": plot_single_smoothed_curve(
                x_values = metrics["iterations"],
                mean_values = metrics["distance_summary"][component_name]["mean"],
                std_values = metrics["distance_summary"][component_name]["std"],
                color = color,
                label = label,
                x_label = "Iteration",
                y_label = "Smoothed Distance to Id",
                basename = f"rotation_demonstration_dist_to_id_adam_{component_name}_smooth",
                output_dir = output_dir,
                dpi = dpi
            ),
        })
    write_local_index(output_dir / "smoothed_figure_index.json", artifacts)
    return artifacts


def render_extension_curves(result_root: Path, preset: str, dpi: int) -> List[Dict[str, Any]]:
    """Render smoothed context-length extension curves.

    Parameters:
        result_root: Root directory that contains experiment outputs.
        preset: Preset directory name.
        dpi: DPI used when exporting smoothed PNG figures.
    """

    experiment_root = result_root / "variable_n" / preset
    metrics = load_metrics(experiment_root)
    if not metrics:
        return []
    output_dir = experiment_root / "figures_smoothed"
    artifacts = [{
        "experiment": "variable_n",
        "basename": "3-step-variable-N-plot_smooth",
        "paths": plot_multi_smoothed_curve(
            x_values = metrics["x_values"],
            curves = [
                {
                    "mean": metrics["gd_summary"]["mean"],
                    "std": metrics["gd_summary"]["std"],
                    "color": "blue",
                    "label": "3-Step GD",
                },
                {
                    "mean": metrics["pgd_summary"]["mean"],
                    "std": metrics["pgd_summary"]["std"],
                    "color": "green",
                    "label": "3-Step Preconditioned GD",
                },
                {
                    "mean": metrics["transformer_summary"]["mean"],
                    "std": metrics["transformer_summary"]["std"],
                    "color": "red",
                    "label": "3-Layer Linear Transformer",
                },
                {
                    "mean": metrics["ols_summary"]["mean"],
                    "std": metrics["ols_summary"]["std"],
                    "color": "purple",
                    "label": "OLS",
                },
            ],
            x_label = "Number of ICL Examples",
            y_label = "Smoothed Loss",
            basename = "3-step-variable-N-plot_smooth",
            output_dir = output_dir,
            dpi = dpi,
            figsize = (9, 9)
        ),
    }]
    write_local_index(output_dir / "smoothed_figure_index.json", artifacts)
    return artifacts


def load_metrics(experiment_root: Path) -> Dict[str, Any]:
    """Load metrics JSON for one experiment if it exists.

    Parameters:
        experiment_root: Root directory for one experiment preset.
    """

    metrics_path = experiment_root / "metrics" / "metrics.json"
    if not metrics_path.exists():
        logger.warning("Skipping missing metrics file: %s", metrics_path)
        return {}
    with metrics_path.open("r", encoding = "utf-8") as handle:
        return json.load(handle)


def plot_single_smoothed_curve(
    x_values: List[float],
    mean_values: List[float],
    std_values: List[float],
    color: str,
    label: str,
    x_label: str,
    y_label: str,
    basename: str,
    output_dir: Path,
    dpi: int
) -> Dict[str, str]:
    """Render one smoothed curve with an unsmoothed uncertainty band.

    Parameters:
        x_values: X-axis values.
        mean_values: Mean curve values.
        std_values: Standard deviation values.
        color: Curve color.
        label: Legend label.
        x_label: X-axis label.
        y_label: Y-axis label.
        basename: Output filename without extension.
        output_dir: Target directory.
        dpi: DPI used when exporting smoothed PNG figures.
    """

    return plot_multi_smoothed_curve(
        x_values = x_values,
        curves = [{
            "mean": mean_values,
            "std": std_values,
            "color": color,
            "label": label,
        }],
        x_label = x_label,
        y_label = y_label,
        basename = basename,
        output_dir = output_dir,
        dpi = dpi,
        figsize = (7, 6)
    )


def plot_multi_smoothed_curve(
    x_values: List[float],
    curves: List[Dict[str, Any]],
    x_label: str,
    y_label: str,
    basename: str,
    output_dir: Path,
    dpi: int,
    figsize = (9, 7)
) -> Dict[str, str]:
    """Render multiple smoothed curves.

    Parameters:
        x_values: Shared X-axis values.
        curves: Curve descriptors with mean, std, color, and label.
        x_label: X-axis label.
        y_label: Y-axis label.
        basename: Output filename without extension.
        output_dir: Target directory.
        dpi: DPI used when exporting smoothed PNG figures.
        figsize: Figure size tuple.
    """

    set_paper_style()
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    x_array = np.array(x_values)
    for curve in curves:
        raw_mean = np.array(curve["mean"], dtype = float)
        raw_std = np.array(curve["std"], dtype = float)
        smooth_mean = np.array(smooth(raw_mean.tolist()), dtype = float)
        smooth_std = np.array(smooth(raw_std.tolist()), dtype = float)
        ax.plot(x_array, smooth_mean, color = curve["color"], label = curve["label"])
        ax.fill_between(x_array, smooth_mean - smooth_std, smooth_mean + smooth_std, color = curve["color"], alpha = 0.16)
        ax.plot(x_array, raw_mean, color = curve["color"], alpha = 0.25, linewidth = 1.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(axis = "both", which = "major", labelsize = 14, width = 2, length = 6)
    ax.tick_params(axis = "both", which = "minor", labelsize = 12, width = 1, length = 3)
    ax.legend()
    return save_figure(fig, output_dir, basename, dpi)


def plot_panel_smoothed_curves(
    x_values: List[int],
    panel_curves: List[Dict[str, Any]],
    basename: str,
    output_dir: Path,
    dpi: int
) -> Dict[str, str]:
    """Render smoothed distance panels for Theorem 4.

    Parameters:
        x_values: Shared iteration axis.
        panel_curves: Panel descriptors containing labels, mean/std values, and styling.
        basename: Output filename without extension.
        output_dir: Target directory.
        dpi: DPI used when exporting smoothed PNG figures.
    """

    set_paper_style()
    fig, axes = plt.subplots(3, 2, figsize = (14, 18))
    x_array = np.array(x_values)
    for panel_index, panel in enumerate(panel_curves):
        row_index = panel_index // 2
        col_index = panel_index % 2
        ax = axes[row_index, col_index]
        raw_mean = np.array(panel["mean"], dtype = float)
        raw_std = np.array(panel["std"], dtype = float)
        smooth_mean = np.array(smooth(raw_mean.tolist()), dtype = float)
        smooth_std = np.array(smooth(raw_std.tolist()), dtype = float)
        ax.plot(x_array, smooth_mean, color = panel["color"], label = panel["label"])
        ax.fill_between(x_array, smooth_mean - smooth_std, smooth_mean + smooth_std, color = panel["color"], alpha = 0.16)
        ax.plot(x_array, raw_mean, color = panel["color"], alpha = 0.25, linewidth = 1.0)
        ax.set_ylim([0.0, 1.0])
        ax.legend()
        ax.tick_params(axis = "both", which = "major", labelsize = 12, width = 2, length = 5)
    axes[2, 1].axis("off")
    return save_figure(fig, output_dir, basename, dpi)


def write_local_index(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Write a local JSON index for smoothed figures.

    Parameters:
        path: Target JSON path.
        entries: Artifact entries to write.
    """

    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding = "utf-8") as handle:
        json.dump(entries, handle, indent = 2, ensure_ascii = False)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
