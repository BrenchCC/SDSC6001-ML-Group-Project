import os
import sys
import logging
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.append(os.getcwd())

from experiments.core.analysis import compute_distance_curves, evaluate_history_losses, extract_final_matrices
from experiments.utils.io_utils import resolve_device
from experiments.core.plotting import plot_distance_panels, plot_error_band, plot_matrix_heatmap, write_figure_index
from experiments.data.generation import build_eval_bundle
from experiments.runners.base import BaseExperimentRunner
from experiments.runners.common import make_model, save_data_asset, save_data_metadata, summarize_component_curves, summarize_seed_curves, train_seed_model

logger = logging.getLogger(__name__)


class SimpleRunner(BaseExperimentRunner):
    """Runner for the identity-covariance sanity-check experiment.

    Parameters:
        None: The runner reads its state from `self.config`.
    """

    def train_stage(self) -> Dict[str, Any]:
        """Train all seeded models for the simple demonstration.

        Parameters:
            None: The runner reads its state from `self.config`.
        """

        device = resolve_device(self.config.device)
        eval_bundle = build_eval_bundle(
            mode = self.config.mode,
            context_length = self.config.context_length,
            dimension = self.config.dimension,
            batch_size = self.config.eval_batch_size,
            shape_k = self.config.shape_k,
            seed = 99,
            device = device
        )
        save_data_asset(self.paths.data_root / "eval_bundle_seed_99.pt", eval_bundle)
        save_data_metadata(self.paths.data_root / "metadata.json", {
            "experiment": self.config.experiment,
            "preset": self.config.preset,
            "eval_seed": 99,
            "seeds": self.config.seeds,
            "context_length": self.config.context_length,
            "dimension": self.config.dimension,
        })

        seed_artifacts = {}
        for seed in self.config.seeds:
            seed_key = f"seed_{seed}"
            seed_artifacts[seed_key] = train_seed_model(
                config = self.config,
                device = device,
                seed = seed,
                progress_label = f"{self.config.experiment}:{seed_key}"
            )
        return {"seed_artifacts": seed_artifacts}

    def eval_stage(self, train_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all checkpoints on a fixed test batch.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
        """

        device = resolve_device(self.config.device)
        eval_bundle = torch.load(self.paths.data_root / "eval_bundle_seed_99.pt", map_location = "cpu")
        loss_curves = {}
        distance_curves = {}
        iterations = None

        for seed_key, payload in train_artifacts["seed_artifacts"].items():
            history = payload["history"]
            iterations = payload["history_iterations"]
            loss_curves[seed_key] = evaluate_history_losses(
                history = history,
                eval_Z = eval_bundle["Z"],
                eval_y = eval_bundle["y"],
                model_builder = lambda: make_model(self.config),
                device = device
            )
            distance_curves[seed_key] = compute_distance_curves(
                history = history,
                n_layer = self.config.n_layer,
                include_p = True
            )

        return {
            "iterations": iterations,
            "loss_curves": loss_curves,
            "loss_summary": summarize_seed_curves(loss_curves),
            "distance_curves": distance_curves,
            "distance_summary": summarize_component_curves(distance_curves),
        }

    def plot_stage(self, train_artifacts: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Render loss, distance, and heatmap figures for the experiment.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
            metrics: Evaluation metrics produced by `eval_stage`.
        """

        iterations = metrics["iterations"]
        figure_index = []
        loss_paths = plot_error_band(
            x_values = iterations,
            mean_values = metrics["loss_summary"]["mean"],
            std_values = metrics["loss_summary"]["std"],
            color = "red",
            label = "3-Layer Linear Transformer",
            x_label = "Iteration",
            y_label = "ICL Test Loss",
            basename = "simple_demonstration_loss_plot",
            output_dir = self.paths.figures_dir,
            dpi = self.config.line_dpi,
            log_y = True
        )
        figure_index.append({
            "experiment": self.config.experiment,
            "basename": "simple_demonstration_loss_plot",
            "paper_use": "Identity-covariance sanity-check loss curve",
            "paths": loss_paths,
        })

        panel_order = [
            ("B0", "red", "$B_0$"),
            ("B1", "orange", "$B_1$"),
            ("A0", "green", "$A_0$"),
            ("A1", "blue", "$A_1$"),
            ("A2", "black", "$A_2$"),
        ]
        panel_curves = []
        for component_name, color, label in panel_order:
            panel_curves.append({
                "mean": metrics["distance_summary"][component_name]["mean"],
                "std": metrics["distance_summary"][component_name]["std"],
                "color": color,
                "label": label,
            })
        distance_paths = plot_distance_panels(
            x_values = iterations,
            panel_curves = panel_curves,
            basename = "simple_demonstration_dist_to_id",
            output_dir = self.paths.figures_dir,
            dpi = self.config.line_dpi
        )
        figure_index.append({
            "experiment": self.config.experiment,
            "basename": "simple_demonstration_dist_to_id",
            "paper_use": "Identity-covariance distance-to-identity summary",
            "paths": distance_paths,
        })

        seed_key = f"seed_{self.config.seeds[0]}"
        matrices = extract_final_matrices(
            history = train_artifacts["seed_artifacts"][seed_key]["history"],
            n_layer = self.config.n_layer,
            include_p = True
        )
        for matrix_name, matrix in matrices.items():
            heatmap_paths = plot_matrix_heatmap(
                matrix = matrix,
                title = f"${matrix_name[0]}_{matrix_name[1:]}$",
                basename = f"simple_demonstration_{matrix_name}",
                output_dir = self.paths.figures_dir,
                dpi = self.config.heatmap_dpi
            )
            figure_index.append({
                "experiment": self.config.experiment,
                "basename": f"simple_demonstration_{matrix_name}",
                "paper_use": f"Final matrix heatmap for {matrix_name}",
                "paths": heatmap_paths,
            })

        write_figure_index(self.paths.figures_dir / "figure_index.json", figure_index)
