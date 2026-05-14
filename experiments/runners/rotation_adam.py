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
from experiments.data.generation import build_eval_bundle, build_rotation_bundle
from experiments.runners.base import BaseExperimentRunner
from experiments.runners.common import make_model, save_data_asset, save_data_metadata, summarize_component_curves, summarize_seed_curves, train_seed_model

logger = logging.getLogger(__name__)


class RotationAdamRunner(BaseExperimentRunner):
    """Runner for the rotated-covariance Adam experiment.

    Parameters:
        None: The runner reads its state from `self.config`.
    """

    def train_stage(self) -> Dict[str, Any]:
        """Train all seeded models for the rotated-covariance Adam setup.

        Parameters:
            None: The runner reads its state from `self.config`.
        """

        device = resolve_device(self.config.device)
        seed_artifacts = {}
        metadata = {
            "experiment": self.config.experiment,
            "preset": self.config.preset,
            "eval_seed": 99,
            "rotation_diagonal": self.config.rotation_diagonal,
            "seeds": self.config.seeds,
        }
        save_data_metadata(self.paths.data_root / "metadata.json", metadata)

        for seed in self.config.seeds:
            seed_key = f"seed_{seed}"
            rotation_bundle = build_rotation_bundle(
                dimension = self.config.dimension,
                diagonal_values = self.config.rotation_diagonal,
                seed = seed,
                device = device
            )
            eval_bundle = build_eval_bundle(
                mode = self.config.mode,
                context_length = self.config.context_length,
                dimension = self.config.dimension,
                batch_size = self.config.eval_batch_size,
                shape_k = self.config.shape_k,
                seed = 99,
                device = device,
                U = rotation_bundle["U"],
                D = rotation_bundle["D"]
            )
            save_data_asset(self.paths.data_root / f"{seed_key}_eval.pt", {
                "U": rotation_bundle["U"].detach().cpu(),
                "D": rotation_bundle["D"].detach().cpu(),
                "Z": eval_bundle["Z"],
                "y": eval_bundle["y"],
            })
            seed_payload = train_seed_model(
                config = self.config,
                device = device,
                seed = seed,
                progress_label = f"{self.config.experiment}:{seed_key}",
                U = rotation_bundle["U"],
                D = rotation_bundle["D"]
            )
            seed_payload["U"] = rotation_bundle["U"].detach().cpu()
            seed_payload["D"] = rotation_bundle["D"].detach().cpu()
            seed_artifacts[seed_key] = seed_payload
        return {"seed_artifacts": seed_artifacts}

    def eval_stage(self, train_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all checkpoints on fixed rotated-covariance test sets.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
        """

        device = resolve_device(self.config.device)
        loss_curves = {}
        log_loss_curves = {}
        distance_curves = {}
        iterations = None

        for seed_key, payload in train_artifacts["seed_artifacts"].items():
            eval_bundle = torch.load(self.paths.data_root / f"{seed_key}_eval.pt", map_location = "cpu")
            history = payload["history"]
            iterations = payload["history_iterations"]
            losses = evaluate_history_losses(
                history = history,
                eval_Z = eval_bundle["Z"],
                eval_y = eval_bundle["y"],
                model_builder = lambda: make_model(self.config),
                device = device
            )
            loss_curves[seed_key] = losses
            log_loss_curves[seed_key] = [float(np.log(max(value, 1e-12))) for value in losses]
            distance_curves[seed_key] = compute_distance_curves(
                history = history,
                n_layer = self.config.n_layer,
                include_p = True,
                rotate_q = True,
                U = eval_bundle["U"],
                D = eval_bundle["D"]
            )

        return {
            "iterations": iterations,
            "loss_curves": loss_curves,
            "log_loss_curves": log_loss_curves,
            "loss_summary_log": summarize_seed_curves(log_loss_curves),
            "distance_curves": distance_curves,
            "distance_summary": summarize_component_curves(distance_curves),
        }

    def plot_stage(self, train_artifacts: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Render all paper-style figures for the Adam rotation experiment.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
            metrics: Evaluation metrics produced by `eval_stage`.
        """

        iterations = metrics["iterations"]
        figure_index = []
        loss_paths = plot_error_band(
            x_values = iterations,
            mean_values = metrics["loss_summary_log"]["mean"],
            std_values = metrics["loss_summary_log"]["std"],
            color = "red",
            label = "Adam",
            x_label = "Iteration",
            y_label = "log(ICL Test Loss)",
            basename = "rotation_demonstration_adam_loss_plot",
            output_dir = self.paths.figures_dir,
            dpi = self.config.line_dpi
        )
        figure_index.append({
            "experiment": self.config.experiment,
            "basename": "rotation_demonstration_adam_loss_plot",
            "paper_use": "Figure log loss for the general linear transformer setting",
            "paths": loss_paths,
        })

        panel_order = [
            ("B0", "red", "$B_0$"),
            ("B1", "orange", "$B_1$"),
            ("A0", "green", "$\\Sigma^{1/2} A_0 \\Sigma^{1/2}$"),
            ("A1", "blue", "$\\Sigma^{1/2} A_1 \\Sigma^{1/2}$"),
            ("A2", "black", "$\\Sigma^{1/2} A_2 \\Sigma^{1/2}$"),
        ]
        panel_curves = []
        for component_name, color, label in panel_order:
            component_paths = plot_error_band(
                x_values = iterations,
                mean_values = metrics["distance_summary"][component_name]["mean"],
                std_values = metrics["distance_summary"][component_name]["std"],
                color = color,
                label = label,
                x_label = "Iteration",
                y_label = "Distance to Id",
                basename = f"rotation_demonstration_dist_to_id_adam_{component_name}",
                output_dir = self.paths.figures_dir,
                dpi = self.config.line_dpi
            )
            figure_index.append({
                "experiment": self.config.experiment,
                "basename": f"rotation_demonstration_dist_to_id_adam_{component_name}",
                "paper_use": f"Distance-to-identity curve for {component_name}",
                "paths": component_paths,
            })
            panel_curves.append({
                "mean": metrics["distance_summary"][component_name]["mean"],
                "std": metrics["distance_summary"][component_name]["std"],
                "color": color,
                "label": label,
            })

        panel_paths = plot_distance_panels(
            x_values = iterations,
            panel_curves = panel_curves,
            basename = "rotation_demonstration_dist_to_id_adam",
            output_dir = self.paths.figures_dir,
            dpi = self.config.line_dpi
        )
        figure_index.append({
            "experiment": self.config.experiment,
            "basename": "rotation_demonstration_dist_to_id_adam",
            "paper_use": "Combined distance panel for the Adam rotation experiment",
            "paths": panel_paths,
        })

        seed_key = f"seed_{self.config.seeds[0]}"
        matrices = extract_final_matrices(
            history = train_artifacts["seed_artifacts"][seed_key]["history"],
            n_layer = self.config.n_layer,
            include_p = True,
            rotate_q = True,
            U = train_artifacts["seed_artifacts"][seed_key]["U"],
            D = train_artifacts["seed_artifacts"][seed_key]["D"]
        )
        for matrix_name, matrix in matrices.items():
            heatmap_paths = plot_matrix_heatmap(
                matrix = matrix,
                title = f"${matrix_name[0]}_{matrix_name[1:]}$",
                basename = f"rotation_demonstration_adam_{matrix_name}",
                output_dir = self.paths.figures_dir,
                dpi = self.config.heatmap_dpi
            )
            figure_index.append({
                "experiment": self.config.experiment,
                "basename": f"rotation_demonstration_adam_{matrix_name}",
                "paper_use": f"Final matrix heatmap for {matrix_name}",
                "paths": heatmap_paths,
            })

        write_figure_index(self.paths.figures_dir / "figure_index.json", figure_index)
