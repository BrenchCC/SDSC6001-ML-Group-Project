import os
import sys
import logging
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.append(os.getcwd())

from experiments.core.analysis import evaluate_history_losses, select_best_checkpoint
from experiments.core.baselines import evaluate_baseline_loss, find_best_eta
from experiments.utils.io_utils import resolve_device
from experiments.core.plotting import plot_multi_curve, write_figure_index
from experiments.data.generation import build_eval_bundle, build_rotation_bundle
from experiments.models.linear_transformer import in_context_loss
from experiments.runners.base import BaseExperimentRunner
from experiments.runners.common import make_model, save_data_asset, save_data_metadata, train_seed_model

logger = logging.getLogger(__name__)


class VariableLRunner(BaseExperimentRunner):
    """Runner for the layer-sweep experiment.

    Parameters:
        None: The runner reads its state from `self.config`.
    """

    def train_stage(self) -> Dict[str, Any]:
        """Train transformer models for each seed and layer count.

        Parameters:
            None: The runner reads its state from `self.config`.
        """

        device = resolve_device(self.config.device)
        seed_artifacts = {}
        save_data_metadata(self.paths.data_root / "metadata.json", {
            "experiment": self.config.experiment,
            "preset": self.config.preset,
            "validation_seed": 999,
            "test_seed": 99,
            "rotation_diagonal": self.config.rotation_diagonal,
            "layers": self.config.variable_layers,
            "seeds": self.config.seeds,
        })

        for seed in self.config.seeds:
            seed_key = f"seed_{seed}"
            rotation_bundle = build_rotation_bundle(
                dimension = self.config.dimension,
                diagonal_values = self.config.rotation_diagonal,
                seed = seed,
                device = device
            )
            save_data_asset(self.paths.data_root / f"{seed_key}_rotation.pt", {
                "U": rotation_bundle["U"].detach().cpu(),
                "D": rotation_bundle["D"].detach().cpu(),
            })
            val_bundle = build_eval_bundle(
                mode = self.config.mode,
                context_length = self.config.context_length,
                dimension = self.config.dimension,
                batch_size = self.config.validation_batch_size,
                shape_k = self.config.shape_k,
                seed = 999,
                device = device,
                U = rotation_bundle["U"],
                D = rotation_bundle["D"]
            )
            test_bundle = build_eval_bundle(
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
            save_data_asset(self.paths.data_root / f"{seed_key}_validation.pt", val_bundle)
            save_data_asset(self.paths.data_root / f"{seed_key}_test.pt", test_bundle)

            layer_artifacts = {}
            for n_layer in self.config.variable_layers:
                layer_key = f"L_{n_layer}"
                layer_artifacts[layer_key] = train_seed_model(
                    config = self.config,
                    device = device,
                    seed = seed,
                    progress_label = f"{self.config.experiment}:{seed_key}:{layer_key}",
                    U = rotation_bundle["U"],
                    D = rotation_bundle["D"],
                    n_layer = n_layer
                )
            seed_artifacts[seed_key] = {
                "U": rotation_bundle["U"].detach().cpu(),
                "D": rotation_bundle["D"].detach().cpu(),
                "layer_artifacts": layer_artifacts,
            }
        return {"seed_artifacts": seed_artifacts}

    def eval_stage(self, train_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate transformer checkpoints and baseline methods across layers.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
        """

        device = resolve_device(self.config.device)
        transformer_log_losses = {}
        gd_log_losses = {f"seed_{seed}": [] for seed in self.config.seeds}
        pgd_log_losses = {f"seed_{seed}": [] for seed in self.config.seeds}
        reference_seed_key = f"seed_{self.config.seeds[0]}"
        reference_rotation = train_artifacts["seed_artifacts"][reference_seed_key]
        reference_val = torch.load(self.paths.data_root / f"{reference_seed_key}_validation.pt", map_location = "cpu")

        gd_best_etas = {}
        pgd_best_etas = {}
        for n_layer in self.config.variable_layers:
            best_eta_gd, _ = find_best_eta(
                Z = reference_val["Z"],
                y = reference_val["y"],
                method_name = "gd",
                eta_grid = self.config.eta_grid_gd,
                dimension = self.config.dimension,
                numstep = n_layer,
                device = device,
                max_samples = self.config.baseline_eval_size
            )
            best_eta_pgd, _ = find_best_eta(
                Z = reference_val["Z"],
                y = reference_val["y"],
                method_name = "pgd",
                eta_grid = self.config.eta_grid_pgd,
                dimension = self.config.dimension,
                numstep = n_layer,
                device = device,
                U = reference_rotation["U"],
                D = reference_rotation["D"],
                max_samples = self.config.baseline_eval_size
            )
            gd_best_etas[n_layer] = best_eta_gd
            pgd_best_etas[n_layer] = best_eta_pgd

        for seed in self.config.seeds:
            seed_key = f"seed_{seed}"
            test_bundle = torch.load(self.paths.data_root / f"{seed_key}_test.pt", map_location = "cpu")
            val_bundle = torch.load(self.paths.data_root / f"{seed_key}_validation.pt", map_location = "cpu")
            seed_payload = train_artifacts["seed_artifacts"][seed_key]
            transformer_log_losses[seed_key] = []
            for n_layer in self.config.variable_layers:
                layer_key = f"L_{n_layer}"
                history = seed_payload["layer_artifacts"][layer_key]["history"]
                validation_losses = evaluate_history_losses(
                    history = history,
                    eval_Z = val_bundle["Z"],
                    eval_y = val_bundle["y"],
                    model_builder = lambda n_layer = n_layer: make_model(self.config, n_layer = n_layer),
                    device = device
                )
                best_index = select_best_checkpoint(validation_losses, tail_window = min(20, len(validation_losses)))
                model = make_model(self.config, n_layer = n_layer).to(device)
                with torch.no_grad():
                    model.allparam.copy_(history[best_index].to(device))
                test_loss = in_context_loss(model, test_bundle["Z"].to(device), test_bundle["y"].to(device))
                transformer_log_losses[seed_key].append(float(torch.log(test_loss.clamp_min(1e-12)).item()))

                gd_loss = evaluate_baseline_loss(
                    Z = test_bundle["Z"],
                    y = test_bundle["y"],
                    method_name = "gd",
                    dimension = self.config.dimension,
                    numstep = n_layer,
                    eta = gd_best_etas[n_layer],
                    device = device,
                    max_samples = self.config.baseline_eval_size
                )
                gd_log_losses[seed_key].append(float(np.log(max(gd_loss, 1e-12))))

                pgd_loss = evaluate_baseline_loss(
                    Z = test_bundle["Z"],
                    y = test_bundle["y"],
                    method_name = "pgd",
                    dimension = self.config.dimension,
                    numstep = n_layer,
                    eta = pgd_best_etas[n_layer],
                    device = device,
                    U = seed_payload["U"],
                    D = seed_payload["D"],
                    max_samples = self.config.baseline_eval_size
                )
                pgd_log_losses[seed_key].append(float(np.log(max(pgd_loss, 1e-12))))

        def summarize(curves: Dict[str, List[float]]) -> Dict[str, List[float]]:
            """Compute mean and standard deviation for seed-wise sweep curves.

            Parameters:
                curves: Mapping from seed key to sweep losses.
            """

            curve_tensor = torch.tensor(list(curves.values()), dtype = torch.float32)
            if curve_tensor.shape[0] == 1:
                return {
                    "mean": curve_tensor.mean(dim = 0).tolist(),
                    "std": torch.zeros_like(curve_tensor[0]).tolist(),
                }
            return {
                "mean": curve_tensor.mean(dim = 0).tolist(),
                "std": curve_tensor.std(dim = 0).tolist(),
            }

        return {
            "x_values": self.config.variable_layers,
            "transformer_log_losses": transformer_log_losses,
            "gd_log_losses": gd_log_losses,
            "pgd_log_losses": pgd_log_losses,
            "transformer_summary": summarize(transformer_log_losses),
            "gd_summary": summarize(gd_log_losses),
            "pgd_summary": summarize(pgd_log_losses),
            "gd_best_etas": gd_best_etas,
            "pgd_best_etas": pgd_best_etas,
        }

    def plot_stage(self, train_artifacts: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Render the layer-sweep comparison figure.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
            metrics: Evaluation metrics produced by `eval_stage`.
        """

        figure_paths = plot_multi_curve(
            x_values = metrics["x_values"],
            curves = [
                {
                    "mean": metrics["gd_summary"]["mean"],
                    "std": metrics["gd_summary"]["std"],
                    "color": "blue",
                    "label": "GD",
                },
                {
                    "mean": metrics["pgd_summary"]["mean"],
                    "std": metrics["pgd_summary"]["std"],
                    "color": "green",
                    "label": "Preconditioned GD",
                },
                {
                    "mean": metrics["transformer_summary"]["mean"],
                    "std": metrics["transformer_summary"]["std"],
                    "color": "red",
                    "label": "Linear Transformer",
                },
            ],
            x_label = "Number of Layers / Steps",
            y_label = "log(Test Loss)",
            basename = "variable-L-plot",
            output_dir = self.paths.figures_dir,
            dpi = self.config.line_dpi
        )
        write_figure_index(self.paths.figures_dir / "figure_index.json", [{
            "experiment": self.config.experiment,
            "basename": "variable-L-plot",
            "paper_use": "Layer-sweep loss comparison",
            "paths": figure_paths,
        }])
