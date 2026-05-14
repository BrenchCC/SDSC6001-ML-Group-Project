import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.append(os.getcwd())

from experiments.configs import ExperimentConfig
from experiments.core.analysis import mean_std_from_seed_curves
from experiments.utils.io_utils import save_json, save_torch
from experiments.core.training import TrainingResult, build_optimizer, run_training_loop
from experiments.data.generation import generate_data, generate_data_inplace
from experiments.models.linear_transformer import TransformerF

logger = logging.getLogger(__name__)


def make_model(config: ExperimentConfig, n_layer: Optional[int] = None) -> TransformerF:
    """Instantiate a transformer model consistent with the experiment config.

    Parameters:
        config: Expanded experiment configuration.
        n_layer: Optional layer-count override.
    """

    return TransformerF(
        n_layer = n_layer or config.n_layer,
        n_head = config.n_head,
        dimension = config.dimension,
        init_var = config.init_var
    )


def training_result_to_dict(result: TrainingResult) -> Dict[str, object]:
    """Convert a training result dataclass into a serialization-friendly payload.

    Parameters:
        result: Training loop result.
    """

    return {
        "history": result.history,
        "history_iterations": result.history_iterations,
        "train_losses": result.train_losses,
        "grad_norms": result.grad_norms,
        "elapsed_time": result.elapsed_time,
    }


def create_batch_fetcher(
    config: ExperimentConfig,
    device: torch.device,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None
) -> Tuple[callable, Optional[Dict[str, torch.Tensor]]]:
    """Create a batch fetcher matching the reference training semantics.

    Parameters:
        config: Expanded experiment configuration.
        device: Device used for training tensors.
        U: Optional orthogonal covariance factor.
        D: Optional diagonal covariance factor.
    """

    logger.info(
        "Preparing training batch fetcher: experiment = %s | context_length = %s | batch_size = %s | fixed_train_batch = %s | resample_interval = %s",
        config.experiment,
        config.context_length,
        config.batch_size,
        config.fixed_train_batch,
        config.resample_interval
    )
    Z, y = generate_data(
        mode = config.mode,
        N = config.context_length,
        d = config.dimension,
        B = config.batch_size,
        shape_k = config.shape_k,
        U = U,
        D = D,
        device = device
    )
    train_snapshot = None
    if config.fixed_train_batch:
        logger.info("Using fixed training batch snapshot for experiment %s", config.experiment)
        train_snapshot = {"Z": Z.detach().cpu(), "y": y.detach().cpu()}

        def fixed_fetcher(_):
            """Return the same batch for every iteration.

            Parameters:
                _: Ignored iteration number.
            """

            return Z, y

        return fixed_fetcher, train_snapshot

    def dynamic_fetcher(iteration: int):
        """Refresh the batch according to the configured resampling interval.

        Parameters:
            iteration: Current training iteration.
        """

        nonlocal Z, y
        if config.resample_interval is not None and iteration > 0 and iteration % config.resample_interval == 0:
            if iteration <= max(config.resample_interval * 5, 10) or iteration % max(config.resample_interval * 20, 1000) == 0:
                logger.info(
                    "Refreshing dynamic training batch: experiment = %s | iteration = %s | resample_interval = %s",
                    config.experiment,
                    iteration,
                    config.resample_interval
                )
            Z, y = generate_data_inplace(Z, U = U, D = D)
        return Z, y

    return dynamic_fetcher, train_snapshot


def train_seed_model(
    config: ExperimentConfig,
    device: torch.device,
    seed: int,
    progress_label: str,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    n_layer: Optional[int] = None
) -> Dict[str, object]:
    """Train one model instance for a specific seed.

    Parameters:
        config: Expanded experiment configuration.
        device: Device used for training.
        seed: Random seed for model and data generation.
        progress_label: Progress-bar label.
        U: Optional orthogonal covariance factor.
        D: Optional diagonal covariance factor.
        n_layer: Optional layer-count override.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(
        "*" * 80
    )
    logger.info(
        "Seed run started: experiment = %s | seed = %s | n_layer = %s | device = %s",
        config.experiment,
        seed,
        n_layer or config.n_layer,
        device
    )
    logger.info("*" * 80)
    model = make_model(config, n_layer = n_layer).to(device)
    optimizer = build_optimizer(
        optimizer_name = config.optimizer_name,
        model = model,
        learning_rate = config.learning_rate,
        beta1 = config.beta1,
        beta2 = config.beta2,
        momentum = config.momentum,
        lbfgs_history_size = config.lbfgs_history_size,
        lbfgs_max_iter = config.lbfgs_max_iter
    )
    batch_fetcher, train_snapshot = create_batch_fetcher(config, device, U = U, D = D)
    training_result = run_training_loop(
        model = model,
        optimizer = optimizer,
        optimizer_name = config.optimizer_name,
        max_iters = config.max_iters,
        checkpoint_stride = config.checkpoint_stride,
        log_stride = config.log_stride,
        batch_fetcher = batch_fetcher,
        clip_value = config.clip_value,
        clip_mode = config.clip_mode,
        zero_p_after_step = config.zero_p_after_step,
        lr_decay_steps = config.lr_decay_steps,
        lr_decay_factor = config.lr_decay_factor,
        progress_label = progress_label
    )
    payload = training_result_to_dict(training_result)
    if train_snapshot is not None:
        payload["train_snapshot"] = train_snapshot
    logger.info(
        "Seed run finished: experiment = %s | seed = %s | elapsed_time = %.2fs",
        config.experiment,
        seed,
        training_result.elapsed_time
    )
    return payload


def summarize_seed_curves(seed_curve_map: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """Aggregate a seed-keyed curve dictionary into mean and standard deviation.

    Parameters:
        seed_curve_map: Mapping from seed identifier to numeric curve.
    """

    return mean_std_from_seed_curves(list(seed_curve_map.values()))


def summarize_component_curves(component_curve_map: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[float]]]:
    """Aggregate component-wise curves across seeds.

    Parameters:
        component_curve_map: Mapping from seed identifier to component curves.
    """

    component_names = list(next(iter(component_curve_map.values())).keys())
    summary: Dict[str, Dict[str, List[float]]] = {}
    for component_name in component_names:
        summary[component_name] = mean_std_from_seed_curves([
            component_curve_map[seed_key][component_name] for seed_key in component_curve_map
        ])
    return summary


def save_data_metadata(path: Path, metadata: Dict[str, object]) -> None:
    """Write JSON metadata for generated data assets.

    Parameters:
        path: Target metadata path.
        metadata: Metadata payload.
    """

    logger.info("Writing data metadata to %s", path)
    save_json(path, metadata)


def save_data_asset(path: Path, payload: Dict[str, torch.Tensor]) -> None:
    """Serialize a data asset bundle with `torch.save`.

    Parameters:
        path: Target file path.
        payload: Tensor payload to serialize.
    """

    logger.info("Writing data asset to %s", path)
    save_torch(path, payload)
