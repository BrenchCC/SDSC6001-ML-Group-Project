import os
import sys
import logging
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Container for a single experiment run configuration.

    Parameters:
        experiment: Public experiment name exposed by the CLI.
        stage: Requested execution stage (`train`, `eval`, `plot`, or `all`).
        preset: Runtime preset name.
        device: Device request from the CLI.
        result_root: Root directory where experiment outputs are stored.
        data_root: Root directory where generated data assets are stored.
        paper_reference: Human-readable paper reference for documentation.
        optimizer_name: Optimizer identifier used by the training loop.
        mode: Synthetic data mode passed to the generator.
        n_layer: Default number of transformer layers for the experiment.
        n_head: Number of attention heads.
        dimension: Covariate dimension.
        context_length: Number of in-context examples.
        batch_size: Training batch size.
        eval_batch_size: Batch size for evaluation datasets.
        validation_batch_size: Batch size used for checkpoint selection.
        baseline_eval_size: Number of tasks used for baseline estimates.
        init_var: Gaussian initialization scale for transformer parameters.
        shape_k: Gamma distribution shape parameter for optional generators.
        learning_rate: Primary optimizer learning rate.
        max_iters: Number of training iterations.
        checkpoint_stride: Iteration interval for history snapshots.
        log_stride: Iteration interval for log messages.
        seeds: Random seeds used for repeated runs.
        clip_value: Gradient clipping threshold, if enabled.
        clip_mode: Clipping strategy (`overall` or `per_matrix`).
        beta1: First optimizer momentum coefficient.
        beta2: Second optimizer momentum coefficient.
        momentum: Momentum used by SGD when relevant.
        resample_interval: Iteration interval for refreshing dynamic batches.
        fixed_train_batch: Whether to reuse a single training batch.
        zero_p_after_step: Whether to zero `P` matrices after each update.
        rotation_diagonal: Diagonal values used to construct the rotated covariance.
        variable_layers: Sweep values for the `variable_l` experiment.
        variable_contexts: Sweep values for the `variable_n` experiment.
        eta_grid_gd: Grid-search values for vanilla GD baselines.
        eta_grid_pgd: Grid-search values for preconditioned GD baselines.
        lbfgs_history_size: LBFGS history size.
        lbfgs_max_iter: LBFGS inner steps per external iteration.
        lr_decay_steps: Iterations at which the learning rate is decayed.
        lr_decay_factor: Multiplicative decay factor.
        line_dpi: DPI for line figures exported as PNG.
        heatmap_dpi: DPI for heatmap figures exported as PNG.
    """

    experiment: str
    stage: str
    preset: str
    device: str
    result_root: Path
    data_root: Path
    paper_reference: str
    optimizer_name: str
    mode: str = "normal"
    n_layer: int = 3
    n_head: int = 1
    dimension: int = 5
    context_length: int = 20
    batch_size: int = 4000
    eval_batch_size: int = 4000
    validation_batch_size: int = 2000
    baseline_eval_size: int = 1000
    init_var: float = 0.0001
    shape_k: float = 0.1
    learning_rate: float = 0.001
    max_iters: int = 10000
    checkpoint_stride: int = 100
    log_stride: int = 100
    seeds: List[int] = field(default_factory = list)
    clip_value: Optional[float] = None
    clip_mode: str = "overall"
    beta1: float = 0.9
    beta2: float = 0.9
    momentum: float = 0.9
    resample_interval: Optional[int] = None
    fixed_train_batch: bool = False
    zero_p_after_step: bool = False
    rotation_diagonal: List[float] = field(default_factory = lambda: [1.0, 1.0, 0.5, 0.25, 1.0])
    variable_layers: List[int] = field(default_factory = lambda: [1, 2, 3, 4])
    variable_contexts: List[int] = field(default_factory = lambda: list(range(2, 21, 2)))
    eta_grid_gd: List[float] = field(default_factory = lambda: [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.16])
    eta_grid_pgd: List[float] = field(default_factory = lambda: [0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.16])
    lbfgs_history_size: int = 600
    lbfgs_max_iter: int = 4
    lr_decay_steps: List[int] = field(default_factory = list)
    lr_decay_factor: float = 0.5
    line_dpi: int = 600
    heatmap_dpi: int = 900


def get_preset_table() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return all preset definitions grouped by experiment name.

    Parameters:
        None: This function does not accept runtime parameters.
    """

    base_paper = {
        "mode": "normal",
        "n_head": 1,
        "dimension": 5,
        "init_var": 0.0001,
        "shape_k": 0.1,
        "line_dpi": 600,
        "heatmap_dpi": 900,
    }
    base_smoke = {
        "mode": "normal",
        "n_head": 1,
        "dimension": 5,
        "init_var": 0.0001,
        "shape_k": 0.1,
        "line_dpi": 600,
        "heatmap_dpi": 900,
        "eval_batch_size": 128,
        "validation_batch_size": 128,
        "baseline_eval_size": 64,
    }
    base_cpu = {
        "mode": "normal",
        "n_head": 1,
        "dimension": 5,
        "init_var": 0.0001,
        "shape_k": 0.1,
        "line_dpi": 240,
        "heatmap_dpi": 300,
        "eval_batch_size": 2048,
        "validation_batch_size": 1024,
        "baseline_eval_size": 256,
    }

    presets = {
        "simple": {
            "paper": {
                **base_paper,
                "paper_reference": "Identity covariance sanity check",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 4000,
                "eval_batch_size": 4000,
                "validation_batch_size": 4000,
                "baseline_eval_size": 1000,
                "learning_rate": 0.001,
                "max_iters": 10000,
                "checkpoint_stride": 100,
                "log_stride": 100,
                "seeds": [0, 1, 2],
                "clip_value": 1000.0,
                "clip_mode": "overall",
                "beta1": 0.9,
                "beta2": 0.9,
            },
            "smoke": {
                **base_smoke,
                "paper_reference": "Identity covariance sanity check",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 128,
                "learning_rate": 0.005,
                "max_iters": 12,
                "checkpoint_stride": 2,
                "log_stride": 2,
                "seeds": [0, 1],
                "clip_value": 1000.0,
                "clip_mode": "overall",
                "beta1": 0.9,
                "beta2": 0.9,
            },
        },
        "rotation_adam": {
            "paper": {
                **base_paper,
                "paper_reference": "General linear transformer with Adam",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 20000,
                "eval_batch_size": 20000,
                "validation_batch_size": 4000,
                "baseline_eval_size": 2000,
                "learning_rate": 0.1,
                "max_iters": 100000,
                "checkpoint_stride": 100,
                "log_stride": 100,
                "seeds": [0, 1, 2],
                "clip_value": 0.01,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.9,
                "resample_interval": 100,
                "lr_decay_steps": list(range(4000, 100000, 4000)),
                "lr_decay_factor": 0.5,
            },
            "smoke": {
                **base_smoke,
                "paper_reference": "General linear transformer with Adam",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 128,
                "learning_rate": 0.02,
                "max_iters": 12,
                "checkpoint_stride": 2,
                "log_stride": 2,
                "seeds": [0, 1],
                "clip_value": 0.05,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.9,
                "resample_interval": 2,
                "lr_decay_steps": [6, 10],
                "lr_decay_factor": 0.5,
            },
            "cpu": {
                **base_cpu,
                "paper_reference": "CPU-scale Theorem 4 relaxed linear transformer reproduction",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 2048,
                "learning_rate": 0.05,
                "max_iters": 3000,
                "checkpoint_stride": 100,
                "log_stride": 100,
                "seeds": [0, 1],
                "clip_value": 0.01,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.9,
                "resample_interval": 50,
                "lr_decay_steps": [1000, 2000],
                "lr_decay_factor": 0.5,
            },
        },
        "rotation_adam_p0": {
            "paper": {
                **base_paper,
                "paper_reference": "Preconditioned GD stationary point",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 20000,
                "eval_batch_size": 20000,
                "validation_batch_size": 4000,
                "baseline_eval_size": 2000,
                "learning_rate": 0.02,
                "max_iters": 30000,
                "checkpoint_stride": 100,
                "log_stride": 100,
                "seeds": [0, 1, 2, 3, 4],
                "clip_value": 0.01,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.9,
                "resample_interval": 100,
                "zero_p_after_step": True,
                "lr_decay_steps": list(range(2000, 30000, 2000)),
                "lr_decay_factor": 0.5,
            },
            "smoke": {
                **base_smoke,
                "paper_reference": "Preconditioned GD stationary point",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 128,
                "learning_rate": 0.02,
                "max_iters": 12,
                "checkpoint_stride": 2,
                "log_stride": 2,
                "seeds": [0, 1],
                "clip_value": 0.05,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.9,
                "resample_interval": 2,
                "zero_p_after_step": True,
                "lr_decay_steps": [6, 10],
                "lr_decay_factor": 0.5,
            },
            "cpu": {
                **base_cpu,
                "paper_reference": "CPU-scale Theorem 3 preconditioned GD stationary point reproduction",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 2048,
                "learning_rate": 0.02,
                "max_iters": 3000,
                "checkpoint_stride": 100,
                "log_stride": 100,
                "seeds": [0, 1],
                "clip_value": 0.01,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.9,
                "resample_interval": 50,
                "zero_p_after_step": True,
                "lr_decay_steps": [1000, 2000],
                "lr_decay_factor": 0.5,
            },
        },
        "rotation_lbfgs": {
            "paper": {
                **base_paper,
                "paper_reference": "General linear transformer with LBFGS",
                "optimizer_name": "lbfgs",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 20000,
                "eval_batch_size": 20000,
                "validation_batch_size": 4000,
                "baseline_eval_size": 2000,
                "learning_rate": 0.1,
                "max_iters": 2000,
                "checkpoint_stride": 10,
                "log_stride": 100,
                "seeds": [0, 1, 2],
                "clip_value": None,
                "fixed_train_batch": True,
                "lbfgs_history_size": 600,
                "lbfgs_max_iter": 4,
            },
            "smoke": {
                **base_smoke,
                "paper_reference": "General linear transformer with LBFGS",
                "optimizer_name": "lbfgs",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 128,
                "learning_rate": 0.1,
                "max_iters": 8,
                "checkpoint_stride": 1,
                "log_stride": 1,
                "seeds": [0, 1],
                "clip_value": None,
                "fixed_train_batch": True,
                "lbfgs_history_size": 50,
                "lbfgs_max_iter": 2,
            },
        },
        "variable_l": {
            "paper": {
                **base_paper,
                "paper_reference": "Loss versus number of layers",
                "optimizer_name": "adamw",
                "n_layer": 4,
                "context_length": 20,
                "batch_size": 20000,
                "eval_batch_size": 20000,
                "validation_batch_size": 4000,
                "baseline_eval_size": 5000,
                "learning_rate": 0.01,
                "max_iters": 20000,
                "checkpoint_stride": 100,
                "log_stride": 100,
                "seeds": [0, 1, 2, 3, 4],
                "clip_value": 0.01,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.9,
                "resample_interval": 100,
                "lr_decay_steps": list(range(4000, 20000, 4000)),
                "lr_decay_factor": 0.5,
            },
            "smoke": {
                **base_smoke,
                "paper_reference": "Loss versus number of layers",
                "optimizer_name": "adamw",
                "n_layer": 4,
                "context_length": 20,
                "batch_size": 128,
                "learning_rate": 0.01,
                "max_iters": 10,
                "checkpoint_stride": 2,
                "log_stride": 2,
                "seeds": [0, 1],
                "clip_value": 0.05,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.9,
                "resample_interval": 2,
                "lr_decay_steps": [6, 8],
                "lr_decay_factor": 0.5,
                "baseline_eval_size": 32,
            },
        },
        "variable_n": {
            "paper": {
                **base_paper,
                "paper_reference": "Loss versus number of ICL examples",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 40000,
                "eval_batch_size": 40000,
                "validation_batch_size": 4000,
                "baseline_eval_size": 5000,
                "learning_rate": 0.01,
                "max_iters": 8000,
                "checkpoint_stride": 100,
                "log_stride": 500,
                "seeds": [0, 1, 2, 3, 4],
                "clip_value": 0.001,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.99,
                "resample_interval": 1,
                "lr_decay_steps": list(range(2000, 8000, 2000)),
                "lr_decay_factor": 0.5,
            },
            "smoke": {
                **base_smoke,
                "paper_reference": "Loss versus number of ICL examples",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 128,
                "learning_rate": 0.01,
                "max_iters": 8,
                "checkpoint_stride": 2,
                "log_stride": 2,
                "seeds": [0, 1],
                "clip_value": 0.01,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.99,
                "resample_interval": 1,
                "lr_decay_steps": [4, 6],
                "lr_decay_factor": 0.5,
                "baseline_eval_size": 32,
            },
            "cpu": {
                **base_cpu,
                "paper_reference": "CPU-scale context-length extension experiment",
                "optimizer_name": "adamw",
                "n_layer": 3,
                "context_length": 20,
                "batch_size": 512,
                "eval_batch_size": 1024,
                "validation_batch_size": 512,
                "baseline_eval_size": 128,
                "learning_rate": 0.01,
                "max_iters": 600,
                "checkpoint_stride": 50,
                "log_stride": 100,
                "seeds": [0, 1],
                "clip_value": 0.01,
                "clip_mode": "per_matrix",
                "beta1": 0.99,
                "beta2": 0.99,
                "resample_interval": 1,
                "variable_contexts": [4, 8, 12, 16, 20],
                "lr_decay_steps": [200, 400],
                "lr_decay_factor": 0.5,
            },
        },
    }
    return presets


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Convert an experiment config into a JSON-friendly dictionary.

    Parameters:
        config: Experiment configuration to serialize.
    """

    payload = asdict(config)
    payload["result_root"] = str(config.result_root)
    payload["data_root"] = str(config.data_root)
    return payload


def build_config(
    experiment: str,
    stage: str,
    preset: str,
    device: str,
    result_dir: str,
    data_dir: str,
    max_iters: Optional[int] = None,
    batch_size: Optional[int] = None,
    stride: Optional[int] = None,
    seeds: Optional[List[int]] = None
) -> ExperimentConfig:
    """Build a fully expanded experiment config from preset and CLI overrides.

    Parameters:
        experiment: Selected experiment name.
        stage: Selected execution stage.
        preset: Selected runtime preset.
        device: Requested runtime device.
        result_dir: Root results directory from the CLI.
        data_dir: Root data directory from the CLI.
        max_iters: Optional override for the number of iterations.
        batch_size: Optional override for the training batch size.
        stride: Optional override for checkpoint and log stride.
        seeds: Optional override for the random seed list.
    """

    preset_table = get_preset_table()
    if experiment not in preset_table:
        raise ValueError(f"Unsupported experiment: {experiment}")
    if preset not in preset_table[experiment]:
        raise ValueError(f"Unsupported preset '{preset}' for experiment '{experiment}'")

    payload = deepcopy(preset_table[experiment][preset])
    if max_iters is not None:
        payload["max_iters"] = max_iters
    if batch_size is not None:
        payload["batch_size"] = batch_size
    if stride is not None:
        payload["checkpoint_stride"] = stride
        payload["log_stride"] = stride
    if seeds:
        payload["seeds"] = seeds

    config = ExperimentConfig(
        experiment = experiment,
        stage = stage,
        preset = preset,
        device = device,
        result_root = Path(result_dir),
        data_root = Path(data_dir),
        **payload
    )
    logger.info("Built config for %s with preset %s", experiment, preset)
    return config
