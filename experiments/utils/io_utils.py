import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import is_dataclass, asdict, dataclass
from typing import Any, Dict

import torch

sys.path.append(os.getcwd())

from experiments.configs import ExperimentConfig, config_to_dict

logger = logging.getLogger(__name__)


@dataclass
class OutputPaths:
    """Resolved filesystem locations for a single experiment.

    Parameters:
        result_root: Experiment result root directory.
        artifact_dir: Directory for serialized training artifacts.
        metrics_dir: Directory for structured metrics.
        figures_dir: Directory for generated figures.
        logs_dir: Directory for persisted runtime logs.
        data_root: Experiment data asset root directory.
    """

    result_root: Path
    artifact_dir: Path
    metrics_dir: Path
    figures_dir: Path
    logs_dir: Path
    data_root: Path


def build_output_paths(config: ExperimentConfig) -> OutputPaths:
    """Resolve and create output directories for an experiment.

    Parameters:
        config: Expanded experiment configuration.
    """

    result_root = config.result_root / config.experiment / config.preset
    data_root = config.data_root / config.experiment / config.preset
    paths = OutputPaths(
        result_root = result_root,
        artifact_dir = result_root / "artifacts",
        metrics_dir = result_root / "metrics",
        figures_dir = result_root / "figures",
        logs_dir = result_root / "logs",
        data_root = data_root
    )
    for path in [paths.result_root, paths.artifact_dir, paths.metrics_dir, paths.figures_dir, paths.logs_dir, paths.data_root]:
        path.mkdir(parents = True, exist_ok = True)
    save_json(paths.result_root / "run_config.json", config_to_dict(config))
    return paths


def resolve_device(device_name: str) -> torch.device:
    """Resolve a user-facing device string into a torch device.

    Parameters:
        device_name: Requested device name from the CLI.
    """

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but no CUDA device is available in the current environment.")
    return torch.device(device_name)


def json_ready(value: Any) -> Any:
    """Convert nested structures into JSON-friendly values.

    Parameters:
        value: Arbitrary Python object to normalize.
    """

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload to disk with UTF-8 encoding.

    Parameters:
        path: Target file path.
        payload: JSON-compatible payload.
    """

    path.parent.mkdir(parents = True, exist_ok = True)
    with path.open("w", encoding = "utf-8") as handle:
        json.dump(json_ready(payload), handle, indent = 2, ensure_ascii = False)


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file from disk.

    Parameters:
        path: Source JSON file path.
    """

    with path.open("r", encoding = "utf-8") as handle:
        return json.load(handle)


def save_torch(path: Path, payload: Dict[str, Any]) -> None:
    """Serialize a Python payload with `torch.save`.

    Parameters:
        path: Target file path.
        payload: Payload to serialize.
    """

    path.parent.mkdir(parents = True, exist_ok = True)
    torch.save(payload, path)


def load_torch(path: Path) -> Dict[str, Any]:
    """Load a `torch.save` payload from disk.

    Parameters:
        path: Source file path.
    """

    return torch.load(path, map_location = "cpu")
