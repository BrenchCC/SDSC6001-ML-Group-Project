"""Backward-compatible IO shim.

Prefer importing from `experiments.utils.io_utils`.
"""

from experiments.utils.io_utils import OutputPaths, build_output_paths, json_ready, load_json, load_torch, resolve_device, save_json, save_torch

__all__ = [
    "OutputPaths",
    "build_output_paths",
    "json_ready",
    "load_json",
    "load_torch",
    "resolve_device",
    "save_json",
    "save_torch",
]
