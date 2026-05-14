"""Backward-compatible config shim.

Prefer importing from `experiments.configs`.
"""

from experiments.configs import ExperimentConfig, build_config, config_to_dict, get_preset_table

__all__ = [
    "ExperimentConfig",
    "build_config",
    "config_to_dict",
    "get_preset_table",
]
