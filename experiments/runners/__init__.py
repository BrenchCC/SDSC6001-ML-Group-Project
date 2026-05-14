"""Experiment runner registry."""

import os
import sys

sys.path.append(os.getcwd())

from experiments.runners.simple import SimpleRunner
from experiments.runners.variable_l import VariableLRunner
from experiments.runners.variable_n import VariableNRunner
from experiments.runners.rotation_adam import RotationAdamRunner
from experiments.runners.rotation_lbfgs import RotationLbfgsRunner
from experiments.runners.rotation_adam_p0 import RotationAdamP0Runner

RUNNER_REGISTRY = {
    "simple": SimpleRunner,
    "variable_l": VariableLRunner,
    "variable_n": VariableNRunner,
    "rotation_adam": RotationAdamRunner,
    "rotation_lbfgs": RotationLbfgsRunner,
    "rotation_adam_p0": RotationAdamP0Runner,
}
