import os
import sys
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.append(os.getcwd())

from experiments.configs import ExperimentConfig
from experiments.utils.io_utils import build_output_paths, load_torch, save_json, save_torch

logger = logging.getLogger(__name__)


class BaseExperimentRunner(ABC):
    """Base class that standardizes stage handling and filesystem layout.

    Parameters:
        config: Expanded experiment configuration.
    """

    def __init__(self, config: ExperimentConfig):
        """Store config and resolve output paths.

        Parameters:
            config: Expanded experiment configuration.
        """

        self.config = config
        self.paths = build_output_paths(config)

    def run(self) -> Dict[str, Any]:
        """Execute the requested stage sequence for one experiment.

        Parameters:
            None: This method does not accept runtime parameters.
        """

        train_artifacts = None
        metrics = None
        stage_start_time = time.time()
        logger.info("=" * 80)
        logger.info("Runner started: experiment = %s | stage = %s", self.config.experiment, self.config.stage)
        logger.info("=" * 80)
        if self.config.stage in {"train", "all"}:
            logger.info("-" * 80)
            logger.info("Stage started: train")
            logger.info("-" * 80)
            train_artifacts = self.train_stage()
            save_torch(self.train_artifact_path, train_artifacts)
            logger.info("Stage finished: train | artifact = %s", self.train_artifact_path)
        else:
            logger.info("Skipping train stage. Loading cached artifact from %s", self.train_artifact_path)
            train_artifacts = load_torch(self.train_artifact_path)

        if self.config.stage in {"eval", "all"}:
            logger.info("-" * 80)
            logger.info("Stage started: eval")
            logger.info("-" * 80)
            metrics = self.eval_stage(train_artifacts)
            save_torch(self.metrics_torch_path, metrics)
            save_json(self.metrics_json_path, metrics)
            logger.info("Stage finished: eval | metrics = %s", self.metrics_json_path)
        elif self.config.stage == "plot":
            logger.info("Skipping eval stage. Loading cached metrics from %s", self.metrics_torch_path)
            metrics = load_torch(self.metrics_torch_path)

        if self.config.stage in {"plot", "all"}:
            if metrics is None:
                metrics = load_torch(self.metrics_torch_path)
            logger.info("-" * 80)
            logger.info("Stage started: plot")
            logger.info("-" * 80)
            self.plot_stage(train_artifacts, metrics)
            logger.info("Stage finished: plot | figures_dir = %s", self.paths.figures_dir)

        logger.info(
            "=" * 80
        )
        logger.info(
            "Runner finished: experiment = %s | elapsed_time = %.2fs",
            self.config.experiment,
            time.time() - stage_start_time
        )
        logger.info("=" * 80)

        return {
            "train_artifact_path": str(self.train_artifact_path),
            "metrics_json_path": str(self.metrics_json_path),
            "metrics_torch_path": str(self.metrics_torch_path),
            "figures_dir": str(self.paths.figures_dir),
            "logs_dir": str(self.paths.logs_dir),
            "data_root": str(self.paths.data_root),
        }

    @property
    def train_artifact_path(self) -> Path:
        """Return the path of the serialized training artifact file.

        Parameters:
            None: This property does not accept runtime parameters.
        """

        return self.paths.artifact_dir / "train_artifacts.pt"

    @property
    def metrics_torch_path(self) -> Path:
        """Return the path of the serialized metrics tensor file.

        Parameters:
            None: This property does not accept runtime parameters.
        """

        return self.paths.metrics_dir / "metrics.pt"

    @property
    def metrics_json_path(self) -> Path:
        """Return the path of the JSON metrics file.

        Parameters:
            None: This property does not accept runtime parameters.
        """

        return self.paths.metrics_dir / "metrics.json"

    @abstractmethod
    def train_stage(self) -> Dict[str, Any]:
        """Run the training stage.

        Parameters:
            None: The implementation reads its state from `self.config`.
        """

    @abstractmethod
    def eval_stage(self, train_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Run the evaluation stage.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
        """

    @abstractmethod
    def plot_stage(self, train_artifacts: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Run the plotting stage.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
            metrics: Evaluation metrics produced by `eval_stage`.
        """
