import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

sys.path.append(os.getcwd())

from experiments.configs import build_config
from experiments.utils.io_utils import build_output_paths, resolve_device
from experiments.runners import RUNNER_REGISTRY

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for the unified experiment CLI.

    Parameters:
        None: Arguments are read from `sys.argv`.
    """

    parser = argparse.ArgumentParser(description = "Unified CLI for LinearTransformer reproduction experiments.")
    parser.add_argument(
        "--experiment",
        required = True,
        choices = ["simple", "rotation_adam", "rotation_adam_p0", "rotation_lbfgs", "variable_l", "variable_n"],
        help = "Experiment runner to execute."
    )
    parser.add_argument(
        "--stage",
        default = "all",
        choices = ["train", "eval", "plot", "all"],
        help = "Execution stage to run."
    )
    parser.add_argument(
        "--preset",
        default = "paper",
        choices = ["paper", "smoke", "cpu"],
        help = "Runtime preset controlling scale and fidelity."
    )
    parser.add_argument(
        "--device",
        default = "auto",
        choices = ["auto", "cpu", "cuda"],
        help = "Requested compute device."
    )
    parser.add_argument(
        "--result-dir",
        default = "results",
        help = "Root directory used to store experiment outputs."
    )
    parser.add_argument(
        "--data-dir",
        default = "data",
        help = "Root directory used to store generated data assets."
    )
    parser.add_argument(
        "--seeds",
        nargs = "+",
        type = int,
        default = None,
        help = "Optional explicit seed list override."
    )
    parser.add_argument(
        "--max-iters",
        type = int,
        default = None,
        help = "Optional override for the number of training iterations."
    )
    parser.add_argument(
        "--batch-size",
        type = int,
        default = None,
        help = "Optional override for the training batch size."
    )
    parser.add_argument(
        "--stride",
        type = int,
        default = None,
        help = "Optional override for checkpoint and log stride."
    )
    parser.add_argument(
        "--checkpoint-stride",
        type = int,
        default = None,
        help = "Optional override for checkpoint and curve-recording stride."
    )
    parser.add_argument(
        "--log-stride",
        type = int,
        default = None,
        help = "Optional override for runtime log stride."
    )
    return parser.parse_args()


def main():
    """Build config, dispatch the selected runner, and report output paths.

    Parameters:
        None: Runtime state is read from parsed CLI arguments.
    """

    args = parse_args()
    logs_dir = Path(args.result_dir) / args.experiment / args.preset / "logs"
    configure_experiment_logging(
        logs_dir = logs_dir,
        experiment = args.experiment,
        preset = args.preset,
        stage = args.stage
    )
    logger.info("=" * 80)
    logger.info("CLI arguments parsed successfully")
    logger.info("=" * 80)

    config = build_config(
        experiment = args.experiment,
        stage = args.stage,
        preset = args.preset,
        device = args.device,
        result_dir = args.result_dir,
        data_dir = args.data_dir,
        max_iters = args.max_iters,
        batch_size = args.batch_size,
        stride = args.stride,
        checkpoint_stride = args.checkpoint_stride,
        log_stride = args.log_stride,
        seeds = args.seeds
    )
    output_paths = build_output_paths(config)
    resolved_device = resolve_device(config.device)
    logger.info("=" * 80)
    logger.info("LinearTransformer reproduction CLI")
    logger.info("=" * 80)
    logger.info("Experiment: %s", config.experiment)
    logger.info("Stage: %s", config.stage)
    logger.info("Preset: %s", config.preset)
    logger.info("Requested device: %s", config.device)
    logger.info("Resolved device: %s", resolved_device)
    logger.info("Result root: %s", output_paths.result_root)
    logger.info("Logs dir: %s", output_paths.logs_dir)
    if config.preset == "paper" and resolved_device.type == "cpu":
        logger.warning("The paper preset is CPU-heavy in the current environment because CUDA is unavailable.")

    try:
        runner_cls = RUNNER_REGISTRY[config.experiment]
        runner = runner_cls(config)
        results = runner.run()
        logger.info("-" * 80)
        logger.info("Artifacts: %s", results["train_artifact_path"])
        logger.info("Metrics JSON: %s", results["metrics_json_path"])
        logger.info("Metrics Torch: %s", results["metrics_torch_path"])
        logger.info("Figures: %s", results["figures_dir"])
        logger.info("Logs: %s", results["logs_dir"])
        logger.info("Data assets: %s", results["data_root"])
        logger.info("Status: SUCCESS")
        logger.info("-" * 80)
    except Exception:
        logger.exception("Experiment run failed.")
        raise


def configure_experiment_logging(logs_dir, experiment: str, preset: str, stage: str) -> None:
    """Configure console and file logging for one experiment run.

    Parameters:
        logs_dir: Directory used to store persisted log files.
        experiment: Experiment name.
        preset: Runtime preset name.
        stage: Requested execution stage.
    """

    logs_dir.mkdir(parents = True, exist_ok = True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = logs_dir / f"{experiment}_{preset}_{stage}_{timestamp}.log"
    latest_log_path = logs_dir / "latest.log"
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(run_log_path, mode = "w", encoding = "utf-8"),
        logging.FileHandler(latest_log_path, mode = "w", encoding = "utf-8"),
    ]
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    logger.info("Persisting runtime logs to %s", run_log_path)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
