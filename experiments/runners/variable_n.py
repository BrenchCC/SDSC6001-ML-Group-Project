import os
import sys
import csv
import json
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


class VariableNRunner(BaseExperimentRunner):
    """Runner for the context-length sweep experiment.

    Parameters:
        None: The runner reads its state from `self.config`.
    """

    def train_stage(self) -> Dict[str, Any]:
        """Train transformer models for each seed and context length.

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
            "contexts": self.config.variable_contexts,
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
            context_artifacts = {}
            for context_length in self.config.variable_contexts:
                original_context_length = self.config.context_length
                self.config.context_length = context_length
                context_key = f"N_{context_length}"
                context_artifacts[context_key] = train_seed_model(
                    config = self.config,
                    device = device,
                    seed = seed,
                    progress_label = f"{self.config.experiment}:{seed_key}:{context_key}",
                    U = rotation_bundle["U"],
                    D = rotation_bundle["D"]
                )
                val_bundle = build_eval_bundle(
                    mode = self.config.mode,
                    context_length = context_length,
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
                    context_length = context_length,
                    dimension = self.config.dimension,
                    batch_size = self.config.eval_batch_size,
                    shape_k = self.config.shape_k,
                    seed = 99,
                    device = device,
                    U = rotation_bundle["U"],
                    D = rotation_bundle["D"]
                )
                save_data_asset(self.paths.data_root / f"{seed_key}_{context_key}_validation.pt", val_bundle)
                save_data_asset(self.paths.data_root / f"{seed_key}_{context_key}_test.pt", test_bundle)
                self.config.context_length = original_context_length
            seed_artifacts[seed_key] = {
                "U": rotation_bundle["U"].detach().cpu(),
                "D": rotation_bundle["D"].detach().cpu(),
                "context_artifacts": context_artifacts,
            }
        return {"seed_artifacts": seed_artifacts}

    def eval_stage(self, train_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate transformer checkpoints and baseline methods across context lengths.

        Parameters:
            train_artifacts: Training artifacts produced by `train_stage`.
        """

        device = resolve_device(self.config.device)
        transformer_losses = {}
        gd_losses = {f"seed_{seed}": [] for seed in self.config.seeds}
        pgd_losses = {f"seed_{seed}": [] for seed in self.config.seeds}
        ols_losses = {f"seed_{seed}": [] for seed in self.config.seeds}
        reference_seed_key = f"seed_{self.config.seeds[0]}"
        reference_rotation = train_artifacts["seed_artifacts"][reference_seed_key]

        gd_best_etas = {}
        pgd_best_etas = {}
        for context_length in self.config.variable_contexts:
            context_key = f"N_{context_length}"
            reference_val = torch.load(self.paths.data_root / f"{reference_seed_key}_{context_key}_validation.pt", map_location = "cpu")
            best_eta_gd, _ = find_best_eta(
                Z = reference_val["Z"],
                y = reference_val["y"],
                method_name = "gd",
                eta_grid = self.config.eta_grid_gd,
                dimension = self.config.dimension,
                numstep = self.config.n_layer,
                device = device,
                max_samples = self.config.baseline_eval_size
            )
            best_eta_pgd, _ = find_best_eta(
                Z = reference_val["Z"],
                y = reference_val["y"],
                method_name = "pgd",
                eta_grid = self.config.eta_grid_pgd,
                dimension = self.config.dimension,
                numstep = self.config.n_layer,
                device = device,
                U = reference_rotation["U"],
                D = reference_rotation["D"],
                max_samples = self.config.baseline_eval_size
            )
            gd_best_etas[context_length] = best_eta_gd
            pgd_best_etas[context_length] = best_eta_pgd

        for seed in self.config.seeds:
            seed_key = f"seed_{seed}"
            transformer_losses[seed_key] = []
            seed_payload = train_artifacts["seed_artifacts"][seed_key]
            for context_length in self.config.variable_contexts:
                context_key = f"N_{context_length}"
                history = seed_payload["context_artifacts"][context_key]["history"]
                val_bundle = torch.load(self.paths.data_root / f"{seed_key}_{context_key}_validation.pt", map_location = "cpu")
                test_bundle = torch.load(self.paths.data_root / f"{seed_key}_{context_key}_test.pt", map_location = "cpu")
                validation_losses = evaluate_history_losses(
                    history = history,
                    eval_Z = val_bundle["Z"],
                    eval_y = val_bundle["y"],
                    model_builder = lambda: make_model(self.config),
                    device = device
                )
                best_index = select_best_checkpoint(validation_losses, tail_window = min(10, len(validation_losses)))
                model = make_model(self.config).to(device)
                with torch.no_grad():
                    model.allparam.copy_(history[best_index].to(device))
                test_loss = in_context_loss(model, test_bundle["Z"].to(device), test_bundle["y"].to(device))
                transformer_losses[seed_key].append(float(test_loss.item()))

                gd_losses[seed_key].append(evaluate_baseline_loss(
                    Z = test_bundle["Z"],
                    y = test_bundle["y"],
                    method_name = "gd",
                    dimension = self.config.dimension,
                    numstep = self.config.n_layer,
                    eta = gd_best_etas[context_length],
                    device = device,
                    max_samples = self.config.baseline_eval_size
                ))
                pgd_losses[seed_key].append(evaluate_baseline_loss(
                    Z = test_bundle["Z"],
                    y = test_bundle["y"],
                    method_name = "pgd",
                    dimension = self.config.dimension,
                    numstep = self.config.n_layer,
                    eta = pgd_best_etas[context_length],
                    device = device,
                    U = seed_payload["U"],
                    D = seed_payload["D"],
                    max_samples = self.config.baseline_eval_size
                ))
                ols_losses[seed_key].append(evaluate_baseline_loss(
                    Z = test_bundle["Z"],
                    y = test_bundle["y"],
                    method_name = "ols",
                    dimension = self.config.dimension,
                    numstep = self.config.n_layer,
                    eta = None,
                    device = device,
                    U = seed_payload["U"],
                    D = seed_payload["D"],
                    max_samples = self.config.baseline_eval_size
                ))

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
            "x_values": self.config.variable_contexts,
            "transformer_losses": transformer_losses,
            "gd_losses": gd_losses,
            "pgd_losses": pgd_losses,
            "ols_losses": ols_losses,
            "transformer_summary": summarize(transformer_losses),
            "gd_summary": summarize(gd_losses),
            "pgd_summary": summarize(pgd_losses),
            "ols_summary": summarize(ols_losses),
            "gd_best_etas": gd_best_etas,
            "pgd_best_etas": pgd_best_etas,
        }

    def plot_stage(self, train_artifacts: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        """Render the context-length sweep comparison figure.

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
                    "label": "3-Step GD",
                },
                {
                    "mean": metrics["pgd_summary"]["mean"],
                    "std": metrics["pgd_summary"]["std"],
                    "color": "green",
                    "label": "3-Step Preconditioned GD",
                },
                {
                    "mean": metrics["transformer_summary"]["mean"],
                    "std": metrics["transformer_summary"]["std"],
                    "color": "red",
                    "label": "3-Layer Linear Transformer",
                },
                {
                    "mean": metrics["ols_summary"]["mean"],
                    "std": metrics["ols_summary"]["std"],
                    "color": "purple",
                    "label": "OLS",
                },
            ],
            x_label = "Number of ICL Examples",
            y_label = "Loss",
            basename = "3-step-variable-N-plot",
            output_dir = self.paths.figures_dir,
            dpi = self.config.line_dpi
        )
        extension_artifacts = write_extension_artifacts(
            metrics = metrics,
            figure_paths = figure_paths,
            output_dir = self.paths.figures_dir,
            seeds = self.config.seeds
        )
        figure_index = [{
            "experiment": self.config.experiment,
            "basename": "3-step-variable-N-plot",
            "paper_use": "Context-length sweep loss comparison",
            "paths": figure_paths,
        }]
        figure_index.append({
            "experiment": self.config.experiment,
            "basename": "extension_variable_n_analysis_package",
            "paper_use": "Report-ready CSV tables and Chinese analysis for the context-length extension experiment",
            "paths": extension_artifacts,
        })
        write_figure_index(self.paths.figures_dir / "figure_index.json", figure_index)


def write_extension_artifacts(
    metrics: Dict[str, Any],
    figure_paths: Dict[str, str],
    output_dir,
    seeds: List[int]
) -> Dict[str, str]:
    """Write report-ready CSV, JSON, and Markdown artifacts for the extension experiment.

    Parameters:
        metrics: Evaluation metrics produced by `eval_stage`.
        figure_paths: Paths returned by the main context-length plot.
        output_dir: Directory where extension artifacts should be written.
        seeds: Seeds used by the current run.
    """

    summary_path = output_dir / "extension_variable_n_summary.csv"
    relative_path = output_dir / "extension_variable_n_relative.csv"
    report_path = output_dir / "extension_variable_n_report.md"
    index_path = output_dir / "extension_variable_n_artifact_index.json"

    summary_rows = build_summary_rows(metrics)
    relative_rows = build_relative_rows(summary_rows)
    write_csv(
        path = summary_path,
        rows = summary_rows,
        fieldnames = [
            "context_length",
            "transformer_mean",
            "transformer_std",
            "gd_mean",
            "gd_std",
            "pgd_mean",
            "pgd_std",
            "ols_mean",
            "ols_std",
        ]
    )
    write_csv(
        path = relative_path,
        rows = relative_rows,
        fieldnames = [
            "context_length",
            "transformer_over_gd",
            "transformer_over_pgd",
            "transformer_over_ols",
            "transformer_improvement_vs_gd",
            "transformer_improvement_vs_pgd",
            "transformer_improvement_vs_ols",
        ]
    )
    report_path.write_text(
        build_extension_report(
            summary_rows = summary_rows,
            relative_rows = relative_rows,
            figure_paths = figure_paths,
            seeds = seeds,
            metrics = metrics
        ),
        encoding = "utf-8"
    )

    artifact_index = {
        "main_plot": figure_paths,
        "summary_csv": str(summary_path),
        "relative_csv": str(relative_path),
        "analysis_report": str(report_path),
        "contexts": metrics["x_values"],
        "seeds": seeds,
    }
    with index_path.open("w", encoding = "utf-8") as handle:
        json.dump(artifact_index, handle, indent = 2, ensure_ascii = False)

    return {
        "summary_csv": str(summary_path),
        "relative_csv": str(relative_path),
        "analysis_report": str(report_path),
        "artifact_index": str(index_path),
    }


def build_summary_rows(metrics: Dict[str, Any]) -> List[Dict[str, float]]:
    """Build one summary row per context length.

    Parameters:
        metrics: Evaluation metrics produced by `eval_stage`.
    """

    rows = []
    for index, context_length in enumerate(metrics["x_values"]):
        rows.append({
            "context_length": int(context_length),
            "transformer_mean": float(metrics["transformer_summary"]["mean"][index]),
            "transformer_std": float(metrics["transformer_summary"]["std"][index]),
            "gd_mean": float(metrics["gd_summary"]["mean"][index]),
            "gd_std": float(metrics["gd_summary"]["std"][index]),
            "pgd_mean": float(metrics["pgd_summary"]["mean"][index]),
            "pgd_std": float(metrics["pgd_summary"]["std"][index]),
            "ols_mean": float(metrics["ols_summary"]["mean"][index]),
            "ols_std": float(metrics["ols_summary"]["std"][index]),
        })
    return rows


def build_relative_rows(summary_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Build relative Transformer performance rows.

    Parameters:
        summary_rows: Rows returned by `build_summary_rows`.
    """

    rows = []
    for row in summary_rows:
        transformer_mean = row["transformer_mean"]
        gd_mean = row["gd_mean"]
        pgd_mean = row["pgd_mean"]
        ols_mean = row["ols_mean"]
        rows.append({
            "context_length": row["context_length"],
            "transformer_over_gd": safe_ratio(transformer_mean, gd_mean),
            "transformer_over_pgd": safe_ratio(transformer_mean, pgd_mean),
            "transformer_over_ols": safe_ratio(transformer_mean, ols_mean),
            "transformer_improvement_vs_gd": safe_improvement(gd_mean, transformer_mean),
            "transformer_improvement_vs_pgd": safe_improvement(pgd_mean, transformer_mean),
            "transformer_improvement_vs_ols": safe_improvement(ols_mean, transformer_mean),
        })
    return rows


def safe_ratio(numerator: float, denominator: float) -> float:
    """Compute a ratio with a zero-denominator guard.

    Parameters:
        numerator: Numerator value.
        denominator: Denominator value.
    """

    if abs(denominator) < 1e-8:
        return float("nan")
    return float(numerator / denominator)


def safe_improvement(reference: float, candidate: float) -> float:
    """Compute relative improvement against a reference loss.

    Parameters:
        reference: Reference method loss.
        candidate: Candidate method loss.
    """

    if abs(reference) < 1e-8:
        return float("nan")
    return float((reference - candidate) / reference)


def write_csv(path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    """Write rows to a CSV file.

    Parameters:
        path: Target CSV path.
        rows: Row dictionaries to write.
        fieldnames: CSV header order.
    """

    with path.open("w", encoding = "utf-8", newline = "") as handle:
        writer = csv.DictWriter(handle, fieldnames = fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_extension_report(
    summary_rows: List[Dict[str, float]],
    relative_rows: List[Dict[str, float]],
    figure_paths: Dict[str, str],
    seeds: List[int],
    metrics: Dict[str, Any]
) -> str:
    """Build a Chinese Markdown analysis report for the extension experiment.

    Parameters:
        summary_rows: Summary loss rows.
        relative_rows: Relative performance rows.
        figure_paths: Paths returned by the main context-length plot.
        seeds: Seeds used by the current run.
        metrics: Evaluation metrics produced by `eval_stage`.
    """

    methods = {
        "Linear Transformer": "transformer_mean",
        "GD": "gd_mean",
        "Preconditioned GD": "pgd_mean",
        "OLS": "ols_mean",
    }
    first_row = summary_rows[0]
    last_row = summary_rows[-1]
    trend_lines = []
    for method_name, key in methods.items():
        first_loss = first_row[key]
        last_loss = last_row[key]
        trend_lines.append(
            f"- {method_name}: N={first_row['context_length']} 时 loss={first_loss:.6f}，"
            f"N={last_row['context_length']} 时 loss={last_loss:.6f}，变化率={safe_improvement(first_loss, last_loss):.2%}。"
        )

    best_methods = []
    for row in summary_rows:
        losses = {
            "Linear Transformer": row["transformer_mean"],
            "GD": row["gd_mean"],
            "Preconditioned GD": row["pgd_mean"],
            "OLS": row["ols_mean"],
        }
        best_method = min(losses, key = losses.get)
        best_methods.append(f"N={row['context_length']}: {best_method} ({losses[best_method]:.6f})")

    final_relative = relative_rows[-1]
    transformer_vs_pgd = final_relative["transformer_over_pgd"]
    transformer_vs_gd = final_relative["transformer_over_gd"]
    monotonic_note = build_monotonic_note(summary_rows)

    return "\n".join([
        "# 上下文长度 N 扩展实验分析",
        "",
        "## 实验问题",
        "本扩展实验考察 in-context examples 数量增加时，Linear Transformer、普通 GD、预条件 GD 和 OLS 的测试损失如何变化。该问题对应课程项目中的样本复杂度/估计误差视角：上下文样本越多，模型可用于恢复线性任务的信息越充分，理论上测试误差应整体下降。",
        "",
        "## 实验设置",
        f"- 上下文长度: {metrics['x_values']}",
        f"- 随机种子: {seeds}",
        "- 方法: 3-layer Linear Transformer、3-step GD、3-step Preconditioned GD、OLS",
        f"- 主图 PNG: `{figure_paths['png']}`",
        f"- 主图 PDF: `{figure_paths['pdf']}`",
        "",
        "## 主要观察",
        *trend_lines,
        f"- 最后一个上下文长度下，Transformer/GD loss ratio = {transformer_vs_gd:.4f}。",
        f"- 最后一个上下文长度下，Transformer/PGD loss ratio = {transformer_vs_pgd:.4f}。",
        f"- 最优方法摘要: {'; '.join(best_methods)}。",
        f"- 单调性说明: {monotonic_note}",
        "",
        "## 可写入报告的分析段落",
        "从样本复杂度角度看，随着上下文样本数 N 增加，各方法获得更多线性回归任务信息，测试损失整体呈下降趋势。预条件 GD 显式使用协方差信息，因此通常比普通 GD 更适合非各向同性输入分布；Linear Transformer 的表现则反映训练后模型能否从上下文中学习接近梯度法的更新机制。OLS 作为闭式解参考，提供了在当前有限样本设定下的强基线。由于本实验受到本地 CPU 预算限制，训练迭代、batch size 和 seed 数均小于原论文规模，因此结论应以趋势分析和方法对比为主，而不是宣称完全复现 paper-scale 数值。",
        "",
        "## CPU 小规模实验限制",
        "本实验默认只使用 2 个 seed，并采用 CPU-friendly batch size 与迭代数。若某些曲线存在局部波动，应解释为训练预算与随机性共同导致的现象；报告中应保留图表并如实说明趋势是否充分。",
        "",
    ])


def build_monotonic_note(summary_rows: List[Dict[str, float]]) -> str:
    """Build a short note about whether losses decrease monotonically.

    Parameters:
        summary_rows: Summary loss rows.
    """

    labels = [
        ("Linear Transformer", "transformer_mean"),
        ("GD", "gd_mean"),
        ("Preconditioned GD", "pgd_mean"),
        ("OLS", "ols_mean"),
    ]
    notes = []
    for label, key in labels:
        values = [row[key] for row in summary_rows]
        is_monotonic = all(values[index] <= values[index - 1] for index in range(1, len(values)))
        if is_monotonic:
            notes.append(f"{label} 单调下降")
        else:
            notes.append(f"{label} 存在局部波动")
    return "；".join(notes)
