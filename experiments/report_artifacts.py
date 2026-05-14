import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments for report artifact indexing.

    Parameters:
        None: Arguments are read from `sys.argv`.
    """

    parser = argparse.ArgumentParser(description = "Build a report-ready artifact index for CPU reproduction runs.")
    parser.add_argument(
        "--result-dir",
        default = "results_cpu",
        help = "Root directory that contains CPU experiment outputs."
    )
    parser.add_argument(
        "--preset",
        default = "cpu",
        help = "Preset directory name used by the experiment outputs."
    )
    parser.add_argument(
        "--seeds",
        nargs = "+",
        type = int,
        default = [0, 1],
        help = "Seeds used by the CPU reproduction run."
    )
    return parser.parse_args()


def main():
    """Build the top-level report artifact Markdown file.

    Parameters:
        None: Runtime state is read from parsed CLI arguments.
    """

    args = parse_args()
    result_root = Path(args.result_dir)
    report_path = result_root / "report_artifacts.md"
    lines = build_report_lines(
        result_root = result_root,
        preset = args.preset,
        seeds = args.seeds
    )
    result_root.mkdir(parents = True, exist_ok = True)
    report_path.write_text("\n".join(lines), encoding = "utf-8")
    logger.info("Report artifact index written to %s", report_path)
    print(report_path)


def build_report_lines(result_root: Path, preset: str, seeds: List[int]) -> List[str]:
    """Build Markdown lines for all report-ready artifacts.

    Parameters:
        result_root: Root directory that contains experiment outputs.
        preset: Preset directory name.
        seeds: Seeds used by the CPU reproduction run.
    """

    theorem3_root = result_root / "rotation_adam_p0" / preset
    theorem4_root = result_root / "rotation_adam" / preset
    extension_root = result_root / "variable_n" / preset
    smooth_index_path = result_root / "smoothed_curve_artifacts.json"
    theorem3_metrics = load_json(theorem3_root / "metrics" / "metrics.json")
    theorem4_metrics = load_json(theorem4_root / "metrics" / "metrics.json")
    extension_metrics = load_json(extension_root / "metrics" / "metrics.json")

    lines = [
        "# CPU 复现实验报告素材清单",
        "",
        "## 运行配置",
        f"- Preset: `{preset}`",
        f"- Seeds: `{seeds}`",
        "- 运行环境: `conda run -n cs_hw python ...`",
        "- 说明: 本素材包面向本地 Mac CPU 预算，重点用于趋势验证和报告截图，不等同于 paper-scale 训练。",
        "",
        "## 报告第 4 部分建议插图",
        "",
        "### Theorem 3：稀疏约束下的预条件梯度下降结构",
        artifact_item(
            path = theorem3_root / "figures" / "rotation_demonstration_adam_pnull_loss_plot.png",
            caption = "Theorem 3 CPU 复现的 log loss 曲线，用于展示训练误差整体下降。"
        ),
        artifact_item(
            path = theorem3_root / "figures" / "rotation_demonstration_dist_to_id_adam_pnull_A0.png",
            caption = "A0 的 raw/rotated distance 对比，用于说明旋转后的矩阵更接近缩放单位阵。"
        ),
        artifact_item(
            path = theorem3_root / "figures" / "rotation_demonstration_pnull_A0.png",
            caption = "训练结束后 Σ^{1/2}A0Σ^{1/2} 的热力图，用于展示近似对角结构。"
        ),
        "",
        "### Theorem 4：放宽参数后的特征变换结构",
        artifact_item(
            path = theorem4_root / "figures" / "rotation_demonstration_adam_loss_plot.png",
            caption = "Theorem 4 CPU 复现的 log loss 曲线，用于展示放宽参数设置下的训练趋势。"
        ),
        artifact_item(
            path = theorem4_root / "figures" / "rotation_demonstration_dist_to_id_adam.png",
            caption = "B_i 与 rotated A_i 的 distance panel，用于展示 B_i 接近 identity 且 A_i 接近 Σ^{-1}。"
        ),
        artifact_item(
            path = theorem4_root / "figures" / "rotation_demonstration_adam_B0.png",
            caption = "训练结束后 B0 的热力图，用于展示特征变换矩阵接近缩放单位阵。"
        ),
        "",
        "### 扩展实验：上下文长度 N 与样本复杂度",
        artifact_item(
            path = extension_root / "figures" / "3-step-variable-N-plot.png",
            caption = "不同上下文长度 N 下 Transformer/GD/PGD/OLS 的测试 loss 对比。"
        ),
        artifact_item(
            path = extension_root / "figures" / "extension_variable_n_summary.csv",
            caption = "扩展实验数值表，包含每个 N 下四类方法的 mean/std loss。"
        ),
        artifact_item(
            path = extension_root / "figures" / "extension_variable_n_report.md",
            caption = "扩展实验中文自动分析，可直接改写进报告。"
        ),
        "",
        "### 平滑曲线补充素材",
        artifact_item(
            path = smooth_index_path,
            caption = "平滑曲线索引；若报告截图希望更缓和，可优先使用 `figures_smoothed` 中的 PNG/PDF。"
        ),
        "",
        "## 核心指标摘要",
        "",
        "### Theorem 3 指标",
        *build_theorem3_summary(theorem3_metrics),
        "",
        "### Theorem 4 指标",
        *build_theorem4_summary(theorem4_metrics),
        "",
        "### 扩展实验指标",
        *build_extension_summary(extension_metrics),
        "",
        "## 报告写作提醒",
        "- 若某些曲线没有完全单调下降，报告中按 CPU 小规模复现实验如实说明，不要写成 paper-scale 结论。",
        "- Theorem 3 的关键表述是 rotated A_i 比 raw A_i 更符合 identity 结构。",
        "- Theorem 4 的关键表述是 B_i 接近 identity，rotated A_i 接近 identity，对应 A_i 接近 Σ^{-1}。",
        "- 扩展实验用于补充样本复杂度视角：N 增大通常带来更多任务信息，使测试 loss 呈下降趋势或总体改善。",
        "",
    ]
    return lines


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file if it exists.

    Parameters:
        path: JSON file path.
    """

    if not path.exists():
        return {}
    with path.open("r", encoding = "utf-8") as handle:
        return json.load(handle)


def artifact_item(path: Path, caption: str) -> str:
    """Build one Markdown artifact line.

    Parameters:
        path: Artifact path.
        caption: Suggested report caption.
    """

    status = "exists" if path.exists() else "missing"
    return f"- `{path}` ({status})：{caption}"


def build_theorem3_summary(metrics: Dict[str, Any]) -> List[str]:
    """Build Theorem 3 metric summary lines.

    Parameters:
        metrics: Parsed metrics JSON for `rotation_adam_p0`.
    """

    if not metrics:
        return ["- Metrics missing. Run `rotation_adam_p0 --preset cpu` first."]
    lines = [
        summarize_curve("log loss", metrics.get("loss_summary_log", {}).get("mean", [])),
    ]
    rotated = metrics.get("rotated_distance_summary", {})
    raw = metrics.get("raw_distance_summary", {})
    for matrix_name in ["A0", "A1", "A2"]:
        lines.append(summarize_curve(f"rotated {matrix_name} distance", rotated.get(matrix_name, {}).get("mean", [])))
        lines.append(summarize_curve(f"raw {matrix_name} distance", raw.get(matrix_name, {}).get("mean", [])))
    return lines


def build_theorem4_summary(metrics: Dict[str, Any]) -> List[str]:
    """Build Theorem 4 metric summary lines.

    Parameters:
        metrics: Parsed metrics JSON for `rotation_adam`.
    """

    if not metrics:
        return ["- Metrics missing. Run `rotation_adam --preset cpu` first."]
    lines = [
        summarize_curve("log loss", metrics.get("loss_summary_log", {}).get("mean", [])),
    ]
    distance = metrics.get("distance_summary", {})
    for matrix_name in ["B0", "B1", "A0", "A1", "A2"]:
        lines.append(summarize_curve(f"{matrix_name} distance", distance.get(matrix_name, {}).get("mean", [])))
    return lines


def build_extension_summary(metrics: Dict[str, Any]) -> List[str]:
    """Build extension experiment metric summary lines.

    Parameters:
        metrics: Parsed metrics JSON for `variable_n`.
    """

    if not metrics:
        return ["- Metrics missing. Run `variable_n --preset cpu` first."]
    lines = [f"- Context lengths: {metrics.get('x_values', [])}"]
    for label, key in [
        ("Transformer", "transformer_summary"),
        ("GD", "gd_summary"),
        ("Preconditioned GD", "pgd_summary"),
        ("OLS", "ols_summary"),
    ]:
        lines.append(summarize_curve(f"{label} loss", metrics.get(key, {}).get("mean", [])))
    return lines


def summarize_curve(name: str, values: List[float]) -> str:
    """Summarize first and final values of one metric curve.

    Parameters:
        name: Human-readable curve name.
        values: Numeric curve values.
    """

    if not values:
        return f"- {name}: missing"
    first_value = float(values[0])
    final_value = float(values[-1])
    delta = final_value - first_value
    return f"- {name}: first={first_value:.6f}, final={final_value:.6f}, delta={delta:.6f}"


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
