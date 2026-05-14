# LinearTransformer 复现实验完整指南

本文档面向其他小组成员或后续复现者，说明如何安装环境、理解代码结构、运行复现实验、查看实验产物，并解释每个实验配置与论文结论的对应关系。

复现对象是论文 *Transformers learn to implement preconditioned gradient descent for in-context learning* 中的线性 Transformer 实验。当前仓库已经把实验统一到 `main.py` 入口，并提供 CPU 友好的 `cpu` preset，适合本地 Mac CPU 生成课程报告所需的图、表和中文分析材料。

## 1. 环境安装

### 1.1 创建 Conda 环境

建议使用项目默认环境名 `cs_hw`：

```bash
conda create -n cs_hw python=3.11 -y
conda activate cs_hw
```

如果已经有同名环境，可直接激活：

```bash
conda activate cs_hw
```

### 1.2 安装依赖

仓库只需要少量 Python 包：

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 内容为：

```text
matplotlib==3.9.0
numpy==1.26.4
torch==2.3.1
tqdm==4.66.4
```

新增的报告索引、CSV、JSON、平滑曲线脚本只使用 Python 标准库和上述依赖，不需要额外安装包。

### 1.3 验证环境

```bash
python - <<'PY'
import matplotlib
import numpy
import torch
import tqdm

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("environment ok")
PY
```

在本地 Mac CPU 上，`torch.cuda.is_available()` 通常是 `False`，这不影响本项目的 CPU 复现配置。

## 2. 代码结构解读

### 2.1 统一入口

- `main.py`
  解析 CLI 参数，构建 `ExperimentConfig`，根据 `--experiment` 选择 runner，并执行 `train`、`eval`、`plot` 或 `all`。

典型执行链路：

```text
main.py
-> experiments.configs.build_config
-> experiments.runners.RUNNER_REGISTRY
-> runner.train_stage / eval_stage / plot_stage
-> results 和 data 目录落盘
```

### 2.2 配置系统

- `experiments/configs/runtime.py`
  定义 `ExperimentConfig` 和所有 preset。

当前支持三类 preset：

- `paper`
  尽量接近论文规模，batch 大、步数多，不推荐本地 CPU 直接跑。
- `smoke`
  快速检查代码能否跑通，步数很少，不能作为报告主结果。
- `cpu`
  本项目为本地 CPU 报告复现新增的折中配置，默认记录每个 iteration 的曲线点，日志仍每 100 步打印一次。

### 2.3 数据生成

- `experiments/data/generation.py`
  生成随机线性回归 in-context learning 任务。

核心对象：

- `Z`
  prompt tensor，包含上下文样本、query 样本和标签列。
- `y`
  query 的真实标签。
- `U, D`
  旋转协方差分解，用于构造非各向同性输入分布。

关键工程细节：

- query 标签在 prompt 中置零，避免标签泄露。
- 线性 Transformer 前向中有 `1 / context_length` 缩放。
- 训练 batch 可按 `resample_interval` 动态重采样。

### 2.4 模型

- `experiments/models/linear_transformer.py`

核心组件：

- `attention(P, Q, Z)`
  实现去掉 softmax 的线性注意力层。
- `TransformerF`
  多层 residual linear Transformer。可学习参数保存在 `allparam` 中，形状对应：

```text
(n_layer, n_head, 2, dimension, dimension)
```

其中 `2` 对应每层的 `P` 矩阵和 `Q` 矩阵。代码中的可视化命名通常把它们解释为 `B_i` 与 `A_i`。

- `in_context_loss`
  使用最终 query 位置的标签维度计算平方误差。由于论文输出定义为负号形式，代码中误差为：

```text
output[:, context_length, dimension] + y
```

### 2.5 训练与评估

- `experiments/core/training.py`
  通用训练循环、optimizer 构造、梯度裁剪、学习率衰减、checkpoint 保存。
- `experiments/core/analysis.py`
  评估历史 checkpoint、计算矩阵到缩放单位阵的距离、提取最终矩阵。
- `experiments/core/baselines.py`
  实现 GD、Preconditioned GD、OLS baseline。
- `experiments/core/plotting.py`
  统一导出 PNG/PDF 图片。

### 2.6 Runner

- `experiments/runners/rotation_adam_p0.py`
  Theorem 3 复现：约束 `P = 0`，验证 `A_i` 学到 `Σ^{-1}` 预条件结构。
- `experiments/runners/rotation_adam.py`
  Theorem 4 复现：放宽参数，验证 `A_i` 与 `B_i` 的临界点结构。
- `experiments/runners/variable_n.py`
  主扩展实验：上下文长度 `N` sweep，对比 Transformer、GD、PGD、OLS。
- `experiments/runners/variable_l.py`
  可选扩展实验：层数 `L` sweep，分析层数是否类似优化步数。
- `experiments/runners/simple.py`
  单位协方差 sanity check。
- `experiments/runners/rotation_lbfgs.py`
  LBFGS 版本的旋转协方差实验。

### 2.7 报告辅助脚本

- `experiments/report_artifacts.py`
  读取各实验 `metrics.json` 和图片路径，生成报告素材总览。
- `experiments/smooth_report_curves.py`
  从 `metrics.json` 重新绘制 EMA 平滑曲线，不修改原始图。
- `scripts/run_cpu_reproduction.sh`
  一键运行 CPU 复现实验。按项目约定，`scripts/` 目录只放 bash 脚本。

## 3. 一键 CPU 复现实验

### 3.1 默认运行

```bash
bash scripts/run_cpu_reproduction.sh
```

脚本实际行为：

- Conda 环境：默认 `cs_hw`
- 设备：强制 `cpu`
- preset：强制 `cpu`
- stage：`all`
- 默认 seed：`0 1`
- 默认结果目录：`results`
- 默认数据目录：`data`
- 顺序运行：
  - `rotation_adam_p0`
  - `rotation_adam`
  - `variable_n`
- 然后运行：
  - `experiments/smooth_report_curves.py`
  - `experiments/report_artifacts.py`

脚本结束后会打印：

```text
results/report_artifacts.md
results/smoothed_curve_artifacts.json
results/rotation_adam_p0/cpu/figures
results/rotation_adam/cpu/figures
results/variable_n/cpu/figures
```

### 3.2 只跑单个 seed

如果本地机器较慢，可只跑 `seed = 0`：

```bash
bash scripts/run_cpu_reproduction.sh 0
```

如果需要多个指定 seed：

```bash
bash scripts/run_cpu_reproduction.sh 0 1 2
```

### 3.3 改输出目录

脚本默认写入 `results/` 和 `data/`。如果希望和普通实验输出分开，可显式设置：

```bash
RESULT_DIR=results_cpu DATA_DIR=data_cpu bash scripts/run_cpu_reproduction.sh
```

如果只想快速生成一份试验结果：

```bash
RESULT_DIR=results_trial DATA_DIR=data_trial bash scripts/run_cpu_reproduction.sh 0
```

### 3.4 改 Conda 环境名

```bash
CONDA_ENV=my_env bash scripts/run_cpu_reproduction.sh
```

## 4. 统一 CLI 用法

基础命令：

```bash
python main.py \
  --experiment {simple,rotation_adam,rotation_adam_p0,rotation_lbfgs,variable_l,variable_n} \
  --stage {train,eval,plot,all} \
  --preset {paper,smoke,cpu} \
  --device {auto,cpu,cuda}
```

常用参数：

| 参数 | 含义 |
|---|---|
| `--experiment` | 选择实验 runner |
| `--stage` | `train`、`eval`、`plot` 或 `all` |
| `--preset` | `paper`、`smoke` 或 `cpu` |
| `--device` | `auto`、`cpu` 或 `cuda` |
| `--result-dir` | 结果根目录 |
| `--data-dir` | 数据根目录 |
| `--seeds` | 覆盖 seed 列表 |
| `--max-iters` | 覆盖训练步数 |
| `--batch-size` | 覆盖训练 batch size |
| `--stride` | 同时覆盖 checkpoint 和日志步长 |
| `--checkpoint-stride` | 只覆盖 checkpoint/曲线记录步长 |
| `--log-stride` | 只覆盖日志打印步长 |

> [!TIP]
> `cpu` preset 默认 `checkpoint_stride = 1`，因此每个 iteration 都会记录一个曲线点；`log_stride = 100`，日志仍然每 100 步打印一次。

## 5. 各实验配置与用途

### 5.1 Preset 总览

| Preset | 用途 | 适合场景 |
|---|---|---|
| `smoke` | 极小规模快速检查 | 修改代码后确认流程能跑通 |
| `cpu` | 本地 CPU 报告复现 | 课程报告图、表、分析材料 |
| `paper` | 更接近论文规模 | 有充足算力时做更完整复现 |

### 5.2 核心复现：`rotation_adam_p0`

理论对应：

- Theorem 3
- 稀疏/约束参数空间
- `P = 0`
- 目标是验证 `A_i ∝ Σ^{-1}`

CPU 配置重点：

| 配置项 | 值 |
|---|---|
| `n_layer` | `3` |
| `context_length` | `20` |
| `dimension` | `5` |
| `batch_size` | `2048` |
| `eval_batch_size` | `2048` |
| `max_iters` | `4000` |
| `seeds` | `[0, 1]` |
| `checkpoint_stride` | `1` |
| `log_stride` | `100` |
| `learning_rate` | `0.02` |
| `lr_decay_steps` | `[1500, 3000]` |
| `zero_p_after_step` | `True` |

主要产物：

- `rotation_demonstration_adam_pnull_loss_plot.png/pdf`
- `rotation_demonstration_dist_to_id_adam_pnull_A0/A1/A2.png/pdf`
- `rotation_demonstration_pnull_A0/A1/A2.png/pdf`

报告解释重点：

```text
如果 Σ^{1/2} A_i Σ^{1/2} 比 raw A_i 更接近缩放单位阵，则说明模型学到的是 Σ^{-1} 方向的预条件器，而不是普通 GD 的 identity 方向。
```

### 5.3 核心复现：`rotation_adam`

理论对应：

- Theorem 4
- 放宽参数空间
- 同时学习 `A_i` 和 `B_i`
- 目标是验证预条件梯度步与特征变换结构

CPU 配置重点：

| 配置项 | 值 |
|---|---|
| `n_layer` | `3` |
| `context_length` | `20` |
| `dimension` | `5` |
| `batch_size` | `2048` |
| `eval_batch_size` | `2048` |
| `max_iters` | `4000` |
| `seeds` | `[0, 1]` |
| `checkpoint_stride` | `1` |
| `log_stride` | `100` |
| `learning_rate` | `0.05` |
| `lr_decay_steps` | `[1500, 3000]` |

主要产物：

- `rotation_demonstration_adam_loss_plot.png/pdf`
- `rotation_demonstration_dist_to_id_adam.png/pdf`
- `rotation_demonstration_dist_to_id_adam_B0/B1.png/pdf`
- `rotation_demonstration_dist_to_id_adam_A0/A1/A2.png/pdf`
- `rotation_demonstration_adam_A0/A1/A2.png/pdf`
- `rotation_demonstration_adam_B0/B1.png/pdf`

报告解释重点：

```text
B_i 接近缩放单位阵，说明模型学习到近似 identity 的特征变换；rotated A_i 接近 identity，说明 A_i 对应 Σ^{-1} 预条件方向。
```

### 5.4 主扩展实验：`variable_n`

研究问题：

```text
上下文样本数 N 增加时，Linear Transformer、GD、Preconditioned GD 和 OLS 的测试损失如何变化？
```

课程对应：

- 样本复杂度
- 估计误差
- in-context examples 数量对泛化误差的影响

CPU 配置重点：

| 配置项 | 值 |
|---|---|
| `n_layer` | `3` |
| `variable_contexts` | `[4, 8, 12, 16, 20]` |
| `batch_size` | `512` |
| `eval_batch_size` | `1024` |
| `validation_batch_size` | `512` |
| `baseline_eval_size` | `128` |
| `max_iters` | `600` |
| `seeds` | `[0, 1]` |
| `checkpoint_stride` | `1` |
| `log_stride` | `100` |

主要产物：

- `3-step-variable-N-plot.png/pdf`
- `extension_variable_n_summary.csv`
- `extension_variable_n_relative.csv`
- `extension_variable_n_report.md`
- `extension_variable_n_artifact_index.json`

报告解释重点：

```text
N 增加时，模型获得更多上下文样本信息，测试损失通常应整体下降。PGD 显式使用协方差信息，通常比普通 GD 更适合非各向同性输入。OLS 是闭式解强基线。
```

### 5.5 可选扩展实验：`variable_l`

研究问题：

```text
层数 L 增加是否类似增加优化步数？更多层是否降低测试损失？
```

课程对应：

- 优化过程
- 计算复杂度
- Transformer depth 与迭代优化步数的关系

注意：

`variable_l` 计算压力更大，因为训练次数约为：

```text
seed 数 × L 取值数量
```

例如 2 个 seed、4 个层数值就是 8 次训练，并且层数越大单次训练越慢。

建议轻量命令：

```bash
python main.py \
  --experiment variable_l \
  --preset smoke \
  --stage all \
  --device cpu \
  --seeds 0 \
  --max-iters 400 \
  --batch-size 256 \
  --checkpoint-stride 1 \
  --log-stride 100
```

主要产物：

- `variable-L-plot.png/pdf`

建议定位：

```text
作为附加扩展趋势图，不作为核心复现证据。
```

### 5.6 其他实验

| 实验 | 用途 | 建议 |
|---|---|---|
| `simple` | 单位协方差 sanity check | 调试模型、mask、符号、loss 时使用 |
| `rotation_lbfgs` | LBFGS 优化器版本 | 可用于对照，但不是当前 CPU 报告主线 |

## 6. 输出目录和报告产物

假设使用脚本默认目录，输出结构为：

```text
results/
  report_artifacts.md
  smoothed_curve_artifacts.json
  rotation_adam_p0/cpu/
    artifacts/
    metrics/
    figures/
    figures_smoothed/
    logs/
  rotation_adam/cpu/
    artifacts/
    metrics/
    figures/
    figures_smoothed/
    logs/
  variable_n/cpu/
    artifacts/
    metrics/
    figures/
    figures_smoothed/
    logs/

data/
  rotation_adam_p0/cpu/
  rotation_adam/cpu/
  variable_n/cpu/
```

如果运行时设置：

```bash
RESULT_DIR=results_cpu DATA_DIR=data_cpu bash scripts/run_cpu_reproduction.sh
```

则根目录相应变为 `results_cpu/` 和 `data_cpu/`。

### 6.1 `report_artifacts.md`

这是报告写作的总索引，包含：

- 建议插入哪些图。
- 每张图的中文 caption。
- 每个实验的核心指标首尾值。
- 平滑曲线索引路径。
- 报告写作提醒。

### 6.2 `metrics.json`

每个实验都有：

```text
results/<experiment>/cpu/metrics/metrics.json
```

它是最重要的数值结果来源。绘图、平滑曲线和报告索引都从这里读取数据。

### 6.3 `figures/`

保存原始图：

- PNG
- PDF
- `figure_index.json`

### 6.4 `figures_smoothed/`

保存 EMA 平滑图。平滑图用于展示趋势，不改变原始实验数值。

## 7. 平滑曲线说明

平滑脚本：

```bash
python experiments/smooth_report_curves.py \
  --result-dir results \
  --preset cpu
```

如果使用 `results_cpu`：

```bash
python experiments/smooth_report_curves.py \
  --result-dir results_cpu \
  --preset cpu
```

平滑策略：

- 从 `metrics.json` 读取曲线。
- 使用 TensorBoard 风格 EMA。
- 同时保留淡色原始曲线和加粗平滑曲线。
- 输出到 `figures_smoothed/`。

报告建议：

- 正文严谨分析优先引用原始 `metrics.json` 和原始图。
- 如果曲线抖动影响观感，可在 PPT 或报告配图中使用平滑图。
- 使用平滑图时建议注明“曲线经过 EMA 平滑，仅用于趋势展示”。

## 8. 常见复现流程

### 8.1 第一次拿到代码

```bash
conda create -n cs_hw python=3.11 -y
conda activate cs_hw
pip install -r requirements.txt
python main.py --experiment simple --preset smoke --stage all --device cpu
```

如果 `simple smoke` 能跑通，再运行：

```bash
bash scripts/run_cpu_reproduction.sh 0
```

确认没问题后，再运行完整 2 seed：

```bash
bash scripts/run_cpu_reproduction.sh
```

### 8.2 只想重画图

前提是已有 `artifacts/` 和 `metrics/`：

```bash
python main.py \
  --experiment rotation_adam \
  --preset cpu \
  --stage plot \
  --device cpu \
  --result-dir results \
  --data-dir data
```

### 8.3 只想重建报告索引

```bash
python experiments/report_artifacts.py \
  --result-dir results \
  --preset cpu \
  --seeds 0 1
```

### 8.4 只想跑更短版本

```bash
python main.py \
  --experiment rotation_adam_p0 \
  --preset cpu \
  --stage all \
  --device cpu \
  --seeds 0 \
  --max-iters 1000 \
  --batch-size 1024 \
  --checkpoint-stride 1 \
  --log-stride 100
```

## 9. 如何判断结果是否可写入报告

### 9.1 Theorem 3

检查：

- loss 曲线是否整体下降。
- `Σ^{1/2} A_i Σ^{1/2}` 的 distance 是否下降。
- rotated distance 是否比 raw distance 更符合理论目标。
- final rotated heatmap 是否呈现较明显对角结构。

可写结论：

```text
CPU 小规模实验中，旋转后的 A_i 矩阵比原始 A_i 更接近缩放单位阵，支持模型学习到 Σ^{-1} 预条件器的解释。
```

### 9.2 Theorem 4

检查：

- loss 曲线是否整体下降。
- `B0/B1` distance 是否下降。
- rotated `A_i` distance 是否下降。
- `B_i` heatmap 是否接近缩放单位阵。

可写结论：

```text
放宽参数后，模型不仅执行预条件梯度步，还学习到近似 identity 的特征变换，这与论文对 Theorem 4 临界点的解释一致。
```

### 9.3 `variable_n`

检查：

- `3-step-variable-N-plot` 是否生成。
- `extension_variable_n_summary.csv` 是否包含所有 `N` 和四种方法。
- `extension_variable_n_report.md` 是否给出中文趋势分析。

可写结论：

```text
随着上下文长度 N 增加，任务信息更充分，测试 loss 的整体趋势可用于讨论样本复杂度和估计误差。若曲线存在局部波动，应如实说明 CPU 预算、seed 数和训练步数限制。
```

## 10. 常见问题

### 10.1 为什么脚本输出到 `results/` 而不是 `results_cpu/`？

`scripts/run_cpu_reproduction.sh` 的实际默认值是：

```bash
RESULT_DIR="${RESULT_DIR:-results}"
DATA_DIR="${DATA_DIR:-data}"
```

如果希望输出到 `results_cpu/` 和 `data_cpu/`，运行：

```bash
RESULT_DIR=results_cpu DATA_DIR=data_cpu bash scripts/run_cpu_reproduction.sh
```

### 10.2 为什么每个 iteration 都保存 checkpoint？

为了让报告曲线更平滑，`cpu` preset 设置：

```text
checkpoint_stride = 1
log_stride = 100
```

这表示每一步都记录曲线点，但日志仍然每 100 步显示一次。

### 10.3 CPU 跑得慢怎么办？

优先减少 seed：

```bash
bash scripts/run_cpu_reproduction.sh 0
```

其次减少步数：

```bash
python main.py \
  --experiment rotation_adam \
  --preset cpu \
  --stage all \
  --device cpu \
  --seeds 0 \
  --max-iters 1000
```

### 10.4 可以直接跑 `paper` preset 吗？

可以，但不建议本地 CPU 直接跑。`paper` preset 的 batch 和迭代数都明显更大，尤其是 `rotation_adam`、`rotation_adam_p0`、`variable_n` 会很慢。

### 10.5 `variable_l` 要不要加入最终报告？

可以作为附加扩展，但不建议替代 `variable_n`。`variable_n` 更适合课程报告中的样本复杂度分析；`variable_l` 更适合补充说明层数与优化步数的关系。

## 11. 推荐最终报告素材清单

完整 CPU 运行后，优先打开：

```text
results/report_artifacts.md
```

报告第 4 部分至少可使用：

- Theorem 3 loss 曲线。
- Theorem 3 raw vs rotated distance 曲线。
- Theorem 3 final rotated `A_i` heatmap。
- Theorem 4 loss 或 distance 曲线。
- Theorem 4 final `B_i` heatmap。
- `variable_n` 主图。
- `variable_n` summary CSV。
- `variable_n` 中文自动分析段落。

如果使用了自定义输出目录，例如 `results_cpu`，则打开：

```text
results_cpu/report_artifacts.md
```

平滑曲线索引：

```text
results/smoothed_curve_artifacts.json
```

或：

```text
results_cpu/smoothed_curve_artifacts.json
```
