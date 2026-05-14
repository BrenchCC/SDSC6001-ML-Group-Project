# SDSC6001 Machine Learning Group Project

本仓库用于 SDSC6001 机器学习课程期末小组项目：复现并分析 NeurIPS 2023 论文 **Transformers learn to implement preconditioned gradient descent for in-context learning**。项目包含复现实验代码、课程规模实验结果、双语 LaTeX 报告、PPT 设计文档和中英双语演讲稿模板。

## 项目主题

- **论文**：Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, Suvrit Sra. *Transformers learn to implement preconditioned gradient descent for in-context learning*. NeurIPS 2023.
- **课程方向**：主要对应优化误差 / 计算复杂度；同时涉及表达能力，并通过上下文长度实验补充样本复杂度视角。
- **核心问题**：Transformer 是否能通过训练自然学会类似梯度下降的算法，而不仅仅是存在一组人工构造权重可以模拟梯度下降。
- **报告术语**：
  - **PSE**：Paper-scale Structural Evidence，原论文官方结构性证据。
  - **CRE**：Course-scale Reproduction Evidence，本项目课程规模复现实验证据。

## 目录结构

```text
.
├── main.py                         # 统一实验入口
├── experiments/                    # 模型、数据、训练、绘图和实验 runner
├── scripts/run_cpu_reproduction.sh # 一键复现实验脚本
├── requirements.txt                # Python 依赖
├── results/                        # 已生成的实验结果和报告素材
├── reports/6001-final-report/      # 双语最终报告与报告图片
├── ppt-design/                     # PPT 设计思路、演讲稿模板和 PPT 图片
└── docs/recording/image_gen.md     # 后续补图用的 AI 生图 prompt
```

## 关键交付物速查

| **交付物** | **路径** | **用途** |
|---|---|---|
| **英文最终报告 PDF** | **`reports/6001-final-report/en/main.pdf`** | **正式英文版课程报告，可直接查看或提交** |
| **中文报告 PDF** | **`reports/6001-final-report/zh/main.pdf`** | **中文版对照阅读材料，便于组内理解和校对** |
| **英文报告源文件** | **`reports/6001-final-report/en/main.tex`** | **需要改作者、文字、图表或参考文献时编辑这里** |
| **中文报告源文件** | **`reports/6001-final-report/zh/main.tex`** | **中文翻阅版的 LaTeX 源文件** |
| **PPT 设计思路** | **`ppt-design/ppt_design_brief_zh.md`** | **约 20 页 PPT 的页面规划、图片安排和设计要求** |
| **零基础理解指南** | **`ppt-design/beginner_guide_zh.md`** | **给组员快速理解论文、实验、延展思考和视频设计使用** |
| **中文演讲稿模板** | **`ppt-design/speech_script_template_zh.md`** | **中文录制视频时的逐页讲稿参考** |
| **英文演讲稿模板** | **`ppt-design/speech_script_template_en.md`** | **英文汇报或英文旁白时的讲稿参考** |
| **PPT 图片素材** | **`ppt-design/figures/`** | **PPT 生成和人工排版时优先使用的全部图片** |
| **报告图片素材** | **`reports/6001-final-report/figures/`** | **LaTeX 报告引用的 PSE、CRE 和 AI 概念图** |
| **实验结果索引** | **`results/report_artifacts.md`** | **快速查找 CRE 实验图、指标和结果说明** |
| **生图 Prompt** | **`docs/recording/image_gen.md`** | **后续补充流程图、结构图、总结图时使用** |

不要把 `docs/paper/`、`docs/course/`、`docs/code_guidance/`、`.codex-planning/`、`.claude/`、`.omc/`、`.omx/` 打进最终提交包。它们是论文源码、课程说明、开发记录或本地工具状态，不是提交物。

## 环境准备

建议先进入项目环境，再安装依赖：

```bash
pip install -r requirements.txt
```

如果只需要制作 PPT、录制视频或查看报告，不需要重新安装依赖，也不需要重新运行实验。

## 已完成交付物

### 最终报告

- 英文报告源文件：`reports/6001-final-report/en/main.tex`
- 英文报告 PDF：`reports/6001-final-report/en/main.pdf`
- 英文参考文献：`reports/6001-final-report/en/references.bib`
- 中文报告源文件：`reports/6001-final-report/zh/main.tex`
- 中文报告 PDF：`reports/6001-final-report/zh/main.pdf`
- 中文参考文献：`reports/6001-final-report/zh/references.bib`
- 报告图片：`reports/6001-final-report/figures/`

报告已经包含：

- 论文研究问题与核心挑战
- linear self-attention 的问题设定
- Transformer forward pass 与 preconditioned gradient descent 的理论对应
- Theorem 3 / Theorem 4 的 PSE 与 CRE 对比
- context length 扩展实验
- `N=12` 附近普通 GD 曲线突刺现象的解释
- 约 10 篇以上参考文献

### PPT 设计材料

- PPT 设计思路：`ppt-design/ppt_design_brief_zh.md`
- 中文演讲稿模板：`ppt-design/speech_script_template_zh.md`
- 英文演讲稿模板：`ppt-design/speech_script_template_en.md`
- PPT 图片素材：`ppt-design/figures/`

`ppt-design/ppt_design_brief_zh.md` 已经按约 20 页结构组织，可直接交给后续 PPT 制作工具或人工设计使用。

### 生图 Prompt

- `docs/recording/image_gen.md`

该文件保存了不依赖仓库路径的 AI 生图 prompt，用于补充流程图、结构图和总结图。当前 `ppt-design/figures/` 中已经包含 5 张 AI 生成概念图，未使用的第 3 和第 6 个 prompt 可以按需继续生成。

## 如何制作 PPT

推荐按以下顺序工作：

1. 打开 `ppt-design/ppt_design_brief_zh.md`。
2. 按 20 页页面规划生成 PPT。
3. 所有图片优先从 `ppt-design/figures/` 读取。
4. 每页只保留一个主结论，避免把报告整段文字搬到 PPT。
5. 实验页必须明确标注 PSE 或 CRE，避免混淆原论文结果和本项目复现实验。

建议页面节奏：

| 页码 | 内容 |
|---|---|
| 1-3 | 标题、课程方向匹配、核心问题 |
| 4-8 | ICL 背景、线性回归 prompt、linear attention、理论机制 |
| 9-12 | Theorem 3 与 CRE 复现 |
| 13-16 | Theorem 4 与 CRE 复现 |
| 17 | 上下文长度扩展实验 |
| 18-20 | PSE vs CRE 总结、局限、结论 |

## 如何录制视频

课程要求视频不超过 15 分钟。建议控制在 12 到 14 分钟，留出缓冲。

推荐分工：

| 成员 | 建议负责部分 |
|---|---|
| 成员 1 | 选题、课程方向、问题设定 |
| 成员 2 | Theorem 1/3 理论机制与 PSE |
| 成员 3 | CRE 实验结果、Theorem 4、扩展实验 |
| 成员 4 | 局限、总结、后续思考 |

录制建议：

- 中文讲解可直接参考 `ppt-design/speech_script_template_zh.md`。
- 英文讲解可参考 `ppt-design/speech_script_template_en.md`。
- Theorem 3/4 不需要逐行证明，重点解释“为什么矩阵结构能支持预条件 GD 解释”。
- 讲到 `cre_variable_n_smooth.png` 时，说明 `N=12` 附近突刺主要来自普通 GD 曲线，是有限步固定步长 GD 在非各向同性协方差下的敏感性；这不是 Transformer 主趋势。
- 最后一页要回到课程主题：优化误差、计算复杂度、表达能力和样本复杂度。

## 如何重新运行实验

一般情况下不需要重新运行。若需要重新生成结果，可以先安装依赖，然后运行：

```bash
bash scripts/run_cpu_reproduction.sh
```

也可以单独运行某个实验：

```bash
python main.py --experiment rotation_adam_p0 --stage all --preset cpu --device cpu --result-dir results --data-dir data --seeds 0 1
python main.py --experiment rotation_adam --stage all --preset cpu --device cpu --result-dir results --data-dir data --seeds 0 1
python main.py --experiment variable_n --stage all --preset cpu --device cpu --result-dir results --data-dir data --seeds 0 1
```

三个核心实验含义：

| 实验 | 对应内容 | 主要输出 |
|---|---|---|
| `rotation_adam_p0` | Theorem 3 稀疏约束设置 | loss、raw/rotated distance、A 矩阵热力图 |
| `rotation_adam` | Theorem 4 放宽参数设置 | loss、A/B distance、B 矩阵热力图 |
| `variable_n` | 上下文长度扩展实验 | Transformer/GD/PGD/OLS loss 曲线 |

结果索引：

- `results/report_artifacts.md`
- `results/smoothed_curve_artifacts.json`
- `results/rotation_adam_p0/cpu/figures/`
- `results/rotation_adam/cpu/figures/`
- `results/variable_n/cpu/figures/`

## 如何重新编译报告

英文报告：

```bash
cd reports/6001-final-report/en
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

中文报告：

```bash
cd reports/6001-final-report/zh
xelatex -interaction=nonstopmode main.tex
bibtex main
xelatex -interaction=nonstopmode main.tex
xelatex -interaction=nonstopmode main.tex
```

编译完成后可以删除中间产物，只保留 `.tex`、`.bib`、`.pdf`、`IEEEtran.cls`：

```bash
find reports/6001-final-report -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.bbl' -o -name '*.blg' -o -name '*.toc' \) -delete
```

## 最终提交包应该包含什么

建议最终压缩包只包含必要代码、结果、报告和 PPT 材料：

```text
README.md
LICENSE
requirements.txt
main.py
experiments/
scripts/
results/
reports/6001-final-report/
ppt-design/
docs/recording/image_gen.md
```

不要包含：

```text
docs/paper/
docs/course/
docs/code_guidance/
.codex-planning/
.claude/
.omc/
.omx/
__pycache__/
data/
```

`data/` 可由实验代码重新生成，通常不需要提交。若老师明确要求提交训练中间数据，再单独补充。

## 推荐打包命令

使用白名单方式打包，避免误打包参考文档：

```bash
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

package_name = "SDSC6001-ML-Group-Project-submission.zip"
include_paths = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    "main.py",
    "experiments",
    "scripts",
    "results",
    "reports/6001-final-report",
    "ppt-design",
    "docs/recording/image_gen.md",
]
skip_dirs = {"__pycache__"}
skip_suffixes = {".aux", ".log", ".out", ".bbl", ".blg", ".toc", ".pyc"}

def should_skip(path: Path) -> bool:
    if any(part in skip_dirs for part in path.parts):
        return True
    if path.suffix in skip_suffixes:
        return True
    return False

with ZipFile(package_name, "w", ZIP_DEFLATED) as zf:
    for item in include_paths:
        path = Path(item)
        if not path.exists():
            continue
        if path.is_file():
            if not should_skip(path):
                zf.write(path, path.as_posix())
            continue
        for file_path in path.rglob("*"):
            if file_path.is_file() and not should_skip(file_path):
                zf.write(file_path, file_path.as_posix())

print(f"Created {package_name}")
PY
```

打包后建议检查压缩包内容：

```bash
python - <<'PY'
from zipfile import ZipFile

package_name = "SDSC6001-ML-Group-Project-submission.zip"
blocked_prefixes = [
    "docs/paper/",
    "docs/course/",
    "docs/code_guidance/",
    ".codex-planning/",
    ".claude/",
    ".omc/",
    ".omx/",
]

with ZipFile(package_name) as zf:
    names = zf.namelist()
    blocked = [
        name for name in names
        if any(name.startswith(prefix) for prefix in blocked_prefixes)
    ]
    print(f"Total files: {len(names)}")
    if blocked:
        print("Unexpected reference/local files:")
        for name in blocked:
            print(name)
    else:
        print("Package check passed: no reference/local tool directories included.")
PY
```

## 提交前检查清单

- `reports/6001-final-report/en/main.pdf` 可以打开。
- `reports/6001-final-report/zh/main.pdf` 可以打开。
- `ppt-design/ppt_design_brief_zh.md` 中引用的图片都在 `ppt-design/figures/`。
- 视频时长不超过 15 分钟。
- 压缩包中不包含 `docs/paper/`、`docs/course/`、`docs/code_guidance/`。
- 压缩包中不包含本地工具目录或开发记录目录。
