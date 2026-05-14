# CPU 复现实验报告素材清单

## 运行配置
- Preset: `cpu`
- Seeds: `[0, 1]`
- 运行环境: `conda run -n cs_hw python ...`
- 说明: 本素材包面向本地 Mac CPU 预算，重点用于趋势验证和报告截图，不等同于 paper-scale 训练。

## 报告第 4 部分建议插图

### Theorem 3：稀疏约束下的预条件梯度下降结构
- `results/rotation_adam_p0/cpu/figures/rotation_demonstration_adam_pnull_loss_plot.png` (exists)：Theorem 3 CPU 复现的 log loss 曲线，用于展示训练误差整体下降。
- `results/rotation_adam_p0/cpu/figures/rotation_demonstration_dist_to_id_adam_pnull_A0.png` (exists)：A0 的 raw/rotated distance 对比，用于说明旋转后的矩阵更接近缩放单位阵。
- `results/rotation_adam_p0/cpu/figures/rotation_demonstration_pnull_A0.png` (exists)：训练结束后 Σ^{1/2}A0Σ^{1/2} 的热力图，用于展示近似对角结构。

### Theorem 4：放宽参数后的特征变换结构
- `results/rotation_adam/cpu/figures/rotation_demonstration_adam_loss_plot.png` (exists)：Theorem 4 CPU 复现的 log loss 曲线，用于展示放宽参数设置下的训练趋势。
- `results/rotation_adam/cpu/figures/rotation_demonstration_dist_to_id_adam.png` (exists)：B_i 与 rotated A_i 的 distance panel，用于展示 B_i 接近 identity 且 A_i 接近 Σ^{-1}。
- `results/rotation_adam/cpu/figures/rotation_demonstration_adam_B0.png` (exists)：训练结束后 B0 的热力图，用于展示特征变换矩阵接近缩放单位阵。

### 扩展实验：上下文长度 N 与样本复杂度
- `results/variable_n/cpu/figures/3-step-variable-N-plot.png` (exists)：不同上下文长度 N 下 Transformer/GD/PGD/OLS 的测试 loss 对比。
- `results/variable_n/cpu/figures/extension_variable_n_summary.csv` (exists)：扩展实验数值表，包含每个 N 下四类方法的 mean/std loss。
- `results/variable_n/cpu/figures/extension_variable_n_report.md` (exists)：扩展实验中文自动分析，可直接改写进报告。

### 平滑曲线补充素材
- `results/smoothed_curve_artifacts.json` (exists)：平滑曲线索引；若报告截图希望更缓和，可优先使用 `figures_smoothed` 中的 PNG/PDF。

## 核心指标摘要

### Theorem 3 指标
- log loss: first=1.641430, final=-1.580373, delta=-3.221803
- rotated A0 distance: first=0.950218, final=0.169304, delta=-0.780914
- raw A0 distance: first=0.999481, final=0.769688, delta=-0.229794
- rotated A1 distance: first=0.975746, final=0.392755, delta=-0.582991
- raw A1 distance: first=0.984114, final=0.811976, delta=-0.172137
- rotated A2 distance: first=0.965415, final=0.212569, delta=-0.752846
- raw A2 distance: first=0.954813, final=0.757261, delta=-0.197552

### Theorem 4 指标
- log loss: first=1.641430, final=-4.221714, delta=-5.863143
- B0 distance: first=0.997822, final=0.558772, delta=-0.439051
- B1 distance: first=0.966573, final=0.477783, delta=-0.488790
- A0 distance: first=0.950218, final=0.373056, delta=-0.577162
- A1 distance: first=0.975746, final=0.376785, delta=-0.598960
- A2 distance: first=0.965415, final=0.394170, delta=-0.571245

### 扩展实验指标
- Context lengths: [4, 8, 12, 16, 20]
- Transformer loss: first=2.058553, final=0.363965, delta=-1.694588
- GD loss: first=3.944510, final=1.513764, delta=-2.430746
- Preconditioned GD loss: first=2.864391, final=0.197570, delta=-2.666821
- OLS loss: first=1.303765, final=0.000000, delta=-1.303765

## 报告写作提醒
- 若某些曲线没有完全单调下降，报告中按 CPU 小规模复现实验如实说明，不要写成 paper-scale 结论。
- Theorem 3 的关键表述是 rotated A_i 比 raw A_i 更符合 identity 结构。
- Theorem 4 的关键表述是 B_i 接近 identity，rotated A_i 接近 identity，对应 A_i 接近 Σ^{-1}。
- 扩展实验用于补充样本复杂度视角：N 增大通常带来更多任务信息，使测试 loss 呈下降趋势或总体改善。
