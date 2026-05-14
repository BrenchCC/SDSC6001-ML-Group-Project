# 上下文长度 N 扩展实验分析

## 实验问题
本扩展实验考察 in-context examples 数量增加时，Linear Transformer、普通 GD、预条件 GD 和 OLS 的测试损失如何变化。该问题对应课程项目中的样本复杂度/估计误差视角：上下文样本越多，模型可用于恢复线性任务的信息越充分，理论上测试误差应整体下降。

## 实验设置
- 上下文长度: [4, 8, 12, 16, 20]
- 随机种子: [0, 1]
- 方法: 3-layer Linear Transformer、3-step GD、3-step Preconditioned GD、OLS
- 主图 PNG: `results/variable_n/cpu/figures/3-step-variable-N-plot.png`
- 主图 PDF: `results/variable_n/cpu/figures/3-step-variable-N-plot.pdf`

## 主要观察
- Linear Transformer: N=4 时 loss=2.058553，N=20 时 loss=0.363965，变化率=82.32%。
- GD: N=4 时 loss=3.944510，N=20 时 loss=1.513764，变化率=61.62%。
- Preconditioned GD: N=4 时 loss=2.864391，N=20 时 loss=0.197570，变化率=93.10%。
- OLS: N=4 时 loss=1.303765，N=20 时 loss=0.000000，变化率=100.00%。
- 最后一个上下文长度下，Transformer/GD loss ratio = 0.2404。
- 最后一个上下文长度下，Transformer/PGD loss ratio = 1.8422。
- 最优方法摘要: N=4: OLS (1.303765); N=8: OLS (0.000000); N=12: OLS (0.000000); N=16: OLS (0.000000); N=20: OLS (0.000000)。
- 单调性说明: Linear Transformer 单调下降；GD 存在局部波动；Preconditioned GD 单调下降；OLS 单调下降

## 可写入报告的分析段落
从样本复杂度角度看，随着上下文样本数 N 增加，各方法获得更多线性回归任务信息，测试损失整体呈下降趋势。预条件 GD 显式使用协方差信息，因此通常比普通 GD 更适合非各向同性输入分布；Linear Transformer 的表现则反映训练后模型能否从上下文中学习接近梯度法的更新机制。OLS 作为闭式解参考，提供了在当前有限样本设定下的强基线。由于本实验受到本地 CPU 预算限制，训练迭代、batch size 和 seed 数均小于原论文规模，因此结论应以趋势分析和方法对比为主，而不是宣称完全复现 paper-scale 数值。

## CPU 小规模实验限制
本实验默认只使用 2 个 seed，并采用 CPU-friendly batch size 与迭代数。若某些曲线存在局部波动，应解释为训练预算与随机性共同导致的现象；报告中应保留图表并如实说明趋势是否充分。
