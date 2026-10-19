# CAB：论文架构适配模块

[方法导航](README.md) · [自设计网络对照](networks.md) · [实验结果](../../experiment_results.md)

本页解释项目实际实现，不声称逐项重现作者原始代码与论文成绩。

## 网络流程

100个历史20日协方差矩阵 → 元素标准化 → 3D CNN → BiLSTM → Multihead Attention → 时间均值池化 → 线性矩阵输出 → 反标准化与PSD修正 → 历史混合。

| 部分 | 当前实现 |
| --- | --- |
| 3D CNN | 输入/输出通道各1，核5，padding2 |
| BiLSTM | 输入为矩阵展平，7层，每方向隐藏128 |
| Attention | 256维，16头 |
| Readout | 时间均值后映射到100个矩阵元素 |
| 输出约束 | 对称化，负特征值截为0 |
| 最终混合 | 0.8神经网络矩阵＋0.2最近历史协方差 |

PSD修正写为：

$$
S=\frac{A+A^\top}{2}=U\operatorname{diag}(\lambda)U^\top,\qquad
S_+=U\operatorname{diag}(\max(\lambda_i,0))U^\top.
$$

最终预测：

$$
\widehat\Sigma=0.8S_++0.2\Sigma_{\mathrm{last}}.
$$

因此CAB也包含历史先验，并非完全从零生成风险结构。

## 与当前其他网络的关系

CAB不套用五种输出变换，保留自己的结构；训练使用Adam1e-4、最多100轮、早停30，骨干float32，而其他模型通常采用不同配方。因此它是并列方法比较，不是只替换编码器的严格消融。

当前实验为10行业、固定训练后测试；不能将它与论文的资产范围、测试时期和更新协议默认视为完全相同。

[实际模型代码](../../src/covariance_research/direct/paper_models/cab.py) · [CAB配置](../../configs/paper/cab.json) · [完整协议](../protocol.md)
