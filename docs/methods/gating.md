# 动态门控：风险序列如何变成权重

[方法导航](README.md) · [统计专家](statistical.md) · [实验结果](../../experiment_results.md)

## 输入与输出

输入是过去100个时点、每个时点6个风险状态指标，形状为 $100\times6$。指标概括波动率比例、平均相关性、吸收比、有效秩、子空间旋转和相关漂移。主成分数量使用3；特征只用已知历史计算，标准化参数只拟合训练数据。

编码器得到32维状态，再输出10个分数：

$$
h_t=f_\theta(X_t),\qquad a_t=Wh_t+b,\qquad
w_{t,k}=\frac{\exp(a_{t,k})}{\sum_{j=1}^{10}\exp(a_{t,j})}.
$$

Softmax不是一组预先指定的权重，而是把网络输出分数转成非负且和为1的权重。参数 $W,b,\theta$通过最终协方差预测损失联合更新。

## 三种门控编码器

| 编码器 | 本项目结构 |
| --- | --- |
| MLP | 展平600 → 64 → ReLU → 32 → ReLU |
| CNN | 6→32时序卷积k5 → ReLU → 32→32卷积k5 → ReLU → 时间均值池化 |
| Transformer | 6→32投影＋位置编码 → 1层4头注意力、FFN64 → 最后时点状态 |

三者最后都接10维线性层及Softmax。这些结构不同于直接预测模块里输入100维协方差Token的网络。

## 如何训练？

权重进入[凸组合](convex.md)或[精度因子融合](precision.md)，得到预测矩阵，再与未来20日标签比较。反向传播从矩阵损失传回融合运算、Softmax和编码器；没有单独提供“正确专家权重”的监督标签。

如果验证选中epoch0，模型仍可能输出随输入变化的权重，但不能说这种变化是训练学到的。损失下降也不自动证明模型利用了完整100步时序。

[Transformer代码](../../src/covariance_research/fusion/gating/transformer.py) · [MLP/CNN代码](../../src/covariance_research/fusion/gating/sequence.py) · [训练与损失](training.md)
