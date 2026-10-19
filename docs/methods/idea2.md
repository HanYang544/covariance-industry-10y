# Idea2：全维风险尺度与方向旋转

[方法导航](README.md) · [Idea1对照](idea1.md) · [动态融合](gating.md) · [实验结果](../../experiment_results.md)

## 核心思路

输入仍是未校准融合矩阵 $C=U\operatorname{diag}(\lambda)U^\top$，但不只修正前q个方向，而是调整全部10个特征值并旋转方向：

$$
\widehat C=(UQ)\operatorname{diag}(\lambda_i m_i)(UQ)^\top,\qquad
m_i=\exp(2\tanh(a_i)).
$$

正的特征值与正交旋转共同保证正定性。

## 如何生成旋转？

网络输出下三角旋转参数。用归一化特征值间距 $d_{ij}$控制幅度：

$$
\theta_{ij}=
\frac{0.1\tanh(b_{ij})\,d_{ij}}{d_{ij}^2+0.01}.
$$

把它填入严格下三角矩阵A，再用反对称矩阵的指数生成正交矩阵：

$$
Q=\exp_m(A-A^\top).
$$

当两个特征值很接近时，方向本身不稳定，因此间距项抑制相应旋转。这是数值与结构约束，不是保证所有重根情形下训练都毫无困难。

## 与Idea1的区别

Idea1主要限制修正子空间的维度；Idea2在全10维调整谱尺度与方向。它的自由度更大，也更需要验证是否产生真实增益。

修正头零初始化时 $m_i=1,Q=I$，预测还原基础融合。训练时，独立修正编码器与融合编码器都通过最终协方差损失更新。

[旋转计算](../../src/covariance_research/fusion/refinement/idea2.py) · [参数构造](../../src/covariance_research/fusion/refinement/core.py)
