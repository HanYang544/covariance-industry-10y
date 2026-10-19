# 凸组合融合

[方法导航](README.md) · [动态门控](gating.md) · [对照：精度因子融合](precision.md) · [实验结果](../../experiment_results.md)

## 核心计算

门控网络提供权重，10个统计专家提供协方差矩阵：

$$
\widehat\Sigma_t=\sum_{k=1}^{10}w_{t,k}\Sigma_t^{(k)},\qquad
w_{t,k}\ge0,\quad \sum_kw_{t,k}=1.
$$

它直接在协方差空间取加权平均。某专家权重越大，融合矩阵越接近该专家。

## 为什么保留正定性？

对任何非零向量 $x$：

$$
x^\top\widehat\Sigma_tx=\sum_kw_{t,k}x^\top\Sigma_t^{(k)}x.
$$

当专家均正定，且至少一个权重为正，右侧严格大于0，因此融合结果正定。若专家只有半正定，只能普遍保证半正定。

## 如何进一步改进？

基础模型直接输出融合结果；联合模型把此矩阵作为未校准基础，再接[Idea1](idea1.md)或[Idea2](idea2.md)。融合网络不冻结，最终矩阵损失同时训练融合与修正参数。

凸组合的解释直观，但输出限制在专家矩阵的凸包内。修正模块能够改变这个限制，代价是增加模型自由度。

[核心代码](../../src/covariance_research/fusion/aggregation/convex.py) · [组装代码](../../src/covariance_research/fusion/model.py)
