# 精度因子融合

[方法导航](README.md) · [动态门控](gating.md) · [对照：凸组合](convex.md) · [实验结果](../../experiment_results.md)

## 先分清三个对象

协方差是 $\Sigma_k$，精度矩阵是 $\Sigma_k^{-1}$，精度Cholesky因子是满足下式的下三角矩阵 $L_k$：

$$
\Sigma_k^{-1}=L_kL_k^\top.
$$

本项目融合的是第三个对象：

$$
\bar L_t=\sum_kw_{t,k}L_{t,k},\qquad
\widehat\Sigma_t=(\bar L_t\bar L_t^\top)^{-1}.
$$

它**不是**直接计算 $\left(\sum_kw_k\Sigma_k^{-1}\right)^{-1}$，也不是把协方差Cholesky因子取逆后直接当成这里的下三角因子。

## 为什么正定？

每个 $L_k$都是正对角的下三角矩阵。非负权重加权后仍保持正对角，因此 $\bar L$可逆，$\bar L\bar L^\top$及其逆矩阵都正定。

## 与凸组合的区别

凸组合直接平均风险矩阵；这里先进入逆矩阵因子空间再融合，最后反变换，因此是不同的非线性融合方式。它可能产生不同的风险方向与尺度，但不存在对所有数据都更优的保证。

Cholesky因子依赖资产排列顺序；本实验保持资产顺序固定，不能宣称这一融合对任意资产重排不变。

## 训练与修正

Softmax权重通过最终矩阵损失训练。基础融合不额外校准；联合版本在该融合结果后接[Idea1](idea1.md)或[Idea2](idea2.md)，两部分端到端更新。

[核心代码](../../src/covariance_research/fusion/aggregation/precision.py) · [因子构造](../../src/covariance_research/statistical/iewma.py)
