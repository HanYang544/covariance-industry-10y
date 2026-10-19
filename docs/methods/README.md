# 方法讲解导航

[返回项目首页](../../README.md) · [查看实验结果](../../experiment_results.md) · [数据与训练协议](../protocol.md)

先了解研究背景与整体逻辑：[从收益预测到多尺度协方差预测](../research_overview.md)。

按研究路线点击阅读，不必先阅读全部源码。

## 基础统计方法

[Historical：样本协方差](statistical.md#historical) · [LW：收缩估计](statistical.md#lw) · [EWMA：指数加权](statistical.md#ewma) · [IEWMA：波动与相关分开更新](statistical.md#iewma)

## 主线一：动态融合与修正

1. [动态门控：输入如何变成Softmax权重](gating.md)
2. 两种融合：[凸组合](convex.md) ／ [精度因子融合](precision.md)
3. 两种改进：[Idea1：低维风险方向修正](idea1.md) ／ [Idea2：全维缩放与旋转](idea2.md)

## 主线二：网络直接预测

1. [输入与完整预测流程](direct.md)
2. [五种输出变换](output_heads.md)：[Cholesky](output_heads.md#cholesky)、[矩阵指数](output_heads.md#matrix-exponential)、[谱Softplus](output_heads.md#spectral-softplus)、[方差–相关双头](output_heads.md#variance-correlation)、[Residual](output_heads.md#spd-residual)
3. [网络架构比较](networks.md)：[MLP](networks.md#mlp)、[CNN](networks.md#cnn)、[TCN](networks.md#tcn)、[GRU](networks.md#gru)、[Transformer](networks.md#transformer)、[Patch Transformer](networks.md#patch-transformer)
4. [CAB论文架构适配](cab.md)

## 共同训练与评价

[损失如何计算、早停如何选择、结果如何阅读](training.md) · [如何运行实验](../running.md) · [源码模块对应](../architecture.md)

这些说明对应本项目实现，不是对所有同名算法版本的概括；参数和结果以冻结协议及实验文档为准。
