# 固定Residual输出的六种网络

[方法导航](README.md) · [输入流程](direct.md) · [Residual输出](output_heads.md#spd-residual) · [CAB独立模块](cab.md) · [实验结果](../../experiment_results.md)

这里均输入100个历史10×10协方差矩阵，每张展平100维。输出统一使用LW60指数Residual；不要与输入100×6风险指标的[门控网络](gating.md)混淆。

## MLP

展平100×100=10000维 → Linear64 → ReLU → Linear32 → ReLU → 输出头。

它一次读取整个窗口，参数量较大，没有显式的时间平移共享结构。

## CNN

将每张矩阵的100个元素视作通道：Conv1d100→32,k5 → ReLU → Conv1d32→32,k5 → ReLU → 时间均值池化 → 输出头。

它通过局部卷积提取变化模式，再汇总整个已知历史。

## TCN

100→32投影，经过六个残差因果卷积块，卷积核3，膨胀率1/2/4/8/16/32，取最后时点状态。

每层左侧补零；感受野127，覆盖100时点输入。所有输入都在预测时已知。

## GRU

单层GRU，输入100维、隐藏32维，取最后隐藏状态。递归处理历史，参数共享，不直接展平10000维。

## Transformer

100→32投影＋正弦时间位置编码 → 一层4头Self-Attention、FFN64、dropout0 → 最后时点状态。

整个窗口均属于已知历史，窗口内双向注意力不等于读取预测日之后的信息。

## Patch Transformer

每5个时点组成一个500维片段，100步变成20个Token；500→32投影＋位置编码 → 同规模Transformer → 最后Token状态。

减少Token数量，但也改变了时间编码粒度。

## 比较边界与实现

六种架构不是等参数量实验。统一学习率也不意味着每种架构都已优化充分；选择epoch0表示仍保留LW60初始化。普通Transformer在五输出头实验与本架构实验间复用同配置结果，不是独立重复证据。

[架构实现](../../src/covariance_research/direct/backbones/residual.py) · [Transformer矩阵输入](../../src/covariance_research/direct/inputs/covariance.py) · [训练协议](training.md)
