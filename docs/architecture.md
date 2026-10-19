# 模块架构

[返回首页](../README.md) · [方法讲解导航](methods/README.md) · [实验结果](../experiment_results.md)

## 共享基础设施

方法说明：[统计基线](methods/statistical.md) · [损失、早停与训练](methods/training.md)。

`common/preprocessing`读取冻结输入，`common/targets`构造中心化未来20日协方差，`common/features`刻画六维风险状态，`common/losses`定义两种距离，`common/training`处理早停、种子和训练。

`statistical`包含Historical、LW、EWMA、IEWMA。它们既是独立基线，也是专家来源。LW为中心化估计并保留原相对对角扰动；EWMA保留原零均值递推，不能为了统一代码而更改均值约定。

## 主线一：动态融合与修正

按步骤阅读：[动态门控](methods/gating.md) → [凸组合](methods/convex.md)／[精度因子融合](methods/precision.md) → [Idea1](methods/idea1.md)／[Idea2](methods/idea2.md)。

```text
100×6风险特征 → 门控编码器 → Softmax权重
                                  ↓
10个统计专家 ──────────→ 凸组合 / 精度因子融合
                                  ↓
100×6风险特征 → 独立修正编码器 → Idea1 / Idea2 / 不修正
                                  ↓
                            最终协方差预测
```

- `fusion/experts`：LW20/60/120/252、五组IEWMA、对角IEWMA，共10专家。
- `fusion/gating`：Transformer及MLP/CNN序列编码，输出Softmax权重。
- `fusion/aggregation/convex.py`：直接加权协方差。
- `fusion/aggregation/precision.py`：加权逆协方差Cholesky因子，再还原矩阵；不是直接平均精度矩阵。
- `fusion/refinement/idea1.py`：历史主方向或学习方向上的低维修正，q取1/3/5。
- `fusion/refinement/idea2.py`：全维特征值调整与特征值间距正则化旋转。
- `fusion/model.py`与`refinement/core.py`：融合与修正联合训练，不冻结融合网络。

`_initialization_*`仅保持历史初始化随机数消耗顺序，不是恢复单时点门控实验。

## 主线二：网络预测矩阵

方法说明：[输入流程](methods/direct.md) · [五种输出变换](methods/output_heads.md) · [六种网络](methods/networks.md) · [CAB](methods/cab.md)。

### 输出变换

`direct/output_heads`分别实现Cholesky、Matrix Exponential、Spectral Softplus、Variance–Correlation、SPD Residual。保留训练标签平均对角元素作为内部尺度；Spectral Softplus保留Loewner连续极限梯度；Residual保留LW60对称平方根。

### 网络架构

`direct/model.py`组装Transformer与输出头；`direct/inputs/covariance.py`将100个10×10矩阵展平成100个时间Token。`direct/backbones/residual.py`维护MLP、CNN、TCN、GRU及Patch Transformer，标准Transformer复用协方差输入模型。

输出变换和架构是两个组合维度，但已有实验没有穷举：五输出固定Transformer，六架构固定Residual。不能把未运行的组合当成已有结果。

### 并列论文模块

`direct/paper_models/cab.py`提取实际行业实验调用的PaperCAB：3D CNN、7层BiLSTM、多头注意力、PSD修正及0.8/0.2历史混合。保留自身输出管线，不替换五输出头；这是架构适配，不是作者代码的完全复现。

## 实验入口与评价

`factory.py`按配置构造模型，`experiments/run.py`负责入口，统一训练器记录结果。`evaluation`保存组合求解与风险指标工具，但有工具不等于所有预测已经完成组合回测。
