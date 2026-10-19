# 损失、早停与结果解释

[方法导航](README.md) · [数据协议](../protocol.md) · [运行方法](../running.md) · [实验结果](../../experiment_results.md)

## 监督标签

未来20个交易日收益组成标签：

$$
\bar r_t=\frac1{20}\sum_{h=1}^{20}r_{t+h},\qquad
S_t=\frac1{19}\sum_{h=1}^{20}(r_{t+h}-\bar r_t)(r_{t+h}-\bar r_t)^\top.
$$

它是有限样本的风险代理，不是无噪声的真实条件协方差。标签不跨训练/验证/测试边界。

## Frobenius

$$
\mathcal L_F=\frac1B\sum_{t=1}^{B}
\sqrt{\sum_{i=1}^{10}\sum_{j=1}^{10}(\widehat\Sigma_{t,ij}-S_{t,ij})^2}.
$$

对称矩阵的非对角误差在完整矩阵中出现两次。这里使用范数本身，不取平方。

## Euclidean

$$
\mathcal L_E=\frac1B\sum_{t=1}^{B}
\sqrt{\sum_{1\le i\le j\le10}(\widehat\Sigma_{t,ij}-S_{t,ij})^2}.
$$

只计含对角线的上三角，各独立元素不额外加权。两种损失分别训练，各自以同名验证指标早停，不把两列数值直接排名。

## 梯度怎么传回去？

动态融合：损失 → 预测矩阵 → 融合运算 → Softmax → 门控编码器。

联合修正：损失 → 修正结果 → 修正参数及基础融合矩阵 → 两个编码器。

直接预测：损失 → SPD输出变换 → 矩阵参数 → 序列编码器。

## 早停与随机种子

一般模型允许epoch0参与选择；30轮没有超过min_delta的验证改善就停止，使用验证最优检查点测试，不使用最后一轮。CAB从第1轮开始。

三种子均值（样本标准差）反映优化随机性，不是三个独立市场样本。第0轮成绩不是训练增益；验证损失下降不自动证明网络利用了时间顺序，测试优于初始化也不保证优于强统计基线。

新训练入口记录训练和验证两条曲线。旧日志缺失的训练曲线不会补造。

[标签实现](../../src/covariance_research/common/targets/__init__.py) · [损失与早停](../../src/covariance_research/common/training/__init__.py) · [统一训练入口](../../src/covariance_research/common/training/runner.py)
