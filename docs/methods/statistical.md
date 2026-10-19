# 基础统计方法

[方法导航](README.md) · [项目首页](../../README.md) · [实验结果](../../experiment_results.md)

这些方法不训练神经网络。窗口或半衰期决定使用历史信息的速度，既可以单独预测，也可以提供动态融合的专家矩阵。

## Historical

**含义：**用过去窗口内的收益，估计接下来一段时间的风险。

$$
S_t^{(W)}=\frac{1}{W-1}\sum_{s=t-W+1}^{t}(r_s-\bar r)(r_s-\bar r)^\top.
$$

对角线是各资产方差，非对角线是资产间协方差。历史基线使用20日窗口。短窗口反应快，但样本估计波动可能较大。

[代码：historical.py](../../src/covariance_research/statistical/historical.py)

## LW

**含义：**将历史样本矩阵向更规则的目标收缩，减轻估计噪声。

$$
\widehat\Sigma=(1-\delta)S+\delta\mu I,\qquad
\mu=\frac{\operatorname{tr}(S)}{N}.
$$

这里的实现调用中心化的scikit-learn LedoitWolf，由估计器确定收缩强度。注意其经验协方差使用窗口长度作为分母，与上面的无偏样本协方差约定不同；之后还加相对对角扰动以稳定矩阵运算。

LW-20、LW-60、LW-120、LW-252分别表示历史窗口长度，不是网络层数，也不是预测期限。

[代码：ledoit_wolf.py](../../src/covariance_research/statistical/ledoit_wolf.py)

## EWMA

**含义：**近期收益权重更大，历史影响按指数衰减。本项目保留零均值二阶矩递推：

$$
C_t=\lambda C_{t-1}+(1-\lambda)r_tr_t^\top.
$$

EWMA-0.94与EWMA-0.97中的数字为衰减系数；越接近1，历史记忆越长。它与中心化样本协方差的均值处理不同，这一差异没有在整理时改掉。

[代码：ewma.py](../../src/covariance_research/statistical/ewma.py)

## IEWMA

**含义：**资产波动率和资产间相关性用不同速度更新。

先用上一时刻的波动率标准化当日收益，再更新相关性状态：

$$
z_{t,i}=\operatorname{clip}\left(\frac{r_{t,i}}{\sqrt{v_{t-1,i}}}\right),\qquad
A_t=\lambda_c A_{t-1}+(1-\lambda_c)z_tz_t^\top.
$$

波动率另行更新，并将相关性状态归一化：

$$
v_t=\lambda_vv_{t-1}+(1-\lambda_v)r_t^2,\qquad
R_t=\operatorname{diag}(A_t)^{-1/2}A_t\operatorname{diag}(A_t)^{-1/2},
$$

$$
C_t=\operatorname{diag}(\sqrt{v_t})R_t\operatorname{diag}(\sqrt{v_t}).
$$

IEWMA-21-63分别指波动率与相关性的半衰期，衰减系数为 $2^{-1/h}$。本项目还包含标准化收益截断、特定专家的对角膨胀与微小对角扰动；不是把所有变体都当成完全相同的经典配方。

[代码：iewma.py](../../src/covariance_research/statistical/iewma.py)

## 这些方法如何组成专家？

四个LW尺度、五组IEWMA半衰期、一个对角IEWMA，共10个专家。EWMA-0.94/0.97和Historical-20是独立基线，不属于这10个融合专家。

[专家定义](../../src/covariance_research/fusion/experts/__init__.py) → [下一步：动态门控](gating.md)
