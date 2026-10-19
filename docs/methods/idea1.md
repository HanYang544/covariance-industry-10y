# Idea1：少数风险方向上的修正

[方法导航](README.md) · [凸组合](convex.md) ／ [精度因子融合](precision.md) · [Idea2](idea2.md) · [实验结果](../../experiment_results.md)

## 它接在什么位置？

先得到未校准的融合矩阵 $C_t$；另一个独立序列编码器读取同样的100×6风险特征，输出修正参数。两个网络共同训练，融合不冻结。已比较 $q=1,3,5$。

正尺度系数使用：

$$
m_i=\exp(2\tanh(a_i)).
$$

方向内的相关结构由单位对角下三角矩阵 $T$生成Gram矩阵，再归一化为相关矩阵 $R$；非对角参数为 $0.5\tanh(b)$。这让修正结构保持正定。

## 版本一：修正最大的q个主风险方向

对融合矩阵按特征值降序分解：

$$
C=U\operatorname{diag}(\lambda)U^\top.
$$

前q个方向记为 $U_q$，其余为 $U_r$。令 $D=\operatorname{diag}(\sqrt{\lambda_i m_i})$，则：

$$
\widehat C=U_qDRDU_q^\top+
U_r\operatorname{diag}(\lambda_{q+1},\ldots,\lambda_N)U_r^\top.
$$

含义：只调整最大的q个风险方向内的尺度与联系，其余方向保留。q=1时没有方向内相关参数，只调一个尺度。

## 版本二：修正网络学习的q个方向

网络输出一个 $N\times q$矩阵，经QR分解得到正交列矩阵 $V$。此处的方向位于基础矩阵Cholesky白化后的坐标中，不应直接等同于原矩阵的前q个特征向量。

令 $C=BB^\top$，正定小矩阵为 $K$，则：

$$
\widehat C=B\left[I+V(K-I_q)V^\top\right]B^\top.
$$

在V张成的子空间内应用K，其余方向保留恒等变换，因此整体保持正定。

## 初始化与解释边界

修正幅度头零初始化，此时两个版本都还原基础融合矩阵。随训练开始学习非零修正，但epoch0不能作为学到修正的证据。历史主方向更受约束，学习方向更灵活；不能预先认定后者一定更好。

[修正计算](../../src/covariance_research/fusion/refinement/idea1.py) · [参数构造](../../src/covariance_research/fusion/refinement/core.py) · [序列联合模型](../../src/covariance_research/fusion/model.py)
