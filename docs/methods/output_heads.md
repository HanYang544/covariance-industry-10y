# 五种协方差输出变换

[方法导航](README.md) · [输入流程](direct.md) · [网络架构](networks.md) · [实验结果](../../experiment_results.md)

网络隐藏状态为32维；下列前三种及Residual输出55个参数，对应10×10下三角或对称矩阵。Variance–Correlation额外输出10个标准差参数。

公式先写无量纲输出。实际实现乘回训练标签平均对角元素u，保持原收益协方差计量单位。ε=10⁻⁶作用于无量纲参数化；浮点环境仍需检查矩阵是否合法。

## Cholesky

将55个参数填成下三角矩阵L，对角使用Softplus：

$$
L_{ii}=\operatorname{softplus}(a_{ii})+\varepsilon,\qquad
\widehat\Sigma=uLL^\top.
$$

L可逆，因此输出正定。无需特征分解，但参数之间存在下三角结构耦合。

[代码](../../src/covariance_research/direct/output_heads/cholesky.py)

## Matrix Exponential

将55个参数组成对称矩阵S：

$$
\widehat\Sigma=u\exp_m(S).
$$

实对称矩阵的指数为正定矩阵。注意这是矩阵指数，不是对每个元素单独求指数。输出尺度过大时仍可能出现数值问题。

[代码](../../src/covariance_research/direct/output_heads/matrix_exp.py)

## Spectral Softplus

设 $S=Q\operatorname{diag}(\lambda)Q^\top$：

$$
\widehat\Sigma=uQ\operatorname{diag}(\operatorname{softplus}(\lambda_i)+\varepsilon)Q^\top.
$$

平滑地把特征值映射到正数，而不是硬截断。本项目使用Loewner分差，并在接近重根时使用连续极限计算梯度。

[代码](../../src/covariance_research/direct/output_heads/spectral_softplus.py)

## Variance Correlation

标准差头输出 $\sigma_i=\operatorname{softplus}(a_i)+\varepsilon$；另一个头输出对称矩阵S。先构造正定矩阵并归一化：

$$
P=\exp_m(S),\qquad R=\operatorname{diag}(P)^{-1/2}P\operatorname{diag}(P)^{-1/2},
$$

$$
\widehat\Sigma=u\operatorname{diag}(\sigma)R\operatorname{diag}(\sigma).
$$

把风险强度与相关结构分开表示。当前仍使用同一个最终矩阵损失，没有额外加入方差或相关性辅助损失。

[代码](../../src/covariance_research/direct/output_heads/variance_correlation.py)

## SPD Residual

用LW60历史基准构造 $B=(\Sigma^{LW60}/u)^{1/2}$，网络输出对称变化S：

$$
\widehat\Sigma=uB\exp_m(S)B^\top.
$$

S=0时输出精确还原LW60；本项目残差头零初始化，所以它从统计基准开始学习，而不是从任意随机风险矩阵开始。这里使用对称平方根，不是Cholesky根。

它增加了历史先验，不能把它优于其他输出头的效果全部解释为“正定变换更好”。

[代码](../../src/covariance_research/direct/output_heads/spd_residual.py) · [下一步：固定Residual比较网络](networks.md)
