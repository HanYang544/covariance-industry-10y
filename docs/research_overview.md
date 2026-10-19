# 多尺度协方差预测研究：从收益预测到风险配置的两条技术主线

[返回项目首页](../README.md) · [方法讲解导航](methods/README.md) · [实验结果](../experiment_results.md) · [完整实验协议](protocol.md)

## 摘要

收益率预测回答“哪些资产可能获得更高收益”，但投资组合构建还必须回答“这些资产一起持有时，会承担多大风险”。仅依据预测收益配置资金，可能把权重集中在共同暴露于同一风险因素的资产上；资产数量增加，也不必然意味着风险充分分散。组合风险不仅取决于各资产自身的波动，还取决于资产之间如何共同变化，因此需要对未来协方差结构进行预测。

本研究关注10个行业资产未来20个交易日的收益协方差预测。针对短期风险估计敏感但噪声较大、长期估计稳定但可能滞后的矛盾，构建两条研究主线：第一条利用多尺度统计估计作为专家，通过风险状态序列学习动态融合权重，并进一步进行结构化矩阵修正；第二条由神经网络编码历史风险信息，直接输出满足协方差约束的预测矩阵。两条路线分别体现“利用统计先验进行受约束学习”和“学习更自由的风险结构映射”，在共同预测任务下比较其精度、稳定性及潜在组合应用价值。

本文从两个层面理解多尺度：一是不同窗口与衰减速度形成的风险估计尺度，二是历史序列中不同跨度的时间依赖。两条主线都可以利用序列信息，但长序列输入本身并不构成网络已学到多尺度规律的证据。

本文介绍研究动机、任务定义和方法体系，不把尚未完成的验证或尚未确认的优势写成研究结论。公式采用Markdown中的标准LaTeX数学语法。

## 一、从收益率预测走向投资组合构建

### 1.1 收益预测只解决了资产配置的一部分问题

设有 $N$ 个资产，下一期收益率向量与投资组合权重分别为：

$$
\mathbf r_{t+1}
=
\begin{bmatrix}
r_{t+1,1}\\
\vdots\\
r_{t+1,N}
\end{bmatrix}
\in\mathbb R^N,
\qquad
\mathbf w_t
=
\begin{bmatrix}
w_{t,1}\\
\vdots\\
w_{t,N}
\end{bmatrix},
\qquad
\mathbf 1^\top\mathbf w_t=1.
$$

在这一期内给定权重，组合收益为：

$$
r_{p,t+1}=\mathbf w_t^\top\mathbf r_{t+1}.
$$

记预测时可获得的信息为 $\mathcal F_t$，条件期望收益向量与条件协方差矩阵分别为：

$$
\boldsymbol\mu_t
=
\mathbb E[\mathbf r_{t+1}\mid\mathcal F_t],
$$

$$
\boldsymbol\Sigma_t
=
\mathbb E
\left[
(\mathbf r_{t+1}-\boldsymbol\mu_t)
(\mathbf r_{t+1}-\boldsymbol\mu_t)^\top
\mid\mathcal F_t
\right].
$$

于是：

$$
\mathbb E[r_{p,t+1}\mid\mathcal F_t]
=
\mathbf w_t^\top\boldsymbol\mu_t,
\qquad
\operatorname{Var}(r_{p,t+1}\mid\mathcal F_t)
=
\mathbf w_t^\top\boldsymbol\Sigma_t\mathbf w_t.
$$

因此，即使已经得到各资产的收益率预测，也不能只按照预测收益排序来决定全部持仓：权重之间如何搭配、风险是否集中，还需要协方差信息。

这里的收益预测是引入资产配置问题的背景。本项目当前并没有训练一个新的预期收益模型，也不声称已经完成收益预测与风险预测的一体化系统。

### 1.2 为什么“买很多资产”不等于有效分散？

将组合方差展开：

$$
\sigma_{p,t}^2
=
\sum_{i=1}^{N}w_{t,i}^2\sigma_{t,i}^2
+
2\sum_{1\le i<j\le N}
w_{t,i}w_{t,j}\rho_{t,ij}\sigma_{t,i}\sigma_{t,j}.
$$

第一项来自单个资产自身的波动，第二项来自资产之间的共同变化。第二项说明，资产间的相关性会直接影响风险分散效果。

例如，两只资产的波动率都为 $\sigma$，各配置一半资金，则：

$$
\sigma_p^2
=
\frac14\sigma^2+\frac14\sigma^2+
2\cdot\frac12\cdot\frac12\rho\sigma^2
=
\frac{\sigma^2}{2}(1+\rho).
$$

当 $\rho=1$ 时，组合波动率仍为 $\sigma$，没有获得这一意义上的分散收益；当 $\rho=0$ 时，波动率降为 $\sigma/\sqrt2$。这个例子不是实际收益保证，而是说明：**分散风险依赖资产之间的联合结构，而不只是资产数量。**

### 1.3 协方差如何进入组合优化？

当收益与风险采用一致的时间尺度时，可以用均值—方差形式表达权衡：

$$
\max_{\mathbf w\in\mathcal W}
\left\{
\mathbf w^\top\widehat{\boldsymbol\mu}_t
-
\frac{\gamma}{2}
\mathbf w^\top\widehat{\boldsymbol\Sigma}_t\mathbf w
\right\},
\qquad \gamma>0.
$$

其中 $\gamma$ 表示风险厌恶程度，$\mathcal W$ 表示预算、做空和持仓上限等约束。

为了单独观察协方差预测的作用，可以不引入收益预测，而使用全局最小方差组合：

$$
\mathbf w_t^\star
=
\arg\min_{\mathbf w\in\mathcal W}
\mathbf w^\top\widehat{\boldsymbol\Sigma}_t\mathbf w.
$$

若只保留预算约束、允许做空且预测矩阵正定，由拉格朗日一阶条件可得：

$$
\mathcal J(\mathbf w,\eta)
=
\mathbf w^\top\widehat{\boldsymbol\Sigma}_t\mathbf w
-
\eta(\mathbf 1^\top\mathbf w-1),
$$

$$
2\widehat{\boldsymbol\Sigma}_t\mathbf w-\eta\mathbf 1=0
\quad\Longrightarrow\quad
\boxed{
\mathbf w_t^\star
=
\frac{
\widehat{\boldsymbol\Sigma}_t^{-1}\mathbf 1
}{
\mathbf 1^\top\widehat{\boldsymbol\Sigma}_t^{-1}\mathbf 1
}
}.
$$

若加入不做空或持仓上限约束，则需要求解相应的约束优化问题，不能直接套用上述闭式解。这也说明，协方差估计不仅影响报告中的误差数值，还可能通过矩阵求逆和优化过程改变资产配置。

## 二、为什么需要预测协方差，而不只是估计历史协方差？

历史估计描述已经发生的收益关系，而投资组合面向未来持有期。用历史矩阵直接预测未来，相当于假设近期风险结构具有足够持续性；这种假设有实用价值，但并不总能成立。

市场风险结构可能在三个方面变化：

- 风险水平变化：整体或个别资产的波动率上升、下降。
- 共同变化结构变化：资产间相关性、共同风险因素的重要性发生变化。
- 风险方向变化：主要波动集中在哪些资产组合方向上发生变化。

对一个正定协方差矩阵作特征分解：

$$
\boldsymbol\Sigma_t
=
\mathbf U_t\boldsymbol\Lambda_t\mathbf U_t^\top,
\qquad
\boldsymbol\Lambda_t
=
\operatorname{diag}(\lambda_{t,1},\ldots,\lambda_{t,N}).
$$

其特征值描述对应方向上的风险强度，特征向量描述风险方向。协方差预测不只是预测每只资产“波动有多大”，还要估计这些风险方向及其相对重要性如何演化。

本研究的目标不是完全预测所有未来冲击，而是在只使用历史信息的条件下，获得比简单沿用历史结构更有效的未来风险估计。能否实现这一点需要通过实验检验。

## 三、为什么采用多尺度视角？

### 3.1 短期敏感性与长期稳定性的矛盾

对长度为 $W$ 的历史窗口，中心化样本协方差可写为：

$$
\bar{\mathbf r}_t^{(W)}
=
\frac1W\sum_{s=t-W+1}^{t}\mathbf r_s,
$$

$$
\mathbf S_t^{(W)}
=
\frac1{W-1}
\sum_{s=t-W+1}^{t}
(\mathbf r_s-\bar{\mathbf r}_t^{(W)})
(\mathbf r_s-\bar{\mathbf r}_t^{(W)})^\top.
$$

较短窗口对新信息更敏感，但容易受到少量极端收益影响；较长窗口利用更多历史观测，但可能混入已不适用的市场状态。两者体现了噪声抑制与状态适应之间的权衡，并不存在对所有时期都最优的固定窗口。

指数加权估计则通过衰减系数控制记忆：

$$
\mathbf C_t
=
\lambda\mathbf C_{t-1}
+
(1-\lambda)\mathbf r_t\mathbf r_t^\top,
\qquad 0<\lambda<1.
$$

若用半衰期 $h$ 表示，则：

$$
\lambda=2^{-1/h}.
$$

因此，多尺度既可以通过不同滚动窗口实现，也可以通过不同指数衰减速度实现。具体统计方法及其均值处理差异见[统计基线讲解](methods/statistical.md)。

### 3.2 风险估计尺度：不同记忆长度的统计专家

本项目将不同时间尺度的统计风险估计组织为专家集合：

$$
\mathcal E_t
=
\left\{
\boldsymbol\Sigma_t^{(1)},
\ldots,
\boldsymbol\Sigma_t^{(K)}
\right\},
\qquad K=10.
$$

专家包括LW-20/60/120/252，五组不同波动率与相关性半衰期的IEWMA，以及一个对角IEWMA专家。不同专家具有不同的历史记忆和结构约束，不只是同一矩阵的重复计算。

核心问题由“哪个固定窗口最好”转变为：

> 在当前市场状态下，应该如何使用不同尺度的信息？在组合这些信息之后，是否还需要对风险结构作进一步修正？

需要区分三个概念：**估计窗口长度、网络输入序列长度、预测期限**。例如“100个历史20日协方差矩阵，预测未来20日协方差”，其中100是序列长度，20是每张历史矩阵的估计窗口，另一个20是预测标签期限。

### 3.3 时间依赖尺度：从历史序列学习风险变化

多尺度视角不只限于多个历史窗口的统计专家，也包括风险变化在不同时间跨度上的依赖。例如，近期的波动冲击、持续数周的相关性变化和更缓慢的风险状态演化，可能同时影响未来协方差。

设每个时刻的风险信息为 $\mathbf Z_s$，它可以是六维风险状态向量，也可以是历史协方差矩阵。序列编码器学习：

$$
\mathbf h_t
=g_\theta\!\left(\mathbf Z_{t-L+1},\ldots,\mathbf Z_t\right),
\qquad L=100.
$$

这里，模型可以利用较近与较远的历史观测，学习不同跨度的时间依赖。因此，使用一段历史进行学习，可以纳入本研究的多尺度视角；它与“同时构造多个估计窗口”是两个不同层面的设计。

但需要区分**学习多尺度依赖的机会**与**显式的多尺度结构**。普通序列MLP或Transformer具有利用不同滞后信息的能力，不能仅凭输入长度为100就断言已经学到了多尺度规律。采用不同膨胀率的TCN或多个时间分辨率分支，才是在结构上显式引入不同感受范围；单一卷积核或单一Patch长度本身也不足以证明多尺度建模。具体架构见[网络结构讲解](methods/networks.md)。

当前实验可以比较序列模型的预测表现，但若要证明提升确实来自不同时间尺度，还需要历史截断、不同滞后区间遮蔽等专门消融，不能将这一机制视为已验证结论。

### 3.4 两种尺度如何贯穿两条主线？

| 多尺度层面 | 关注的问题 | 在本研究中的体现 |
| --- | --- | --- |
| 风险估计尺度 | 用多长的历史、以多快的衰减估计当前风险？ | 不同窗口的LW专家与不同半衰期的IEWMA专家 |
| 时间依赖尺度 | 较近与较远的风险状态变化如何影响未来？ | 风险特征序列或历史协方差序列的网络编码 |

主线一结合这两个层面：多尺度专家提供风险估计，历史风险状态序列指导动态融合与矩阵修正。主线二主要通过历史协方差序列学习风险演化，即使每张矩阵都采用固定20日窗口，网络仍可利用不同跨度的时间依赖；因此不能把主线二简单理解为“没有多尺度视角”。

同时，两条主线并未获得完全相同的信息范围。连续100张滚动20日协方差矩阵覆盖119个交易日的收益，而部分长记忆专家使用更长历史。**估计窗口、序列长度和预测期限不能互相替代**，比较结果时应保留这一差异。

## 四、研究任务与监督目标

### 4.1 任务定义

本项目设置：

$$
N=10,\qquad F=20,\qquad L=100.
$$

其中 $N$ 是行业资产数，$F$ 是预测期限，$L$ 是网络输入历史序列长度。预测日为 $t$，模型只能使用 $\mathcal F_t$ 中的信息，输出：

$$
\widehat{\boldsymbol\Sigma}_{t,F}
=
f_\theta(\mathcal F_t)
\in\mathbb R^{N\times N}.
$$

合法的协方差预测必须满足对称性和半正定性：

$$
\widehat{\boldsymbol\Sigma}_{t,F}
=
\widehat{\boldsymbol\Sigma}_{t,F}^{\top},
\qquad
\mathbf x^\top\widehat{\boldsymbol\Sigma}_{t,F}\mathbf x\ge0
\quad
\forall\,\mathbf x\in\mathbb R^N.
$$

对需要稳定求逆的应用，本项目多数参数化进一步保证数学上的严格正定；浮点实现仍需数值检查。

### 4.2 未来20日标签如何构造？

用未来20日实际收益计算窗口均值与样本协方差：

$$
\bar{\mathbf r}_{t,F}
=
\frac1F\sum_{h=1}^{F}\mathbf r_{t+h},
$$

$$
\boxed{
\mathbf S_{t,F}
=
\frac1{F-1}
\sum_{h=1}^{F}
(\mathbf r_{t+h}-\bar{\mathbf r}_{t,F})
(\mathbf r_{t+h}-\bar{\mathbf r}_{t,F})^\top
}.
$$

这里 $F=20$，分母为19。训练时可以利用已经实现的未来窗口构造历史样本标签，实际预测时不能看到这个窗口。数据划分还要求标签不跨越训练、验证和测试边界。

$\mathbf S_{t,F}$ 是有限样本的未来风险代理，不是无噪声的真实条件协方差。只有20个收益观测，标签可能受偶发冲击影响；10个资产下它可以满秩，但不意味着估计一定稳定。

### 4.3 这个标签不是20日累计收益的协方差

本任务预测的是未来窗口内的**日收益风险结构**。若定义收益之和 $\mathbf R_{t,F}=\sum_{h=1}^{F}\mathbf r_{t+h}$，其条件协方差一般为：

$$
\operatorname{Cov}(\mathbf R_{t,F}\mid\mathcal F_t)
=
\sum_{h=1}^{F}
\operatorname{Cov}(\mathbf r_{t+h}\mid\mathcal F_t)
+
\sum_{h\ne k}
\operatorname{Cov}(\mathbf r_{t+h},\mathbf r_{t+k}\mid\mathcal F_t).
$$

因此不能无条件地把本项目输出当作20日累计收益协方差，也不能不加假设地乘以20。上述收益之和与复利累计收益也不是同一个定义。将预测用于组合时，需要明确风险尺度、调仓方式与实际收益计算口径。

## 五、主线一：多尺度统计专家的动态融合与结构修正

### 5.1 风险状态驱动的动态门控

设每个时点的六维市场风险状态为：

$$
\mathbf x_t\in\mathbb R^6,
\qquad
\mathbf X_t
=
\begin{bmatrix}
\mathbf x_{t-L+1}^{\top}\\
\vdots\\
\mathbf x_t^{\top}
\end{bmatrix}
\in\mathbb R^{L\times6}.
$$

六个指标概括波动率比例、平均相关性、吸收比、有效秩、子空间旋转和相关漂移，使用过去信息构造。门控网络读取整段状态历史：

$$
\mathbf h_t=g_\theta(\mathbf X_t),\qquad
\mathbf a_t=\mathbf W\mathbf h_t+\mathbf b,
$$

$$
\boxed{
\alpha_{t,k}
=
\frac{\exp(a_{t,k})}
{\sum_{j=1}^{K}\exp(a_{t,j})}
},
\qquad
\alpha_{t,k}>0,\quad
\sum_{k=1}^{K}\alpha_{t,k}=1.
$$

这里用 $\boldsymbol\alpha_t$ 表示专家权重，以区别于投资组合资产权重 $\mathbf w_t$。专家权重不是最终持仓比例，而是构造预测协方差的中间变量。

MLP、CNN、Transformer提供不同的序列编码方式。网络不需要“正确专家权重”标签，而是通过最终矩阵预测损失学习权重。详见[动态门控](methods/gating.md)。

### 5.2 内部融合方法一：协方差凸组合

$$
\boxed{
\mathbf C_t^{\mathrm{convex}}
=
\sum_{k=1}^{K}
\alpha_{t,k}\boldsymbol\Sigma_t^{(k)}
}.
$$

这是直接在协方差空间融合。若专家矩阵均正定，则对任何非零向量 $\mathbf z$：

$$
\mathbf z^\top\mathbf C_t^{\mathrm{convex}}\mathbf z
=
\sum_{k=1}^{K}
\alpha_{t,k}
\mathbf z^\top\boldsymbol\Sigma_t^{(k)}\mathbf z
>0.
$$

因此结果仍正定。它具有直观解释，但输出受限于专家矩阵的凸组合范围。[详细讲解](methods/convex.md)

### 5.3 内部融合方法二：精度因子融合

先对各专家的逆协方差作Cholesky分解：

$$
\left(\boldsymbol\Sigma_t^{(k)}\right)^{-1}
=
\mathbf L_{t,k}\mathbf L_{t,k}^{\top}.
$$

加权因子，再还原协方差：

$$
\bar{\mathbf L}_t
=
\sum_{k=1}^{K}\alpha_{t,k}\mathbf L_{t,k},
$$

$$
\boxed{
\mathbf C_t^{\mathrm{precision}}
=
\left(\bar{\mathbf L}_t\bar{\mathbf L}_t^\top\right)^{-1}
}.
$$

正对角下三角因子的非负加权仍可逆，因此结果正定。这不是直接加权精度矩阵，也不是与凸组合等价的写法；Cholesky因子还依赖固定的资产顺序。两种融合的优劣是实验问题，不是理论上预先确定的结论。[详细讲解](methods/precision.md)

### 5.4 为什么还要修正融合结果：从融合到风险结构外推

修正模块的主要动机是：**以融合得到的历史风险结构为起点，进一步预测未来风险强度与方向的变化，而不只是重新分配专家权重。**这里的“外推”指风险结构的预测性调整，不是简单沿时间直线延长，也不意味着输出一定落在专家范围之外。

需要区分两种融合的表达范围。凸组合满足：

$$
\mathbf C_t^{\mathrm{convex}}
\in\operatorname{conv}\!\left\{
\boldsymbol\Sigma_t^{(1)},\ldots,\boldsymbol\Sigma_t^{(K)}
\right\}.
$$

因此，对任意固定组合权重 $\mathbf w$，其预测方差被专家预测方差的最小值与最大值夹住：

$$
\min_k\mathbf w^\top\boldsymbol\Sigma_t^{(k)}\mathbf w
\leq\mathbf w^\top\mathbf C_t^{\mathrm{convex}}\mathbf w
\leq\max_k\mathbf w^\top\boldsymbol\Sigma_t^{(k)}\mathbf w.
$$

精度因子融合则不同：虽然因子权重仍是非负且和为1，但从因子到协方差的非线性映射，使其输出**可能位于专家协方差矩阵的凸包之外**。所以，不能说“所有融合都只能插值，只有修正才能外推”；精度因子融合本身就具有一定的非线性结构外推能力，但不保证每次都超出凸包，也不保证外推更准确。

增加Idea修正的意义，是在融合规则允许的变化之外，提供由历史状态指导的风险强度缩放、子空间调整或方向旋转。对于凸组合，它允许突破专家凸包的限制；对于精度因子融合，它提供额外的结构调整自由度，而不是首次赋予外推能力。专家也可能共享偏差，仅调整融合权重不足以适应未来变化。因此，将未校准融合矩阵作为外推起点：

$$
\widehat{\boldsymbol\Sigma}_{t,F}
=
\mathcal R_\phi\!\left(\mathbf C_t,\mathbf X_t\right).
$$

融合编码器与修正编码器读取同样的历史风险状态，但参数独立，并通过最终损失共同训练：

$$
\min_{\theta,\phi}
\frac1{|\mathcal T_{\mathrm{train}}|}
\sum_{t\in\mathcal T_{\mathrm{train}}}
\ell\!\left(
\mathcal R_\phi\left(\mathbf C_t(\theta),\mathbf X_t\right),
\mathbf S_{t,F}
\right).
$$

没有先把融合网络永久冻结再单独训练修正网络。下面两种改进在保持正定性的同时，允许对基础风险结构作不同程度的外推调整：Idea1主要在少数风险方向上修改，Idea2进一步调整全维风险强度与方向。修正可以放大也可以收缩风险，并非必须增加风险；其目标是降低未来协方差预测误差，而不是刻意远离专家或融合结果。是否确实改善预测，需要与对应的未修正融合模型比较，不能仅凭表达能力更强就认定有效。

### 5.5 Idea1：少数风险方向上的修正

对基础矩阵分解，并将特征值按从大到小排列：

$$
\mathbf C_t
=
\mathbf U_t\operatorname{diag}(\lambda_{t,1},\ldots,\lambda_{t,N})
\mathbf U_t^\top.
$$

网络生成正的尺度系数：

$$
m_{t,i}=\exp\!\left(2\tanh(a_{t,i})\right).
$$

方向内相关矩阵由单位对角下三角矩阵 $\mathbf T_t$ 构造，其非对角元素为 $0.5\tanh(b_{t,ij})$：

$$
\mathbf G_t=\mathbf T_t\mathbf T_t^\top,\qquad
\mathbf D_{G,t}=\operatorname{diag}
\left(\sqrt{G_{t,11}},\ldots,\sqrt{G_{t,qq}}\right),
\qquad
\mathbf R_t=\mathbf D_{G,t}^{-1}\mathbf G_t\mathbf D_{G,t}^{-1}.
$$

**主方向版本**只修改前 $q$ 个特征方向。记其余方向为 $\mathbf U_{t,r}$，则：

$$
\mathbf D_t
=
\operatorname{diag}
\left(
\sqrt{\lambda_{t,1}m_{t,1}},
\ldots,
\sqrt{\lambda_{t,q}m_{t,q}}
\right),
$$

$$
\boxed{
\widehat{\boldsymbol\Sigma}_{t,F}
=
\mathbf U_{t,q}\mathbf D_t\mathbf R_t\mathbf D_t\mathbf U_{t,q}^{\top}
+
\mathbf U_{t,r}
\operatorname{diag}(\lambda_{t,q+1},\ldots,\lambda_{t,N})
\mathbf U_{t,r}^{\top}
}.
$$

**学习方向版本**不固定使用前 $q$ 个主方向。网络输出矩阵后经QR正交化得到 $\mathbf V_t^\top\mathbf V_t=\mathbf I_q$。令 $\mathbf C_t=\mathbf B_t\mathbf B_t^\top$ 为Cholesky分解：

$$
\mathbf D_{m,t}=\operatorname{diag}(\sqrt{m_{t,1}},\ldots,\sqrt{m_{t,q}}),
\qquad
\mathbf K_t=\mathbf D_{m,t}\mathbf R_t\mathbf D_{m,t},
$$

$$
\boxed{
\widehat{\boldsymbol\Sigma}_{t,F}
=
\mathbf B_t
\left[
\mathbf I_N+\mathbf V_t(\mathbf K_t-\mathbf I_q)\mathbf V_t^\top
\right]
\mathbf B_t^\top
}.
$$

这相当于在基础矩阵白化后的 $q$ 维子空间中修改风险，其余方向保持不变。本项目比较 $q=1,3,5$；修正幅度头零初始化时，预测还原基础融合矩阵。[详细讲解](methods/idea1.md)

### 5.6 Idea2：全维风险强度与方向旋转

Idea2进一步允许调整全部特征值和特征方向。记归一化特征值与间距：

$$
\widetilde\lambda_{t,i}
=
\frac{\lambda_{t,i}}{\frac1N\sum_{j=1}^{N}\lambda_{t,j}},
\qquad
d_{t,ij}
=
\left|\widetilde\lambda_{t,i}-\widetilde\lambda_{t,j}\right|.
$$

将网络参数转成下三角旋转幅度：

$$
A_{t,ij}
=
\frac{0.1\tanh(b_{t,ij})\,d_{t,ij}}
{d_{t,ij}^{2}+0.01},
\quad i>j,
\qquad A_{t,ij}=0,\quad i\le j.
$$

通过反对称矩阵指数得到正交旋转：

$$
\mathbf Q_t
=
\exp_m(\mathbf A_t-\mathbf A_t^\top),
\qquad
\mathbf Q_t^\top\mathbf Q_t=\mathbf I_N.
$$

最终：

$$
\boxed{
\widehat{\boldsymbol\Sigma}_{t,F}
=
(\mathbf U_t\mathbf Q_t)
\operatorname{diag}(\lambda_{t,1}m_{t,1},\ldots,\lambda_{t,N}m_{t,N})
(\mathbf U_t\mathbf Q_t)^\top
}.
$$

正的谱尺度与正交旋转保持正定性。特征值间距项在方向难以区分时抑制相应旋转，但不应被解读为已经解决所有谱分解的数值问题。与Idea1相比，它具有更大的全维调整空间，也需要更严格地检验是否产生额外泛化收益。[详细讲解](methods/idea2.md)

## 六、主线二：由网络直接学习协方差矩阵映射

### 6.1 为什么设置这条对照路线？

第一条主线把统计专家作为强先验，主要学习如何选择与修正已有风险估计。第二条主线进一步提出：

> 如果不依赖多专家融合，而是直接让网络学习历史风险结构与未来风险结构的关系，能否得到更准确的预测？

这可以检验统计先验与网络自由度之间的权衡。这条路线通过历史序列学习风险演化，允许模型利用不同跨度的时间依赖，而不是仅从某一天的矩阵作静态映射。直接预测具有更大的表达空间，但可能需要更多有效数据、更适合的优化方法和更稳健的输出参数化；序列学习是否真正利用了多个时间尺度，仍需机制消融支持。

### 6.2 历史协方差序列输入

将截止日为 $s$ 的历史20日样本协方差记为 $\mathbf H_s$。构造：

$$
\mathcal H_t
=
\left(
\mathbf H_{t-L+1},\ldots,\mathbf H_t
\right)
\in\mathbb R^{L\times N\times N},
\qquad L=100.
$$

在Transformer方案中，每张矩阵展平并映射为一个时间Token：

$$
\mathbf z_s
=
\mathbf W_{\mathrm{in}}\operatorname{vec}(\mathbf H_s)
+
\mathbf b_{\mathrm{in}}
+
\mathbf p_s
\in\mathbb R^{32}.
$$

这里的矩阵先经过训练集拟合的元素级标准化，公式为简洁起见省略了该预处理符号。随后：

$$
\mathbf h_t
=
\operatorname{Last}
\left[
\operatorname{Transformer}
(\mathbf z_{t-L+1},\ldots,\mathbf z_t)
\right],
$$

$$
\widehat{\boldsymbol\Sigma}_{t,F}
=
\mathcal D_\psi(\mathbf h_t).
$$

当前矩阵输入实现是“每张矩阵一个时间Token”，不是“每个资产一个Token”的双重时空Attention。另一组对照使用100×6风险特征序列作为输入。[输入流程讲解](methods/direct.md)

### 6.3 为什么输出矩阵不能直接使用普通线性层？

普通线性层可以输出任意 $N^2$ 个数，但不能自动保证矩阵对称、半正定或可稳定求逆。因此将矩阵合法性作为输出参数化的一部分，而不是仅依靠训练自行学会。

以下公式使用数学尺度描述；实际实现以训练标签平均对角元素 $u>0$ 为内部单位，再还原到原协方差单位。

**方法一：Softplus-Cholesky。**

输出 $N(N+1)/2=55$ 个参数组成下三角矩阵：

$$
L_{t,ii}=\operatorname{softplus}(a_{t,ii})+\varepsilon,\qquad
\widehat{\boldsymbol\Sigma}_{t,F}=u\mathbf L_t\mathbf L_t^\top.
$$

**方法二：矩阵指数。**

网络输出对称矩阵 $\mathbf A_t$：

$$
\widehat{\boldsymbol\Sigma}_{t,F}=u\exp_m(\mathbf A_t).
$$

**方法三：谱Softplus。**

若 $\mathbf A_t=\mathbf Q_t\operatorname{diag}(\nu_{t,i})\mathbf Q_t^\top$：

$$
\widehat{\boldsymbol\Sigma}_{t,F}
=
u\mathbf Q_t
\operatorname{diag}\!\left(\operatorname{softplus}(\nu_{t,i})+\varepsilon\right)
\mathbf Q_t^\top.
$$

**方法四：方差—相关双头。**

网络分别输出正标准差与对称矩阵：

$$
\sigma_{t,i}=\operatorname{softplus}(a_{t,i})+\varepsilon,
\qquad
\mathbf P_t=\exp_m(\mathbf A_t),
$$

$$
\mathbf D_{P,t}
=
\operatorname{diag}\!\left(\sqrt{P_{t,11}},\ldots,\sqrt{P_{t,NN}}\right),
\qquad
\mathbf R_t=\mathbf D_{P,t}^{-1}\mathbf P_t\mathbf D_{P,t}^{-1},
$$

$$
\widehat{\boldsymbol\Sigma}_{t,F}
=
u\operatorname{diag}(\boldsymbol\sigma_t)
\mathbf R_t
\operatorname{diag}(\boldsymbol\sigma_t).
$$

**方法五：SPD Residual。**

以LW-60为历史基准，令 $\mathbf B_t=(\boldsymbol\Sigma_t^{LW60}/u)^{1/2}$ 为对称平方根：

$$
\boxed{
\widehat{\boldsymbol\Sigma}_{t,F}
=
u\mathbf B_t\exp_m(\mathbf A_t)\mathbf B_t^\top
}.
$$

当 $\mathbf A_t=0$，输出还原LW-60。该方法学习的是基准上的结构变化，而非完全从零生成矩阵。因此它包含统计先验，需要与前四种方法区别解释。五种变换的数学与实现细节见[输出头讲解](methods/output_heads.md)。

### 6.4 编码器比较与论文架构模块

输出变换与网络编码器是两个不同的设计维度。为避免一次改变太多因素，本项目分别进行：

- 固定Transformer，比较五种输出变换。
- 固定LW-60 Residual输出，比较MLP、CNN、TCN、GRU、Transformer与Patch Transformer。
- 并列加入CAB论文架构适配，保留其3D CNN、BiLSTM、Attention与原有PSD修正和历史混合流程。

CAB最终使用：

$$
\widehat{\boldsymbol\Sigma}_{t,F}^{CAB}
=
0.8\,\boldsymbol\Sigma_t^{DL,+}
+
0.2\,\mathbf H_t,
$$

其中 $\boldsymbol\Sigma_t^{DL,+}$ 是对称化并截断负特征值后的网络输出。CAB并非完全无历史先验的网络；其训练配方也与其他网络不同，不能被视作严格的单因素编码器消融。

[网络结构讲解](methods/networks.md) · [CAB适配边界](methods/cab.md)

## 七、统一训练、验证与评价

### 7.1 两种损失分别训练

对于一组训练样本 $\mathcal T$，Frobenius目标为：

$$
\boxed{
\mathcal L_F
=
\frac1{|\mathcal T|}
\sum_{t\in\mathcal T}
\left\|
\widehat{\boldsymbol\Sigma}_{t,F}-\mathbf S_{t,F}
\right\|_F
=
\frac1{|\mathcal T|}
\sum_{t\in\mathcal T}
\sqrt{
\sum_{i=1}^{N}\sum_{j=1}^{N}
\left(
\widehat\Sigma_{t,F}^{ij}-S_{t,F}^{ij}
\right)^2
}
}.
$$

Euclidean目标只取含对角线的上三角：

$$
\boxed{
\mathcal L_E
=
\frac1{|\mathcal T|}
\sum_{t\in\mathcal T}
\sqrt{
\sum_{1\le i\le j\le N}
\left(
\widehat\Sigma_{t,F}^{ij}-S_{t,F}^{ij}
\right)^2
}
}.
$$

两者均不平方。Frobenius完整矩阵会把对称非对角误差计入两次，Euclidean只计一次。因此采用两套独立训练与验证选择，不把两个指标的绝对数值直接互相排名。

### 7.2 以验证集选择，不以测试集反复优化

参数按训练损失更新，用验证集选择检查点：

$$
e^\star
=
\arg\min_e
\mathcal L_{\mathrm{validation}}(\theta_e),
\qquad
\theta^\star=\theta_{e^\star}.
$$

测试阶段使用锁定后的 $\theta^\star$。如果另行选择学习率等超参数，也应依据验证集而非测试排序。

当前训练、验证、测试年份分别为2008–2017、2018–2020、2021–2025。多种子结果采用均值与样本标准差：

$$
\bar m=\frac1S\sum_{s=1}^{S}m_s,
\qquad
s_m=
\sqrt{
\frac1{S-1}
\sum_{s=1}^{S}(m_s-\bar m)^2
},
\qquad S=3.
$$

这些种子衡量训练随机性，不等于独立市场时期的重复实验。具体优化器、早停及CAB差异见[实验协议](protocol.md)。

### 7.3 预测误差与组合用途分开评价

较低的矩阵误差不必然带来较低的组合风险。对某个固定权重：

$$
\left|
\mathbf w^\top
(\widehat{\boldsymbol\Sigma}-\boldsymbol\Sigma)
\mathbf w
\right|
\le
\left\|
\widehat{\boldsymbol\Sigma}-\boldsymbol\Sigma
\right\|_2
\|\mathbf w\|_2^2
\le
\left\|
\widehat{\boldsymbol\Sigma}-\boldsymbol\Sigma
\right\|_F
\|\mathbf w\|_2^2.
$$

这个界说明矩阵误差与风险误差存在联系，但并不保证不同预测模型的组合排名一致。实际优化得到的权重本身也依赖预测矩阵，求逆、持仓约束与交易规则会进一步影响结果。

因此研究应同时关注两类证据：矩阵预测质量，以及在固定组合规则下的波动、回撤、收益与换手表现。本文不把尚未完成的三种子组合回测描述为已经完成；已有结果及完成状态以[实验结果文档](../experiment_results.md)为准。

## 八、代表性预测结果：两条主线的实证比较

### 8.1 展示口径

以下摘录[完整实验结果](../experiment_results.md)，仅展示协方差预测误差，不包含交易、收益、回撤或换手指标。训练期为2008—2017年，验证期为2018—2020年，测试期为2021—2025年；预测目标均为未来20日中心化样本协方差。

Frobenius与Euclidean是**分别训练、分别验证早停的两个实验分支**，表中两列不是同一个检查点的两种评分。Euclidean取含对角线的上三角元素距离，两种指标都越低越好，但不能直接比较两列的绝对大小。神经网络结果为三个种子20261012/13/14的均值（样本标准差），统计基线为确定性结果。

本节按研究问题摘选代表配置，也保留修正失败的反例；“最好”仅指所述已完成实验组内的测试均值最低，是事后描述，不是用测试集选出的可部署模型，也不是统计显著性结论。完整表包含未展示的配置。本节网络输出头与架构比较固定使用原学习率0.01，不混入尚未完成的学习率搜索。

### 8.2 从统计估计到动态融合

| 方法 | Frobenius | Euclidean |
| --- | ---: | ---: |
| Historical-20 | 0.001115056 | 0.000892859 |
| LW-60 | 0.001015495 | 0.000813958 |
| EWMA-0.97 | 0.001002044 | 0.000802098 |
| IEWMA-21-63 | 0.000977765 | 0.000792913 |
| Transformer门控＋凸组合 | 0.000946294（0.000041828） | 0.000763165（0.000031611） |
| Transformer门控＋精度因子融合 | 0.000901160（0.000003638） | 0.000727080（0.000002918） |

IEWMA-21-63是完整统计基线表中两项误差均最低的方法。相对于它，Transformer精度因子融合的Frobenius和Euclidean均值分别降低约7.83%和8.30%。相同Transformer门控下，精度因子融合也优于凸组合，说明融合空间的选择值得研究，而不只是更换编码器。

但这些差异不能直接证明网络已经学到了有效的动态机制：部分基础融合运行在验证集选中了epoch0，统计先验和初始化也可能贡献效果。还需结合训练过程与机制消融判断。

### 8.3 两种外推Idea：哪些修正有帮助？

下表选取CNN分支，便于在同一种编码器内部观察两种融合及修正的关系。Idea1分别展示主方向与学习方向；Idea2调整全维风险强度与方向。修正网络与融合网络联合训练，不冻结融合网络。

| CNN配置 | Frobenius | Euclidean | 相对对应纯融合改善率（F / E） |
| --- | ---: | ---: | ---: |
| 凸组合，不修正 | 0.000972660（0.000019043） | 0.000789606（0.000006673） | — |
| 凸组合＋Idea1主方向，q=1 | 0.000909912（0.000007463） | 0.000738533（0.000006957） | 6.45% / 6.47% |
| 凸组合＋Idea1学习方向，q=5 | 0.000910768（0.000003997） | 0.000759977（0.000041614） | 6.36% / 3.75% |
| 凸组合＋Idea2，全10维 | 0.000904002（0.000001492） | 0.000733703（0.000001569） | 7.06% / 7.08% |
| 精度因子融合，不修正 | 0.000926907（0.000044833） | 0.000747718（0.000036512） | — |
| 精度因子融合＋Idea1主方向，q=5 | 0.000901871（0.000001764） | 0.000727126（0.000001198） | 2.70% / 2.75% |
| 精度因子融合＋Idea2，全10维 | 0.000899882（0.000000768） | 0.000725465（0.000000932） | 2.92% / 2.98% |

改善率按测试均值计算：

$$
\operatorname{Improvement}
=\frac{\bar e_{\mathrm{base}}-\bar e_{\mathrm{refined}}}
{\bar e_{\mathrm{base}}}\times100\%.
$$

这里有三点值得展示。第一，凸组合上的修正改善较大，与“在受限融合基础上增加风险结构外推能力”的设计动机相符，但尚不能据此证明改善就是由超出凸包造成的。第二，精度因子融合已有非线性表达能力，叠加修正仍可能改善，并非只能在凸组合上使用。第三，CNN＋精度因子融合＋Idea2在已完成的联合修正配置中，两种测试均值均最低。

不过，**相对自身基础模型的改善，不等于大幅超过最强未修正模型**。上述Idea2相对Transformer精度因子融合，Frobenius仅降低约0.14%，Euclidean约0.22%；这点差距不能宣称为显著优势。三个种子的标准差也只反映训练随机性，不是市场样本的置信区间。

修正并非普遍有效。例如，Transformer精度因子融合加Idea2后，Frobenius为0.000935051（0.000017744），Euclidean为0.000759007（0.000015917），较对应纯融合分别变差3.76%和4.39%。因此，目前更合理的结论是“合适的编码器与修正结构搭配能够改善部分融合配置”，而不是“增加外推模块必然更好”。

### 8.4 直接预测：五种矩阵输出方式

固定输入为100个历史20日协方差矩阵，固定Transformer编码器，比较五种输出方式：

| 输出方式 | Frobenius | Euclidean |
| --- | ---: | ---: |
| Softplus-Cholesky | 0.001179262（0.000022920） | 0.001015124（0.000027734） |
| Matrix Exponential | 0.001169420（0.000030338） | 0.000999208（0.000042741） |
| Spectral Softplus | 0.001214182（0.000074523） | 0.001003875（0.000044778） |
| Variance–Correlation | 0.001217000（0.000015954） | 0.001038344（0.000016507） |
| SPD Residual（LW-60基准） | 0.000964482（0.000021059） | 0.000804295（0.000035343） |

当前配方下，Residual两项误差最低；相对LW-60本身，分别降低约5.02%和1.19%。这支持进一步研究“从历史风险基准出发学习变化”，但Residual额外使用LW-60且初始化不同，不能将优势全部归因于正定输出变换。

输入也有影响：同一Transformer Residual使用100×6风险特征时，Frobenius为0.001059256（0.000043190），Euclidean为0.000837579（0.000028672），均高于历史矩阵输入版本。这是输入表达值得研究的线索，但两组输入的信息量、投影参数量及标准化方式不同，并非只改变单个因素的严格消融。

### 8.5 直接预测：固定Residual后的网络比较

前六行使用相同历史协方差输入与LW-60 Residual输出。CAB单列为论文架构适配参照，其学习率、批次更新和输出处理不同，不属于严格只替换编码器的消融。

| 网络 | Frobenius | Euclidean |
| --- | ---: | ---: |
| Transformer＋Residual | 0.000964482（0.000021059） | 0.000804295（0.000035343） |
| 序列MLP＋Residual | 0.000975807（0.000004385） | 0.000781826（0.000003536） |
| CNN＋Residual | 0.000969112（0.000006534） | 0.000779183（0.000001335） |
| TCN＋Residual | 0.000973473（0.000019112） | 0.000792692（0.000022048） |
| GRU＋Residual | 0.000987051（0.000010440） | 0.000799440（0.000010972） |
| Patch Transformer＋Residual | 0.000991939（0.000010143） | 0.000867064（0.000065779） |
| CAB论文架构适配 | 0.001062480（0.000024985） | 0.000989439（0.000100253） |

在六种Residual架构中，Transformer的Frobenius测试均值最低，CNN的Euclidean测试均值最低，没有一个编码器在两项任务上都占优。五输出头表与本表的Transformer Residual是同一组结果复用，不构成两次独立验证。

### 8.6 预测层面的阶段性结论

目前较清晰的结果是：动态融合优于所列单个统计基线；部分Idea修正进一步改善其对应基础融合；直接预测中的Residual比无历史基准的输出方式更有竞争力，但上述直接预测配置尚未超过Transformer精度因子融合。CNN＋精度因子融合＋Idea2取得较低误差，但对最强纯融合的增益很小。

这些观察更支持“统计结构先验＋适度数据驱动调整”的研究路线，而不是“网络越复杂越好”。同时，专家历史范围更长、架构参数量不等、测试集已反复查看，均限制了因果解释和泛化结论。本节不据矩阵误差推断实际投资表现。

## 九、两条主线共同回答什么问题？

| 研究问题 | 主要对照 |
| --- | --- |
| 单个历史尺度是否足够？ | 统计基线与多尺度专家体系 |
| 网络是否利用了不同跨度的历史依赖？ | 现有序列架构比较提供初步线索；历史截断与滞后区间消融仍需专门验证 |
| 状态依赖的融合是否有效？ | 动态融合、初始化及相应机制检查 |
| 选择专家之后是否仍需修正？ | 基础融合与Idea1/Idea2 |
| 统计先验能否降低学习难度？ | 专家融合、直接输出与LW-60 Residual |
| 输出合法性与网络结构如何影响学习？ | 五输出头与六架构两组实验 |
| 预测改善是否具有组合意义？ | 相同组合规则下的风险与交易指标 |

研究的重点不应是预先证明某一类网络一定最好，而是识别：在日频、有限样本、重叠标签的条件下，哪些结构约束能帮助模型稳定利用历史信息，哪些额外自由度确实带来可验证的增益。

需要注意，本项目已经多次查看测试集；验证集选模和多种子汇总不能把这个测试集重新变成全新的独立验证样本。不同方法的信息范围、训练配方和参数量也并不完全相同，解释结果时必须保留这些边界。

## 十、研究逻辑总结

整个研究可以概括为：

$$
\boxed{
\text{收益预测与资产选择}
\;\longrightarrow\;
\text{组合风险与分散化需求}
\;\longrightarrow\;
\text{未来协方差预测}
}
$$

在此基础上，两条方法路线分别是：

$$
\boxed{
\text{多尺度统计专家}
\;\longrightarrow\;
\text{状态驱动动态融合}
\;\longrightarrow\;
\text{结构化风险修正}
}
$$

$$
\boxed{
\text{历史风险序列}
\;\longrightarrow\;
\text{神经网络编码}
\;\longrightarrow\;
\text{合法协方差矩阵输出}
}
$$

两条路线共同置于“风险估计尺度＋时间依赖尺度”的研究视角下：第一条路线强调多尺度统计先验与序列驱动的融合修正，第二条路线强调从历史序列学习风险结构映射的表达能力。这里的多尺度是研究视角与建模设计，不意味着所有网络都已被证明学到了多尺度机制。通过在统一任务上比较两者，研究希望回答的不是“神经网络能否取代所有统计方法”，而是“如何将历史统计结构与数据驱动学习结合，形成更可靠、对组合配置更有意义的风险预测”。

## 十一、局限性与未来研究

### 11.1 数据规模有限，未来协方差标签本身也有噪声

当前训练集包含2498个预测日，但相邻100步输入高度重叠，相邻20日标签共享19日收益，因此独立信息量低于表面样本数。未来20日样本协方差也只是潜在风险的带噪声代理，而非无误差的“真实风险”。这可能增加完整矩阵预测的学习难度，但尚不能认定“日频数据少”是直接预测落后于融合的唯一或主要原因，更不能仅凭模型效果判断原始数据质量差。

未来可以固定模型，比较不同训练历史长度的学习曲线，并检查缺失值、异常值、收益单位和市场时期差异；再在其他行业或ETF数据上检验稳定性。更高频数据可用于构造实现协方差标签，但这会改变数据与监督任务，需要处理不同步交易、微观结构噪声等问题，不能视为简单增加训练行数。跨数据集比较也应先统一标签、采样和验证规则。

### 11.2 预测较好不等于已经证明动态机制有效

当前部分门控与修正模型很早就达到验证最优，甚至选中epoch0。较好的测试误差可能同时来自统计专家、初始化和训练更新；损失下降也不等于模型确实利用了时间顺序或市场状态。专家间高度相似时，不同权重还可能产生近似相同的矩阵，不能仅凭权重曲线解释经济机制。

未来应将训练后的模型与其自身初始化预测对照，并补充训练集拟合的线性或Ridge对照、历史截断和滞后区间遮蔽实验。线性对照若输出专家权重，需要明确是否约束非负及和为1，并检查最终矩阵合法性。时间打乱或常量状态输入可作为诊断，但可能引入分布偏移，不宜单独作为机制结论。还可检查状态变化时权重和预测矩阵是否同步改变，而不只展示注意力或Softmax权重。

### 11.3 两条主线的信息范围与训练条件尚不完全对齐

100个滚动20日矩阵只覆盖119个交易日收益，而专家包括LW-252和更长记忆的IEWMA。不同网络的参数量、初始化及训练配方也有差异。因此，当前结果不能把动态融合的优势全部归因于结构设计，也不能证明Transformer或其他编码器普遍不适合该任务。

未来可设置两组更严格的对照：一组限制专家与网络使用相同原始历史范围；另一组向直接预测网络提供与专家对应的多窗口矩阵信息。在此基础上，再控制参数规模和调参预算，比较固定窗口序列、显式多尺度分支与不同时间感受野。这样才能逐步区分长期信息、统计先验与序列建模各自的贡献。

### 11.4 更强的外推自由度也可能放大估计误差

Idea1与Idea2允许融合矩阵之外的结构调整，但正定约束只保证输出合法，并不保证条件数适宜、风险方向准确或预测更好。当前修正既有改善也有退化，说明额外自由度可能适应未来变化，也可能拟合标签噪声。精度因子融合还依赖固定资产顺序，其优势不能直接推广到任意资产排列。

未来可以记录修正前后的特征值、条件数、风险方向变化与修正幅度，检验改善来自哪些时期、是否确实超出专家凸包，而不是把“外推”仅作为结构描述。还可研究向恒等修正收缩的正则化，例如对于指数残差形式：

$$
\widehat{\boldsymbol\Sigma}_t
=\mathbf B_t\exp_m(\mathbf S_t)\mathbf B_t^\top,
\qquad
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{prediction}}
+\lambda_{\mathrm{reg}}\|\mathbf S_t\|_F^2,
\qquad \lambda_{\mathrm{reg}}\geq0.
$$

这里 $\mathbf S_t=\mathbf0$ 对应不修正，惩罚项限制偏离基础矩阵的程度。这是后续可验证的方案，不是本文已报告实验的训练目标。对Idea1/Idea2可相应约束尺度变化和旋转幅度，并通过验证集选择约束强度。精度因子融合则可增加一致的资产排列敏感性检查，判断结果是否过度依赖排序。

### 11.5 统一学习率不代表每个网络都已充分优化

多数实验使用统一训练配方，便于比较，但不保证每种网络和输出头都处于合适的优化条件。早期验证最优可能与学习率、过拟合、初始化或时期差异有关；仅凭早停轮次无法区分这些原因。

未来应保留训练与验证损失、梯度范数和输出谱的记录，在相同验证预算下比较学习率、正则化和初始化。已有学习率搜索的结果应与原固定配方结果分开呈现，完成状态以完整实验报告为准。重点是判断网络是否稳定学到了可泛化的信息，而不是持续依据测试成绩更换配置。

### 11.6 独立验证与统计不确定性仍需补强

三个随机种子衡量训练随机性，不代表三个独立市场样本。当前测试集已被多次查看，精选较优配置存在事后选择偏差；尤其Idea2相对最强纯融合的均值优势较小，不足以支持显著优越的结论。

未来应先锁定模型与调参规则，再使用未参与开发的新时期或独立数据集验证。也可采用严格按时间推进的滚动验证，确保每次标准化和训练只使用当时可用数据，且未来标签不跨越划分边界。比较逐日误差差值时，应使用考虑序列相关性的区块重采样等方法，并检查区块长度敏感性；不能将高度重叠的每日标签当作独立样本直接推断显著性。Frobenius与Euclidean之外，可补充方差、相关性和谱结构误差，以定位改善来源，而不把不同尺度的指标混成一个排名。

### 11.7 后续研究的优先顺序

建议先补强证据，再扩大模型复杂度：

1. **确认学到了什么**：检查训练过程、初始化贡献，补充线性对照和时序信息消融。
2. **对齐比较条件**：统一历史信息范围与调参预算，再判断动态融合和直接预测的差异。
3. **约束并解释外推**：分析修正幅度、风险方向与分时期表现，检验收缩正则化是否有效。
4. **验证能否迁移**：锁定配置，在新时期或其他资产数据上检验，并报告考虑时间相关性的误差不确定性。

上述方向均以预测研究为主，属于后续计划或待补强证据，不作为已完成实验的结论。当前最值得继续追问的是：**统计先验在哪些条件下真正降低学习难度，网络又能在什么条件下可靠地补充历史估计无法表达的未来风险变化？**

[继续阅读具体方法](methods/README.md) · [查看实验结果](../experiment_results.md) · [返回项目首页](../README.md)
