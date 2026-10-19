# 单种子序列融合＋Idea修正快速比较

固定种子20261012，共84项（21配置×2融合×2损失）；当前完成84/84项。已完成的同种子验证检查点直接复用，其他种子不再补跑。

融合和修正均使用100×6历史特征，独立编码器、端到端联合训练。MLP/CNN/Transformer各做Idea1谱方向q1/3/5、学习方向q1/3/5、Idea2全10维。设置与此前联合修正一致，未更改学习率、精度、早停或数据。

以下为单次测试成绩，不是三种子均值，无标准差。纯融合参考也取同一固定种子，未挑选最优种子。已完成的三种子历史结果保留，但不混入本表。

### 优先展示：优于对应基础动态融合的代表方案

筛选口径：固定同一种子20261012、同编码器、同融合方式，两种损失的测试误差都下降，且最佳轮次均大于0。以下按测试结果事后挑选用于展示，不是验证集选出的最优模型，也不证明显著提升；所有84项结果仍完整保留在后面。Frobenius与Euclidean是分别训练的两个模型。

| 代表方案 | Frobenius：基础→修正 | 改善率 | Euclidean：基础→修正 | 改善率 | 展示理由 |
| --- | ---: | ---: | ---: | ---: | --- |
| CNN＋凸组合＋Idea1主方向q=1 | 0.000990501 → 0.000901297 | 9.01% | 0.000797290 → 0.000731105 | 8.30% | 修正幅度最少，且两项相对改善明显 |
| CNN＋精度因子融合＋Idea1主方向q=1 | 0.000900866 → 0.000898499 | 0.26% | 0.000726459 → 0.000724822 | 0.23% | 本轮联合修正单种子测试中，两项绝对误差最低；增益较小 |
| CNN＋凸组合＋Idea2 | 0.000990501 → 0.000903712 | 8.76% | 0.000797290 → 0.000731965 | 8.19% | 展示全维旋转修正的效果；结果接近更简单的Idea1 |
| Transformer＋精度因子融合＋Idea1学习方向q=5 | 0.000905171 → 0.000899043 | 0.68% | 0.000730448 → 0.000726009 | 0.61% | 展示学习方向方案的正向例子；不代表该配置在凸组合下也有效 |

建议主展示CNN＋Idea1(q=1)在两种融合下的结果：凸组合体现较大改善，精度融合体现较低绝对误差；Idea2和学习方向作为补充。不能把这些代表项概括成“所有修正均优于基础融合”，部分配置实际变差。不同结构即使种子相同，初始融合权重也未必相同，改善不能完全归因于修正模块。

### 全部配置结果（含无改善项）

| 配置 | 融合 | 损失 | 联合修正 | 同种子纯融合 | 改善率 | 最佳轮次 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| mlp_spectral_q1 | precision | frobenius | 0.000900934 | 0.000904776 | 0.42% | 1 |
| mlp_spectral_q1 | precision | euclidean | 0.000726537 | 0.000729803 | 0.45% | 1 |
| mlp_spectral_q1 | covariance | frobenius | 0.000923771 | 0.000969393 | 4.71% | 6 |
| mlp_spectral_q1 | covariance | euclidean | 0.000755453 | 0.000782429 | 3.45% | 1 |
| mlp_spectral_q3 | precision | frobenius | 0.000925120 | 0.000904776 | -2.25% | 3 |
| mlp_spectral_q3 | precision | euclidean | 0.000741071 | 0.000729803 | -1.54% | 3 |
| mlp_spectral_q3 | covariance | frobenius | 0.000955951 | 0.000969393 | 1.39% | 6 |
| mlp_spectral_q3 | covariance | euclidean | 0.000770622 | 0.000782429 | 1.51% | 6 |
| mlp_spectral_q5 | precision | frobenius | 0.000917382 | 0.000904776 | -1.39% | 0 |
| mlp_spectral_q5 | precision | euclidean | 0.000739045 | 0.000729803 | -1.27% | 0 |
| mlp_spectral_q5 | covariance | frobenius | 0.000984071 | 0.000969393 | -1.51% | 0 |
| mlp_spectral_q5 | covariance | euclidean | 0.000792627 | 0.000782429 | -1.30% | 0 |
| mlp_learned_q1 | precision | frobenius | 0.000908497 | 0.000904776 | -0.41% | 0 |
| mlp_learned_q1 | precision | euclidean | 0.000732656 | 0.000729803 | -0.39% | 0 |
| mlp_learned_q1 | covariance | frobenius | 0.000919157 | 0.000969393 | 5.18% | 4 |
| mlp_learned_q1 | covariance | euclidean | 0.000750500 | 0.000782429 | 4.08% | 2 |
| mlp_learned_q3 | precision | frobenius | 0.000913224 | 0.000904776 | -0.93% | 2 |
| mlp_learned_q3 | precision | euclidean | 0.000734739 | 0.000729803 | -0.68% | 2 |
| mlp_learned_q3 | covariance | frobenius | 0.000921475 | 0.000969393 | 4.94% | 2 |
| mlp_learned_q3 | covariance | euclidean | 0.000745497 | 0.000782429 | 4.72% | 2 |
| mlp_learned_q5 | precision | frobenius | 0.000906877 | 0.000904776 | -0.23% | 0 |
| mlp_learned_q5 | precision | euclidean | 0.000731256 | 0.000729803 | -0.20% | 0 |
| mlp_learned_q5 | covariance | frobenius | 0.000933140 | 0.000969393 | 3.74% | 1 |
| mlp_learned_q5 | covariance | euclidean | 0.000753856 | 0.000782429 | 3.65% | 1 |
| mlp_idea2_q10 | precision | frobenius | 0.000903122 | 0.000904776 | 0.18% | 1 |
| mlp_idea2_q10 | precision | euclidean | 0.000727965 | 0.000729803 | 0.25% | 1 |
| mlp_idea2_q10 | covariance | frobenius | 0.000941709 | 0.000969393 | 2.86% | 1 |
| mlp_idea2_q10 | covariance | euclidean | 0.000757109 | 0.000782429 | 3.24% | 1 |
| cnn_spectral_q1 | precision | frobenius | 0.000898499 | 0.000900866 | 0.26% | 1 |
| cnn_spectral_q1 | precision | euclidean | 0.000724822 | 0.000726459 | 0.23% | 1 |
| cnn_spectral_q1 | covariance | frobenius | 0.000901297 | 0.000990501 | 9.01% | 2 |
| cnn_spectral_q1 | covariance | euclidean | 0.000731105 | 0.000797290 | 8.30% | 2 |
| cnn_spectral_q3 | precision | frobenius | 0.000902925 | 0.000900866 | -0.23% | 1 |
| cnn_spectral_q3 | precision | euclidean | 0.000727580 | 0.000726459 | -0.15% | 1 |
| cnn_spectral_q3 | covariance | frobenius | 0.000914409 | 0.000990501 | 7.68% | 2 |
| cnn_spectral_q3 | covariance | euclidean | 0.000740530 | 0.000797290 | 7.12% | 2 |
| cnn_spectral_q5 | precision | frobenius | 0.000900999 | 0.000900866 | -0.01% | 1 |
| cnn_spectral_q5 | precision | euclidean | 0.000727008 | 0.000726459 | -0.08% | 1 |
| cnn_spectral_q5 | covariance | frobenius | 0.000926359 | 0.000990501 | 6.48% | 2 |
| cnn_spectral_q5 | covariance | euclidean | 0.000753550 | 0.000797290 | 5.49% | 2 |
| cnn_learned_q1 | precision | frobenius | 0.000903983 | 0.000900866 | -0.35% | 2 |
| cnn_learned_q1 | precision | euclidean | 0.000728217 | 0.000726459 | -0.24% | 2 |
| cnn_learned_q1 | covariance | frobenius | 0.000912888 | 0.000990501 | 7.84% | 2 |
| cnn_learned_q1 | covariance | euclidean | 0.000738583 | 0.000797290 | 7.36% | 2 |
| cnn_learned_q3 | precision | frobenius | 0.000906886 | 0.000900866 | -0.67% | 0 |
| cnn_learned_q3 | precision | euclidean | 0.000780642 | 0.000726459 | -7.46% | 14 |
| cnn_learned_q3 | covariance | frobenius | 0.001058827 | 0.000990501 | -6.90% | 28 |
| cnn_learned_q3 | covariance | euclidean | 0.000778559 | 0.000797290 | 2.35% | 12 |
| cnn_learned_q5 | precision | frobenius | 0.000909982 | 0.000900866 | -1.01% | 0 |
| cnn_learned_q5 | precision | euclidean | 0.000733789 | 0.000726459 | -1.01% | 0 |
| cnn_learned_q5 | covariance | frobenius | 0.000907187 | 0.000990501 | 8.41% | 2 |
| cnn_learned_q5 | covariance | euclidean | 0.000731191 | 0.000797290 | 8.29% | 2 |
| cnn_idea2_q10 | precision | frobenius | 0.000899743 | 0.000900866 | 0.12% | 1 |
| cnn_idea2_q10 | precision | euclidean | 0.000724968 | 0.000726459 | 0.21% | 1 |
| cnn_idea2_q10 | covariance | frobenius | 0.000903712 | 0.000990501 | 8.76% | 2 |
| cnn_idea2_q10 | covariance | euclidean | 0.000731965 | 0.000797290 | 8.19% | 2 |
| transformer_spectral_q1 | precision | frobenius | 0.000916567 | 0.000905171 | -1.26% | 1 |
| transformer_spectral_q1 | precision | euclidean | 0.000740840 | 0.000730448 | -1.42% | 1 |
| transformer_spectral_q1 | covariance | frobenius | 0.000919189 | 0.000923568 | 0.47% | 1 |
| transformer_spectral_q1 | covariance | euclidean | 0.000739027 | 0.000746184 | 0.96% | 30 |
| transformer_spectral_q3 | precision | frobenius | 0.000948538 | 0.000905171 | -4.79% | 1 |
| transformer_spectral_q3 | precision | euclidean | 0.000773811 | 0.000730448 | -5.94% | 1 |
| transformer_spectral_q3 | covariance | frobenius | 0.000931197 | 0.000923568 | -0.83% | 1 |
| transformer_spectral_q3 | covariance | euclidean | 0.000757246 | 0.000746184 | -1.48% | 1 |
| transformer_spectral_q5 | precision | frobenius | 0.000945478 | 0.000905171 | -4.45% | 1 |
| transformer_spectral_q5 | precision | euclidean | 0.000763976 | 0.000730448 | -4.59% | 1 |
| transformer_spectral_q5 | covariance | frobenius | 0.000936234 | 0.000923568 | -1.37% | 1 |
| transformer_spectral_q5 | covariance | euclidean | 0.000755593 | 0.000746184 | -1.26% | 1 |
| transformer_learned_q1 | precision | frobenius | 0.000912465 | 0.000905171 | -0.81% | 1 |
| transformer_learned_q1 | precision | euclidean | 0.000735939 | 0.000730448 | -0.75% | 1 |
| transformer_learned_q1 | covariance | frobenius | 0.000918629 | 0.000923568 | 0.53% | 2 |
| transformer_learned_q1 | covariance | euclidean | 0.000744707 | 0.000746184 | 0.20% | 2 |
| transformer_learned_q3 | precision | frobenius | 0.000909117 | 0.000905171 | -0.44% | 0 |
| transformer_learned_q3 | precision | euclidean | 0.000732302 | 0.000730448 | -0.25% | 0 |
| transformer_learned_q3 | covariance | frobenius | 0.000914669 | 0.000923568 | 0.96% | 3 |
| transformer_learned_q3 | covariance | euclidean | 0.000736551 | 0.000746184 | 1.29% | 3 |
| transformer_learned_q5 | precision | frobenius | 0.000899043 | 0.000905171 | 0.68% | 1 |
| transformer_learned_q5 | precision | euclidean | 0.000726009 | 0.000730448 | 0.61% | 1 |
| transformer_learned_q5 | covariance | frobenius | 0.001025073 | 0.000923568 | -10.99% | 0 |
| transformer_learned_q5 | covariance | euclidean | 0.000820426 | 0.000746184 | -9.95% | 0 |
| transformer_idea2_q10 | precision | frobenius | 0.000953148 | 0.000905171 | -5.30% | 1 |
| transformer_idea2_q10 | precision | euclidean | 0.000774716 | 0.000730448 | -6.06% | 1 |
| transformer_idea2_q10 | covariance | frobenius | 0.000919156 | 0.000923568 | 0.48% | 1 |
| transformer_idea2_q10 | covariance | euclidean | 0.000743825 | 0.000746184 | 0.32% | 1 |

### 待完成配置

全部完成。

单种子只用于快速探索，不反映随机稳定性；测试集已查看，尚无独立验证或显著性检验。最佳轮次为0的结果没有经过训练更新，不能证明学到了动态修正。
