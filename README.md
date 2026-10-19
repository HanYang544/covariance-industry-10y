# 协方差预测研究：统计专家融合与网络直接预测

[研究介绍：动机与两条主线](docs/research_overview.md) · [方法讲解导航](docs/methods/README.md) · [统一实验结果](experiment_results.md) · [模块架构](docs/architecture.md) · [实验协议](docs/protocol.md) · [运行方法](docs/running.md) · [兼容说明](docs/migration.md)

预测10个行业未来20日中心化收益协方差。项目按“共享基础设施＋两条研究主线＋统一评估”组织；目录英文，文档中文。

初次阅读建议从[研究介绍](docs/research_overview.md)开始：从收益率预测与风险分散引出协方差预测、多尺度动机，再介绍融合与网络两条路线，附完整LaTeX公式。

## 研究路线

| 层次 | 模块 |
| --- | --- |
| 共同起点 | [Historical](docs/methods/statistical.md#historical) · [LW](docs/methods/statistical.md#lw) · [EWMA](docs/methods/statistical.md#ewma) · [IEWMA](docs/methods/statistical.md#iewma) |
| 主线一：动态融合 | [序列门控与Softmax](docs/methods/gating.md) → [凸组合](docs/methods/convex.md)／[精度因子融合](docs/methods/precision.md) |
| 主线一：进一步改进 | [Idea1：低维方向修正](docs/methods/idea1.md)／[Idea2：全维缩放与旋转](docs/methods/idea2.md) |
| 主线二：输出变换 | [输入流程](docs/methods/direct.md) → [Cholesky](docs/methods/output_heads.md#cholesky) · [矩阵指数](docs/methods/output_heads.md#matrix-exponential) · [谱Softplus](docs/methods/output_heads.md#spectral-softplus) · [方差–相关](docs/methods/output_heads.md#variance-correlation) · [Residual](docs/methods/output_heads.md#spd-residual) |
| 主线二：网络架构 | [MLP](docs/methods/networks.md#mlp) · [CNN](docs/methods/networks.md#cnn) · [TCN](docs/methods/networks.md#tcn) · [GRU](docs/methods/networks.md#gru) · [Transformer](docs/methods/networks.md#transformer) · [Patch Transformer](docs/methods/networks.md#patch-transformer) |
| 主线二：论文模型 | [CAB架构适配](docs/methods/cab.md) |

点击方法名即可查看讲解。建议先读[统计基线](docs/methods/statistical.md)，再按两条主线展开；[损失、早停和梯度如何传递](docs/methods/training.md)单独说明。每页都能返回导航、结果文档或对应代码。

Residual使用单个LW-60先验，不是完全从零预测；CAB保留论文自己的输出流程，不强行替换输出头。

## 目录

```text
covariance-industry-10y/
├── README.md
├── experiment_results.md
├── configs/                       # 协议与模型配置
├── data/                          # 原始数据、冻结输入入口
├── src/covariance_research/
│   ├── common/                    # 数据、标签、特征、损失、训练
│   ├── statistical/               # 基础统计估计
│   ├── fusion/                    # 门控、两种融合、两种修正
│   ├── direct/                    # 输入、输出头、架构、论文模型
│   ├── evaluation/                # 组合与评价工具
│   └── factory.py                 # 按配置组装模型
├── experiments/                   # 新实验统一入口
├── outputs/                       # 分类结果入口与新运行
├── docs/                          # 中文说明
├── scripts/                       # 运行、报告、验证
├── tests/                         # 数值等价与数据检查
└── modules/fusion/                # 冻结历史实现及在跑任务兼容层
```

大部分data/outputs分类入口是项目内部目录联接，不复制检查点，也不移动在跑任务。联接内就是原文件，不要递归删除这些入口。

## 常用操作

```powershell
# 查看配置，不训练
./scripts/workspace.ps1 -Task inspect -Config configs/fusion/precision.json

# 更新报告
./scripts/workspace.ps1 -Task report

# 使用CPU统计环境验证新旧模型、梯度、标签及历史分数
./scripts/workspace.ps1 -Task verify -Runtime cpu

# 明确启动新实验，输出名不得已存在
./scripts/workspace.ps1 -Task train -Config configs/direct/residual_transformer.json -Name my_new_run -Runtime gpu
```

示例是单次运行配置，不会自动跑网格；三个种子及两个损失分别设置。新结果写入outputs/new_runs，不会自动混入冻结研究表格。

## 当前状态与边界

- 原116个单种子实验已补齐三个种子，新增232条记录；基础融合、CAB原已有三种子。
- 学习率搜索继续使用原队列，主报告自动更新，不要重复启动。
- 新源码保留参数名和初始化顺序；旧来源路径、协议和哈希保持原貌。
- 新训练入口同时保存训练与验证曲线；不会追溯伪造旧实验没有记录的训练曲线。
- 本项目未新增三种子组合回测，不将其他项目的单种子组合指标混作这里的三种子结果。
- 本次整理不启动额外研究训练，不安装依赖；运行、缓存、临时文件都在E盘。
