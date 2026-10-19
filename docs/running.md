# 运行与结果定位

| 内容 | 新入口 |
| --- | --- |
| 统计基线与冻结输入 | `outputs/statistical`、`data/frozen` |
| 动态门控 | `outputs/fusion/gating/{mlp,cnn,transformer}` |
| Idea联合修正 | `outputs/fusion/refinement` |
| 两个新增种子 | `outputs/replicates` |
| 五输出：风险特征输入 | `outputs/direct/feature_heads` |
| 五输出：协方差输入 | `outputs/direct/covariance_heads` |
| Residual六架构 | `outputs/direct/architectures` |
| 学习率搜索 | `outputs/direct/learning_rates` |
| CAB | `outputs/paper/cab` |

原单种子目录保留seed12，另外两个种子在replicates；报告合并它们。目录入口不冒充已将三个种子物理合并。

新实验先查看或复制configs中的示例，修改encoder、fusion、refinement、q、head、input、loss、seed、lr等。示例不是新选定的最优配置。需要明确指定train和未存在的输出名；新运行保存在outputs/new_runs，包括协议、曲线、选模、检查点及测试预测。

```powershell
./scripts/workspace.ps1 -Task inspect -Config configs/fusion/idea1.json
./scripts/workspace.ps1 -Task train -Config configs/fusion/idea1.json -Name idea1_new_seed -Runtime gpu
./scripts/workspace.ps1 -Task verify -Runtime cpu
./scripts/workspace.ps1 -Task report
```

新入口拒绝覆盖，不自动恢复中断训练。旧队列使用原恢复机制；当前学习率队列不要重复启动。更新后的报告生成器保留新文档顺序。

CPU环境含统计依赖，用于全套验证；GPU环境用于网络训练。没有安装新依赖，所有入口显式使用E盘缓存和临时目录。
