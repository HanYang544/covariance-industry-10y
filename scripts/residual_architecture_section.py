import json

def section(out):
    folder=out/'residual_architecture_100'
    if not (folder/'protocol.json').exists():return []
    protocol=json.loads((folder/'protocol.json').read_text())
    records=json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else []
    lines=['','## 固定Residual输出：六种网络架构比较','','输入为100个滚动20日中心化协方差矩阵。固定输出为 B exp(S) Bᵀ，其中B为LW60的对称平方根（内部训练单位缩放后再还原），S由网络输出55参数构成对称矩阵。不加入专家融合。所有残差头零初始化，因此初始预测相同，均为LW60。','','训练划分、输入标准化、标签、seed20261012、Adam0.01、float64、最多500轮、验证早停30、含epoch0、128微批累积全量梯度，均沿用上一轮。每项验证选模后测试，逐项更新。已有完全同配置Transformer残差检查点复用，其他架构从头训练。','','| 架构 | 结构 |','| --- | --- |']
    for name,structure in protocol['architectures'].items():lines.append(f'| {name} | {structure} |')
    lines+=['','完整展平MLP参数量显著大于其他模型，故本实验不是等参数量对照。TCN使用左侧补零、膨胀率1/2/4/8/16/32，时间感受野127，最后时点可以覆盖全部100时点。Patch Transformer将每5个时点组成一个片段，共20个Token。','',f'已完成 {len(records)}/12 项（包含复用Transformer2项）。','','| 架构 | 损失 | 验证分数 | 测试分数 | 最佳轮次 | 参数量 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for name in protocol['methods']:
        for loss in ['frobenius','euclidean']:
            r=next((r for r in records if r['method']==name and r['objective']==loss),None)
            if r:lines.append(f'| {name} | {loss} | {r["validation"]:.9f} | {r["test"]:.9f} | {r["best_epoch"]} | {r["parameters"]} |')
            else:lines.append(f'| {name} | {loss} | — | 训练中／待运行 | — | — |')
    if len(records)==12:
        lines+=['','### 按验证集选择（非测试集挑选）','']
        for loss in ['frobenius','euclidean']:
            r=min([r for r in records if r['objective']==loss],key=lambda r:r['validation'])
            lines.append(f'- {loss}：验证选择 {r["method"]}，对应测试 {r["test"]:.9f}。')
    lines+=['','单种子探索，不代表跨种子稳定性。所有候选预先确定，不按中途测试成绩调整剩余训练；固定学习率不保证对每种架构都最优。最佳轮次为0意味着保持LW60初始化，不能宣称网络已学到有效修正。']
    return lines
