import json

def section(out):
    folder=out/'covariance_spd_100'
    if not (folder/'protocol.json').exists():return []
    records=json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else []
    previous=json.loads((out/'direct_spd_100/results.json').read_text())
    lines=['','## 100个20日协方差矩阵 → Transformer → 五种SPD输出','','本轮输入不是六个风险特征，而是100个历史滚动20日中心化样本协方差矩阵（分母19）。在预测日t，输入矩阵截止日期为t−99到t，每张只使用截止当日的过去20日收益；总共覆盖119个交易日收益。标签仍为t+1到t+20日收益的样本协方差。','','形状[B,100,10,10] → 每张矩阵展平100维 → Linear100→32＋正弦位置编码 → 单层Transformer（4头、FFN64、dropout0）→ 最后时点状态32 → 五种输出头。不是资产行Token，不加资产Attention；本轮每张矩阵对应一个时间Token。','','输入按矩阵元素使用训练历史窗口的均值和标准差标准化，不截断。沿用已冻结CAB输入及其训练标准化器，并验证样本日期、标签与现有实验完全一致。输出头沿用上一轮定义及训练标量单位；残差式仍使用LW60对称平方根，不把它改为历史20日样本矩阵。','','训练2008–2017/验证2018–2020/测试2021–2025；单种子20261012；Adam0.01、最多500轮、验证早停30、epoch0参与、float64、128微批累积全量梯度。每项完成验证选模后即测试并更新本文档。历史矩阵输入与六特征输入的信息量、输入投影参数量和标准化方式均不同，不能将差异全部归因于协方差结构。','',f'已完成并记录 {len(records)}/10 项。','','| 输出方式 | 损失 | 历史矩阵输入测试 | 六特征序列输入测试 | 最佳轮次 |','| --- | --- | ---: | ---: | ---: |']
    for method in ['cholesky','matrix_exp','spectral_softplus','variance_correlation','residual']:
        for loss in ['frobenius','euclidean']:
            r=next((r for r in records if r['method']==method and r['objective']==loss),None)
            old=next(r for r in previous if r['method']==method and r['objective']==loss)
            score=f'{r["test"]:.9f}' if r else '训练中／待运行'
            epoch=str(r['best_epoch']) if r else '—'
            lines.append(f'| {method} | {loss} | {score} | {old["test"]:.9f} | {epoch} |')
    lines+=['','前四种不做专家融合、没有历史基准旁路；第五种有LW60先验。当前是固定配方、单种子的探索性比较，不据测试分数中途调整配置。']
    return lines
