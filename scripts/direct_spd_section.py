"""Summary fragment for the five SPD-output experiments."""
import json

def section(out):
    folder=out/'direct_spd_100'
    if not (folder/'protocol.json').exists():return []
    lines=['','## Transformer直接输出矩阵：五种正定参数化','','输入100×6风险特征 → Linear6→32＋正弦位置编码 → 1层4头Transformer（FFN64）→ 最后时点状态 → SPD输出头。无专家融合、无Idea修正。第五种额外使用LW60历史基准，单列，不能将其优势全部归因于输出参数化。','','| 方法 | 输出与约束 |','| --- | --- |','| Softplus-Cholesky | 55参数，下三角对角Softplus+ε，输出LLᵀ |','| Matrix Exponential | 55参数组成对称S，输出exp(S) |','| Spectral Softplus | 对称S的特征值经Softplus+ε映射；使用Loewner矩阵函数梯度，重根用连续极限 |','| Variance–Correlation | 10标准差参数＋55相关结构参数；exp(S)归一化成相关矩阵，再用标准差缩放 |','| SPD Residual Update | LW60的对称平方根B，输出B exp(S) Bᵀ |','','为控制数值量级，使用训练标签的平均对角元素u作固定标量输出单位。前四种输出u×SPD(head)；第五种先对LW60/u开平方，再乘回u，S=0时精确等于LW60。没有按资产重新标准化收益，也没有修改原标签或损失的计量单位。ε=1e-6作用于无量纲参数化，未做硬特征值截断。','','前四种初始输出接近uI（输出头权重标准差0.001）；残差头零初始化，输出从LW60开始。两类初始化和先验不同，不是严格只改变正定约束的消融。数学上输出正定；浮点运算仍检查有限性与Cholesky分解，不把数值失败隐藏成有效成绩。','','固定种子20261012。Frobenius/Euclidean分别训练，Adam0.01、500轮上限、验证早停30、epoch0参与、float64、128微批累积完整训练集梯度。按用户要求，每项验证选模锁定后即可测试并展示；全部配置预先固定，不根据中途测试成绩调整后续模型。未搜索学习率；同一学习率不保证对每种输出头都最优。']
    path=folder/'results.json'
    if not path.exists():path=folder/'live_results.json'
    if not path.exists():
        count=len(list(folder.glob('*/*/selection.json')))
        lines+=['',f'当前完成验证选模 {count}/10；测试成绩尚未生成，不能判断哪个最好。']
        return lines
    records=json.loads(path.read_text())
    lines+=['',f'当前已记录 {len(records)}/10 项测试结果；其余训练中或待运行。']
    lines+=['','### 单种子测试成绩','','| 方法 | Frobenius | 最佳轮次 | Euclidean | 最佳轮次 |','| --- | ---: | ---: | ---: | ---: |']
    for method in ['cholesky','matrix_exp','spectral_softplus','variance_correlation','residual']:
        f=next((r for r in records if r['method']==method and r['objective']=='frobenius'),None)
        e=next((r for r in records if r['method']==method and r['objective']=='euclidean'),None)
        cells=[]
        for r in [f,e]:cells.extend([f'{r["test"]:.9f}',str(r['best_epoch'])] if r else ['训练中／待运行','—'])
        lines.append(f'| {method} | '+' | '.join(cells)+' |')
    lines+=['','单种子成绩仅用于探索，不代表跨种子稳定性；不能将最好的一次测试结果当成独立确认的结论。']
    return lines
