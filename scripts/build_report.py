"""Current report: sequence encoders only, plus CAB/statistical baselines."""
import csv,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'modules/fusion/outputs'
def main():
    lines=['# 100 日序列风险门控：实验结果','','单时点模型及其对照结果已从本项目移除。统计基线、CAB 和序列门控保留。','','## 统一设置','','10 个行业；训练2008–2017/验证2018–2020/测试2021–2025，样本2498/736/1235；全部每日采样，未来20日中心化样本协方差（分母19），标签不跨划分边界，LW中心化。标准化只拟合训练日期，截断[-3,3]。','','三个编码器都输入过去100×6风险特征，输出当日10专家Softmax权重，分别做精度Cholesky因子融合与凸组合，不加校准或修正。Adam0.01、最多500轮、早停30、包含epoch0、float64、128微批累积完整训练集梯度；Frobenius与Euclidean分别训练、验证选模和测试。前者取完整矩阵距离，后者取含对角线上三角距离。','','## 网络结构','','| 编码器 | 结构 |','| --- | --- |','| Transformer | 6→32+位置编码 → 1层4头FFN64 → 最后时点状态32 → 10 |','| 序列MLP | Flatten100×6 → Linear600→64 → ReLU → Linear64→32 → ReLU → 10 |','| 时序CNN | Conv1d6→32,k5,pad2 → ReLU → Conv1d32→32,k5,pad2 → ReLU → 时间均值池化 → 10 |','','最终都接Softmax。CNN只访问已知历史窗口，不使用未来数据。参数量不同，未做超参数搜索。','','## 测试结果','','三种子20261012/13/14，均值（样本标准差），越低越好。','','| 编码器 | 融合 | Frobenius | Euclidean |','| --- | --- | ---: | ---: |']
    for label,folder in [('Transformer','temporal_gate_100'),('序列MLP','sequence_mlp_100'),('时序CNN','sequence_cnn_100')]:
        path=OUT/folder/'results.json'
        if not path.exists():
            lines.append(f'| {label} | 两种融合 | 训练中 | 训练中 |');continue
        records=json.loads(path.read_text())
        assert len(records)==12
        for fusion in ['precision','covariance']:
            cells=[]
            for loss in ['frobenius','euclidean']:
                v=[r['test'] for r in records if r['fusion']==fusion and r['objective']==loss]
                assert len(v)==3
                cells.append(f'{statistics.mean(v):.9f}（{statistics.stdev(v):.9f}）')
            lines.append(f'| {label} | {fusion} | '+' | '.join(cells)+' |')
        (OUT/folder/'README.md').write_text(f'# {label}：100日风险特征门控\n\n输入100×6历史特征。\n\n[统一结构、设置与结果](../../../../experiment_results.md) · [逐种子成绩](results.json)\n\n模型检查点、验证历史和每日专家权重保留在运行子目录。\n',encoding='utf-8')
    lines+=['','### 早停与参数量检查','','| 编码器 | 12次运行中选中epoch0的次数 | 参数量 |','| --- | ---: | ---: |']
    for label,folder,parameters in [('Transformer','temporal_gate_100',9098),('序列MLP','sequence_mlp_100',40874),('时序CNN','sequence_cnn_100',6474)]:
        path=OUT/folder/'results.json'
        if path.exists():
            records=json.loads(path.read_text())
            lines.append(f'| {label} | {sum(r["best_epoch"]==0 for r in records)} | {parameters} |')
    lines+=['','这些epoch0结果保留在均值中，没有按测试成绩剔除；它们没有经过训练更新，不能作为学到动态机制的证据。当前配方下Transformer均值最低，但不能据此证明其架构普遍优越。','','## CAB 与统计基线','','CAB输入100个历史协方差矩阵而非风险特征序列；Adam0.0001、batch128、最多100轮、早停30、不含epoch0。因此它不是只替换编码器的严格消融。','','| 方法 | Frobenius | Euclidean |','| --- | ---: | ---: |']
    cab=json.loads((OUT/'industry_10y_aligned/cab/results.json').read_text())
    cells=[]
    for loss in ['frobenius','euclidean']:
        v=[r['test'] for r in cab if r['objective']==loss]
        cells.append(f'{statistics.mean(v):.9f}（{statistics.stdev(v):.9f}）')
    lines.append('| CAB | '+' | '.join(cells)+' |')
    for r in csv.DictReader((OUT/'industry_10y_aligned/baselines.csv').open()):
        lines.append(f'| {r["method"]} | {float(r["frobenius"]):.9f} | {float(r["euclidean"]):.9f} |')
    lines+=['','## 解释边界','','测试集此前已查看，当前为探索性比较，没有显著性检验。20日目标的每日样本高度重叠。三个种子的标准差不是置信区间。早停包含epoch0，若选中未训练初始化，不能称为学到了有效动态权重，应检查selection.json。','','单时点门控、校准、专家网格、Idea1/2、联合修正和相关Ridge对照的结果已移出当前项目，原项目仍可恢复；不再计算以单时点MLP为基准的改善率。']
    single=OUT/'sequence_joint_single_100'
    joint=single if (single/'plan.json').exists() else OUT/'sequence_joint_100'
    if (joint/'plan.json').exists():
        if joint==single:
            from build_single_joint_report import main as refresh_joint
        else:
            from build_joint_report import main as refresh_joint
        refresh_joint()
        section=(joint/'README.md').read_text(encoding='utf-8').splitlines()
        lines+=['','## 序列融合＋Idea修正补充实验','']
        lines += ['#'+line if line.startswith('## ') else line for line in section[1:]]
        plan=json.loads((joint/'plan.json').read_text())
        pending=[c['id'] for c in plan['configurations'] if not (joint/c['id']/'completion.json').exists()]
        if pending:
            lines+=['','### 尚未完成的配置','','以下配置仍在训练或排队，未填写测试成绩：','',', '.join(pending)]
    from direct_spd_section import section as spd_section
    lines+=spd_section(OUT)
    from covariance_spd_section import section as covariance_section
    lines+=covariance_section(OUT)
    from residual_architecture_section import section as residual_section
    lines+=residual_section(OUT)
    lines += [
        '', '## 简要分析：为什么动态专家融合目前更有优势？', '',
        '动态融合本身也是神经网络。当前差异更应理解为“学习如何组合已有风险估计”与“学习生成或修正完整风险结构”的区别，而不是统计方法与神经网络的简单对立。以下是可能原因，尚非逐项消融确认的因果结论。', '',
        '- **样本少且重叠明显**：训练集2498个预测日，相邻100步输入共享99步，相邻20日标签共享19日收益，独立信息量低于表面样本数；不能简单除以20估算有效样本量。20日日频样本协方差本身也有估计噪声。',
        '- **专家提供结构约束**：融合主要学习10个专家的搭配，专家已经完成多尺度风险估计；直接输出或Residual需要学习更自由的矩阵结构变化，更容易受噪声影响。优势不只是输出参数更少。',
        '- **信息范围不完全相同**：100个滚动20日矩阵覆盖119个交易日收益，而融合专家包含LW-252和长历史IEWMA。当前对照不能把长期信息带来的优势全部归因于网络架构。',
        '- **训练配方可能不匹配**：各架构统一使用Adam学习率0.01并不保证优化充分。部分模型在第0—2轮验证最优，提示需要检查学习率、过拟合与时期差异；早停轮次本身不足以确定原因。', '',
        '**当前判断**：部分Residual模型能改善其LW-60基准，但尚不能据此认定优于动态融合；“日频数据太少”只是合理假设，不能作为已证实的唯一或主要原因。', '',
        '**建议验证顺序**：先仅用验证集比较较小学习率和适度正则化，再给直接预测模型补齐同样的多尺度信息，最后固定模型比较不同训练历史长度的学习曲线。保持测试集只用于报告，不根据测试排名反复调参。'
    ]
    from three_seed_report import enrich
    lines=enrich(lines,OUT)
    from covariance_lr_section import section as learning_rate_section
    lines+=learning_rate_section(OUT)
    from organize_report import arrange
    lines=arrange(lines)
    (ROOT/'experiment_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
