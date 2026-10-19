"""Render current reports only from completed, audited three-seed experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT.parent.parent
OUT=ROOT/'outputs/multiseed'


def main():
    completion=json.loads((OUT/'completion.json').read_text())
    assert completion['neural_fits']==492 and completion['ridge_fits']==6 and completion['legacy_matches']==166
    protocol=json.loads((OUT/'protocol.json').read_text())
    configs=protocol['configurations'];summary=pd.read_csv(OUT/'summary.csv')
    seeds=pd.read_csv(OUT/'per_seed.csv')
    assert len(summary)==166 and len(seeds)==498 and (summary.n==3).all()
    by_id=summary.set_index(['id','objective'])
    def cell(name,o):
        r=by_id.loc[name,o]
        return f'{r["mean"]:.9f}（{r["std"]:.9f}）'
    def pair(name):return cell(name,'frobenius')+' | '+cell(name,'euclidean')
    def title(c):
        if c['kind']=='idea':return f'Idea{c["idea"]} / LW-{c["window"]}'
        fusion='凸组合' if c['fusion']=='covariance' else '精度因子融合'
        if c['kind']=='joint':
            v={'spectral':'主风险方向','learned':'学习方向','idea2':'Idea2'}[c['variant']]
            return f'{fusion}＋{v}（q={c["q"]}）'
        if c['group']=='grid':
            g=c['grid'];return f'{g["lw"]}L＋{g["iewma"]}I＋{g["diagonal"]}D'
        return f'{c["experts"]}专家{fusion}'+('＋校准' if c['calibrated'] else '，无校准')
    def table(items):
        return ['| 方法 | Frobenius 均值（标准差）↓ | Euclidean 均值（标准差）↓ |',
                '| --- | ---: | ---: |']+[f'| {title(c)} | {pair(c["id"])} |' for c in items]
    relative='../modules/fusion/outputs/multiseed/'
    lines=['# 训练协议与实验结果：三个随机种子','',
        '本页是 **A：旧10行业协议** 的历史结果，不是新对齐实验。请通过[三套实验导航](experiment_index.md)查看B新行业与C ETF；本页分数保留不变。','',
        '**当前主结果统一为均值（样本标准差），不再以单个种子的成绩作结论。** 统计基线为确定性估计，仅列一个数值。', '',
        '## 1 实验范围与统计口径','',
        '- 随机种子：20261012、20261013、20261014。每个保留配置、每种损失均独立重训3次，没有挑选最好种子。',
        '- 去重后82个神经配置 × 2种损失 × 3个种子 = **492次训练**；另有4/10专家Ridge各3次确定性重拟合，共498次。56套专家网格中的4LW和全10专家与主线重复，复用同一组结果，避免重复计数。',
        '- 包含当前六特征主线、全部56套专家网格、Idea1/2的LW-20/60/120、两种融合的主方向/学习方向q=1/3/5，以及两种融合＋Idea2。',
        '- 不恢复此前要求移除的4/14输入特征、LW-100/252/504、历史学习率/q搜索、NLL及归档模型；旧文件仅为追溯记录。',
        '- 先锁定全部训练/验证选择，再计算测试分数。20261012的166项对应成绩与旧记录逐项一致，通过重新训练验证，而非直接复制旧分数。', '',
        '对每个种子先在相同62个测试窗口取平均，得到一个分数 $m_s$，再计算：','',
        r'$$\bar m=\frac{1}{3}\sum_{s=1}^{3}m_s,\qquad \mathrm{SD}=\sqrt{\frac{1}{3-1}\sum_{s=1}^{3}(m_s-\bar m)^2}.$$','',
        '**括号内是跨种子的样本标准差（ddof=1）**，不是62个窗口的标准差、均值标准误、置信区间，也不是先平均三个预测矩阵后再计算损失。', '',
        f'[每个种子成绩]({relative}per_seed.csv) · [均值与标准差]({relative}summary.csv) · [旧种子复现核对]({relative}legacy_seed_audit.csv) · [完整协议]({relative}protocol.json)', '',
        '## 2 共同数据、损失与训练设置','',
        '| 项目 | 设置 |','| --- | --- |',
        '| 数据 | 固定Kenneth French 10行业日收益快照 |',
        '| 训练 / 验证 / 测试 | 2005–2017 / 2018–2020 / 2021–2025，651 / 37 / 62个样本 |',
        '| 标签 | 未来20日未中心化平均日频二阶矩；不是减均值的样本协方差 |',
        '| 采样与边界 | 训练间隔5日，验证/测试20日；20日未来标签不跨分区边界 |',
        '| 状态输入 | 六指标、k=3；仅训练集拟合标准化参数，截断到[-3,3] |',
        '| 优化 | Adam，学习率0.01，全批量，最多500轮，patience=30，min_delta=1e-12，float64 |',
        '| 早停 | 每个种子独立用对应验证损失选择检查点，包含第0轮 |',
        '| 新增训练改动 | 仅随机种子；架构、基础矩阵、专家集合及训练超参数保持对应原实验设置 |', '',
        r'$$Y_t=\frac1{20}\sum_{h=1}^{20}r_{t+h}r_{t+h}^{\top},\qquad \mathcal L_F=\frac1B\sum_t\|\widehat C_t-Y_t\|_F,$$', '',
        r'$$\mathcal L_E=\frac1B\sum_t\sqrt{\sum_{i\leq j}(\widehat C_{t,ij}-Y_{t,ij})^2}.$$','',
        'Frobenius完整矩阵误差未平方；Euclidean为含对角线上三角55元素的未平方L2距离，非对角计一次。两列分别训练、分别早停、分别评估，只在同一列内比较。', '',
        '校准仍沿用原实现：单独使用历史5/20/60/252日对数方差特征，学习正的整体缩放。六特征指门控/修正网络输入，不代表开启校准时没有额外历史信息。所有联合方向修正均不开启该额外校准层。', '',
        '## 3 统计基线（确定性，不重复随机种子）','',
        '| 方法 | Frobenius ↓ | Euclidean ↓ |','| --- | ---: | ---: |']
    base=pd.read_csv(ROOT/'outputs/baseline_catalog/summary.csv')
    base=base[base.segment=='2021_2025'].merge(pd.read_csv(ROOT/'outputs/euclidean_study/baselines.csv'),on='method',validate='one_to_one')
    assert len(base)==11
    for r in base.itertuples():lines.append(f'| {r.method} | {r.frobenius:.9f} | {r.euclidean:.9f} |')
    lines+=['','同一统计预测分别评分，无神经训练和随机初始化；不虚构随机种子标准差。', '',
        '## 4 动态融合：专家数量与校准消融','',
        '4专家为LW-20/60/120/252；10专家在此基础上加入IEWMA-21-63、63-125、125-250、250-500、500-1000及Diagonal-IEWMA-21。EWMA-0.94/0.97只作统计基线。所有权重由六特征动态门控产生。', '',
        *table([c for c in configs if c['group']=='fusion']), '',
        '协方差凸组合直接对专家矩阵加权；精度因子融合对逆矩阵的Cholesky因子加权，再重建并求逆。它不是直接加权精度矩阵。不同种子可能选中第0轮，因此不能将所有分数差异归因于训练收益。', '',
        '## 5 Idea1与Idea2：固定LW基础矩阵','',
        '只改变基础矩阵窗口为20/60/120日，六特征历史仍为120日。Idea1修正前三主方向，Idea2旋转全部10个方向并调整其风险大小。', '',
        *table([c for c in configs if c['group']=='idea']), '',
        '## 6 专家网格：按三种子测试均值探索排序','',
        '固定精度因子融合＋校准。L按LW-20/60/120/252取前几个；I按上述五个IEWMA取前几个；D表示是否加入对角专家。L数0–4、I数0–5、D数0/1，至少2专家，共56套。', '',
        '**每个候选组合先聚合三个种子，再按测试均值排序。不会先在每个种子里挑不同赢家，再平均赢家成绩。测试集仍参与选择，多种子不能消除这一选择偏差。**', '',
        '| 专家数 | Frobenius最优固定组合 | 均值（标准差）↓ | Euclidean最优固定组合 | 均值（标准差）↓ |',
        '| ---: | --- | ---: | --- | ---: | ---: |']
    grid=[]
    for c in configs:
        if 'grid' not in c:continue
        for o in ['frobenius','euclidean']:
            r=by_id.loc[c['id'],o]
            grid.append(dict(id=c['id'],objective=o,experts=c['experts'],**c['grid'],mean=r['mean'],std=r['std'],n=3))
    grid=pd.DataFrame(grid).sort_values(['objective','mean','id'],kind='stable')
    grid.to_csv(OUT/'grid_summary.csv',index=False)
    winners=grid.groupby(['objective','experts'],sort=True).head(1).sort_values(['experts','objective'])
    winners.to_csv(OUT/'grid_best_by_count.csv',index=False)
    def composition(r):return f'{r.lw}L＋{r.iewma}I＋{r.diagonal}D'
    for k in range(2,11):
        f=winners[(winners.experts==k)&(winners.objective=='frobenius')].iloc[0]
        e=winners[(winners.experts==k)&(winners.objective=='euclidean')].iloc[0]
        lines.append(f'| {k} | {composition(f)} | {cell(f.id,"frobenius")} | {composition(e)} | {cell(e.id,"euclidean")} |')
    lines+=['',f'[全部56套双损失三种子统计]({relative}grid_summary.csv) · [按专家数选择的组合]({relative}grid_best_by_count.csv)', '',
        '同一个数量可能包含不同专家组成；候选数不同，参数量也随专家数变化。只能说本轮网格的测试均值更低，不能证明专家数量的独立因果作用或全局最优。', '',
        '## 7 精度因子融合＋Idea1式方向修正','',
        '基础为十专家精度因子融合，不冻结、不校准；融合网络与修正网络端到端共同训练。独立6→16 ReLU修正编码器决定正的风险倍数和方向间关系。', '',
        '- 主风险方向（spectral）：对融合矩阵可微特征分解，只修正前q个主方向子空间，剩余部分不变。',
        '- 学习方向（learned）：网络输出10×q方向矩阵，经QR正交化，在融合矩阵的Cholesky归一化空间进行正定低秩修正。', '',
        *table([c for c in configs if c['kind']=='joint' and c['fusion']=='precision' and c['variant']!='idea2']), '',
        '## 8 协方差凸组合＋Idea1式方向修正','',
        '与上一节对应结构一致，仅替换基础融合方式；修正后矩阵保持正定，但不一定仍在专家矩阵的凸包内。', '',
        *table([c for c in configs if c['kind']=='joint' and c['fusion']=='covariance' and c['variant']!='idea2']), '',
        '两节的q=1没有方向间相关参数。主方向q=1/3/5参数量为411/496/649，学习方向为581/1006/1499；参数量不同，不视为严格等容量消融。', '',
        '## 9 两种动态融合＋Idea2','',
        '在融合矩阵上进行全10方向旋转与风险大小调整，共1329个参数；不冻结、不加整体校准。网络仍读取六特征，融合矩阵通过特征分解参与可微修正，不是将矩阵展平输入MLP。', '',
        *table([c for c in configs if c['kind']=='joint' and c['variant']=='idea2']), '',
        r'$$C=U\operatorname{diag}(\lambda)U^\top,\quad \lambda_i^\prime=\lambda_i\exp(2\tanh(a_i)),\quad Q=\exp(\Omega),\quad \widehat C=(UQ)\operatorname{diag}(\lambda^\prime)(UQ)^\top.$$','',
        r'$$d_{ij}=\frac{|\lambda_i-\lambda_j|}{\overline\lambda},\quad \Omega_{ij}=\frac{0.1d_{ij}\tanh(b_{ij})}{d_{ij}^2+0.01}\ (i>j),\quad \Omega_{ji}=-\Omega_{ij}.$$','',
        '反对称矩阵指数提供正交旋转，正特征值保证正定。输出头零初始化，初始为恒等修正。近重复特征值仍可能造成梯度不稳定；本轮成功不意味着所有市场数据都稳定。', '',
        '## 10 Ridge确定性线性融合对照','',
        '按三个种子各实际重拟合一次，但闭式/最小二乘求解不使用随机初始化，因此标准差为零不是神经网络稳定性的证据。仍按平方误差＋L2训练，验证Frobenius选alpha；没有改造成Frobenius或Euclidean训练。', '',
        '| 专家数 | Ridge 测试Frobenius 均值（标准差）↓ | 精度融合＋校准 均值（标准差）↓ |',
        '| ---: | ---: | ---: |']
    for k in [4,10]:lines.append(f'| {k} | {cell(f"ridge_{k}","frobenius")} | {cell(f"precision_{k}_calibrated","frobenius")} |')
    lines+=['','## 11 如何解读这轮结果','',
        '三种子统计衡量随机初始化带来的训练波动，不能替代时间序列独立验证。对比均值时必须同时看括号内的波动；三次运行不足以仅凭均值差异宣称显著优越。', '',
        '所有模型共用相同62个历史测试窗口，窗口损失可能有时间依赖，且测试集已反复用于研发与网格选择。没有改变数据划分，也没有新增真正未见的测试市场区间。', '',
        '本轮没有新增网络结构、调学习率或挑选种子。原单种子结论不直接继承为多种子结论；所有保留配置均报告，不隐藏表现较差的随机种子。', '',
        f'[完整多种子结果]({relative}README.md) · [全部实验目录](../outputs/experiment_results.md) · [历史单种子解释（非当前主结果）](historical_single_seed_report.md)', '']
    observations=['本轮专家网格按三个种子的测试均值排序：']
    for o in ['frobenius','euclidean']:
        winner=grid[grid.objective==o].iloc[0]
        observations.append(f'- {o}最低组合为{composition(winner)}，共{winner.experts}专家：{cell(winner.id,o)}。')
    observations+=['', '该均值排名不同于旧单种子排名，不能直接沿用过去的“最优专家数”结论；相邻组合差距可能小于种子间波动。', '']
    at=lines.index('## 11 如何解读这轮结果')+2
    lines[at:at]=observations
    cab_root=ROOT/'outputs/cab_holdout'
    if (cab_root/'completion.json').exists():
        cab=pd.read_csv(cab_root/'summary.csv')
        assert len(cab)==2 and (cab.n==3).all()
        lines+=['## 12 新增：CAB论文架构适配复现','',
            '输入100个历史20日风险矩阵，目标、651/37/62个日期与标签不变。采用Conv3D＋7层BiLSTM＋16头注意力，输出PSD修正后与最近历史矩阵按0.8/0.2混合。该方法没有额外六维状态输入。', '',
            '两个损失各自3个种子、验证早停；Adam0.01、全批量累积、最多500轮、patience30。GPU主干float32，PSD与损失float64；这是相对旧模型float64主干的数值精度差异。原论文Adam1e-4/batch128/100轮未照搬。', '',
            '| 方法 | Frobenius 均值（标准差）↓ | Euclidean 均值（标准差）↓ |','| --- | ---: | ---: |']
        cells=[]
        for o in ['frobenius','euclidean']:
            row=cab[cab.objective==o].iloc[0];cells.append(f'{row["mean"]:.9f}（{row["std"]:.9f}）')
        lines.append('| CAB-adapted | '+' | '.join(cells)+' |')
        for key,label in [('precision_10_calibrated','十专家精度融合＋校准'),('joint_covariance_idea2_q10','凸组合＋Idea2')]:
            lines.append(f'| {label} | {pair(key)} |')
        lines+=['','CAB新增6次训练单独记录，不并入前述492次历史主线计数。100个输入覆盖119个不同历史收益日，输入与目标均按项目未中心化二阶矩口径；这是任务适配，不是论文ETF实验或作者实现的完全复现。', '',
                '[架构与适配说明](cab.md) · [全部种子及对照](../modules/fusion/outputs/cab_holdout/README.md) · [论文原文](https://onlinelibrary.wiley.com/doi/full/10.1002/for.70110)', '']
    (SUITE/'docs/comparison_report.md').write_text('\n'.join(lines),encoding='utf-8')
    detailed=['# 三种子完整结果','',
        '本页保留 **A：旧10行业协议**。新对齐行业和ETF结果见[三套实验导航](../../../../docs/experiment_index.md)，不要混用标签和采样设置。','',
        '20261012、20261013、20261014；492次神经训练＋6次确定性Ridge重拟合。每个种子独立验证早停，所有选择锁定后测试。均值（样本标准差，ddof=1），不是预测集成。', '',
        '[结构化主报告](../../../../docs/comparison_report.md) · [完整协议](protocol.json) · [全部种子分数](per_seed.csv) · [聚合CSV](summary.csv) · [复现审计](legacy_seed_audit.csv)', '',
        '## 非网格方法','',*table([c for c in configs if c['group']!='grid']), '',
        '## 全部专家网格（含与主线重复的两套）','',
        '| 专家组成 | 专家数 | Frobenius 均值（标准差）↓ | Euclidean 均值（标准差）↓ |',
        '| --- | ---: | ---: | ---: |']
    for c in configs:
        if 'grid' in c:
            g=c['grid'];detailed.append(f'| {g["lw"]}L＋{g["iewma"]}I＋{g["diagonal"]}D | {c["experts"]} | {pair(c["id"])} |')
    detailed+=['','## Ridge（确定性重复）','','| 专家数 | Frobenius 均值（标准差） |','| ---: | ---: |']
    for k in [4,10]:detailed.append(f'| {k} | {cell(f"ridge_{k}","frobenius")} |')
    detailed+=['','## 单次运行记录','','路径为 `runs/<配置>/<损失>/<种子>/`，包含模型、验证选择、训练历史、测试预测和逐窗口损失。Ridge保存验证alpha搜索而非epoch历史。', '',
        '统计基线无需随机重跑，见主报告。原单种子文件保留用于追溯；此前移除的模型/窗口不在本轮范围。专家排名按固定组合三种子测试均值，测试选择偏差仍存在。', '']
    (OUT/'README.md').write_text('\n'.join(detailed),encoding='utf-8')
    print('Rendered comparison_report.md and multiseed/README.md; 166 complete mean/std records.',flush=True)


if __name__=='__main__':main()
