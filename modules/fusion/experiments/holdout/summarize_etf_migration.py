"""One comparison report, with independent F/E training and validation selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];SUITE=ROOT.parent.parent
OUT=ROOT/'outputs/paper_etf_migration';CAB=ROOT/'outputs/paper_etf'

def main():
    complete=json.loads((OUT/'completion.json').read_text());assert complete['neural_fits']==492 and complete['ridge_fits']==12
    protocol=json.loads((OUT/'protocol.json').read_text());configs=protocol['configurations']
    summary=pd.read_csv(OUT/'summary.csv');per=pd.read_csv(OUT/'per_seed.csv');base=pd.read_csv(OUT/'baselines.csv')
    assert len(summary)==168 and len(per)==504 and (summary.n==3).all()
    for (_,o),g in per.groupby(['id','objective']):assert sorted(g.seed)==protocol['seeds']
    table=summary.set_index(['id','objective'])
    def cell(name,o):
        r=table.loc[name,o];return f'{r["mean"]:.9f}（{r["std"]:.9f}）'
    def pair(name):return cell(name,'frobenius')+' | '+cell(name,'euclidean')
    def title(c):
        if c['kind']=='idea':return f'Idea{c["idea"]} / LW-{c["window"]}'
        fusion='凸组合' if c['fusion']=='covariance' else '精度因子融合'
        if c['kind']=='joint':return f'{fusion}＋'+{'spectral':'主风险方向','learned':'学习方向','idea2':'Idea2'}[c['variant']]+f'（q={c["q"]}）'
        if c['group']=='grid':
            g=c['grid'];return f'{g["lw"]}L＋{g["iewma"]}I＋{g["diagonal"]}D'
        return f'{c["experts"]}专家{fusion}'+('＋校准' if c['calibrated'] else '，无校准')
    def block(items):
        return ['| 方法 | Frobenius 均值（标准差）↓ | Euclidean 均值（标准差）↓ |','| --- | ---: | ---: |']+[f'| {title(c)} | {pair(c["id"])} |' for c in items]
    cab=json.loads((CAB/'results.json').read_text());cab_cells={}
    target=np.load(OUT/'inputs_test.npz')['target'];cab_target=np.load(CAB/'test.npz')['target']
    np.testing.assert_array_equal(target,cab_target)
    for o in protocol['objectives']:
        a=[r['test'] for r in cab if r['objective']==o];assert len(a)==3
        cab_cells[o]=f'{np.mean(a):.9f}（{np.std(a,ddof=1):.9f}）'
    lines=['# 14 ETF：历史方法迁移与 CAB 统一任务比较','',
        '**全部完成：492次神经训练＋12次Ridge重拟合，另有11项统计基线、历史20日持续预测，以及此前完成的6次CAB训练。**','',
        '## 1 比较口径','',
        '- 两种损失分别训练、分别验证选模；各配置三个固定种子20261012/20261013/20261014。表格为三次测试平均损失的均值（样本标准差，ddof=1），不是逐日期标准差或置信区间。',
        '- 与CAB逐元素共用666/182/733个训练/验证/测试标签及日期。目标为未来20日中心化样本协方差，分母19；输入只使用预测日及之前的信息。测试年份2021—2023，不使用2024标签。',
        '- Yahoo同名14 ETF公开调整价格＋FRED利率，不是论文原始Refinitiv数据。只给长历史专家额外补入2016-01-05至2017-01-03的252个收益日，不增加训练样本、不改变CAB。',
        '- 六个门控特征不变，主成分k=3；有效秩除以14、平均相关性使用91对资产。仅训练日期拟合均值/标准差并截断到[-3,3]。校准另用5/20/60/252日对数二阶矩及截距。',
        '- 迁移保留原模型Adam0.01、全批666、最多500轮、patience30、包含第0轮、float64；CAB为Adam1e-4、batch128、最多100轮、不含第0轮。**统一的是任务和信息边界，不是完全相同的优化器或历史长度。**',
        '- 所有504项验证选择锁定后统一评分。模型测试期参数固定。专家网格的测试排序只能视为探索；另列按验证均值选出的配置，不混淆两者。',
        '- LW估计改为中心化Ledoit-Wolf，保留其原生1/window归一化及1e-8相对稳定项；EWMA/IEWMA保留原零均值递推估计，作为未来中心化协方差的预测器，不改动评价标签。',
        '- 不恢复已删除的4/14输入特征、其他损失、Idea基础LW100/252/504或其他归档网络。LW252仍属于此前保留的基础专家，不等于恢复已删Idea窗口消融。','',
        '[完整迁移设置](../../../../docs/etf_migration.md) · [逐种子分数](per_seed.csv) · [均值标准差](summary.csv) · [训练协议](protocol.json) · [数据协议](data_protocol.json)','',
        '## 2 统计基线与 CAB','',
        '| 方法 | Frobenius ↓ | Euclidean ↓ |','| --- | ---: | ---: |']
    lines += [f'| {r.method} | {r.frobenius:.9f} | {r.euclidean:.9f} |' for r in base.itertuples()]
    lines += [f'| CAB（各自损失独立训练） | {cab_cells["frobenius"]} | {cab_cells["euclidean"]} |','',
        '统计估计器无随机训练，不虚构标准差；两列使用同一套统计预测。CAB沿用已保存的原ETF结果，没有针对迁移结果重新调参。','',
        '## 3 动态融合与校准消融','']+block([c for c in configs if c['group']=='fusion'])
    lines += ['','## 4 Idea1 / Idea2 基础窗口','',
        'Idea1保留q=3；Idea2仍修正全部风险方向，因此从10维自然扩展到14维，网络隐藏层仍为16。','']+block([c for c in configs if c['group']=='idea'])
    lines += ['','## 5 融合后联合修正','',
        '融合门控不冻结，修正损失可反传到门控。所有联合模型均无额外校准。','']+block([c for c in configs if c['group']=='joint'])
    lines += ['','## 6 Ridge 线性融合对照','',
        '**Ridge仍最小化平方误差＋L2，不声称用未平方F/E直接训练。** 此次两个分支分别按对应验证指标选择alpha，再在同一测试集评价；三次确定性重拟合应给出相同结果。','',
        '| 方法 | Frobenius 均值（标准差）↓ | Euclidean 均值（标准差）↓ |','| --- | ---: | ---: |']
    lines += [f'| Ridge / {k}专家 | {pair(f"ridge_{k}")} |' for k in [4,10]]
    grid=[c for c in configs if 'grid' in c];assert len(grid)==56
    lines += ['','## 7 专家网格：先看验证选择，再看测试探索','',
        'L为按20/60/120/252顺序取前若干LW，I为按21-63/63-125/125-250/250-500/500-1000顺序取前若干IEWMA，D为快速IEWMA对角专家。不是全部任意子集。56套网格均为精度因子融合＋校准；两套与主线重复，复用结果。','',
        '| 损失 | 验证选择配置 | 验证均值 | 该配置测试均值（标准差） | 测试探索最优配置 | 其测试均值（标准差） |','| --- | --- | ---: | ---: | --- | ---: |']
    selections=[]
    for o in protocol['objectives']:
        vg=min(grid,key=lambda c:table.loc[c['id'],o]['validation_mean']);tg=min(grid,key=lambda c:table.loc[c['id'],o]['mean'])
        lines.append(f'| {o} | {title(vg)} | {table.loc[vg["id"],o]["validation_mean"]:.9f} | {cell(vg["id"],o)} | {title(tg)} | {cell(tg["id"],o)} |')
        selections.append(dict(objective=o,validation_selected=vg['id'],test_exploratory_best=tg['id']))
    lines += ['','### 全部56套配置','']+block(grid)
    lines += ['','## 8 训练诊断与解释边界','', '| 项目 | 数量 |','| --- | ---: |',
        f'| 第0轮即最佳的神经运行 | {int((per[per.kind!="ridge"].best_epoch==0).sum())} / 492 |',
        f'| 最佳轮不超过1轮的神经运行 | {int((per[per.kind!="ridge"].best_epoch<=1).sum())} / 492 |',
        f'| 训练达到500轮上限 | {int((per[per.kind!="ridge"].stopped_epoch==500).sum())} / 492 |','',
        'CAB六次最佳轮均为1；验证期从2020-03-17开始，包含疫情冲击，与训练风险状态可能有明显差异。没有因测试结果调整划分或挑选种子。',
        '有限预热对500/1000日半衰期专家仍可能保留初始化影响；增加更长历史的敏感性实验尚未进行。',
        '这里比较预测矩阵损失，不等于组合收益或风险表现；未新增组合回测、统计显著性检验，也不构成论文严格数值复现。相邻20日标签重叠，不能将733个日期视为独立样本。',
        '旧10行业结果保留，不与此处数值直接拼接。测试排名不是可泛化优势证明。','']
    text='\n'.join(lines)
    (OUT/'README.md').write_text(text,encoding='utf-8')
    (OUT/'grid_selection.json').write_text(json.dumps(selections,indent=2),encoding='utf-8')
    (SUITE/'docs/etf_comparison_report.md').write_text('# 14 ETF 迁移结果\n\n[查看完整统一比较报告](../modules/fusion/outputs/paper_etf_migration/README.md)\n\n492次神经训练、12次Ridge重拟合已完成，含统计基线及CAB对照。两种损失分别训练，全部三个种子的均值（标准差）集中在上述文档。\n',encoding='utf-8')
    print('ETF migration report complete: '+str(OUT/'README.md'),flush=True)

if __name__=='__main__':main()
