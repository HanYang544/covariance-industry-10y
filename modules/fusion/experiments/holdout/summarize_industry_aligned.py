"""Full-industry report plus exact ETF test-date subset, without refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/industry_10y_aligned'

def main():
    assert json.loads((OUT/'completion.json').read_text())['neural_fits']==492
    cab=json.loads((OUT/'cab/results.json').read_text()) if (OUT/'cab/results.json').exists() else []
    assert len(cab) in [0,6]
    protocol=json.loads((OUT/'protocol.json').read_text());configs=protocol['configurations']
    per=pd.read_csv(OUT/'per_seed.csv');summary=pd.read_csv(OUT/'summary.csv');base=pd.read_csv(OUT/'baselines.csv')
    assert len(per)==504 and len(summary)==168
    ref=dict(np.load(OUT/'inputs_test.npz'));dates=ref['dates'];etf=np.load(ROOT/'outputs/paper_etf_migration/inputs_test.npz')['dates']
    mask=np.isin(dates,etf);np.testing.assert_array_equal(dates[mask],etf)
    common=[]
    for r in per.itertuples():
        daily=pd.read_csv(OUT/'runs'/r.id/r.objective/str(r.seed)/'test_by_origin.csv')
        np.testing.assert_array_equal(daily.date.to_numpy(),dates)
        common.append(dict(id=r.id,objective=r.objective,seed=r.seed,test_score=float(daily.loss.to_numpy()[mask].mean())))
    for r in cab:
        a=dict(np.load(OUT/'cab'/r['objective']/str(r['seed'])/'predictions.npz'));np.testing.assert_array_equal(a['target'],ref['target'])
        e=a['prediction']-a['target'];i,j=np.triu_indices(10)
        loss=np.linalg.norm(e,axis=(1,2)) if r['objective']=='frobenius' else np.linalg.norm(e[:,i,j],axis=1)
        np.testing.assert_allclose(loss.mean(),r['test'],rtol=1e-12)
        common.append(dict(id='CAB',objective=r['objective'],seed=r['seed'],test_score=float(loss[mask].mean())))
    common=pd.DataFrame(common);common.to_csv(OUT/'common_dates_per_seed.csv',index=False)
    cs=common.groupby(['id','objective']).test_score.agg(['mean','std']).reset_index();cs.to_csv(OUT/'common_dates_summary.csv',index=False)
    z=dict(np.load(OUT/'baselines.npz'));cb=[]
    for k,name in enumerate(z['names']):
        e=z['prediction'][mask,k]-z['target'][mask];i,j=np.triu_indices(10)
        cb.append(dict(method=str(name),frobenius=np.linalg.norm(e,axis=(1,2)).mean(),euclidean=np.linalg.norm(e[:,i,j],axis=1).mean()))
    pd.DataFrame(cb).to_csv(OUT/'common_dates_baselines.csv',index=False)
    by=summary.set_index(['id','objective'])
    def cell(name,o):
        r=by.loc[name,o];return f'{r["mean"]:.9f}（{r["std"]:.9f}）'
    def label(c):
        if c['kind']=='idea':return f'Idea{c["idea"]} / LW-{c["window"]}'
        f='凸组合' if c['fusion']=='covariance' else '精度因子融合'
        if c['kind']=='joint':return f+{'spectral':'＋主方向','learned':'＋学习方向','idea2':'＋Idea2'}[c['variant']]+f' q={c["q"]}'
        if c['group']=='grid':
            g=c['grid'];return f'{g["lw"]}L＋{g["iewma"]}I＋{g["diagonal"]}D'
        return f'{c["experts"]}专家'+f+('＋校准' if c['calibrated'] else '，无校准')
    def table(items):return ['| 方法 | Frobenius 均值（标准差）↓ | Euclidean 均值（标准差）↓ |','| --- | ---: | ---: |']+[f'| {label(c)} | {cell(c["id"],"frobenius")} | {cell(c["id"],"euclidean")} |' for c in items]
    lines=['# 10行业：十年训练历史＋ETF标签/采样/LW口径','',
        ('**状态：全部原方法和CAB均已完成。**' if cab else f'**已有结果报告：492次原方法神经训练、12次Ridge重拟合及统计基线已完成。CAB已完成{len(list((OUT/"cab").glob("*/*/selection.json")))}/6次验证选模，仍在训练；本报告不展示其未完成的测试成绩。**'),' ',
        '**新对齐协议，不覆盖旧行业或ETF结果。** [三套实验总导航](../../../../docs/experiment_index.md) · [详细协议](../../../../docs/industry_aligned.md)','',
        '## 1 设置与范围','',
        '- 训练2008—2017（十个完整年度），验证2018—2020，测试2021—2025；2498/736/1235个日频样本。十年不是单样本回看十年，也不是逐年滚动重训。',
        '- 未来20日中心化样本协方差，分母19；全部每日采样，完整标签不跨分区；LW中心化、原生1/window归一化和1e-8稳定项。',
        '- 沿用原10行业原始收益和2003年起预热，没有改成超额收益。六特征k=3、专家集合不变。',
        '- 原方法492次神经训练＋12次Ridge重拟合，CAB六次GPU训练；三个种子20261012/13/14，均值（跨种子样本标准差ddof=1）。',
        '- 两损失独立训练及选模。原方法Adam0.01/fullbatch/500轮/patience30/含第0轮；CAB对齐ETF为Adam1e-4/batch128/100轮/patience30/不含第0轮。测试期固定参数。',
        '- 同时改训练年数、标签、采样及LW，不是逐项因果消融。资产、训练/验证年份、收益定义仍不同于ETF。','',
        '## 2 统计基线与CAB：完整2021—2025测试','',
        '| 方法 | Frobenius ↓ | Euclidean ↓ |','| --- | ---: | ---: |']
    lines += [f'| {r.method} | {r.frobenius:.9f} | {r.euclidean:.9f} |' for r in base.itertuples()]
    if cab:
        cc={o:np.array([r['test'] for r in cab if r['objective']==o]) for o in ['frobenius','euclidean']}
        lines += [f'| CAB | {cc["frobenius"].mean():.9f}（{cc["frobenius"].std(ddof=1):.9f}） | {cc["euclidean"].mean():.9f}（{cc["euclidean"].std(ddof=1):.9f}） |']
    else:lines+=['','**原方法已完成，CAB仍在训练；暂不报告CAB测试分数。**']
    for heading,group in [('3 动态融合与校准','fusion'),('4 Idea1/2基础窗口','idea'),('5 融合后联合修正','joint')]:lines+=['','## '+heading,'']+table([c for c in configs if c['group']==group])
    lines+=['','## 6 Ridge','', '训练平方误差＋L2，仅alpha分别按F/E验证指标选择；不是未平方F/E直接训练。','',
        '| 方法 | Frobenius 均值（标准差）↓ | Euclidean 均值（标准差）↓ |','| --- | ---: | ---: |']
    lines += [f'| Ridge{k}专家 | {cell(f"ridge_{k}","frobenius")} | {cell(f"ridge_{k}","euclidean")} |' for k in [4,10]]
    grid=[c for c in configs if 'grid' in c]
    lines+=['','## 7 专家网格','',
        'L/I/D为有序LW前缀、IEWMA前缀、对角专家；56套均为精度融合＋校准，两套与主线重复。验证选择与测试探索分开。','',
        '| 损失 | 验证选中配置 | 对应测试 | 测试探索最佳配置 | 对应测试 |','| --- | --- | ---: | --- | ---: |']
    for o in ['frobenius','euclidean']:
        v=min(grid,key=lambda c:by.loc[c['id'],o]['validation_mean']);t=min(grid,key=lambda c:by.loc[c['id'],o]['mean'])
        lines.append(f'| {o} | {label(v)} | {cell(v["id"],o)} | {label(t)} | {cell(t["id"],o)} |')
    lines+=['','### 全部固定配置','']+table(grid)
    lines+=['','## 8 与ETF完全相同的测试日期子集','',
        f'严格使用ETF的{mask.sum()}个起点，排除标签跨入2024年的起点；仅重新评分，不重训/重选模。全表见 common_dates_summary.csv 和 common_dates_baselines.csv。','',
        '| 方法 | 行业Frobenius | 行业Euclidean |','| --- | ---: | ---: |']
    cby=cs.set_index(['id','objective'])
    for name in ['precision_10_pure','precision_10_calibrated','covariance_10_pure','covariance_10_calibrated','idea1_lw120','idea2_lw120','CAB']:
        if name=='CAB' and not cab:continue
        cells=[f'{cby.loc[name,o]["mean"]:.9f}（{cby.loc[name,o]["std"]:.9f}）' for o in ['frobenius','euclidean']]
        lines.append(f'| {name} | {cells[0]} | {cells[1]} |')
    lines+=['','## 9 诊断与文件','',
        f'原方法第0轮最优：{int((per[per.kind!="ridge"].best_epoch==0).sum())}/492；达到500轮：{int((per[per.kind!="ridge"].stopped_epoch==500).sum())}/492。','',
        '| CAB损失 | 种子 | 最佳轮 | 停止轮 |','| --- | ---: | ---: | ---: |']
    lines += [f'| {r["objective"]} | {r["seed"]} | {r["best_epoch"]} | {r["stop_epoch"]} |' for r in cab]
    lines+=['','## 10 校准效果：仅比较各协议内部的相对变化','',
        '下表为（无校准误差－有校准误差）／无校准误差。正数代表校准改善，负数代表变差；不是不同资产绝对分数的排行榜。','',
        '| 协议与测试范围 | 精度融合F改善 | 精度融合E改善 | 凸组合F改善 | 凸组合E改善 |','| --- | ---: | ---: | ---: | ---: |']
    comparisons=[('A旧行业2021—2025',pd.read_csv(ROOT/'outputs/multiseed/summary.csv').set_index(['id','objective'])),
                 ('B新行业2021—2025',by),('B新行业ETF相同733日起点',cby),
                 ('C ETF 2021—2023',pd.read_csv(ROOT/'outputs/paper_etf_migration/summary.csv').set_index(['id','objective']))]
    for name,t in comparisons:
        gains=[]
        for fusion in ['precision','covariance']:
            for o in ['frobenius','euclidean']:
                pure=t.loc[f'{fusion}_10_pure',o]['mean'];cal=t.loc[f'{fusion}_10_calibrated',o]['mean']
                gains.append(f'{100*(pure-cal)/pure:+.2f}%')
        lines.append('| '+name+' | '+' | '.join(gains)+' |')
    lines+=['','联合改动后的模式变化不能单独归因于训练长度、中心化或日频采样；如要识别原因，需要逐项消融。','',
        '[逐种子](per_seed.csv) · [均值标准差](summary.csv) · [数据协议](data_protocol.json) · [训练协议](protocol.json)',
        '三套结果各自解释。标签重叠，种子标准差不是置信区间；测试时期此前已被观察，本轮不是全新独立测试。本轮不包含新增组合回测或统计显著性证明。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    if cab:
        cab_lines=['# 新行业对齐协议：CAB逐种子结果','', '[完整方法比较](../README.md) · [训练协议](protocol.json) · [全部数值](results.json)','',
            '与ETF同架构/训练配方，仅资产维度改为10。训练2008—2017，验证2018—2020，测试2021—2025；中心化标签、日频采样，验证早停。','',
            '| 损失 | 种子 | 最佳轮 | 停止轮 | 验证 | 测试 |','| --- | ---: | ---: | ---: | ---: | ---: |']
        cab_lines += [f'| {r["objective"]} | {r["seed"]} | {r["best_epoch"]} | {r["stop_epoch"]} | {r["validation"]:.9f} | {r["test"]:.9f} |' for r in cab]
        (OUT/'cab/README.md').write_text('\n'.join(cab_lines)+'\n',encoding='utf-8')
        (OUT/'report_completion.json').write_text(json.dumps(dict(neural_fits=492,ridge_fits=12,cab_fits=6,common_dates=int(mask.sum())),indent=2),encoding='utf-8')
    print('Industry aligned report ready; CAB '+('complete' if cab else 'pending'),flush=True)

if __name__=='__main__':main()
