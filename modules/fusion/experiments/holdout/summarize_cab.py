"""CAB three-seed results alongside unchanged existing models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/cab_holdout'


def main():
    done=json.loads((OUT/'completion.json').read_text());assert done['fits']==6
    rows=pd.read_csv(OUT/'per_seed.csv');summary=pd.read_csv(OUT/'summary.csv')
    assert len(rows)==6 and (summary.n==3).all()
    for o in ['frobenius','euclidean']:
        group=rows[rows.objective==o]
        assert sorted(group.seed)==[20261012,20261013,20261014]
        r=summary[summary.objective==o].iloc[0]
        np.testing.assert_allclose([r['mean'],r['std']],[group.test_score.mean(),group.test_score.std(ddof=1)],rtol=1e-10)
    diagnostics=[]
    for r in rows.itertuples():
        history=pd.read_csv(OUT/r.objective/str(r.seed)/'training_history.csv')
        updates=history[history.epoch>0]
        diagnostics.append(dict(objective=r.objective,seed=r.seed,zero_gradient_epochs=int((updates.gradient_norm==0).sum()),
                                optimizer_steps=len(updates)))
    pd.DataFrame(diagnostics).to_csv(OUT/'training_diagnostics.csv',index=False)
    old=pd.read_csv(ROOT/'outputs/multiseed/summary.csv').set_index(['id','objective'])
    comparisons=[]
    labels=[('precision_10_calibrated','十专家精度融合＋校准'),('covariance_10_calibrated','十专家凸组合＋校准'),
            ('idea1_lw120','Idea1 / LW-120'),('idea2_lw120','Idea2 / LW-120'),
            ('joint_precision_idea2_q10','精度融合＋Idea2'),('joint_covariance_idea2_q10','凸组合＋Idea2')]
    for key,label in labels:
        for o in ['frobenius','euclidean']:
            r=old.loc[key,o]
            comparisons.append(dict(id=key,method=label,objective=o,mean=r['mean'],std=r['std'],n=3))
    for r in summary.to_dict('records'):comparisons.append(dict(id='cab_adapted',**r))
    pd.DataFrame(comparisons).to_csv(OUT/'comparison.csv',index=False)
    lines=['# CAB架构适配：100个历史风险矩阵','',
        '复现论文核心架构，保持项目10资产、未来20日二阶矩目标、样本日期与验证早停。不是作者代码或原论文成绩复现。', '',
        '[模型与适配细节](../../../../docs/cab.md) · [统一报告](../../../../docs/comparison_report.md) · [论文原文](https://onlinelibrary.wiley.com/doi/full/10.1002/for.70110)', '',
        '## 1 共同任务与差异','',
        '- 输入100个日滚动20日未中心化二阶矩，每个10×10，共119个不同历史收益日；不使用LW矩阵或额外六维状态向量。',
        '- 标签与旧实验逐元素相同，训练/验证/测试651/37/62。标准化只拟合训练输入。',
        '- 单通道Conv3D(kernel5) → 7层BiLSTM(hidden128/方向) → 16头Attention → 时间均值 → 100元素矩阵 → PSD投影 → 0.8网络＋0.2最近历史。2,896,098参数。',
        '- Adam0.01、逻辑全批量651、最多500轮、patience30；包含第0轮。GPU分块32累积梯度，不是每32样本更新。训练配方与论文Adam1e-4/batch128/100轮不同。',
        '- 主干float32，标准化/逆变换/PSD/标签/损失float64，关闭AMP与TF32；旧融合主干float64。没有用测试成绩调学习率或选择种子。',
        '- 每种损失独立训练三个种子，所有检查点锁定后测试，不采用论文测试期间更新模型的策略。', '',
        '## 2 三种子均值（样本标准差）','',
        '| 方法 | Frobenius ↓ | Euclidean ↓ |','| --- | ---: | ---: |']
    table=pd.DataFrame(comparisons)
    for key in [k for k,_ in labels]+['cab_adapted']:
        group=table[table.id==key];cells=[]
        for o in ['frobenius','euclidean']:
            r=group[group.objective==o].iloc[0];cells.append(f'{r["mean"]:.9f}（{r["std"]:.9f}）')
        lines.append(f'| {group.iloc[0]["method"]} | '+ ' | '.join(cells)+' |')
    lines+=['','## 3 每个种子的验证选择与测试','',
        '| 损失 | 种子 | 最佳/停止轮次 | 验证分数 | 测试分数 | 训练秒数 |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for r in rows.itertuples():lines.append(f'| {r.objective} | {r.seed} | {r.best_epoch}/{r.stopped_epoch} | {r.validation_score:.9f} | {r.test_score:.9f} | {r.seconds:.1f} |')
    lines+=['','均值与标准差对三个种子的测试平均损失计算，ddof=1，不是预测集成；三个种子为20261012/20261013/20261014。旧方法不重训、不覆盖。', '',
        '### 训练诊断','',
        '| 损失 | 种子 | 零梯度更新轮数 / 总更新轮数 |','| --- | ---: | --- |']
    for r in diagnostics:lines.append(f'| {r["objective"]} | {r["seed"]} | {r["zero_gradient_epochs"]}/{r["optimizer_steps"]} |')
    lines+=['','零梯度轮次仅统计实际更新轮，不含初始化第0轮。出现零梯度意味着这些轮次没有来自损失的新梯度信号；Adam动量仍可能移动参数。硬PSD截断存在平坦区，但仅凭此记录不能排除其他梯度饱和原因。此训练配方下的成绩不能视为CAB充分调优后的上限。', '',
        '## 4 解释边界','',
        '在相同历史测试集比较适配架构；输入信息形式、网络参数量、数值精度与模型归纳假设不同，不是只改变单一层的消融。共用学习率并不保证各架构达到最佳效果，不能据此否定原论文方法。测试集已反复查看，多种子不等于独立泛化验证。', '',
        '首次分块128因GPU显存分配失败，只完成初始化验证；已另存[失败记录](attempts/oom_micro128/README.md)，不计入六次成功训练。后续降低计算分块而未改变逻辑训练批量。', '',
        '[逐种子CSV](per_seed.csv) · [均值标准差CSV](summary.csv) · [对照表](comparison.csv) · [训练协议](protocol.json) · [数据协议](data_protocol.json)', '']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':main()
