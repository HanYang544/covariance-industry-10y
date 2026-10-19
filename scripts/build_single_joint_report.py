import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'modules/fusion/outputs'
JOINT=OUT/'sequence_joint_single_100'
def main():
    plan=json.loads((JOINT/'plan.json').read_text())
    complete=[c for c in plan['configurations'] if (JOINT/c['id']/'completion.json').exists()]
    lines=['# 单种子序列融合＋Idea修正快速比较','','固定种子20261012，共84项（21配置×2融合×2损失）；当前完成'+str(len(complete)*4)+'/84项。已完成的同种子验证检查点直接复用，其他种子不再补跑。','','融合和修正均使用100×6历史特征，独立编码器、端到端联合训练。MLP/CNN/Transformer各做Idea1谱方向q1/3/5、学习方向q1/3/5、Idea2全10维。设置与此前联合修正一致，未更改学习率、精度、早停或数据。','','以下为单次测试成绩，不是三种子均值，无标准差。纯融合参考也取同一固定种子，未挑选最优种子。已完成的三种子历史结果保留，但不混入本表。','','| 配置 | 融合 | 损失 | 联合修正 | 同种子纯融合 | 改善率 | 最佳轮次 |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    full_header=lines[-2:]
    lines=lines[:-2]
    lines+=['### 优先展示：优于对应基础动态融合的代表方案','','筛选口径：固定同一种子20261012、同编码器、同融合方式，两种损失的测试误差都下降，且最佳轮次均大于0。以下按测试结果事后挑选用于展示，不是验证集选出的最优模型，也不证明显著提升；所有84项结果仍完整保留在后面。Frobenius与Euclidean是分别训练的两个模型。','','| 代表方案 | Frobenius：基础→修正 | 改善率 | Euclidean：基础→修正 | 改善率 | 展示理由 |','| --- | ---: | ---: | ---: | ---: | --- |']
    featured=[
        ('cnn_spectral_q1','covariance','CNN＋凸组合＋Idea1主方向q=1','修正幅度最少，且两项相对改善明显'),
        ('cnn_spectral_q1','precision','CNN＋精度因子融合＋Idea1主方向q=1','本轮联合修正单种子测试中，两项绝对误差最低；增益较小'),
        ('cnn_idea2_q10','covariance','CNN＋凸组合＋Idea2','展示全维旋转修正的效果；结果接近更简单的Idea1'),
        ('transformer_learned_q5','precision','Transformer＋精度因子融合＋Idea1学习方向q=5','展示学习方向方案的正向例子；不代表该配置在凸组合下也有效')
    ]
    for identifier,fusion,label,reason in featured:
        config=next((c for c in complete if c['id']==identifier),None)
        if config is None:continue
        records=json.loads((JOINT/identifier/'results.json').read_text())
        folder={'mlp':'sequence_mlp_100','cnn':'sequence_cnn_100','transformer':'temporal_gate_100'}[config['architecture']]
        baseline=json.loads((OUT/folder/'results.json').read_text())
        cells=[]
        for loss in ['frobenius','euclidean']:
            r=next(r for r in records if r['fusion']==fusion and r['objective']==loss)
            ref=next(b['test'] for b in baseline if b['seed']==20261012 and b['fusion']==fusion and b['objective']==loss)
            assert r['seed']==20261012 and r['best_epoch']>0 and r['test']<ref
            cells += [f'{ref:.9f} → {r["test"]:.9f}',f'{(ref-r["test"])/ref:.2%}']
        lines.append(f'| {label} | '+' | '.join(cells)+f' | {reason} |')
    lines+=['','建议主展示CNN＋Idea1(q=1)在两种融合下的结果：凸组合体现较大改善，精度融合体现较低绝对误差；Idea2和学习方向作为补充。不能把这些代表项概括成“所有修正均优于基础融合”，部分配置实际变差。不同结构即使种子相同，初始融合权重也未必相同，改善不能完全归因于修正模块。','','### 全部配置结果（含无改善项）','']+full_header
    for c in complete:
        records=json.loads((JOINT/c['id']/'results.json').read_text())
        folder={'mlp':'sequence_mlp_100','cnn':'sequence_cnn_100','transformer':'temporal_gate_100'}[c['architecture']]
        baseline=json.loads((OUT/folder/'results.json').read_text())
        assert len(records)==4
        for r in records:
            assert r['seed']==20261012
            ref=next(b['test'] for b in baseline if b['seed']==20261012 and b['fusion']==r['fusion'] and b['objective']==r['objective'])
            lines.append(f'| {c["id"]} | {r["fusion"]} | {r["objective"]} | {r["test"]:.9f} | {ref:.9f} | {(ref-r["test"])/ref:.2%} | {r["best_epoch"]} |')
    lines+=['','### 待完成配置','',', '.join(c['id'] for c in plan['configurations'] if c not in complete) or '全部完成。','','单种子只用于快速探索，不反映随机稳定性；测试集已查看，尚无独立验证或显著性检验。最佳轮次为0的结果没有经过训练更新，不能证明学到了动态修正。']
    (JOINT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Single seed report: {len(complete)*4}/84.')
if __name__=='__main__':main()
