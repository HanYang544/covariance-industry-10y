"""Read only completed sequence-joint runs and compare same-encoder fusion."""
import json
import statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'modules/fusion/outputs'
JOINT=OUT/'sequence_joint_100'

def main():
    plan=json.loads((JOINT/'plan.json').read_text())
    complete=[c for c in plan['configurations'] if (JOINT/c['id']/'completion.json').exists()]
    lines=['# 100日序列动态融合＋Idea联合修正','','当前完成 '+str(len(complete))+'/21 个配置（'+str(len(complete)*12)+'/252 次训练及测试）。未完成配置不展示测试成绩。','','## 结构与设置','','同一段历史风险特征X∈R^(100×6)，送入两个参数独立的序列编码器：融合编码器输出10专家Softmax权重，构造未经校准的基础矩阵；修正编码器输出32维状态，经过修正头调整该矩阵。两支网络从头联合训练，不冻结、不额外预训练、不加校准。','','三种编码器均沿用纯融合实验：MLP展平600→64→32；CNN两层k5卷积6→32→32后时间均值；Transformer一层4头、32维、FFN64、正弦位置编码、取最后时点。修正编码器与融合编码器不共享参数。','','| 设置 | 取值 |','| --- | --- |','| Idea1 spectral | 修正融合矩阵最大的q个主风险方向；q=1/3/5 |','| Idea1 learned | 修正网络学习的q个方向，QR正交化，在Cholesky白化坐标中做正定低秩更新；q=1/3/5 |','| Idea2 | 全10维特征值缩放和反对称矩阵指数旋转，q固定10 |','| 融合 | 精度Cholesky因子融合 / 协方差凸组合 |','| 损失 | Frobenius / Euclidean，分开训练、早停、测试 |','| 种子 | 20261012/20261013/20261014 |','','Idea1保留exp(2tanh)风险倍数、0.5tanh下三角相关结构；Idea2保留归一化特征值间距、0.1tanh旋转幅度与tau=0.01。修正头零初始化，初始结果等于融合矩阵。原LW-20/60/120是独立Idea的基础矩阵窗口，本轮由融合矩阵替代，不重复该窗口消融。','','数据、标准化、每日采样、未来20日中心化协方差标签与纯序列融合一致。训练2008–2017，验证2018–2020，测试2021–2025。Adam0.01、最多500轮、早停30、epoch0参与选模、float64、128微批累积完整训练集梯度；只对最终修正矩阵计算损失。','','全部21个配置预先固定。每个配置的12次验证选模锁定后才评估该配置测试集；不根据已完成配置的测试结果调整剩余设置。','','## 已完成配置结果','','均值（样本标准差）；改善率=(同编码器纯融合−联合修正)/同编码器纯融合。正数表示改善。纯融合参考均为100日序列，绝非单时点基准。','','| 配置 | 融合 | 损失 | 联合修正测试 | 纯序列融合测试 | 改善率 | epoch0次数 |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for c in complete:
        records=json.loads((JOINT/c['id']/'results.json').read_text())
        folder={'mlp':'sequence_mlp_100','cnn':'sequence_cnn_100','transformer':'temporal_gate_100'}[c['architecture']]
        baseline=json.loads((OUT/folder/'results.json').read_text())
        for fusion in ['precision','covariance']:
            for loss in ['frobenius','euclidean']:
                chosen=[r for r in records if r['fusion']==fusion and r['objective']==loss]
                values=[r['test'] for r in chosen]
                refs=[r['test'] for r in baseline if r['fusion']==fusion and r['objective']==loss]
                assert len(values)==len(refs)==3
                mean=statistics.mean(values);ref=statistics.mean(refs)
                lines.append(f'| {c["id"]} | {fusion} | {loss} | {mean:.9f}（{statistics.stdev(values):.9f}） | {ref:.9f}（{statistics.stdev(refs):.9f}） | {(ref-mean)/ref:.2%} | {sum(r["best_epoch"]==0 for r in chosen)}/3 |')
    lines+=['','## 注意事项','','测试集已经用于探索，不能将测试最优配置视为独立确认的结论。相同种子不意味着不同结构拥有完全相同的初始融合权重。选中epoch0的运行不构成修正机制学到有效规律的证据。谱修正依赖特征分解，近重根可能导致梯度敏感；当前不添加额外梯度裁剪或正则，以保持原设置。','','各配置目录保存protocol.json、输入/代码哈希、12次验证历史、模型检查点、选择记录、逐日预测和专家权重。']
    (JOINT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Joint report: {len(complete)}/21 complete.')

if __name__=='__main__':main()
