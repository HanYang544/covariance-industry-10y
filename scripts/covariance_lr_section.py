"""Read-only report fragment; never ranks learning rates on test scores."""
import json
import statistics

def section(out):
    root=out/'covariance_lr_search'
    if not (root/'plan.json').exists():return []
    plan=json.loads((root/'plan.json').read_text())
    def base(method,loss,seed):
        parent=out/'covariance_spd_100' if seed==20261012 else out/'seed_completion'/str(seed)/'covariance_spd_100'
        return parent/method/loss
    def fmt(v):return f'{statistics.mean(v):.9f}（{statistics.stdev(v):.9f}）'
    done=len(list(root.glob('*/*/*/*/*/*/selection.json')))
    lines=['','## 历史协方差输入 Transformer：仅调整学习率','','保持100个历史20日协方差矩阵输入、五种输出头、数据划分、初始化种子、Adam、全量梯度累积、500轮上限及验证早停30不变；只比较0.01/0.003/0.001/0.0001。0.01复用原三种子结果，新增90次训练。每种输出×损失分别以三种子平均验证误差选定学习率，再评估测试；不是挑选最好的测试分数。', '',f'新增训练已完成验证选模 **{done}/90**；学习率已锁定 **{len(list(root.glob("*/*/lr_selection.json")))}/10** 个分支。较小学习率若触及500轮上限，不能据此认定充分收敛。原有表格为0.01基准，不覆盖。','','| 输出 | 损失 | 验证选择学习率 | 原0.01测试均值（标准差） | 新测试均值（标准差） | 均值改善率 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for method in plan['methods']:
        for loss in plan['objectives']:
            ref=[json.loads((base(method,loss,s)/'result.json').read_text())['test'] for s in plan['seeds']]
            path=root/method/loss/'selected_results.json'
            if path.exists():
                r=json.loads(path.read_text());values=[v['test'] for v in r['records']]
                cells=[str(r['lr']),fmt(ref),fmt(values),f'{1-statistics.mean(values)/statistics.mean(ref):.2%}']
            else:cells=['待选定',fmt(ref),'未评估','—']
            lines.append(f'| {method} | {loss} | '+' | '.join(cells)+' |')
    lines+=['','### 学习率候选：只展示验证成绩','','| 输出 | 损失 | 学习率 | 完成种子数 | 验证均值（标准差） | 最佳轮次（12/13/14） |','| --- | --- | ---: | ---: | ---: | --- |']
    for method in plan['methods']:
        for loss in plan['objectives']:
            for rate in plan['rates']:
                rows=[]
                for s in plan['seeds']:
                    path=base(method,loss,s)/'selection.json' if rate==.01 else root/method/loss/str(rate)/str(s)/method/loss/'selection.json'
                    if path.exists():rows.append(json.loads(path.read_text()))
                value=fmt([r['validation'] for r in rows]) if len(rows)==3 else '待齐全'
                epochs='/'.join(str(r['best_epoch']) for r in rows) if len(rows)==3 else '—'
                lines.append(f'| {method} | {loss} | {rate} | {len(rows)}/3 | {value} | {epochs} |')
    lines+=['','本轮使用此前已多次查看过的测试区间，仍属探索性研究；验证选学习率避免本轮按测试挑选，但不会将原测试集恢复成全新独立样本。']
    return lines
