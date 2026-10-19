"""Aggregate original seed and verified additive seeds in the primary report."""
import json
import statistics

SEEDS=[20261012,20261013,20261014]

def enrich(lines,out):
    def records(family,method,loss,config=''):
        result=[]
        for seed in SEEDS:
            root=out/family if seed==SEEDS[0] else out/'seed_completion'/str(seed)/family
            path=root/config/method/loss/str(seed)/'result.json' if config else root/method/loss/'result.json'
            r=json.loads(path.read_text())
            assert r['seed']==seed
            result.append(r)
        return result
    def fmt(rows,key='test'):
        values=[r[key] for r in rows]
        return f'{statistics.mean(values):.9f}（{statistics.stdev(values):.9f}）'
    def epochs(rows):return '/'.join(str(r['best_epoch']) for r in rows)
    result=[];section='';mode=None;skip=False;skip_validation=False;count=0
    for line in lines:
        if line.startswith('## '):
            section=line;mode=None;skip_validation=False
        if skip_validation:continue
        if line.startswith('### 优先展示：'):
            skip=True
            result+=['### 三种子完整比较','','原按单种子测试结果挑选的代表表不再作为三种子结论；以下保留全部配置，包括变差项。改善率按三个种子的测试均值计算，不是逐种子改善率的均值。','']
        if skip:
            if line.startswith('### 全部配置结果'):skip=False
            else:continue
        if line.startswith('### 按验证集选择'):
            skip_validation=True
            result+=['### 验证选模说明','','每个模型、每个种子仍独立按验证集早停。原先按单种子验证成绩选出的架构不直接沿用为三种子最优架构。','']
            continue
        # Reset this skip at the next top-level section.
        if skip and line.startswith('## '):skip=False
        if line.startswith('| 配置 | 融合 | 损失 |'):mode='joint'
        elif line.startswith('| 方法 | Frobenius | 最佳轮次 |'):mode='direct'
        elif line.startswith('| 输出方式 | 损失 |'):mode='cov'
        elif line.startswith('| 架构 | 损失 |'):mode='residual'
        elif not line.startswith('|'):mode=None
        if mode and line.startswith('|') and not line.startswith('| ---'):
            c=[s.strip() for s in line.strip('|').split('|')]
            if mode=='joint' and c[0]!='配置':
                rows=records('sequence_joint_single_100',c[1],c[2],c[0])
                family={'cnn':'sequence_cnn_100','mlp':'sequence_mlp_100','transformer':'temporal_gate_100'}[c[0].split('_')[0]]
                refs=[r for r in json.loads((out/family/'results.json').read_text()) if r['fusion']==c[1] and r['objective']==c[2]]
                assert sorted(r['seed'] for r in refs)==SEEDS
                a=statistics.mean(r['test'] for r in refs);b=statistics.mean(r['test'] for r in rows)
                c[3:]=[fmt(rows),fmt(refs),f'{(a-b)/a:.2%}',epochs(rows)];count+=1
            elif mode=='direct' and c[0]!='方法':
                f=records('direct_spd_100',c[0],'frobenius');e=records('direct_spd_100',c[0],'euclidean')
                c[1:]=[fmt(f),epochs(f),fmt(e),epochs(e)];count+=2
            elif mode=='cov' and c[0]!='输出方式':
                rows=records('covariance_spd_100',c[0],c[1]);old=records('direct_spd_100',c[0],c[1])
                c[2:]=[fmt(rows),fmt(old),epochs(rows)];count+=1
            elif mode=='residual' and c[0]!='架构':
                rows=records('residual_architecture_100',c[0],c[1])
                c[2:]=[fmt(rows,'validation'),fmt(rows),epochs(rows),str(rows[0]['parameters'])];count+=1
            line='| '+' | '.join(c)+' |'
        if line.startswith('固定种子20261012，共84项'):
            line='已完成21配置×2融合×2损失×3种子，共252条结果。融合网络和修正网络共同训练，原始单种子文件保留用于追溯。'
        elif line.startswith('以下为单次测试成绩'):
            line='下表为三种子测试均值（样本标准差）。基础融合使用相同三个种子，不挑选最优种子；最佳轮次依次列出20261012/13/14。'
        elif line.startswith('单种子只用于快速探索') or line.startswith('单种子成绩仅用于探索'):
            line='三个种子衡量训练随机性，不是独立市场样本或置信区间；测试集已多次查看，不能据此宣称独立验证或统计显著性。epoch0表示没有训练更新。'
        elif line.startswith('单时点门控、校准、专家网格'):
            continue
        line=line.replace('### 单种子测试成绩','### 三种子测试成绩').replace('单种子20261012','三个种子20261012/13/14').replace('固定种子20261012。','固定三个种子20261012/13/14。').replace('seed20261012','seeds20261012/13/14').replace('单种子的探索性比较','三种子的探索性比较').replace('单种子探索，不代表跨种子稳定性。','三种子探索，不代表跨时期稳健性。')
        if line.startswith('当前已记录 10/10') or line.startswith('已完成并记录 10/10'):line='已完成5种输出×2种损失×3种子，共30条结果。'
        if line.startswith('已完成 12/12'):line='已完成6种架构×2种损失×3种子，共36条结果（含复用Transformer的6条，不作为独立重复证据）。'
        result.append(line)
    assert count==116,count
    note=['','## 三随机种子补全口径','','本文件已纳入全部补跑结果：原20261012加20261013/14。新增232条记录已完成，116个原单种子实验均补齐到3个种子；基础融合和CAB原已具备3个种子，统计基线不重复训练。所有矩阵误差均显示均值（样本标准差），最佳轮次以种子顺序列出。','','数据划分、模型、优化器、两种损失与验证早停均保持原配置。复用检查点已核验来源；相同Transformer Residual结果跨表复用，不算独立证据。原始结果不覆盖，补跑文件保存在[seed_completion](modules/fusion/outputs/seed_completion/)。当前补全针对矩阵预测误差，不把旧单种子组合指标当成三种子回测结果。','']
    return result[:1]+note+result[1:]
