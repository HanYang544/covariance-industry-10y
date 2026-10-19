"""Group the existing numerical tables by research line, preserving every row."""
from collections import Counter

def arrange(lines):
    blocks={};name=None;preface=[]
    for line in lines[1:]:
        if line.startswith('## '):name=line[3:];blocks[name]=[]
        elif name is None:preface.append(line)
        else:blocks[name].append(line)
    def take(name,title,level=3):
        body=blocks.pop(name)
        return ['', '#'*level+' '+title,'']+[('#'*(level-2)+s if s.startswith('###') else s) for s in body]
    mixed=blocks.pop('CAB 与统计基线')
    header=next(s for s in mixed if s.startswith('| 方法 |'))
    separator=next(s for s in mixed if s.startswith('| ---'))
    cab=next(s for s in mixed if s.startswith('| CAB |'))
    base=[s for s in mixed if s.startswith('|') and s not in [header,separator,cab]]
    cab_note=[s for s in mixed if not s.startswith('|')]
    result=['# 协方差预测：统计融合与网络直接预测','','本报告沿两条研究主线编排，所有既有结果保持原实验口径。代码结构与运行方法见[项目说明](README.md)。', '',
        '主线一：统计基线 → 动态门控 → 凸组合／精度因子融合 → Idea1／Idea2可选修正。', '',
        '主线二：序列输入 → 网络预测矩阵；分别比较五种输出变换、网络架构，并列呈现CAB论文架构适配。Residual使用LW-60基准，不等于无统计先验的纯网络。']
    result+=take('统一设置','一、统一实验协议',2)
    if '三随机种子补全口径' in blocks:result+=take('三随机种子补全口径','1.1 随机种子与汇总口径')
    result+=['','## 二、统计基线与主线一：动态融合及修正','','### 2.1 基础统计方法','','统计估计既是独立基线，也是融合专家的来源；历史尺度属于配置，不重复定义方法。', '',header,separator]+base
    result+=take('网络结构','2.2 动态门控网络')
    result+=['','两种内部融合模块：covariance为协方差凸组合；precision为逆协方差Cholesky因子的加权融合，再还原协方差。两者共用门控网络，当前不加校准。']
    result+=take('测试结果','2.3 两种动态融合的结果')
    result+=take('序列融合＋Idea修正补充实验','2.4 两种内部改进：Idea1与Idea2')
    result+=['','## 三、主线二：网络直接预测','','本主线将输入、网络编码器、矩阵输出方式分开。架构比较固定Residual输出；输出方式比较固定Transformer；CAB保留论文自身的输出处理，不强行替换成五种输出头。']
    result+=take('Transformer直接输出矩阵：五种正定参数化','3.1 五种输出变换：风险特征序列输入')
    result+=take('100个20日协方差矩阵 → Transformer → 五种SPD输出','3.2 五种输出变换：历史协方差序列输入')
    result+=take('固定Residual输出：六种网络架构比较','3.3 网络架构模块：固定Residual比较')
    result+=['','### 3.4 论文架构适配模块：CAB','']+cab_note+[header,separator,cab]
    if '历史协方差输入 Transformer：仅调整学习率' in blocks:result+=take('历史协方差输入 Transformer：仅调整学习率','3.5 学习率搜索（运行进度自动更新）')
    result+=take('解释边界','四、解释边界与综合分析',2)
    result+=take('简要分析：为什么动态专家融合目前更有优势？','4.1 结构先验、数据量与训练配方')
    assert not blocks,list(blocks)
    assert not (Counter(s for s in lines if s.startswith('|'))-Counter(s for s in result if s.startswith('|'))),'Numerical rows changed'
    clean=[]
    for s in result:
        if s or not clean or clean[-1]:clean.append(s)
    links={
        '## 一、统一实验协议':'[标签、损失与早停讲解](docs/methods/training.md)',
        '### 2.1 基础统计方法':'[Historical](docs/methods/statistical.md#historical) · [LW](docs/methods/statistical.md#lw) · [EWMA](docs/methods/statistical.md#ewma) · [IEWMA](docs/methods/statistical.md#iewma)',
        '### 2.2 动态门控网络':'[风险序列如何生成Softmax权重](docs/methods/gating.md)',
        '### 2.3 两种动态融合的结果':'[凸组合原理](docs/methods/convex.md) · [精度因子融合原理](docs/methods/precision.md)',
        '### 2.4 两种内部改进：Idea1与Idea2':'[Idea1：主方向／学习方向](docs/methods/idea1.md) · [Idea2：全维缩放与旋转](docs/methods/idea2.md)',
        '### 3.1 五种输出变换：风险特征序列输入':'[输入与预测流程](docs/methods/direct.md) · [五种输出变换公式](docs/methods/output_heads.md)',
        '### 3.2 五种输出变换：历史协方差序列输入':'[历史协方差输入](docs/methods/direct.md) · [五种输出变换公式](docs/methods/output_heads.md)',
        '### 3.3 网络架构模块：固定Residual比较':'[六种网络结构](docs/methods/networks.md) · [Residual原理](docs/methods/output_heads.md#spd-residual)',
        '### 3.4 论文架构适配模块：CAB':'[CAB逐层结构与实现边界](docs/methods/cab.md)',
        '### 3.5 学习率搜索（运行进度自动更新）':'[训练与验证选模口径](docs/methods/training.md)',
    }
    linked=[]
    for s in clean:
        linked.append(s)
        if s in links:linked.extend(['','方法讲解：'+links[s],''])
    return linked
