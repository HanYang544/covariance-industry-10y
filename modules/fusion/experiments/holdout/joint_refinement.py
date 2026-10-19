"""Four joint fits; no frozen fusion or detached spectral computations."""
import copy
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from portfolio_risk.config import load_config
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.state_research import prepare_research
from portfolio_risk.models.joint_refinement import JointRefinement
from portfolio_risk.losses.euclidean import euclidean_loss

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/joint_refinement'
def loss(p,y,o):
    return torch.linalg.matrix_norm(p-y,ord='fro').mean() if o=='frobenius' else euclidean_loss(p,y)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--q',type=int,choices=[1,3,5],default=3)
    parser.add_argument('--fusion',choices=['precision','covariance'],default='precision')
    parser.add_argument('--idea2',action='store_true')
    args=parser.parse_args();q=args.q;fusion=args.fusion
    global OUT
    if q!=3:OUT=ROOT/f'outputs/joint_refinement_q{q}'
    if fusion=='covariance':OUT=ROOT/f'outputs/joint_convex_q{q}'
    if args.idea2:q=10;OUT=ROOT/f'outputs/joint_idea2_{fusion}'
    if OUT.exists():raise RuntimeError('Refuse overwrite')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    archive=ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest==json.loads((ROOT/'outputs/euclidean_study/protocol.json').read_text())['source_sha256']
    OUT.mkdir()
    protocol=dict(q=q,variants=['spectral','learned'],objectives=['frobenius','euclidean'],features=6,experts=10,
        fusion=f'{fusion}, uncalibrated, trainable',refinement=f'q{q}, independent 6->16 ReLU encoder',
        spectral='torch eigh; descending eigenvalues, largest-component-positive signs; no detach',
        learned=f'6->16->{10*q} direction head, thin QR; Cholesky normalized-space SPD rank{q} update',
        initialization='zero variance/correlation heads; initial prediction equals fusion',
        seed=20261012,optimizer='Adam',learning_rate=.01,max_epochs=500,patience=30,min_delta=1e-12,
        batch='full train',dtype='float64',gradient_clipping=False,extra_regularization=False,
        source_sha256=digest,target='future20 uncentered mean outer product',
        selection='per-run validation early stopping; all four checkpoints locked before test evaluation',
        limitations='single seed, previously inspected test; differing parameter counts; differentiable eig/QR may be ill-conditioned',
        code_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/portfolio_risk/models/joint_refinement.py']})
    if args.idea2:
        protocol['variants']=['idea2']
        protocol['refinement']='Idea2: full 10-direction gap-regularized skew-exponential rotation and positive variance scaling; tau=.01'
        protocol['selection']='per-run validation early stopping; both checkpoints locked before test evaluation'
        protocol.pop('learned')
        protocol['initialization']='zero variance/rotation heads; initial prediction equals fusion'
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    base=load_config(ROOT/'configs/preliminary.yaml');config=load_config(ROOT/'configs/state_whitener.yaml')
    frame,_=parse_industry_zip(archive);frame=frame.loc[base['data']['history_start']:'2025-12-31']
    prepared=prepare_research(frame,base,config);a=prepared['arrays'];tr,va,te=[a['splits'][s] for s in ['train','validation','test']]
    assert (len(tr),len(va),len(te))==(651,37,62)
    assert tr[-1]+20<va[0] and va[-1]+20<te[0]
    ref=ROOT/'outputs/dynamic_holdout_six_features_k3'
    raw=pd.read_csv(ref/'market_risk_features.csv',index_col=0,parse_dates=True);saved=np.load(ref/'model.npz')
    np.testing.assert_array_equal(raw.index,frame.index)
    np.testing.assert_allclose(raw.iloc[tr].mean(axis=0),saved['gate_mean'],rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(raw.iloc[tr].std(axis=0,ddof=0),saved['gate_scale'],rtol=1e-12,atol=1e-14)
    state=np.clip((raw.to_numpy()-saved['gate_mean'])/saved['gate_scale'],-3,3)
    dates=frame.index[te].astype(str).to_numpy(dtype=str);old=np.load(ref/'test_predictions.npz')
    np.testing.assert_array_equal(dates,old['dates']);np.testing.assert_array_equal(a['realized'][te],old['target'])
    input_key='whiteners' if fusion=='precision' else 'experts'
    b={n:tuple(torch.tensor(v[idx],dtype=torch.float64) for v in [prepared[input_key],state,a['realized']]) for n,idx in a['splits'].items()}
    np.savez_compressed(OUT/'verification_inputs.npz',**{input_key:b['test'][0].numpy()},state=b['test'][1].numpy(),target=b['test'][2].numpy(),dates=dates,names=np.array(prepared['names']))
    np.savez(OUT/'scaler.npz',mean=saved['gate_mean'],scale=saved['gate_scale'])
    records=[]
    for variant in protocol['variants']:
        for o in protocol['objectives']:
            name=variant+'_'+o;folder=OUT/name;folder.mkdir();torch.manual_seed(20261012)
            model=JointRefinement(variant,q,fusion);initial=copy.deepcopy(model.state_dict())
            optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            def score(split):
                w,x,y=b[split];return loss(model(w,x),y,o)
            best=float('inf');best_epoch=0;history=[]
            for epoch in range(501):
                gate_norm=0.;total_norm=0.
                if epoch:
                    optimizer.zero_grad();score('train').backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                    gate_norm=float(sum(p.grad.square().sum() for p in model.fusion.parameters()).sqrt())
                    total_norm=float(sum(p.grad.square().sum() for p in model.parameters()).sqrt())
                    optimizer.step()
                with torch.no_grad():
                    t,v=float(score('train')),float(score('validation'))
                    _,diag=model(b['train'][0],b['train'][1],True)
                    stability=float(diag['relative_boundary_gap' if variant in ['spectral','idea2'] else 'qr_min_abs_diagonal'].min())
                assert np.isfinite([t,v,stability]).all()
                history.append(dict(epoch=epoch,train_loss=t,validation_loss=v,gate_grad_norm=gate_norm,total_grad_norm=total_norm,stability_min=stability))
                if v<best-1e-12:best=v;best_epoch=epoch;checkpoint=copy.deepcopy(model.state_dict())
                if epoch-best_epoch>=30:break
            torch.save(checkpoint,folder/'model.pt');pd.DataFrame(history).to_csv(folder/'training_history.csv',index=False)
            delta=float(sum((checkpoint[k]-initial[k]).square().sum() for k in initial if k.startswith('fusion.')).sqrt())
            assert max(r['gate_grad_norm'] for r in history)>0
            r=dict(name=name,variant=variant,q=q,fusion=fusion,objective=o,best_epoch=best_epoch,stopped_epoch=epoch,validation_score=best,gate_parameter_change_l2=delta,parameters=sum(p.numel() for p in model.parameters()))
            records.append(r);print(f'{name}: validation={best:.9f}, best/stop={best_epoch}/{epoch}, fusion delta={delta:.6f}',flush=True)
    (OUT/'selection.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    for r in records:
        folder=OUT/r['name'];model=JointRefinement(r['variant'],q,fusion);model.load_state_dict(torch.load(folder/'model.pt',weights_only=True))
        w,x,y=b['test']
        with torch.no_grad():p,d=model(w,x,True)
        p=p.numpy();np.linalg.cholesky(p);err=p-y.numpy();ii,jj=np.triu_indices(10)
        scores=np.linalg.norm(err,axis=(1,2)) if r['objective']=='frobenius' else np.linalg.norm(err[:,ii,jj],axis=1)
        r['test_score']=float(scores.mean());np.testing.assert_allclose(r['test_score'],loss(torch.tensor(p),y,r['objective']).item(),rtol=1e-12)
        np.savez_compressed(folder/'test_predictions.npz',prediction=p,target=y.numpy(),dates=dates,base=d['base'].numpy(),weights=d['weights'].numpy())
        pd.DataFrame(dict(date=dates,loss=scores)).to_csv(folder/'test_by_origin.csv',index=False)
        (folder/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    pd.DataFrame(records).to_csv(OUT/'comparison.csv',index=False)
    lines=[f'# 十专家{fusion}融合＋{q}方向修正：端到端联合训练','','固定六特征与原10专家。融合层不校准、不冻结；修正层采用独立6→16 ReLU编码器，无额外整体校准。covariance表示协方差凸组合，precision表示精度因子融合。若最佳轮次为0则保留初始化，不解释为训练增益。', '',
        f'spectral：torch.linalg.eigh选前{q}主方向，保持Idea1方差/相关修正；learned：网络输出10×{q}矩阵，经QR得到{q}个方向，在Cholesky归一化空间修正。方差/相关头零初始化，初始等于融合矩阵。q=1时无方向间相关参数。全程无detach、无NumPy切断计算图。', '',
        '训练/验证/测试651/37/62，Adam 0.01、全批量、500轮、patience30、seed20261012，float64；无梯度裁剪、无额外正则。两种损失各自验证早停，四组检查点全部锁定后测试。', '',
        '| 结构 | 损失 | 测试误差 ↓ | 验证误差 | 最佳/停止轮次 | 融合参数变化L2 | 参数数 |','| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for r in records:lines.append(f'| {r["variant"]} | {r["objective"]} | {r["test_score"]:.9f} | {r["validation_score"]:.9f} | {r["best_epoch"]}/{r["stopped_epoch"]} | {r["gate_parameter_change_l2"]:.6f} | {r["parameters"]} |')
    lines+=['','保存了融合层梯度、参数变化和训练稳定性指标；spectral稳定性列为第q/q+1特征值相对间隔，learned为QR对角绝对值最小值。它们只是诊断，不证明所有数据上都稳定。模型参数量不同，单种子重复查看测试集，不宣称显著性或普遍优势。', '',
        '[分数](comparison.csv) · [协议](protocol.json) · [轮次选择](selection.json) · [统一结果](../../../../docs/comparison_report.md)']
    if args.idea2:
        lines=[f'# 十专家{fusion}融合＋Idea2联合训练','',
            '输入六个风险状态特征，十个专家。融合不冻结、不校准；对融合矩阵进行可微特征分解，独立6→16 ReLU网络输出10个方差调整与45个旋转参数。与原Idea2相同，采用特征值间隔正则化、反对称矩阵指数旋转和正的方差倍数，tau=0.01。初始修正为恒等映射。', '',
            '训练/验证/测试651/37/62，Adam 0.01、全批量、最多500轮、patience30、seed20261012、float64。两种损失分别训练与验证选轮，两组检查点锁定后评估测试。标签为未来20日未中心化二阶矩。无额外校准、梯度裁剪或额外正则项。', '',
            '| 损失 | 测试误差 ↓ | 验证误差 | 最佳/停止轮次 | 融合参数变化L2 |',
            '| --- | ---: | ---: | --- | ---: |']
        for r in records:lines.append(f'| {r["objective"]} | {r["test_score"]:.9f} | {r["validation_score"]:.9f} | {r["best_epoch"]}/{r["stopped_epoch"]} | {r["gate_parameter_change_l2"]:.6f} |')
        lines+=['','稳定性记录为最小相邻特征值间隔/平均特征值；近重复特征值可能导致特征分解梯度不稳定。单种子且测试集已多次查看，仅作探索，不声称统计显著性。', '', '[结果数据](comparison.csv) · [协议](protocol.json) · [统一报告](../../../../docs/comparison_report.md)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(pd.DataFrame(records)[['name','test_score','best_epoch']].to_string(index=False),flush=True)

if __name__=='__main__':main()
