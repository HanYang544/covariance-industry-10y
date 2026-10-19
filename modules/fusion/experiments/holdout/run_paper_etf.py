"""CAB public ETF reconstruction: independent F/E training, held-out selection."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

OUT=Path(__file__).resolve().parents[2]/'outputs/paper_etf'
SEEDS=[20261012,20261013,20261014]

class PaperCAB(nn.Module):
    def __init__(self,mean,scale):
        super().__init__();self.n=len(mean)
        self.register_buffer('mean',torch.tensor(mean,dtype=torch.float64))
        self.register_buffer('scale',torch.tensor(scale,dtype=torch.float64))
        self.conv=nn.Conv3d(1,1,5,padding=2)
        self.lstm=nn.LSTM(self.n**2,128,7,batch_first=True,bidirectional=True)
        self.attention=nn.MultiheadAttention(256,16,batch_first=True)
        self.readout=nn.Linear(256,self.n**2)
    def forward(self,x):
        z=((x-self.mean)/self.scale).float()
        z=self.conv(z[:,None]).squeeze(1).flatten(2)
        z,_=self.lstm(z);z,_=self.attention(z,z,z,need_weights=False)
        z=self.readout(z.mean(1)).reshape(-1,self.n,self.n).double()
        z=(z+z.transpose(-1,-2))/2*self.scale+self.mean
        z=(z+z.transpose(-1,-2))/2
        v,u=torch.linalg.eigh(z)
        return .8*((u*v.clamp_min(0).unsqueeze(-2))@u.transpose(-1,-2))+.2*x[:,-1]

def losses(p,y,objective):
    e=p-y
    if objective=='frobenius':return torch.linalg.matrix_norm(e,ord='fro')
    i,j=torch.triu_indices(e.shape[-1],e.shape[-1],device=e.device)
    return torch.linalg.vector_norm(e[:,i,j],dim=-1)

def dump(path,value):path.write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark',action='store_true');args=ap.parse_args()
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scaler=np.load(OUT/'scaler.npz');data={}
    for split in ['train','validation','test']:
        a=np.load(OUT/f'{split}.npz');data[split]=(torch.tensor(a['history'],device='cuda'),torch.tensor(a['target'],device='cuda'))
    create=lambda:PaperCAB(scaler['mean'],scaler['scale']).cuda()
    micro=16
    if args.benchmark:
        model=create();t=time.time();x,y=data['train'];p=model(x[:micro]);l=losses(p,y[:micro],'frobenius').mean();l.backward()
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
        assert torch.linalg.eigvalsh(p).min()>-1e-12
        torch.cuda.synchronize();print(dict(seconds=time.time()-t,loss=float(l),parameters=sum(q.numel() for q in model.parameters()),gpu=torch.cuda.get_device_name()),flush=True);return
    protocol=dict(seeds=SEEDS,objectives=['frobenius','euclidean'],lr=1e-4,optimizer='Adam',batch=128,microbatch=micro,
        max_epochs=100,patience=30,selection='minimum matching validation loss, no epoch0 selection',
        evaluation='fixed holdout; no test-period parameter updates; NOT full daily-refit paper reproduction',
        extensions='Euclidean training and validation patience30 are project extensions, not claimed paper settings',
        architecture='Conv3D(1,1,5) -> BiLSTM(196,128,7) -> MHA(256,16) -> time mean -> Linear(256,196) -> inverse scale -> PSD -> .8DL+.2history',
        normalization='per-entry training history mean/std only',runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name()),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        data_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')})
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:dump(OUT/'protocol.json',protocol)
    def predict(model,split):
        model.eval();x,_=data[split]
        with torch.no_grad():return torch.cat([model(x[s:s+micro]) for s in range(0,len(x),micro)])
    records=[]
    for objective in protocol['objectives']:
        for seed in SEEDS:
            folder=OUT/objective/str(seed);folder.mkdir(parents=True,exist_ok=True)
            if (folder/'selection.json').exists():continue
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);rng=np.random.default_rng(seed)
            model=create();opt=torch.optim.Adam(model.parameters(),lr=1e-4)
            history=[];best=float('inf');best_epoch=0;t0=time.time();x,y=data['train']
            for epoch in range(1,101):
                model.train();order=rng.permutation(len(x));total=0.
                for start in range(0,len(x),128):
                    batch=order[start:start+128];opt.zero_grad()
                    for s in range(0,len(batch),micro):
                        ix=batch[s:s+micro];loss=losses(model(x[ix]),y[ix],objective).sum()/len(batch)
                        assert torch.isfinite(loss);loss.backward();total+=float(loss.detach())*len(batch)
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
                    opt.step()
                val=float(losses(predict(model,'validation'),data['validation'][1],objective).mean())
                history.append(dict(epoch=epoch,train=total/len(x),validation=val,seconds=time.time()-t0))
                if val<best-1e-12:
                    best=val;best_epoch=epoch;torch.save(model.state_dict(),folder/'model.pt')
                with (folder/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
                if epoch==1 or epoch%10==0:print(objective,seed,history[-1],flush=True)
                if epoch-best_epoch>=30:break
            dump(folder/'selection.json',dict(objective=objective,seed=seed,best_epoch=best_epoch,stop_epoch=epoch,validation=best))
            del model,opt;torch.cuda.empty_cache()
    # Test is accessed only after all six validation selections have been locked.
    for objective in protocol['objectives']:
        for seed in SEEDS:
            folder=OUT/objective/str(seed);record=json.loads((folder/'selection.json').read_text())
            model=create();model.load_state_dict(torch.load(folder/'model.pt',weights_only=True));p=predict(model,'test')
            assert torch.isfinite(p).all() and torch.linalg.eigvalsh(p).min()>-1e-10
            record['test']=float(losses(p,data['test'][1],objective).mean())
            np.savez_compressed(folder/'predictions.npz',prediction=p.cpu().numpy(),target=data['test'][1].cpu().numpy())
            dump(folder/'result.json',record);records.append(record)
    lines=['# CAB：14 ETF 公开数据重建实验','',
        '**状态：固定测试集架构复现，不是论文数值或逐日重训练的严格复现。**','',
        'Yahoo 调整价格替代 Refinitiv；FRED DGS1MO。2017—2020 按时间 80%/20% 划分；2021—2023 测试。缺少完整未来窗口的末尾 20 日不评估。',
        '输入为 100 个历史 20 日中心化样本协方差，预测未来 20 日样本协方差（分母 19）。标准化仅拟合训练输入。',
        'Adam 1e-4，逻辑 batch 128（microbatch16 梯度累积），最多100轮；验证早停 patience30 是本项目补充。三个种子，标准差 ddof=1。',
        'Frobenius 与 Euclidean 分别训练、分别选模，测试只在全部选模结束后读取。Euclidean 取含对角线上三角的未平方 L2 范数。','',
        '| 训练／评估损失 | CAB 测试均值（标准差） | 历史20日协方差持续预测 |','| --- | ---: | ---: |']
    for o in protocol['objectives']:
        values=[r['test'] for r in records if r['objective']==o]
        baseline=float(losses(data['test'][0][:,-1],data['test'][1],o).mean())
        lines.append(f'| {o} | {np.mean(values):.9f} ({np.std(values,ddof=1):.9f}) | {baseline:.9f} |')
    lines+=['','## 各次运行','', '| 损失 | 种子 | 最佳轮 | 停止轮 | 验证损失 | 测试损失 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {r['objective']} | {r['seed']} | {r['best_epoch']} | {r['stop_epoch']} | {r['validation']:.9f} | {r['test']:.9f} |" for r in records]
    lines+=['','## 尚未复现的部分','','论文测试期逐日更新细节不充分，本轮不自行假设训练步数。作者原始数据、超参数搜索、其余基线和组合回测尚未复现。已有融合/Idea 方法尚未迁移到此数据协议，不能与旧10行业分数直接比较。','',
        '[论文](https://doi.org/10.1002/for.70110)；完整数据约定见 data_protocol.json，训练与文件哈希见 protocol.json。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');dump(OUT/'results.json',records)
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
