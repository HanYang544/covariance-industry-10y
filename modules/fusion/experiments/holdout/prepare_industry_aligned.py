"""Ten training years, daily origins, centered labels/LW; preserve old snapshots."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.covariance.iewma import iterated_ewma,precision_cholesky
from portfolio_risk.market_risk_features import MarketRiskFeatures
from prepare_etf_migration import HALVES,EXPERTS

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/industry_10y_aligned'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'data_protocol.json').exists():print('Frozen inputs already prepared');return
    path=ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==json.loads((ROOT/'outputs/multiseed/protocol.json').read_text())['source_sha256']
    frame,_=parse_industry_zip(path);frame=frame.loc['2003-01-01':'2025-12-31'];r=frame.to_numpy();n,d=r.shape
    cov=np.full((n,d,d),np.nan);lw=np.full((n,4,d,d),np.nan)
    for t in range(19,n):cov[t]=np.cov(r[t-19:t+1],rowvar=False,ddof=1)
    for t in range(251,n):
        for k,w in enumerate([20,60,120,252]):
            c=LedoitWolf(assume_centered=False).fit(r[t-w+1:t+1]).covariance_
            lw[t,k]=c+np.eye(d)*np.trace(c)/d*1e-8
    splits={}
    for s,start,end in [('train','2008-01-01','2017-12-31'),('validation','2018-01-01','2020-12-31'),('test','2021-01-01','2025-12-31')]:
        ix=np.flatnonzero((frame.index>=start)&(frame.index<=end));ix=ix[ix+20<n];ix=ix[frame.index[ix+20]<=end];splits[s]=ix
    assert splits['train'][-1]+20<splits['validation'][0] and splits['validation'][-1]+20<splits['test'][0]
    ie=np.stack([iterated_ewma(r,v,c,4.2,.05 if k==0 else 0) for k,(v,c) in enumerate(HALVES)],axis=1)
    diag=np.zeros_like(ie[:,0]);j=np.arange(d);diag[:,j,j]=np.diagonal(ie[:,0],axis1=-2,axis2=-1)
    experts=np.concatenate([lw,ie,diag[:,None]],axis=1)
    ewmas=[]
    for decay in [.94,.97]:
        state=np.outer(r[0],r[0]);series=[]
        for row in r:
            state=decay*state+(1-decay)*np.outer(row,row);series.append(state+np.eye(d)*np.trace(state)/d*1e-8)
        ewmas.append(np.stack(series))
    feature=MarketRiskFeatures(top_k=3,min_assets=10);raw=feature.transform(frame)
    state,mean,scale=feature.standardize(raw,frame.index[splits['train']]);raw.to_csv(OUT/'raw_features.csv')
    calibration=np.full((n,4),np.nan)
    for t in range(251,n):calibration[t]=[np.log(max(float(np.mean(r[t-w+1:t+1]**2)),1e-16)) for w in [5,20,60,252]]
    fm=calibration[splits['train']].mean(0);fs=np.maximum(calibration[splits['train']].std(0),1e-12)
    design=np.column_stack([np.ones(n),np.clip((calibration-fm)/fs,-3,3)])
    cab=OUT/'cab';cab.mkdir(exist_ok=True)
    for s,ix in splits.items():
        dates=frame.index[ix].strftime('%Y-%m-%d').to_numpy(dtype='U10');target=cov[ix+20]
        b=dict(experts=experts[ix],whiteners=precision_cholesky(experts[ix]),state=state.to_numpy()[ix],design=design[ix],target=target)
        for k,w in enumerate([20,60,120]):
            v,u=np.linalg.eigh(lw[ix,k]);v=v[:,::-1].copy();u=u[:,:,::-1].copy()
            pivot=np.argmax(np.abs(u),axis=-2);sign=np.take_along_axis(u,pivot[:,None,:],axis=-2).squeeze(-2)
            b[f'values_{w}']=v;b[f'basis_{w}']=u*np.where(sign<0,-1,1)[:,None,:]
        assert all(np.isfinite(v).all() for v in b.values())
        np.savez_compressed(OUT/f'inputs_{s}.npz',**b,dates=dates,origins=ix)
        x=np.stack([cov[t-99:t+1] for t in ix])
        np.savez_compressed(cab/f'{s}.npz',history=x,target=target,dates=dates,origins=ix)
        if s=='train':np.savez(cab/'scaler.npz',mean=x.mean(axis=(0,1)),scale=np.maximum(x.std(axis=(0,1)),1e-12))
    ix=splits['test'];base=np.concatenate([experts[ix,:9],np.stack(ewmas,axis=1)[ix],cov[ix,None]],axis=1)
    np.savez_compressed(OUT/'baselines.npz',prediction=base,names=np.array(EXPERTS[:9]+['EWMA-0.94','EWMA-0.97','Historical-20']),target=cov[ix+20],dates=frame.index[ix].strftime('%Y-%m-%d').to_numpy(dtype='U10'))
    np.savez(OUT/'scaler.npz',gate_mean=mean.to_numpy(),gate_scale=scale.to_numpy(),feature_mean=fm,feature_scale=fs)
    np.savez_compressed(OUT/'timeline.npz',returns=r,dates=frame.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'))
    protocol=dict(source_sha256=digest,assets=list(frame.columns),training='2008-2017 ten complete calendar years, fixed fit not annual rolling refit',validation='2018-2020',test='2021-2025',
        preprocessing='Original French fractional RAW returns retained; unlike ETF excess returns, no new riskfree subtraction',
        warmup='2003-2007 past-only inputs and recursive initialization; not supervised training; same historical start as original industry experiment',
        aligned='centered future20 covariance ddof1; DAILY sampling all splits; centered sklearn LW + relative jitter1e-8',
        unchanged='six features k3, 20/120 windows, 10 experts, EWMA/IEWMA/calibration definitions; architecture and own-family optimizer retained',
        counts={s:len(ix) for s,ix in splits.items()},boundaries={s:dict(first=str(frame.index[ix[0]].date()),last=str(frame.index[ix[-1]].date()),last_label=str(frame.index[ix[-1]+20].date())) for s,ix in splits.items()},
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*.npz')})
    (OUT/'data_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');print(json.dumps(protocol,indent=2),flush=True)

if __name__=='__main__':main()
