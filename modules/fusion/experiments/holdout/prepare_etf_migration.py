"""Align retained models to immutable CAB ETF dates/labels, adding past warmup only."""
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from portfolio_risk.covariance.iewma import iterated_ewma,precision_cholesky
from portfolio_risk.market_risk_features import MarketRiskFeatures
from prepare_paper_etf import fetch,ASSETS

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/paper_etf'
OUT=ROOT/'outputs/paper_etf_migration'
HALVES=[(21,63),(63,125),(125,250),(250,500),(500,1000)]
EXPERTS=[f'LW-{w}' for w in [20,60,120,252]]+[f'IEWMA-{v}-{c}' for v,c in HALVES]+['Diagonal-IEWMA-21']

def main():
    OUT.mkdir(parents=True,exist_ok=True);raw=OUT/'raw_warmup';raw.mkdir(exist_ok=True)
    if (OUT/'data_protocol.json').exists():print('Retaining frozen migration inputs');return
    source_hash={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.npz')}
    def download(ticker):
        url=f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1451606400&period2=1483488000&interval=1d'
        p=raw/f'{ticker}.json'
        if not p.exists():p.write_bytes(fetch(url))
        z=json.loads(p.read_bytes())['chart']['result'][0]
        dates=pd.to_datetime(z['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        return pd.Series(z['indicators']['adjclose'][0]['adjclose'],index=dates,name=ticker)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:prices=pd.concat(pool.map(download,ASSETS),axis=1).sort_index()
    assert prices.notna().all().all() and (prices>0).all().all()
    p=raw/'DGS1MO.csv'
    if not p.exists():p.write_bytes(fetch('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO&cosd=2015-12-01&coed=2017-01-03'))
    rf=pd.read_csv(io.BytesIO(p.read_bytes()),index_col=0,parse_dates=True).iloc[:,0]
    rf=pd.to_numeric(rf,errors='coerce').reindex(prices.index.union(rf.index)).sort_index().ffill().reindex(prices.index)
    warm=prices.pct_change(fill_method=None).sub(rf.shift(1)/100/252,axis=0).iloc[1:]
    timeline=np.load(SOURCE/'timeline.npz');current=pd.DataFrame(timeline['returns'],index=pd.to_datetime(timeline['dates']),columns=ASSETS)
    assert warm.index[-1]<current.index[0]
    frame=pd.concat([warm,current]);r=frame.to_numpy();offset=len(warm)
    assert frame.index.is_unique and frame.index.is_monotonic_increasing and np.isfinite(r).all()
    # Original target and split arrays are copied, not recomputed/reselected.
    indices={s:np.load(SOURCE/f'{s}.npz')['origins']+offset for s in ['train','validation','test']}
    assert min(indices['train'])>=251
    lw=np.full((len(r),4,14,14),np.nan)
    for t in range(251,len(r)):
        for k,w in enumerate([20,60,120,252]):
            c=LedoitWolf(assume_centered=False).fit(r[t-w+1:t+1]).covariance_
            lw[t,k]=c+np.eye(14)*np.trace(c)/14*1e-8
    ie=np.stack([iterated_ewma(r,v,c,4.2,.05 if k==0 else 0) for k,(v,c) in enumerate(HALVES)],axis=1)
    diagonal=np.zeros_like(ie[:,0]);j=np.arange(14);diagonal[:,j,j]=np.diagonal(ie[:,0],axis1=-2,axis2=-1)
    experts=np.concatenate([lw,ie,diagonal[:,None]],axis=1)
    ewmas=[]
    for decay in [.94,.97]:
        state=np.outer(r[0],r[0]);series=[]
        for row in r:
            state=decay*state+(1-decay)*np.outer(row,row)
            series.append(state+np.eye(14)*np.trace(state)/14*1e-8)
        ewmas.append(np.stack(series))
    feature_model=MarketRiskFeatures(top_k=3,min_assets=14)
    features=feature_model.transform(frame)
    state,mean,scale=feature_model.standardize(features,frame.index[indices['train']])
    features.to_csv(OUT/'raw_features.csv');feature_model.diagnostics_.to_csv(OUT/'feature_diagnostics.csv')
    calibration=np.full((len(r),4),np.nan)
    for t in range(251,len(r)):
        calibration[t]=[np.log(max(float(np.mean(r[t-w+1:t+1]**2)),1e-16)) for w in [5,20,60,252]]
    fm=calibration[indices['train']].mean(0);fs=np.maximum(calibration[indices['train']].std(0),1e-12)
    design=np.column_stack([np.ones(len(r)),np.clip((calibration-fm)/fs,-3,3)])
    for split,ix in indices.items():
        ref=dict(np.load(SOURCE/f'{split}.npz'))
        np.testing.assert_array_equal(frame.index[ix].strftime('%Y-%m-%d'),ref['dates'])
        b=dict(experts=experts[ix],whiteners=precision_cholesky(experts[ix]),state=state.to_numpy()[ix],design=design[ix],target=ref['target'])
        for k,w in enumerate([20,60,120]):
            v,u=np.linalg.eigh(lw[ix,k]);v=v[:,::-1].copy();u=u[:,:,::-1].copy()
            pivot=np.argmax(np.abs(u),axis=-2);sign=np.take_along_axis(u,pivot[:,None,:],axis=-2).squeeze(-2)
            b[f'values_{w}']=v;b[f'basis_{w}']=u*np.where(sign<0,-1,1)[:,None,:]
        assert all(np.isfinite(v).all() for v in b.values());np.linalg.cholesky(b['experts'])
        np.savez_compressed(OUT/f'inputs_{split}.npz',**b,dates=ref['dates'],origins=ix)
    test=indices['test'];ref=np.load(SOURCE/'test.npz')
    base=np.concatenate([experts[test,:9],np.stack(ewmas,axis=1)[test],ref['history'][:,-1,None]],axis=1)
    names=EXPERTS[:9]+['EWMA-0.94','EWMA-0.97','Historical-20']
    np.savez_compressed(OUT/'baselines.npz',prediction=base,names=np.array(names),target=ref['target'],dates=ref['dates'])
    np.savez(OUT/'scaler.npz',gate_mean=mean.to_numpy(),gate_scale=scale.to_numpy(),feature_mean=fm,feature_scale=fs)
    np.savez_compressed(OUT/'extended_returns.npz',returns=r,dates=frame.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'),warmup_count=offset)
    protocol=dict(source='frozen paper_etf task with pre-2017 warmup only',source_sha256=source_hash,
        warmup_count=offset,warmup_first=str(warm.index[0].date()),warmup_last=str(warm.index[-1].date()),assets=ASSETS,
        samples={s:len(ix) for s,ix in indices.items()},features=list(features.columns),top_k=3,
        experts=EXPERTS,calibration='intercept + standardized clipped log raw second moment 5/20/60/252; original definition retained',
        lw='centered sklearn LedoitWolf (native 1/window normalization) + relative jitter1e-8; zero-mean assumption changed to centered for ETF covariance task',
        iewma='legacy causal zero-mean risk recursion unchanged; first20 initialization, clipping4.2, fast diagonal inflation.05, jitter1e-8',
        ewma='legacy zero-mean RiskMetrics recursion .94/.97; jitter1e-8',
        comparison='exact CAB dates and labels; different models retain their own history lengths and architecture',
        raw_warmup_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.iterdir()},
        input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.npz')})
    (OUT/'data_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    assert source_hash=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.glob('*.npz')}
    print(json.dumps(protocol,indent=2),flush=True)

if __name__=='__main__':main()
