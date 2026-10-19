"""Build CAB inputs on the exact existing origins, without new downloads."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from portfolio_risk.data.french import parse_industry_zip
from portfolio_risk.models.cab import history_sequences

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/cab_holdout'


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'data_protocol.json').exists():raise RuntimeError('Prepared data already exists')
    archive=ROOT/'data/raw/10_Industry_Portfolios_daily_CSV.zip'
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest==json.loads((ROOT/'outputs/multiseed/protocol.json').read_text())['source_sha256']
    frame,_=parse_industry_zip(archive);frame=frame.loc['2003-01-01':'2025-12-31']
    data={};split_origins={}
    for name,n in [('train',651),('validation',37),('test',62)]:
        ref=np.load(ROOT/f'outputs/multiseed/inputs_{name}.npz')
        dates=pd.to_datetime(ref['dates']);origins=frame.index.get_indexer(dates)
        assert len(origins)==n and (origins>=118).all()
        x=history_sequences(frame.to_numpy(),origins)
        y=np.stack([frame.iloc[t+1:t+21].to_numpy().T@frame.iloc[t+1:t+21].to_numpy()/20 for t in origins])
        np.testing.assert_array_equal(y,ref['target'])
        assert x.shape==(n,100,10,10) and np.isfinite(x).all()
        data[name]=(x,y,ref['dates'],origins)
        split_origins[name]=origins.tolist()
    assert split_origins['train'][-1]+20<split_origins['validation'][0]
    assert split_origins['validation'][-1]+20<split_origins['test'][0]
    train=data['train'][0]
    mean=train.mean(axis=(0,1));scale=np.maximum(train.std(axis=(0,1),ddof=0),1e-12)
    np.savez(OUT/'scaler.npz',mean=mean,scale=scale)
    for name,(x,y,dates,origins) in data.items():
        np.savez_compressed(OUT/f'{name}.npz',history=x,target=y,dates=dates,origins=origins)
    protocol=dict(source_sha256=digest,assets=list(frame.columns),input_length=100,input_risk_window=20,
                  input_definition='100 trailing uncentered 20-day second-moment matrices, ending t-99 through t inclusive; 119 distinct return days',
                  target='unchanged future20 uncentered mean outer product',splits=split_origins,
                  scaling='per matrix entry mean/std over TRAIN histories only, axes sample/time; symmetric; std floor 1e-12; no clipping',
                  paper_deviation='Uncentered second moments rather than centered sample covariance, matching existing suite labels.',
                  no_lookahead='Latest input date t, future label t+1:t+20. BiLSTM reverse direction sees only the historical input window.',
                  input_sha256={f'{s}.npz':hashlib.sha256((OUT/f'{s}.npz').read_bytes()).hexdigest() for s in data})
    (OUT/'data_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print('CAB data prepared: 651/37/62 identical labels and dates; input [100,10,10], all input information <= origin.',flush=True)


if __name__=='__main__':main()
