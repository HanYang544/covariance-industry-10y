"""Public-data reconstruction of the Reis ETF task; never overwrite old runs."""
import concurrent.futures
import hashlib
import io
import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parents[2]/'outputs/paper_etf'
ASSETS='IXN IXG RXI EXI IXJ KXI IYR IFGL MXI IXC IXP JXI AGG IAGG'.split()

def fetch(url):
    headers={} if 'stlouisfed.org' in url else {'User-Agent':'Mozilla/5.0'}
    return urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=40).read()

def main():
    raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
    if (OUT/'data_protocol.json').exists():
        print('Existing prepared snapshot retained');return
    def download(ticker):
        url=f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1483228800&period2=1704067200&interval=1d'
        p=raw/f'{ticker}.json'
        if not p.exists():p.write_bytes(fetch(url))
        r=json.loads(p.read_bytes())['chart']['result'][0]
        dates=pd.to_datetime(r['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
        return pd.Series(r['indicators']['adjclose'][0]['adjclose'],index=dates,name=ticker)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        prices=pd.concat(list(pool.map(download,ASSETS)),axis=1)
    prices=prices.sort_index();assert not prices.index.duplicated().any()
    assert prices.notna().all().all(), 'Missing ETF prices: stop instead of silently filling'
    assert (prices>0).all().all() and len(prices)==1760
    p=raw/'DGS1MO.csv'
    if not p.exists():p.write_bytes(fetch('https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS1MO&cosd=2016-12-01&coed=2023-12-31'))
    rf=pd.read_csv(io.BytesIO(p.read_bytes()),index_col=0,parse_dates=True).iloc[:,0]
    rf=pd.to_numeric(rf,errors='coerce').reindex(prices.index.union(rf.index)).sort_index().ffill().reindex(prices.index)
    # Conservative lag of available yield; simple annual percentage / 252 convention.
    returns=prices.pct_change(fill_method=None).sub(rf.shift(1)/100/252,axis=0).iloc[1:]
    assert np.isfinite(returns.to_numpy()).all()
    r=returns.to_numpy();n=len(r)
    cov=np.full((n,14,14),np.nan)
    for t in range(19,n):cov[t]=np.cov(r[t-19:t+1],rowvar=False,ddof=1)
    # Chronological 80/20 split of pre-2021 return dates; labels may not cross boundaries.
    pre=np.flatnonzero(returns.index.year<=2020);boundary=pre[int(.8*len(pre))]
    test_start=np.flatnonzero(returns.index.year>=2021)[0]
    origins=np.arange(118,n-20)
    splits={'train':origins[origins+20<boundary],
            'validation':origins[(origins>=boundary)&(origins+20<test_start)],
            'test':origins[origins>=test_start]}
    for name,ix in splits.items():
        x=np.stack([cov[t-99:t+1] for t in ix]);y=cov[ix+20]
        assert np.isfinite(x).all() and np.isfinite(y).all()
        np.savez_compressed(OUT/f'{name}.npz',history=x,target=y,origins=ix,dates=returns.index[ix].strftime('%Y-%m-%d').to_numpy(dtype='U10'))
    train=np.load(OUT/'train.npz')['history']
    np.savez(OUT/'scaler.npz',mean=train.mean(axis=(0,1)),scale=np.maximum(train.std(axis=(0,1)),1e-12))
    np.savez_compressed(OUT/'timeline.npz',returns=r,covariance=cov,dates=returns.index.strftime('%Y-%m-%d').to_numpy(dtype='U10'))
    protocol=dict(assets=ASSETS,source='Yahoo adjusted close, NOT original Refinitiv; FRED DGS1MO',
        paper='https://doi.org/10.1002/for.70110',price_days=len(prices),return_days=n,
        returns='simple fractional returns minus lagged DGS1MO /100/252; no backfill',
        covariance='centered sample covariance ddof=1, 20 days; same definition inputs/targets',
        input=[100,14,14],target='t+1..t+20; daily stride; no 2024 observations',
        split_boundary=str(returns.index[boundary].date()),counts={k:len(v) for k,v in splits.items()},
        label_purge='last train label < validation start; last validation label < test start',
        test_end='last 20 origins of 2023 omitted because complete future labels unavailable',
        ambiguities=['simple returns','ddof=1','riskfree daily conversion/lag','chronological 80/20 before supervised window construction','per-entry train-only scaling'],
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raw.iterdir())})
    (OUT/'data_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    print(json.dumps(protocol,indent=2),flush=True)

if __name__=='__main__':main()
