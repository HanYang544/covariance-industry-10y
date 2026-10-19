"""Recompute reported scores from all original/additional predictions."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'modules/fusion/outputs'
target=np.load(OUT/'industry_10y_aligned/inputs_test.npz')
paths=list((OUT/'seed_completion').glob('*/**/result.json'))
assert len(paths)==232
for family in ['sequence_joint_single_100','direct_spd_100','covariance_spd_100','residual_architecture_100']:
    paths+=list((OUT/family).rglob('result.json'))
assert len(paths)==348,len(paths)
for path in paths:
    r=json.loads(path.read_text())
    p=np.load(path.parent/'predictions.npz')
    np.testing.assert_array_equal(p['dates'],target['dates'])
    delta=p['prediction']-target['target']
    assert np.isfinite(delta).all()
    if r['objective']=='euclidean':
        i,j=np.triu_indices(10);delta=delta[:,i,j]
    score=np.linalg.norm(delta.reshape(len(delta),-1),axis=1).mean()
    np.testing.assert_allclose(score,r['test'],rtol=1e-10,atol=1e-14,err_msg=str(path))
width=None
for line in (ROOT/'experiment_results.md').read_text(encoding='utf-8').splitlines():
    if not line.startswith('|'):width=None;continue
    n=line.count('|')
    if width is None:width=n
    assert width==n,line
print('Verified 348 scores/date alignments and all Markdown table widths.')
