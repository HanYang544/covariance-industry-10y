"""Load real archived checkpoints through the new modules and match saved predictions."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from covariance_research.factory import create_model
torch.set_num_threads(1)
OUT=ROOT/'modules/fusion/outputs';data=OUT/'industry_10y_aligned'
a=np.load(data/'inputs_test.npz');cab=np.load(data/'cab/test.npz');scale=dict(np.load(data/'cab/scaler.npz'))
target=np.load(data/'inputs_train.npz')['target'];unit=float(np.diagonal(target,axis1=-2,axis2=-1).mean())
raw=np.genfromtxt(data/'raw_features.csv',delimiter=',',skip_header=1,usecols=range(1,7));s=np.load(data/'scaler.npz')
features=np.clip((raw-s['gate_mean'])/s['gate_scale'],-3,3)
seq=np.stack([features[t-99:t+1] for t in a['origins'][:2]])
tensor=lambda x:torch.tensor(x,dtype=torch.float64)
fusion_args=(tensor(seq),tensor(a['experts'][:2]),tensor(a['whiteners'][:2]))
v,u=np.linalg.eigh(a['experts'][:2,1]/unit);root=(u*np.sqrt(v)[:,None,:])@u.transpose(0,2,1)
cov_args=(tensor((cab['history'][:2]-scale['mean'])/scale['scale']),tensor(root))
cases=[
 (dict(family='fusion',encoder='transformer',fusion='precision'),OUT/'temporal_gate_100/precision/frobenius/20261012',fusion_args),
 (dict(family='fusion',encoder='cnn',fusion='precision',refinement='spectral',q=1),OUT/'sequence_joint_single_100/cnn_spectral_q1/precision/frobenius/20261012',fusion_args),
 (dict(family='direct',encoder='transformer',input='covariance',head='residual'),OUT/'covariance_spd_100/residual/frobenius',cov_args),
 (dict(family='direct',encoder='cnn',input='covariance',head='residual'),OUT/'residual_architecture_100/cnn/frobenius',cov_args),
 (dict(family='paper'),data/'cab/frobenius/20261012',(tensor(cab['history'][:2]),))]
for config,path,args in cases:
    model=create_model(config,unit,scale)
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
    with torch.no_grad():p=model(*args)
    if isinstance(p,tuple):p=p[0]
    expected=np.load(path/'predictions.npz')['prediction'][:2]
    np.testing.assert_allclose(p.numpy(),expected,rtol=2e-5 if config['family']=='paper' else 1e-8,atol=1e-10)
print('PASS: 5 real archived model checkpoints reproduce stored predictions via new modules (CPU/GPU numerical tolerance).')
