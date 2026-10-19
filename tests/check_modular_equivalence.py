"""Numerical migration checks; CPU only, no retraining or checkpoint writes."""
import ast
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'modules/fusion/src'),str(ROOT/'modules/fusion/experiments/holdout')]
from portfolio_risk.models.temporal_gate import TemporalGate as OldT
from portfolio_risk.models.sequence_gates import SequenceGate as OldS
from portfolio_risk.models.sequence_joint import SequenceJoint as OldJ
from portfolio_risk.models.direct_spd import DirectSPD as OldD
from portfolio_risk.models.covariance_spd import CovarianceSPD as OldC
from portfolio_risk.models.residual_architectures import build_model as old_build
from run_paper_etf import PaperCAB as OldCAB
from covariance_research.fusion.gating.transformer import TemporalGate
from covariance_research.fusion.gating.sequence import SequenceGate
from covariance_research.fusion.model import SequenceJoint
from covariance_research.direct.model import DirectSPD
from covariance_research.direct.inputs.covariance import CovarianceSPD
from covariance_research.direct.backbones.residual import build_model
from covariance_research.direct.paper_models.cab import PaperCAB
from covariance_research.common.targets import future_covariance
from covariance_research.statistical.ledoit_wolf import ledoit_wolf_covariance
from covariance_research.common.training import EarlyStopping

torch.set_num_threads(1);torch.manual_seed(44)
sequence=torch.randn(2,100,6,dtype=torch.float64)
matrix=torch.randn(2,100,10,10,dtype=torch.float64)*.1
a=torch.randn(2,10,10,10,dtype=torch.float64)
experts=a@a.transpose(-1,-2)+torch.eye(10,dtype=torch.float64)
whiteners=torch.linalg.cholesky(torch.linalg.inv(experts))
base=torch.eye(10,dtype=torch.float64).expand(2,10,10)
count=0
def check(old,new,args,tolerance=1e-10):
    global count
    torch.manual_seed(20261012);left=old()
    torch.manual_seed(20261012);right=new()
    assert list(left.state_dict())==list(right.state_dict())
    for name,v in left.state_dict().items():torch.testing.assert_close(v,right.state_dict()[name],rtol=0,atol=0)
    # Perturb output heads so identity initializations do not hide refinement errors.
    torch.manual_seed(45)
    with torch.no_grad():
        for name,p in left.named_parameters():
            if any(s in name for s in ['variance','change','head']):p.add_(torch.randn_like(p)*.003)
    right.load_state_dict(left.state_dict(),strict=True)
    left.eval();right.eval()
    x=left(*args);y=right(*args)
    if isinstance(x,tuple):x=x[0];y=y[0]
    torch.testing.assert_close(x,y,rtol=tolerance,atol=tolerance)
    x.square().mean().backward();y.square().mean().backward()
    for (name,p),(_,q) in zip(left.named_parameters(),right.named_parameters()):
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(),name
            torch.testing.assert_close(p.grad,q.grad,rtol=max(tolerance,1e-8),atol=tolerance)
    count+=1

for fusion in ['precision','covariance']:
    check(lambda:OldT(fusion),lambda:TemporalGate(fusion),(sequence,experts,whiteners))
    for encoder in ['mlp','cnn']:
        check(lambda:OldS(fusion,encoder),lambda:SequenceGate(fusion,encoder),(sequence,experts,whiteners))
    for encoder in ['mlp','cnn','transformer']:
        for variant,q in [('spectral',1),('spectral',3),('spectral',5),('learned',1),('learned',3),('learned',5),('idea2',10)]:
            check(lambda:OldJ(fusion,encoder,variant,q),lambda:SequenceJoint(fusion,encoder,variant,q),(sequence,experts,whiteners))
for head in ['cholesky','matrix_exp','spectral_softplus','variance_correlation','residual']:
    check(lambda:OldD(head,1e-4),lambda:DirectSPD(head,1e-4),(sequence,base))
    check(lambda:OldC(head,1e-4),lambda:CovarianceSPD(head,1e-4),(matrix,base))
for architecture in ['transformer','mlp','cnn','tcn','gru','patch_transformer']:
    check(lambda:old_build(architecture,1e-4),lambda:build_model(architecture,1e-4),(matrix,base))
check(lambda:OldCAB(np.eye(10)*1e-4,np.ones((10,10))*1e-5),lambda:PaperCAB(np.eye(10)*1e-4,np.ones((10,10))*1e-5),(matrix,),1e-6)
out=ROOT/'modules/fusion/outputs/industry_10y_aligned'
timeline=np.load(out/'timeline.npz');frozen=np.load(out/'inputs_train.npz')
origins=frozen['origins'][:3];returns=timeline['returns']
np.testing.assert_allclose(future_covariance(returns,origins),frozen['target'][:3],rtol=1e-12,atol=1e-16)
for row,t in enumerate(origins):
    np.testing.assert_allclose(ledoit_wolf_covariance(returns[t-59:t+1]),frozen['experts'][row,1],rtol=1e-12,atol=1e-16)
stop=EarlyStopping();assert stop.update(0,1)==(True,False);assert stop.update(30,2)==(False,True)
for path in (ROOT/'src').rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
print(f'PASS: {count} model configurations: initialization, strict state loading, predictions and gradients; frozen targets/LW; early stopping; source syntax.')

# End-to-end training smoke test on two synthetic examples only, in a temporary E-drive directory.
import tempfile
from unittest.mock import patch
from covariance_research.common.training import runner
class TinyData:
    def __init__(self,*args):
        self.unit=1.;self.scaler={};self.loaded=[]
    def load(self,split):
        self.loaded.append(split)
        return dict(sequence=sequence,experts=experts,whiteners=whiteners,target=experts[:,0]),np.array(['2000-01-01','2000-01-02'])
with tempfile.TemporaryDirectory(prefix='covariance-refactor-test-',dir='E:/Codex/Cache/Temp/portfolio-risk') as temporary:
    with patch.object(runner,'FrozenData',TinyData):
        runner.run(Path(temporary),dict(family='fusion',encoder='mlp',fusion='covariance',loss='frobenius',max_epochs=1),'smoke',device='cpu')
    path=Path(temporary)/'outputs/new_runs/smoke'
    assert all((path/name).exists() for name in ['protocol.json','selection.json','result.json','model.pt','predictions.npz','training_history.csv'])
    assert 'epoch,train,validation' in (path/'training_history.csv').read_text()
print('PASS: new unified runner one-step synthetic smoke test and output files.')
