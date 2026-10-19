"""Read frozen experiment inputs; never recalculate historical labels implicitly."""
import numpy as np
import torch

class FrozenData:
    def __init__(self,root,config,device):
        self.root,self.config,self.device=root,config,device
        self.scaler=dict(np.load(root/'cab/scaler.npz'))
        target=np.load(root/'inputs_train.npz')['target']
        self.unit=float(np.diagonal(target,axis1=-2,axis2=-1).mean())
        self.features=None
        if config['family']=='fusion' or config.get('input')=='features':
            raw=np.genfromtxt(root/'raw_features.csv',delimiter=',',skip_header=1,usecols=range(1,7))
            scaler=np.load(root/'scaler.npz')
            self.features=np.clip((raw-scaler['gate_mean'])/scaler['gate_scale'],-3,3)

    def load(self,split):
        a=dict(np.load(self.root/f'inputs_{split}.npz'));family=self.config['family']
        if self.features is not None:
            sequence=np.stack([self.features[t-99:t+1] for t in a['origins']])
            np.testing.assert_allclose(sequence[:,-1],a['state'],atol=1e-12)
        else:
            h=np.load(self.root/'cab'/f'{split}.npz')
            np.testing.assert_array_equal(h['dates'],a['dates']);np.testing.assert_array_equal(h['target'],a['target'])
            sequence=h['history']
            if family!='paper':sequence=(sequence-self.scaler['mean'])/self.scaler['scale']
        batch=dict(sequence=sequence,target=a['target'])
        if family=='fusion':batch.update(experts=a['experts'],whiteners=a['whiteners'])
        elif family=='direct':
            v,u=np.linalg.eigh(a['experts'][:,1]/self.unit)
            assert (v>0).all()
            batch['root']=(u*np.sqrt(v)[:,None,:])@u.transpose(0,2,1)
        assert all(np.isfinite(v).all() for v in batch.values())
        return {k:torch.tensor(v,dtype=torch.float64,device=self.device) for k,v in batch.items()},a['dates']
