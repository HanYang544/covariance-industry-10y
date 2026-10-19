"""Low-dimensional spectral or learned-direction correction."""
import torch

def spectral(values,u,multiplier,correlation,q,detail):
    amplitude=(values[:,:q]*multiplier).sqrt()
    core=correlation*amplitude.unsqueeze(-1)*amplitude.unsqueeze(-2)
    selected=u[:,:,:q];rest=u[:,:,q:]
    prediction=selected@core@selected.transpose(-1,-2)+(rest*values[:,None,q:])@rest.transpose(-1,-2)
    if q<values.shape[-1]:detail['relative_boundary_gap']=(values[:,q-1]-values[:,q])/values.mean(-1)
    return prediction

def learned(base,raw,multiplier,correlation,q,detail):
    v,r=torch.linalg.qr(raw,mode='reduced')
    v=v*torch.where(r.diagonal(dim1=-2,dim2=-1)<0,-1.,1.).unsqueeze(-2)
    amplitude=multiplier.sqrt()
    core=correlation*amplitude.unsqueeze(-1)*amplitude.unsqueeze(-2)
    identity=torch.eye(base.shape[-1],dtype=base.dtype,device=base.device)
    middle=identity+v@(core-torch.eye(q,dtype=base.dtype,device=base.device))@v.transpose(-1,-2)
    factor=torch.linalg.cholesky(base);prediction=factor@middle@factor.transpose(-1,-2)
    detail['qr_min_abs_diagonal']=r.diagonal(dim1=-2,dim2=-1).abs().min(-1).values
    return prediction
