"""Full eigenvalue scaling and eigengap-regularized rotation."""
import torch

def rotate(base,values,u,multiplier,change,rows,cols,detail):
    normalized=values/values.mean(-1,keepdim=True)
    gap=(normalized[:,rows]-normalized[:,cols]).abs()
    angles=gap*(.1*torch.tanh(change))/(gap.square()+.01)
    lower=torch.zeros_like(base);lower[:,rows,cols]=angles
    rotation=torch.matrix_exp(lower-lower.transpose(-1,-2));updated=u@rotation
    prediction=(updated*(values*multiplier).unsqueeze(-2))@updated.transpose(-1,-2)
    detail['relative_boundary_gap']=((values[:,:-1]-values[:,1:])/values.mean(-1,keepdim=True)).min(-1).values
    detail['rotation']=rotation
    return prediction
