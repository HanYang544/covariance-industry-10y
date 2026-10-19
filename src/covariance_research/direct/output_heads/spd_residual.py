import torch

def transform(symmetric,base_root):
    if base_root is None:raise ValueError('LW60 normalized symmetric square root required')
    return base_root@torch.matrix_exp(symmetric)@base_root.transpose(-1,-2)
