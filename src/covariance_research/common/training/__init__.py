from dataclasses import dataclass
import torch

@dataclass
class EarlyStopping:
    patience: int = 30
    min_delta: float = 1e-12
    best: float = float('inf')
    best_epoch: int = 0

    def update(self,epoch,value):
        improved=value < self.best-self.min_delta
        if improved: self.best,self.best_epoch=value,epoch
        return improved,epoch-self.best_epoch >= self.patience

def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def matrix_scores(prediction,target,objective):
    error=prediction-target
    if objective=='euclidean':
        i,j=torch.triu_indices(error.shape[-1],error.shape[-1],device=error.device)
        error=error[...,i,j]
    elif objective!='frobenius':raise ValueError(objective)
    return torch.linalg.vector_norm(error.flatten(1),dim=1)
