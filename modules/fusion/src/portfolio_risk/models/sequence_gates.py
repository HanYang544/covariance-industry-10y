"""MLP/CNN gates for the same 100x6 historical input."""
import torch
from torch import nn


class SequenceGate(nn.Module):
    def __init__(self, fusion='precision', architecture='mlp'):
        super().__init__()
        self.fusion = fusion
        self.architecture = architecture
        if architecture == 'mlp':
            self.encoder = nn.Sequential(nn.Flatten(1), nn.Linear(600, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        elif architecture == 'cnn':
            self.encoder = nn.Sequential(nn.Conv1d(6, 32, 5, padding=2), nn.ReLU(), nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(1))
        else:
            raise ValueError(architecture)
        self.head = nn.Linear(32, 10)
        self.double()

    def forward(self, sequence, experts, whiteners):
        assert sequence.shape[1:] == (100, 6)
        x = sequence.transpose(1, 2) if self.architecture == 'cnn' else sequence
        weights = self.head(self.encoder(x)).softmax(-1)
        if self.fusion == 'covariance':
            prediction = torch.einsum('bk,bkij->bij', weights, experts)
        else:
            factor = torch.einsum('bk,bkij->bij', weights, whiteners)
            prediction = torch.linalg.inv(factor @ factor.transpose(-1, -2))
        return prediction, weights
