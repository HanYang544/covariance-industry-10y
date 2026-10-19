"""Sequence-conditioned gate; independent of the frozen MLP experiments."""
import math
import torch
from torch import nn


class TemporalGate(nn.Module):
    def __init__(self, fusion='precision', experts=10, length=100):
        super().__init__()
        self.fusion = fusion
        self.input = nn.Linear(6, 32)
        position = torch.arange(length, dtype=torch.float64)[:, None]
        frequency = torch.exp(torch.arange(0, 32, 2, dtype=torch.float64) * (-math.log(10000.) / 32))
        encoding = torch.zeros(length, 32, dtype=torch.float64)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer('position', encoding)
        layer = nn.TransformerEncoderLayer(32, 4, dim_feedforward=64, dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.head = nn.Linear(32, experts)
        self.double()

    def forward(self, sequence, experts, whiteners):
        tokens = self.input(sequence) + self.position
        # All tokens are observed at forecast time; bidirectional historical attention is causal.
        state = self.encoder(tokens)[:, -1]
        weights = self.head(state).softmax(-1)
        if self.fusion == 'covariance':
            prediction = torch.einsum('bk,bkij->bij', weights, experts)
        else:
            factor = torch.einsum('bk,bkij->bij', weights, whiteners)
            prediction = torch.linalg.inv(factor @ factor.transpose(-1, -2))
        return prediction, weights
