"""End-to-end sequence fusion + unchanged spectral/learned/rotation refinement."""
import torch
from torch import nn
from covariance_research.fusion.gating.transformer import TemporalGate
from covariance_research.fusion.gating.sequence import SequenceGate
from covariance_research.fusion.refinement.core import JointRefinementND


class SequenceEncoder(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        if architecture == 'transformer':
            model = TemporalGate()
            self.input = model.input
            self.encoder = model.encoder
            self.register_buffer('position', model.position)
        else:
            self.encoder = SequenceGate(architecture=architecture).encoder

    def forward(self, sequence):
        if self.architecture == 'transformer':
            return self.encoder(self.input(sequence) + self.position)[:, -1]
        return self.encoder(sequence.transpose(1, 2) if self.architecture == 'cnn' else sequence)


class FusionAdapter(nn.Module):
    def __init__(self, architecture, fusion):
        super().__init__()
        self.gate = TemporalGate(fusion) if architecture == 'transformer' else SequenceGate(fusion, architecture)

    def forward(self, matrices, state, return_weights=False):
        prediction, weights = self.gate(state, *matrices)
        return (prediction, weights) if return_weights else prediction


class SequenceJoint(nn.Module):
    def __init__(self, fusion, architecture, variant, q):
        super().__init__()
        core = JointRefinementND(variant, q, fusion, dimension=10)
        core.fusion = FusionAdapter(architecture, fusion)
        core.encoder = SequenceEncoder(architecture)
        core.variance = nn.Linear(32, q).double()
        core.change = nn.Linear(32, q*(q-1)//2).double() if q > 1 else None
        for head in [core.variance, core.change]:
            if head is not None:
                nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        if variant == 'learned': core.directions = nn.Linear(32, 10*q).double()
        self.core = core.double()

    def forward(self, sequence, experts, whiteners):
        prediction, detail = self.core((experts, whiteners), sequence, diagnostics=True)
        return prediction, detail['weights']
