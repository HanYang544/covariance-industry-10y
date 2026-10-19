"""100 rolling20 covariance matrices -> temporal Transformer -> five SPD heads."""
from torch import nn
from covariance_research.direct.model import DirectSPD

class CovarianceSPD(DirectSPD):
    def __init__(self, method, unit):
        super().__init__(method,unit)
        self.input=nn.Linear(100,32).double()

    def forward(self, sequence, base_root=None):
        assert sequence.shape[1:]==(100,10,10)
        return super().forward(sequence.flatten(2),base_root)
