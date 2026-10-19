"""Compose the two research lines without changing checkpoint parameter names."""
from .fusion.gating.transformer import TemporalGate
from .fusion.gating.sequence import SequenceGate
from .fusion.model import SequenceJoint
from .direct.model import DirectSPD
from .direct.inputs.covariance import CovarianceSPD
from .direct.backbones.residual import build_model
from .direct.paper_models.cab import PaperCAB

def create_model(config,unit,scaler):
    family=config['family'];encoder=config.get('encoder','transformer')
    if family=='fusion':
        fusion=config.get('fusion','precision');refinement=config.get('refinement','none')
        if refinement!='none':return SequenceJoint(fusion,encoder,refinement,config.get('q',3))
        return TemporalGate(fusion) if encoder=='transformer' else SequenceGate(fusion,encoder)
    if family=='paper':return PaperCAB(scaler['mean'],scaler['scale'])
    if family!='direct':raise ValueError(family)
    head=config.get('head','residual');inputs=config.get('input','covariance')
    if encoder=='transformer':return (CovarianceSPD if inputs=='covariance' else DirectSPD)(head,unit)
    if inputs!='covariance' or head!='residual':raise ValueError('Existing non-Transformer experiment uses covariance input and Residual output')
    return build_model(encoder,unit)
