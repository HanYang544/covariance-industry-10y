"""Reuse the temporal loop; keep architecture metadata accurate."""
import hashlib
from pathlib import Path
import run_temporal_gate as engine
from portfolio_risk.models.sequence_gates import SequenceGate


def main():
    original_save = engine.save
    for architecture in ['mlp', 'cnn']:
        engine.OUT = engine.ROOT / f'outputs/sequence_{architecture}_100'
        engine.OUT.mkdir(exist_ok=True)
        engine.TemporalGate = lambda fusion, kind=architecture: SequenceGate(fusion, kind)
        def save(path, value):
            if path.name == 'protocol.json':
                for key in ['heads', 'layers', 'feedforward', 'pooling', 'positional_encoding']:
                    value.pop(key, None)
                value['architecture'] = architecture
                value['structure'] = 'Flatten600-64-ReLU-32-ReLU-10' if architecture == 'mlp' else 'Conv1d6-32-k5-ReLU-Conv1d32-32-k5-ReLU-GlobalMean-10'
                for file in [Path(__file__), engine.ROOT/'src/portfolio_risk/models/sequence_gates.py']:
                    value['hashes'][str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            original_save(path, value)
        engine.save = save
        engine.main()
    engine.save = original_save


if __name__ == '__main__':
    main()
