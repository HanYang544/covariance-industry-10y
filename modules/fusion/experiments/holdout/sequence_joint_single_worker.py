"""Additive 100-day feature Transformer experiment; frozen inputs are read-only."""
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path
import numpy as np
import torch
from portfolio_risk.models.sequence_joint import SequenceJoint

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/industry_10y_aligned'
OUT = ROOT / 'outputs/temporal_gate_100'
SEEDS = [20261012]


def score(prediction, target, objective):
    error = prediction - target
    if objective == 'euclidean':
        i, j = torch.triu_indices(10, 10, device=error.device)
        error = error[:, i, j]
    return torch.linalg.vector_norm(error.flatten(1), dim=1)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def main(configuration):
    global OUT
    OUT = ROOT / 'outputs/sequence_joint_single_100' / configuration['id']
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    raw = np.genfromtxt(SOURCE / 'raw_features.csv', delimiter=',', skip_header=1, usecols=range(1, 7))
    scaler = np.load(SOURCE / 'scaler.npz')
    features = np.clip((raw - scaler['gate_mean']) / scaler['gate_scale'], -3, 3)
    data = {}
    dates = {}
    for split in ['train', 'validation']:
        a = dict(np.load(SOURCE / f'inputs_{split}.npz'))
        history = np.stack([features[t-99:t+1] for t in a['origins']])
        assert np.isfinite(history).all()
        np.testing.assert_allclose(history[:, -1], a['state'], atol=1e-12)
        data[split] = {k: torch.tensor(v, device=device, dtype=torch.float64) for k, v in dict(sequence=history, experts=a['experts'], whiteners=a['whiteners'], target=a['target']).items()}
        dates[split] = a['dates']
    paths = [SOURCE / f'inputs_{s}.npz' for s in ['train', 'validation', 'test']] + [SOURCE / 'raw_features.csv', SOURCE / 'scaler.npz', Path(__file__), ROOT / 'src/portfolio_risk/models/temporal_gate.py']
    paths += [ROOT / ('src/portfolio_risk/models/' + name + '.py') for name in ['sequence_joint', 'sequence_gates', 'joint_refinement_nd']]
    protocol = dict(history=100, features=6, experts=10, calibration=False, refinement=True, configuration=configuration, independent_encoders=True, freeze_fusion=False, dimension=32, dropout=0, optimizer='Adam', lr=.01, batch='full batch gradients accumulated in microbatches of 128', epochs=500, patience=30, min_delta=1e-12, epoch0=True, dtype='float64', seeds=SEEDS, source=str(SOURCE), hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    if (OUT / 'protocol.json').exists():
        assert json.loads((OUT / 'protocol.json').read_text(encoding='utf-8')) == protocol
    else:
        save(OUT / 'protocol.json', protocol)

    def predict(model, batch):
        predictions, weights = [], []
        with torch.no_grad():
            for start in range(0, len(batch['target']), 128):
                p, w = model(*[batch[k][start:start+128] for k in ['sequence', 'experts', 'whiteners']])
                predictions.append(p); weights.append(w)
        return torch.cat(predictions), torch.cat(weights)

    records = []
    started = time.time()
    for fusion, objective, seed in itertools.product(['precision', 'covariance'], ['frobenius', 'euclidean'], SEEDS):
        folder = OUT / fusion / objective / str(seed)
        folder.mkdir(parents=True, exist_ok=True)
        selection = folder / 'selection.json'
        original = ROOT / 'outputs/sequence_joint_100' / configuration['id'] / fusion / objective / str(seed)
        if not selection.exists() and (original / 'selection.json').exists():
            import shutil
            for name in ['selection.json', 'model.pt', 'training_history.csv']:
                shutil.copy2(original / name, folder / name)
            save(folder / 'reused_from.json', dict(source=str(original), seed=seed))
        if selection.exists():
            records.append(json.loads(selection.read_text())); continue
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        model = SequenceJoint(fusion, configuration['architecture'], configuration['variant'], configuration['q']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        best = float('inf'); best_epoch = 0; history = []
        for epoch in range(501):
            if epoch:
                model.train(); optimizer.zero_grad()
                batch = data['train']; count = len(batch['target'])
                for start in range(0, count, 128):
                    p, _ = model(*[batch[k][start:start+128] for k in ['sequence', 'experts', 'whiteners']])
                    (score(p, batch['target'][start:start+128], objective).sum() / count).backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                value = score(predict(model, data['validation'])[0], data['validation']['target'], objective).mean().item()
            assert np.isfinite(value)
            history.append(dict(epoch=epoch, validation=value))
            if value < best - 1e-12:
                best = value; best_epoch = epoch
                torch.save(model.cpu().state_dict(), folder / 'model.pt'); model.to(device)
            if epoch % 10 == 0:
                progress = dict(fusion=fusion, objective=objective, seed=seed, epoch=epoch, best_epoch=best_epoch, validation=value, selected=len(records), total=4, seconds=time.time()-started)
                save(OUT / 'progress.json', progress); print(json.dumps(progress), flush=True)
            if epoch - best_epoch >= 30: break
        with (folder / 'training_history.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'validation']); writer.writeheader(); writer.writerows(history)
        record = dict(fusion=fusion, objective=objective, seed=seed, best_epoch=best_epoch, stop_epoch=epoch, validation=best)
        save(selection, record); records.append(record)
    save(OUT / 'selection.json', records)
    # Test is loaded only after every validation checkpoint is locked.
    a = dict(np.load(SOURCE / 'inputs_test.npz'))
    history = np.stack([features[t-99:t+1] for t in a['origins']])
    batch = {k: torch.tensor(v, device=device, dtype=torch.float64) for k, v in dict(sequence=history, experts=a['experts'], whiteners=a['whiteners'], target=a['target']).items()}
    for record in records:
        folder = OUT / record['fusion'] / record['objective'] / str(record['seed'])
        model = SequenceJoint(record['fusion'], configuration['architecture'], configuration['variant'], configuration['q']).to(device)
        model.load_state_dict(torch.load(folder / 'model.pt', weights_only=True, map_location=device)); model.eval()
        prediction, weights = predict(model, batch)
        torch.linalg.cholesky(prediction)
        record['test'] = score(prediction, batch['target'], record['objective']).mean().item()
        np.savez_compressed(folder / 'predictions.npz', prediction=prediction.cpu().numpy(), weights=weights.cpu().numpy(), dates=a['dates'])
        save(folder / 'result.json', record)
    save(OUT / 'results.json', records)
    save(OUT / 'completion.json', dict(runs=4, seconds=time.time()-started))
    print('Completed all 4 temporal gate runs', flush=True)


if __name__ == '__main__':
    raise SystemExit('Use run_sequence_joint.py')
