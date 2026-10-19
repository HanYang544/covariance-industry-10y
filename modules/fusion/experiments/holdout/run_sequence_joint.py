"""Fixed, nonadaptive 21-configuration queue, 12 runs per configuration."""
import json
from pathlib import Path
from sequence_joint_worker import main as fit

ROOT=Path(__file__).resolve().parents[2]/'outputs/sequence_joint_100'
ROOT.mkdir(parents=True,exist_ok=True)
configs=[dict(id=f'{a}_{v}_q{q}',architecture=a,variant=v,q=q) for a in ['mlp','cnn','transformer'] for v,q in [(v,q) for v in ['spectral','learned'] for q in [1,3,5]]+[('idea2',10)]]
(ROOT/'plan.json').write_text(json.dumps(dict(configurations=configs,total_fits=252,selection='Each configuration locks all12 validation checkpoints before its test evaluation; entire grid fixed before any testing'),indent=2),encoding='utf-8')
for index,c in enumerate(configs):
    if not (ROOT/c['id']/'completion.json').exists():fit(c)
    (ROOT/'progress.json').write_text(json.dumps(dict(completed_configurations=index+1,total_configurations=21,completed_fits=(index+1)*12,last=c['id']),indent=2),encoding='utf-8')
(ROOT/'completion.json').write_text(json.dumps(dict(configurations=21,fits=252)),encoding='utf-8')
print('All252 sequence joint runs complete',flush=True)
