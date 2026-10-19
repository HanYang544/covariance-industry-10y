"""Single fixed seed quick comparison; reuse completed validation selections."""
import json
import subprocess
import sys
from pathlib import Path
from sequence_joint_single_worker import main as fit
MODULE=Path(__file__).resolve().parents[2]
ROOT=MODULE/'outputs/sequence_joint_single_100'
ROOT.mkdir(parents=True,exist_ok=True)
configs=json.loads((MODULE/'outputs/sequence_joint_100/plan.json').read_text())['configurations']
(ROOT/'plan.json').write_text(json.dumps(dict(configurations=configs,seed=20261012,total_fits=84,reason='User requested single-seed quick exploration; completed matching checkpoints reused'),indent=2),encoding='utf-8')
for index,c in enumerate(configs):
    if not (ROOT/c['id']/'completion.json').exists():fit(c)
    (ROOT/'progress.json').write_text(json.dumps(dict(completed_configurations=index+1,total_configurations=21,completed_fits=(index+1)*4,last=c['id']),indent=2),encoding='utf-8')
    subprocess.run([sys.executable,'-B',str(MODULE.parent.parent/'scripts/build_report.py')],check=True)
(ROOT/'completion.json').write_text(json.dumps(dict(configurations=21,fits=84,seed=20261012)),encoding='utf-8')
