"""Read an explicit model configuration; training requires --train and a fresh name."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
parser=argparse.ArgumentParser()
parser.add_argument('--config',required=True)
parser.add_argument('--train',action='store_true')
parser.add_argument('--name')
parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
args=parser.parse_args()
config=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
if not args.train:print(json.dumps(config,indent=2))
else:
    if not args.name:parser.error('--train requires --name for a new output directory')
    from covariance_research.common.training.runner import run
    run(ROOT,config,args.name,args.device)
