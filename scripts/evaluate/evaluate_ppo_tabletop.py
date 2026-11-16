#!/usr/bin/env python3
"""Read-only PPO evaluation from the saved pose, or the full tabletop task."""
import argparse
import hashlib
from pathlib import Path

from realman_rl_simtoreal.rl.ppo_tabletop import load_checkpoint,evaluate_model
from realman_rl_simtoreal.rl.common.artifacts import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--json',required=True,type=Path)
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed-start',type=int,default=350000)
    p.add_argument('--full',action='store_true',help='Force original initial joint pose, open jaws, 30mm lift and 2s hold')
    args=p.parse_args()
    if args.json.exists():p.error('Choose a new evaluation output path')
    if args.episodes<1 or args.seed_start<0:p.error('Positive episode count and nonnegative seed required')
    model,config=load_checkpoint(args.model)
    result=evaluate_model(model,config,range(args.seed_start,args.seed_start+args.episodes),full=args.full)
    result.update(model=str(args.model.resolve()),model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  training_stage=config['curriculum_stage'],full_task_override=args.full)
    write_json(args.json,result);print(result['summary']);print('Saved',args.json)
    return 0


if __name__=='__main__':raise SystemExit(main())
