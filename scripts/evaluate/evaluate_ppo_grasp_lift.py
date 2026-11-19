#!/usr/bin/env python3
"""Independent evaluation of four-action PPO, with no staged-controller fallback."""
import argparse
from pathlib import Path

from realman_rl_simtoreal.utils.artifacts import file_sha256
from realman_rl_simtoreal.rl.ppo_grasp import CURRICULUM,environment_config,load_checkpoint,evaluate_model
from realman_rl_simtoreal.rl.common.artifacts import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--stage',choices=list(CURRICULUM),help='Explicit task override; default saved training task')
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed-start',type=int,default=250000)
    p.add_argument('--json',type=Path,required=True)
    args=p.parse_args()
    if args.episodes<1 or args.seed_start<0:p.error('Positive episode count and nonnegative seed required')
    if args.json.exists():p.error('Output exists; select a new JSON path')
    try:
        model,config=load_checkpoint(args.model)
        stage=config['curriculum_stage'] if args.stage is None else args.stage
        if args.stage is not None:config=dict(config,environment=environment_config(stage))
        result=evaluate_model(model,config,range(args.seed_start,args.seed_start+args.episodes))
        result.update(model=str(args.model.resolve()),model_sha256=file_sha256(args.model),
            training_stage=config['curriculum_stage'],evaluation_stage=stage,staged_controller_used=False)
        write_json(args.json,result)
        print(result['summary']);print('Saved',args.json)
        return 0
    except (ValueError,RuntimeError,OSError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
