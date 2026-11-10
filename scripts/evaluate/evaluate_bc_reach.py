#!/usr/bin/env python3
"""Paired frozen rule/untrained/BC evaluation on held-out targets."""
import argparse
from pathlib import Path
import torch
from realman_rl_simtoreal.rl.common.artifacts import empty_directory, write_json
from realman_rl_simtoreal.imitation.bc import load_policy, evaluate_policy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--untrained',type=Path,help='Defaults to untrained_bc.pt next to --model')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed-start',type=int,default=20000)
    p.add_argument('--episodes',type=int,default=100)
    a=p.parse_args()
    if a.episodes < 1 or a.seed_start < 0:p.error('Positive episodes and nonnegative seed required')
    try:
        torch.set_num_threads(1)
        trained,config=load_policy(a.model)
        untrained,initial_config=load_policy(a.untrained or a.model.parent/'untrained_bc.pt')
        if config != initial_config:raise ValueError('BC and untrained configs differ')
        seeds=list(range(a.seed_start,a.seed_start+a.episodes))
        if set(seeds) & (set(config['train_seeds']) | set(config['validation_seeds'])):
            raise ValueError('Evaluation targets overlap supervised train/validation seeds')
        out=empty_directory(a.out)
        summaries={};goals=None
        for name,policy in [('rule',None),('untrained',untrained),('bc',trained)]:
            result=evaluate_policy(name,seeds,policy=policy,environment=config['environment'])
            current=[e['goal_m'] for e in result['episodes']]
            if goals is not None and current != goals:raise ValueError('Unpaired targets')
            goals=current
            write_json(out/(name+'.json'),result)
            summaries[name]=result['summary']
            print(name,result['summary'],flush=True)
        write_json(out/'comparison.json',dict(model=str(a.model.resolve()),seeds=seeds,
                   paired_goals_identical=True,no_supervised_target_overlap=True,summaries=summaries))
        return 0
    except (ValueError,RuntimeError) as exc:
        p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
