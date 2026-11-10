#!/usr/bin/env python3
"""Independent deterministic evaluation of an explicitly marked PPO checkpoint."""
import argparse
from pathlib import Path

from realman_rl_simtoreal.rl.ppo import load_ppo, evaluate_ppo
from realman_rl_simtoreal.rl.common.artifacts import write_json


def run(model_path,*,episodes=1,seed_start=88000):
    if episodes<1 or seed_start<0:raise ValueError('Positive episodes and nonnegative seed required')
    model,transform,config=load_ppo(model_path)
    result=evaluate_ppo(model,transform,config['environment'],range(seed_start,seed_start+episodes))
    result.update(model=str(Path(model_path).resolve()),seeds=list(range(seed_start,seed_start+episodes)),
                  model_actual_timesteps=int(model.num_timesteps),environment=config['environment'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed-start',type=int,default=88000);p.add_argument('--json',type=Path)
    a=p.parse_args()
    try:
        destination=a.json or a.model.parent/'independent_evaluation.json'
        if destination.exists():raise ValueError('Evaluation output exists; choose a new --json path')
        result=run(a.model,episodes=a.episodes,seed_start=a.seed_start)
        write_json(destination,result)
        print(result['summary']);print(result['goal_scope']);print('Saved',destination)
        return 0
    except (ValueError,RuntimeError,OSError,ImportError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
