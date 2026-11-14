#!/usr/bin/env python3
"""Independent deterministic evaluation of an explicitly marked PPO checkpoint."""
import argparse
from pathlib import Path

from realman_rl_simtoreal.rl.ppo import load_ppo, evaluate_ppo
from realman_rl_simtoreal.rl.common.artifacts import write_json
from realman_rl_simtoreal.envs.reach_env import OBSERVATION_MODES


def run(model_path,*,episodes=None,seed_start=88000,observation_mode=None):
    model,transform,config=load_ppo(model_path,observation_mode=observation_mode)
    n=len(config['environment']['target_offsets']) if config['environment']['target_mode']=='discrete' else None
    if episodes is None:episodes=n or 1
    if episodes<1 or seed_start<0:raise ValueError('Positive episodes and nonnegative seed required')
    if n is not None and episodes!=n:raise ValueError(f'Discrete evaluation requires exactly {n} episodes, one per target')
    result=evaluate_ppo(model,transform,config['environment'],range(seed_start,seed_start+episodes))
    result.update(model=str(Path(model_path).resolve()),seeds=list(range(seed_start,seed_start+episodes)),
                  model_actual_timesteps=int(model.num_timesteps),environment=config['environment'],
                  observation_mode=transform.observation_mode,features_dim=transform.features_dim)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--observation-mode',choices=OBSERVATION_MODES,
                   help='Defaults to checkpoint mode; an explicit mismatch is rejected')
    p.add_argument('--episodes',type=int,help='Default one for legacy tasks; all candidates for discrete tasks; repeats rejected')
    p.add_argument('--seed-start',type=int,default=88000);p.add_argument('--json',type=Path)
    a=p.parse_args()
    try:
        destination=a.json or a.model.parent/'independent_evaluation.json'
        if destination.exists():raise ValueError('Evaluation output exists; choose a new --json path')
        result=run(a.model,episodes=a.episodes,seed_start=a.seed_start,observation_mode=a.observation_mode)
        write_json(destination,result)
        print(result['summary']);print(result['goal_scope'])
        for label,row in result.get('per_goal',{}).items():
            print(label,'success=',row['success'],'distance_mm=',1000*row['final_distance_m'],
                  'steps=',row['episode_length'],'first_action=',row['first_action'],
                  'reason=',row['termination_reason'],'actual_path_m=',row['actual_path_m'],
                  'toward_goal_fraction=',row['action_direction_fraction'])
        print('Saved',destination)
        return 0
    except (ValueError,RuntimeError,OSError,ImportError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
