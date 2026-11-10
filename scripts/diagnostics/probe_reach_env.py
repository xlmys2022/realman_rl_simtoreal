#!/usr/bin/env python3
"""Bounded reward/action/target probes only: no learner, expert or BC policy."""
import argparse
import json
from pathlib import Path

import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.tasks.reach import TASKS, task_environment, ENV_CONFIG
from realman_rl_simtoreal.rl.common.metrics import MotionStats
from realman_rl_simtoreal.rl.common.artifacts import write_json


def trajectory(config, name, seed=81000):
    rows=[];motion=MotionStats()
    with ReachEnv(**config) as env:
        obs,_=env.reset(seed=seed)
        initial=obs['achieved_goal'].copy()
        rng=np.random.default_rng([2026,seed])
        for step in range(env.max_episode_steps):
            # Fixed open-loop probes of a fixed +X target, NOT a training policy.
            if name=='forward':action=np.array([1.,0,0] if step < 5 else [0,0,0],dtype=np.float32)
            elif name=='idle':action=np.zeros(3,dtype=np.float32)
            elif name in ('fail','reject'):action=np.ones(3,dtype=np.float32)
            elif name=='random':action=rng.uniform(-1,1,3).astype(np.float32)
            else:raise ValueError('Unknown probe')
            before=obs
            obs,reward,t,tr,info=env.step(action)
            assert reward==env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info)
            motion.observe(before['achieved_goal'],obs['achieved_goal'],obs['desired_goal'],
                           before['observation'][18:21],info['command_position_m'],action,t or tr)
            rows.append(dict(step=step+1,action=action.tolist(),previous_m=before['achieved_goal'].tolist(),
                achieved_m=obs['achieved_goal'].tolist(),goal_m=obs['desired_goal'].tolist(),reward=reward,
                physics_steps=info['physics_steps'],distance_m=info['distance_m'],
                rejected=info.get('is_action_rejected',False),safety_failure=info['is_safety_failure'],
                success=info['is_success'],terminated=t,truncated=tr,reason=info['termination_reason']))
            if t or tr:break
        rewards=np.array([r['reward'] for r in rows])
        gamma=.99
        result=dict(config=config,seed=seed,steps=len(rows),reason=info['termination_reason'],
            success=info['is_success'],final_distance_m=info['distance_m'],return_=float(rewards.sum()),
            discounted_return=float(rewards@gamma**np.arange(len(rows))),
            rejected_actions=sum(r['rejected'] for r in rows),motion=motion.summary(),records=rows)
        # Counterexample: post-hoc naive bounded negative exponential + +/-1
        # on the SAME physical transitions. It is NOT enabled in the environment.
        naive=[np.expm1(-r['distance_m']/.005)+int(r['success'])-int(r['safety_failure']) for r in rows]
        result['naive_exponential_discounted_return']=float(np.array(naive)@gamma**np.arange(len(rows)))
        if env.reward_mode=='reach_shaped_v1':
            phi0=np.expm1(-np.linalg.norm(initial-obs['desired_goal'])/env.reward_sigma_m)
            phin=0. if rows[-1]['terminated'] else np.expm1(-info['distance_m']/env.reward_sigma_m)
            terminal=env.shaped_terminal_value*(int(rows[-1]['success'])-int(rows[-1]['safety_failure']))
            expected=(-phi0+gamma**len(rows)*phin+gamma**(len(rows)-1)*terminal
                      -(1-gamma)*sum(gamma**i*r['rejected'] for i,r in enumerate(rows)))
            if abs(expected-result['discounted_return'])>1e-10:raise ValueError('Discounted potential identity failed')
            result.update(terminal_value=env.shaped_terminal_value,telescoping_error=abs(expected-result['discounted_return']))
        return result


def run(folder,episodes=4):
    folder=Path(folder)
    if folder.exists() and any(folder.iterdir()):raise ValueError('Output must be empty')
    folder.mkdir(parents=True,exist_ok=True)
    rewards={}
    for mode in ('safe_dense','reach_shaped_v1'):
        rewards[mode]={}
        for name in ('forward','idle','fail','reject'):
            config=task_environment('fixed_reach',ENV_CONFIG,reward_mode=mode)
            if name in ('fail','reject'):
                # Deliberate invalid command, never physically executed.
                config['action_scale']=.01
            if name=='reject':config['boundary_mode']='reject_and_hold'
            rewards[mode][name]=trajectory(config,name)
    for name in rewards['safe_dense']:
        a,b=rewards['safe_dense'][name]['records'],rewards['reach_shaped_v1'][name]['records']
        if [(r['achieved_m'],r['reason']) for r in a] != [(r['achieved_m'],r['reason']) for r in b]:
            raise ValueError('Reward changed physical transitions')
    new=rewards['reach_shaped_v1']
    if not new['forward']['discounted_return']>new['idle']['discounted_return']>new['reject']['discounted_return']>new['fail']['discounted_return']:
        raise ValueError('Measured reward ranking failed')
    scale_results={}
    for scale in (.001,.00025):
        config=task_environment('random_reach',ENV_CONFIG,reward_mode='safe_dense',action_scale=scale)
        scale_results[str(scale)]=[trajectory(config,'random',seed=82000+i) for i in range(episodes)]
    targets={}
    for task in TASKS:
        config=task_environment(task,ENV_CONFIG)
        with ReachEnv(**config) as env:
            offsets=[]
            for seed in range(83000,83020):
                obs,_=env.reset(seed=seed)
                offsets.append((obs['desired_goal']-env.kinematics.home_position).tolist())
            targets[task]=dict(config=config,offsets_m=offsets)
    result=dict(reward_probes=rewards,action_scales=scale_results,target_probes=targets,
                note='No training. Forward is a fixed open-loop reward probe only; random actions use an independent RNG.')
    write_json(folder/'results.json',result)
    for mode,cases in rewards.items():
        print(mode,{k:dict(reason=v['reason'],steps=v['steps'],discounted_return=v['discounted_return']) for k,v in cases.items()},flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=4,help='Random action episodes per scale')
    args=p.parse_args()
    if args.episodes<1:p.error('Positive episode count required')
    try:run(args.out,args.episodes);return 0
    except (ValueError,RuntimeError,OSError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
