#!/usr/bin/env python3
"""Bounded real-physics reward direction and analytic checks; never train."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.rl.ppo import PPO_CONFIG
from realman_rl_simtoreal.tasks.reach import ENV_CONFIG, task_environment
from realman_rl_simtoreal.tasks.rewards import REWARD_MODES


OLD_MODES = ('distance', 'safe_dense', 'reach_shaped_v1')


def native(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):return [native(v) for v in value]
    return value


def trace(mode, goal_index, direction, *, steps, pulse_steps=None):
    config = task_environment('axis_pair_reach', ENV_CONFIG, reward_mode=mode,
                              action_scale=.00025, observation_mode='fetch_like_v1')
    records = []
    with ReachEnv(**config) as env:
        obs, info = env.reset(seed=2026, options={'goal_index':goal_index})
        goal = obs['desired_goal'].copy();initial = obs['achieved_goal'].copy()
        reset_state = np.r_[env.data.qpos, env.data.qvel, env.data.ctrl, env.data.time]
        for step in range(steps):
            action = np.array([direction if pulse_steps is None or step < pulse_steps else 0, 0, 0], np.float32)
            before = obs['achieved_goal'].copy();command = info['command_position_m'].copy()
            obs, reward, terminated, truncated, info = env.step(action)
            current = obs['achieved_goal'].copy()
            record = dict(step=step+1, action=action.tolist(), previous_achieved_goal=before,
                achieved_goal=current, desired_goal=goal, initial_distance_m=float(np.linalg.norm(goal-before)),
                current_distance_m=float(np.linalg.norm(goal-current)), actual_displacement_m=current-before,
                command_displacement_m=info['command_position_m']-command, reward=reward,
                is_success=info['is_success'], is_safety_failure=info['is_safety_failure'],
                is_action_rejected=bool(info.get('is_action_rejected',False)),
                termination_reason=info['termination_reason'], terminated=terminated, truncated=truncated,
                physics_steps=info['physics_steps'])
            if 'reward_components' in info:record['reward_components']=info['reward_components']
            if reward != env.compute_reward(current, goal, info):raise ValueError('Reward cannot be reproduced')
            records.append(native(record))
            if terminated or truncated:break
        rewards=np.array([r['reward'] for r in records])
        return dict(mode=mode, goal_index=goal_index, direction=direction, config=config,
            reset_state=reset_state.tolist(), initial_position_m=initial.tolist(), goal_position_m=goal.tolist(),
            records=records, steps=len(records), return_=float(rewards.sum()),
            discounted_return=float(rewards @ (PPO_CONFIG['gamma']**np.arange(len(rewards)))),
            success=info['is_success'], reason=info['termination_reason'])


def old_formula(env, previous_distance, current_distance, *, failure=False, rejected=False):
    """Independent scalar calculation, not a call to _shaped_reward."""
    success = current_distance <= env.success_threshold and not failure
    phi_previous = math.expm1(-previous_distance/env.reward_sigma_m)
    phi_next = 0. if success or failure else math.expm1(-current_distance/env.reward_sigma_m)
    terminal = 1.01/env.reward_gamma**(env.max_episode_steps-1)
    return env.reward_gamma*phi_next-phi_previous+terminal*(success-failure)-(1-env.reward_gamma)*rejected


def analytic_cases():
    cases = [('toward_0.25mm', .005, .00475, False, False),
             ('away_0.25mm', .005, .00525, False, False),
             ('stationary', .005, .005, False, False),
             ('near_threshold', .001251, .001001, False, False),
             ('success_boundary', .00125, .001, False, False),
             ('success_inside', .00125, .00075, False, False),
             ('safety_failure', .005, .005, True, False),
             ('timeout', .005, .005, False, False),
             ('rejected_hold', .005, .005, False, True)]
    rows=[]
    with ReachEnv(reward_mode='reach_shaped_v1') as env:
        if env.reward_gamma != PPO_CONFIG['gamma']:raise ValueError('Shaping gamma differs from PPO gamma')
        for name, previous, current, failure, rejected in cases:
            info=dict(previous_achieved_goal=np.array([previous,0,0]), is_safety_failure=failure,
                      is_action_rejected=rejected, is_success=False,
                      termination_reason='time_limit' if name=='timeout' else None)
            actual=env.compute_reward(np.array([current,0,0]), np.zeros(3), info)
            expected=old_formula(env,previous,current,failure=failure,rejected=rejected)
            if not math.isclose(actual,expected,rel_tol=0,abs_tol=1e-12):raise ValueError('Analytic mismatch')
            rows.append(dict(case=name, previous_distance_m=previous,current_distance_m=current,
                             reward=actual,analytic_reward=expected,absolute_error=abs(actual-expected)))
        parameters=dict(gamma=env.reward_gamma,sigma_m=env.reward_sigma_m,
                        terminal_value=env.shaped_terminal_value,horizon=env.max_episode_steps,
                        success_threshold_m=env.success_threshold)
    return dict(parameters=parameters,cases=rows)


def run(out, *, modes=OLD_MODES):
    out=Path(out)
    if out.exists():raise ValueError('Diagnostic output must be a new directory')
    out.mkdir(parents=True)
    traces=[]
    for mode in modes:
        for goal_index in (0,1):
            for direction in (1,-1,0):
                traces.append(dict(kind='single_step',**trace(mode,goal_index,direction,steps=1)))
                traces.append(dict(kind='short_pulse',**trace(mode,goal_index,direction,steps=25,pulse_steps=20)))
            for kind,direction,pulse in [('toward_episode',1 if goal_index==0 else -1,20),
                                          ('idle_timeout',0,None),('outward_failure',-1 if goal_index==0 else 1,None)]:
                traces.append(dict(kind=kind,**trace(mode,goal_index,direction,steps=200,pulse_steps=pulse)))
    reference=traces[0]['reset_state']
    if any(t['reset_state']!=reference for t in traces):raise ValueError('Direction tests did not start from identical Home')
    result=dict(schema=1,training_steps=0,action_scale_m=.00025,control_period_s=.05,
                old_analytic=analytic_cases(),traces=traces,
                environment_source_sha256=hashlib.sha256(Path(inspect.getfile(ReachEnv)).read_bytes()).hexdigest())
    (out/'diagnostics.json').write_text(json.dumps(native(result),indent=2,allow_nan=False)+'\n')
    for t in traces:
        if t['kind']=='single_step':
            r=t['records'][0];print(t['mode'],t['goal_index'],t['direction'],
                'distance_mm',r['initial_distance_m']*1000,r['current_distance_m']*1000,
                'actual_dx_mm',r['actual_displacement_m'][0]*1000,'reward',r['reward'])
        elif t['kind'] in ('toward_episode','idle_timeout','outward_failure'):
            print(t['mode'],t['goal_index'],t['kind'],'steps',t['steps'],'return',t['return_'],
                  'discounted',t['discounted_return'],'reason',t['reason'])
    return result


def compare_recorded(source, out):
    """Score the SAME frozen physical transitions; do not run new trajectories."""
    source=Path(source);out=Path(out)
    if out.exists():raise ValueError('Comparison output must be a new directory')
    document=json.loads(source.read_text());compared=[]
    with ReachEnv(reward_mode='reach_shaped_v1') as old, ReachEnv(reward_mode='reach_progress_v1') as new:
        for trace_ in document['traces']:
            if trace_['mode']!='reach_shaped_v1':continue
            records=[]
            for row in trace_['records']:
                info=dict(previous_achieved_goal=row['previous_achieved_goal'],
                          is_safety_failure=row['is_safety_failure'],is_action_rejected=row['is_action_rejected'])
                a=old.compute_reward(row['achieved_goal'],row['desired_goal'],info)
                terms=new.reward_components(row['achieved_goal'],row['desired_goal'],info)
                if a!=row['reward']:raise ValueError('Old reward changed from pre-implementation snapshot')
                records.append(dict(row, old_reward=a, new_reward=terms['total_reward'], new_components=terms))
            rewards=np.array([r['new_reward'] for r in records])
            compared.append(dict(kind=trace_['kind'],goal_index=trace_['goal_index'],direction=trace_['direction'],
                records=records,old_return=trace_['return_'],old_discounted_return=trace_['discounted_return'],
                new_return=float(rewards.sum()),new_discounted_return=float(rewards @ (.99**np.arange(len(rewards))))))
        analytic=[]
        for case in document['old_analytic']['cases']:
            name=case['case'];info=dict(previous_achieved_goal=[case['previous_distance_m'],0,0],
                is_safety_failure=name=='safety_failure',is_action_rejected=name=='rejected_hold')
            terms=new.reward_components([case['current_distance_m'],0,0],[0,0,0],info)
            analytic.append(dict(case, new_reward=terms['total_reward'],new_components=terms))
        def cycle(env):
            return [env.compute_reward([d,0,0],[0,0,0],dict(previous_achieved_goal=[p,0,0],
                is_safety_failure=False,is_action_rejected=False)) for p,d in ((.005,.00475),(.00475,.005))]
        default=cycle(new)
        with ReachEnv(reward_mode='reach_progress_v1',reward_progress_parameters=dict(w_distance=0.,step_cost=0.)) as pure:
            progress_only=cycle(pure)
        loopholes=dict(default_cycle_rewards=default,default_discounted_return=default[0]+.99*default[1],
            pure_progress_cycle_rewards=progress_only,pure_progress_discounted_return=progress_only[0]+.99*progress_only[1],
            note='Discounted progress differences do not generally cancel. Defaults penalize this tested cycle; this is not a global proof.')
    out.mkdir(parents=True)
    result=dict(schema=1,source=str(source.resolve()),physical_steps=0,training_steps=0,
                old_rewards_exactly_preserved=True,analytic=analytic,loopholes=loopholes,traces=compared)
    (out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def pick(kind,direction=1,goal_index=0):
        return next(t for t in compared if t['kind']==kind and t['direction']==direction and t['goal_index']==goal_index)
    records=pick('short_pulse')['records'];time=[r['step']*.05 for r in records]
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
    for key,label in [('old_reward','reach_shaped_v1'),('new_reward','reach_progress_v1')]:
        ax.plot(time,[r[key] for r in records],marker='o',label=label)
    ax.set(xlabel='Simulation time (s)',ylabel='Reward per env.step',title='Same +X trajectory: old and new rewards')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'reward_trace_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(8,6.5),layout='constrained',sharex=True)
    for key in ('progress_reward','distance_penalty','step_penalty'):
        axes[0].plot(time,[r['new_components'][key] for r in records],label=key)
    for key in ('success_bonus','safety_penalty','rejection_penalty'):
        axes[1].plot(time,[r['new_components'][key] for r in records],label=key)
    for ax in axes:ax.set(ylabel='Signed reward term');ax.legend();ax.grid(alpha=.2)
    axes[0].set_title('reach_progress_v1 components: same +X trajectory')
    axes[1].set_xlabel('Simulation time (s)');fig.savefig(out/'new_reward_components.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
    for direction,label in [(1,'Toward +X'),(-1,'Away -X'),(0,'Idle')]:
        r=pick('short_pulse',direction)['records']
        ax.plot([0]+[v['step']*.05 for v in r],[5]+[v['current_distance_m']*1000 for v in r],label=label)
    ax.axhline(1,color='gray',linestyle='--',label='Success threshold')
    ax.set(xlabel='Simulation time (s)',ylabel='Actual end-effector distance (mm)',title='Actual distance: independent identical Home resets')
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'distance_trace.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6.5),layout='constrained')
    for index in (0,1):
        rows=[pick('single_step',d,index)['records'][0] for d in (1,0,-1)]
        for column,key in enumerate(('old_reward','new_reward')):
            ax=axes[index,column];bars=ax.bar(['+X action','Zero action','-X action'],[r[key] for r in rows])
            ax.bar_label(bars,fmt='%.5f',padding=3,fontsize=9);ax.margins(y=.25)
            ax.set(ylabel='Reward per env.step',title=f"{'+' if index==0 else '-'}X 5mm goal: {key}")
    fig.savefig(out/'direction_reward_comparison.png',dpi=170);plt.close(fig)
    print('Recorded-transition comparison:',len(compared),'traces; no physical steps')
    for r in analytic:print(r['case'],'old',r['reward'],'new',r['new_reward'])
    print('Cycle audit:',loopholes)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--reward-modes',nargs='+',choices=REWARD_MODES,default=list(OLD_MODES))
    p.add_argument('--compare-from',type=Path,help='Score previously recorded physical transitions and export four diagnostic figures')
    args=p.parse_args()
    try:
        if args.compare_from:compare_recorded(args.compare_from,args.out)
        else:run(args.out,modes=args.reward_modes)
        return 0
    except (ValueError,RuntimeError,OSError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
