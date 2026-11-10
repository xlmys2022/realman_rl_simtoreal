#!/usr/bin/env python3
"""Controlled 10k boundary experiment, reusing the official SAC training entry.

Development uses 10000..10019. --final only evaluates frozen supplied models
on 100..199. No hyperparameter sweep or automatic extra training.
"""
import argparse
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from realman_rl_simtoreal.paths import OUTPUTS, PROJECT_ROOT
from realman_rl_simtoreal.experiments.sac_reach import evaluate,write_json
from realman_rl_simtoreal.experiments.sac_diagnostics import load_bundle,trace_policy,td_diagnostic


def development(model_path, seeds, folder, *, replay_path=None, prefix=None):
    model,transform,config=load_bundle(model_path)
    result=trace_policy('sac',seeds,model=model,transform=transform,config=config['environment'])
    write_json(folder/'trajectory.json',result)
    replay_path=Path(replay_path) if replay_path else Path(model_path).parent/'replay_buffer.pkl.gz'
    with gzip.open(replay_path,'rb') as file:model.load_replay_buffer(io.BytesIO(file.read()))
    if model.replay_buffer.full:
        raise ValueError('Wrapped replay cannot establish an original training prefix')
    count=model.replay_buffer.size() if prefix is None else prefix
    if count!=model.num_timesteps or count>model.replay_buffer.size():
        raise ValueError('Diagnostic replay prefix must match model training budget')
    rejpath=Path(model_path).parent.parent/'rejected_actions.jsonl'
    rejected=[json.loads(x)['step']-1 for x in rejpath.read_text().splitlines()] if rejpath.exists() else []
    summary,arrays=td_diagnostic(model,indices=np.arange(count),rejected_indices=rejected)
    summary['replay_source']=str(replay_path.resolve())
    summary['replay_prefix']=count
    write_json(folder/'td_summary.json',summary)
    np.savez_compressed(folder/'td_samples.npz',**arrays)
    return result['summary'],result['episodes'],config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',type=Path,default=OUTPUTS/'rl/sac_reach/formal_seed2026/checkpoints/step_010000/model.zip')
    p.add_argument('--a-replay',type=Path,default=OUTPUTS/'rl/sac_reach/formal_seed2026/final/replay_buffer.pkl.gz')
    p.add_argument('--b',type=Path,help='Existing frozen B model; omit to train B from scratch exactly 10000 steps')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--final',action='store_true')
    a=p.parse_args()
    if a.seed<0:p.error('Nonnegative seed required')
    if a.final and a.b is None:p.error('--final requires a frozen --b; it never trains/selects on final seeds')
    if a.out.exists() and any(a.out.iterdir()):p.error('Output directory must be empty')
    a.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(factor='task command boundary termination vs rejection+hold',training_budget=10000,
                  training_seed=a.seed,boundary_mode='reject_and_hold',rejection_penalty=1.,
                  development_seeds=list(range(10000,10020)),final_seeds=list(range(100,200)),
                  replication_rule='If B dev success improves >=20 percentage points AND mean distance decreases '
                                   'AND no collision/numerical/actual limit/drive/speed/orientation failure, '
                                   'run one independent 10000-step seed 2027; otherwise stop. No penalty tuning.',
                  a_model=str(a.a.resolve()),a_replay=str(a.a_replay.resolve()),
                  no_reward_return_ranking_across_modes=True)
    write_json(a.out/'protocol.json',protocol)
    model_b=a.b
    if model_b is None:
        # A20 is frozen before B training; the 30k circular buffer never wrapped
        # (capacity100k), so its first10k are exactly A10k's training transitions.
        sa,ea,ca=development(a.a,protocol['development_seeds'],a.out/'A',replay_path=a.a_replay,prefix=10000)
        print('A development',sa,flush=True)
        command=[sys.executable,'-I',str(PROJECT_ROOT/'scripts/train_sac_reach.py'),
                 '--run-dir',str(a.out/'training'),'--timesteps','10000','--seed',str(a.seed),
                 '--eval-every','5000','--eval-episodes','20','--eval-seed-start','10000',
                 '--boundary-mode','reject_and_hold','--rejection-penalty','1','--label','boundary_B']
        with (a.out/'training_console.log').open('w') as file:
            code=subprocess.run(command,stdout=file,stderr=subprocess.STDOUT).returncode
        if code:
            write_json(a.out/'status.json',dict(status='training_failed',exit_code=code))
            return 1
        model_b=a.out/'training/final/model.zip'
    elif not a.final:
        sa,ea,ca=development(a.a,protocol['development_seeds'],a.out/'A',replay_path=a.a_replay,prefix=10000)
    if a.final:
        seeds=protocol['final_seeds'];results={};configs={}
        for name,path in [('A',a.a),('B',model_b)]:
            model,transform,config=load_bundle(path)
            results[name]=evaluate('sac',seeds,model=model,transform=transform,config=config['environment'])
            configs[name]=config
            print(name,results[name]['summary'],flush=True)
        ea,eb=results['A']['episodes'],results['B']['episodes']
        ca,cb=configs['A'],configs['B']
        payload=dict(results=results)
    else:
        sb,eb,cb=development(model_b,protocol['development_seeds'],a.out/'B')
        print('B development',sb,flush=True)
        real_faults=sum(sb['termination_counts'].get(k,0) for k in ('collision','numerical_failure','joint_limit','velocity_limit','drive_failure','orientation_limit'))
        improve=bool(sb['success_rate']-sa['success_rate']>=.2-1e-12
                     and sb['mean_final_distance_m']<sa['mean_final_distance_m'] and real_faults==0)
        payload=dict(A=sa,B=sb,clear_improvement=improve,replication_required=improve)
    # Equality checks isolate the only allowed task-level mechanism change.
    if ca['sac']!=cb['sac'] or ca['observation_transform']!=cb['observation_transform']:
        raise ValueError('SAC hyperparameters or observation transform changed')
    b_env=dict(cb['environment']);mode=b_env.pop('boundary_mode',None);penalty=b_env.pop('rejection_penalty',None)
    if mode!='reject_and_hold' or penalty!=1. or b_env!=ca['environment']:
        raise ValueError('Unexpected environment factor change')
    if [e.get('goal_m',e.get('target_position_m')) for e in ea]!=[e.get('goal_m',e.get('target_position_m')) for e in eb]:
        raise ValueError('Different target sets')
    payload.update(a_model=str(a.a.resolve()),b_model=str(Path(model_b).resolve()),
                   paired_goals_identical=True,only_declared_factor_changed=True)
    write_json(a.out/'comparison.json',payload)
    return 0


if __name__=='__main__':raise SystemExit(main())
