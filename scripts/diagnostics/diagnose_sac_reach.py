#!/usr/bin/env python3
"""Read-only SAC trajectory/soft-TD diagnosis, never overwrite a previous report."""
import argparse
from pathlib import Path
import numpy as np
from realman_rl_simtoreal.paths import OUTPUTS
from realman_rl_simtoreal.rl.common.artifacts import write_json
from realman_rl_simtoreal.rl.sac_diagnostics import load_bundle, trace_policy, td_diagnostic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=OUTPUTS/'rl/sac_reach/formal_seed2026/final/model.zip')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=20)
    p.add_argument('--seed-start',type=int,default=10000)
    p.add_argument('--skip-td',action='store_true')
    a=p.parse_args()
    if a.episodes<1 or a.seed_start<0:p.error('Invalid seeds/count')
    if a.out.exists() and any(a.out.iterdir()):p.error('Output directory must be empty')
    a.out.mkdir(parents=True,exist_ok=True)
    model,transform,config=load_bundle(a.model,replay=not a.skip_td)
    seeds=list(range(a.seed_start,a.seed_start+a.episodes))
    summaries={}
    for name in ('sac','rule'):
        result=trace_policy(name,seeds,model=model,transform=transform,config=config['environment'])
        write_json(a.out/(name+'_trajectory.json'),result)
        summaries[name]=result['summary']
        print(name,result['summary'],flush=True)
    if not a.skip_td:
        summary,arrays=td_diagnostic(model)
        write_json(a.out/'td_summary.json',summary)
        np.savez_compressed(a.out/'td_samples.npz',**arrays)
        summaries['td']=summary
        print('TD',summary,flush=True)
    rewards={}
    for name in ('idle','rule','outward'):
        result=trace_policy(name,[seeds[0]],config=config['environment'])
        write_json(a.out/(name+'_reward_trace.json'),result)
        rewards[name]=result['episodes']
    write_json(a.out/'summary.json',dict(model=str(a.model.resolve()),environment=config['environment'],
                                      seeds=seeds,diagnostics=summaries,reward_trajectories=rewards))
    return 0


if __name__=='__main__':raise SystemExit(main())
