#!/usr/bin/env python3
"""Approach, close, lift and hold a dynamic cube through four-action controls."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import time

from realman_rl_simtoreal.viewer import configure_backend, passive_viewer
configure_backend()

from realman_rl_simtoreal.envs.grasp_lift_env import GraspLiftEnv
from realman_rl_simtoreal.paths import OUTPUTS
from realman_rl_simtoreal.rl.common.artifacts import write_json
from realman_rl_simtoreal.rl.grasp_lift import rollout,summarize,experiment_metadata
from realman_rl_simtoreal.tasks.grasp_lift import GRASP_LIFT_CONFIG


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--seed-start',type=int,default=2027)
    p.add_argument('--mass',type=float,default=.01,help='Cube mass in kilograms')
    p.add_argument('--friction',type=float,default=.7)
    p.add_argument('--object-radius',type=float,default=.002,help='Seeded in-plane object/support displacement, metres')
    p.add_argument('--lift-height',type=float,default=.03,help='Lift along the support normal, metres')
    p.add_argument('--no-close',action='store_true',help='Negative control; an unsuccessful grasp exits with status 1')
    p.add_argument('--no-lift',action='store_true',help='Negative control; an unsuccessful lift exits with status 1')
    p.add_argument('--record-physics',action='store_true')
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--json',type=Path,default=OUTPUTS/'grasp_lift/evaluation.json')
    a=p.parse_args()
    if a.episodes<1 or a.seed_start<0:p.error('Positive episodes and nonnegative seeds required')
    if a.viewer and a.episodes!=1:p.error('Viewer requires exactly one episode')
    if a.json.exists():p.error('Output exists; select a new --json path')
    config=dict(GRASP_LIFT_CONFIG,mass=a.mass,friction=a.friction,object_radius=a.object_radius,
                lift_height=a.lift_height,record_physics=a.record_physics)
    episodes=[]
    try:
        with GraspLiftEnv(**config) as env:
            window_context=nullcontext(None)
            if a.viewer:
                window_context=passive_viewer(env.model,env.data)
            with window_context as window:
                def tick(current,info):
                    if window is not None:
                        if not window.is_running():raise KeyboardInterrupt()
                        window.sync()
                        time.sleep(max(0,current.data.time-(time.perf_counter()-wall_start)))
                if window is not None:
                    window.cam.lookat[:]=env.data.xpos[env.object_body]
                    window.cam.distance=.45
                    window.cam.azimuth,window.cam.elevation=145,-15
                for seed in range(a.seed_start,a.seed_start+a.episodes):
                    wall_start=time.perf_counter()
                    result=rollout(env,seed,close=not a.no_close,lift=not a.no_lift,tick=tick)
                    episodes.append(result)
                    print(f"seed={seed} success={result['success']} clearance_mm={1000*result['final_support_clearance_m']:.3f} "
                          f"world_z_mm={1000*result['world_z_lift_m']:.3f} hold_s={result['hold_time_s']:.3f} "
                          f"reason={result['reason']} detail={result['detail']}",flush=True)
        document=dict(schema=1,task='grasp_lift',policy='staged_controller',environment=config,
                      controls=dict(close=not a.no_close,lift=not a.no_lift),summary=summarize(episodes),episodes=episodes,
                      **experiment_metadata())
        write_json(a.json,document)
        print(document['summary']);print('Saved',a.json)
        return int(any(not e['success'] for e in episodes))
    except (ValueError,RuntimeError,KeyError) as exc:p.exit(1,str(exc)+'\n')
    except KeyboardInterrupt:p.exit(130,'Grasp demonstration interrupted before completion\n')


if __name__=='__main__':raise SystemExit(main())
