"""Opt-in existing-model trajectory regression; never trains or modifies outputs.

Run in the recorded realman_rl CPU environment. All four cases take several minutes.
The pre-refactor fingerprints freeze every physical tick and every policy step.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
from dataclasses import replace
from realman_rl_simtoreal.paths import PROJECT_ROOT as ROOT
from realman_rl_simtoreal.rl import ppo_tabletop_cartesian as rl
from realman_rl_simtoreal.rl.common.artifacts import optimizer_state_hash
from realman_rl_simtoreal.envs.factory import make_tabletop_cartesian_env
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig as Config


def array_digest(value):
    return hashlib.sha256(str(value.dtype).encode()+str(tuple(value.shape)).encode()+value.tobytes()).hexdigest()


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def result_digest(result,frames):
    result=dict(result);result.pop('sources')  # Current-source hashes necessarily change during refactoring.
    payload=json.loads(json.dumps(dict(result=result,frames=frames),default=plain,allow_nan=False))
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def run_case(case,contract,checkpoints):
    path=ROOT/checkpoints[contract['checkpoint']]['path']
    if not path.is_file():raise FileNotFoundError('Required historical checkpoint is unavailable: '+str(path))
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if actual!=checkpoints[contract['checkpoint']]['file_sha256']:raise AssertionError('Checkpoint bytes changed: '+str(path))
    model,config=rl.load(path)
    cfg=replace(Config(**config['environment']),cache_startup=False,object_radius=contract['object_radius'])
    def policy_state():return (rl.hashes(model),model.num_timesteps,model._n_updates,optimizer_state_hash(model))
    before=policy_state();env=make_tabletop_cartesian_env(cfg)
    physics=[];states=[];observations=[];frames=[];original=mujoco.mj_step
    def advance(m,d,*args,**kwargs):
        result=original(m,d,*args,**kwargs)
        if m is env.robot.model:
            physics.append(np.r_[d.time,d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),d.qfrc_actuator.copy()])
        return result
    step=env.step
    def observed(action):
        result=step(action);obs,reward,terminated,truncated,info=result;robot=env.robot
        position,rotation=robot.tcp_pose()
        states.append(np.r_[robot.data.time,robot.data.qpos.copy(),robot.data.qvel.copy(),position,rotation.ravel(),
            robot.gripper.get_width(),robot.forces(),info['ik_fraction'],reward])
        observations.append(obs.copy())
        if hasattr(env,'camera') and env.camera.frame is not None:
            frame=env.camera.frame
            frames.append(dict(timestamp=frame.timestamp,rgb=hashlib.sha256(frame.rgb.tobytes()).hexdigest(),
                depth=hashlib.sha256(frame.depth_m.tobytes()).hexdigest()))
        return result
    env.step=observed;mujoco.mj_step=advance
    try:
        result=rl.evaluate(model,dict(config,environment=cfg.to_dict()),[contract['seed']],environment=env)
    finally:
        mujoco.mj_step=original;env.close()
    if policy_state()!=before:raise AssertionError('Model, optimizer or training counters changed')
    episode=result['episodes'][0]
    arrays=dict(physics=np.array(physics),states=np.array(states),observations=np.array(observations),
        raw_actions=np.array([r['raw_action'] for r in episode['records']]),
        actions=np.array([r['action'] for r in episode['records']]))
    for name,value in arrays.items():
        if list(value.shape)!=contract['arrays'][name]['shape']:raise AssertionError(case+'/'+name+' shape changed')
        if array_digest(value)!=contract['arrays'][name]['sha256']:raise AssertionError(case+'/'+name+' values changed')
    if result_digest(result,frames)!=contract['result_sha256']:raise AssertionError(case+' JSON, contacts, rewards or camera frames changed')
    return dict(case=case,seed=contract['seed'],policy_steps=episode['length'],physics_steps=len(physics),
        success=episode['success'],reason=episode['reason'],exact_array_and_json_match=True,
        model_optimizer_counters_unchanged=True)


def main():
    fixture=json.loads((ROOT/'tests/fixtures/refactor_contracts.json').read_text())
    contracts=fixture['trajectory_cases'];parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases',nargs='+',choices=list(contracts),default=list(contracts))
    parser.add_argument('--json',type=Path,help='Optional new report path; existing files are refused')
    args=parser.parse_args()
    if args.json is not None and args.json.exists():parser.error('Choose a new report path')
    rows=[]
    for case in args.cases:
        row=run_case(case,contracts[case],fixture['checkpoint_cases']);rows.append(row);print(json.dumps(row),flush=True)
    if args.json is not None:
        args.json.parent.mkdir(parents=True,exist_ok=True)
        args.json.write_text(json.dumps(rows,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
