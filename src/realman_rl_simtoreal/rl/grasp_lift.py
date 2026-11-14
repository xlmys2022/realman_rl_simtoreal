"""Evaluation for four-action grasp/lift controllers and PPO policies."""
from collections import Counter
import hashlib
from importlib.metadata import version
from time import perf_counter

import numpy as np

from ..envs.grasp_lift_env import GraspLiftEnv
from ..simulation.grasp_controller import GraspLiftController
from ..paths import PROJECT_ROOT, GRASP_CONTACT_SCENE


def experiment_metadata():
    """Fingerprint the actual combined robot, gripper and contact task."""
    assets={GRASP_CONTACT_SCENE}
    for folder in ('assets/robots/rml63','assets/grippers/eg2_4c2'):
        assets.update(p for p in (PROJECT_ROOT/folder).rglob('*') if p.is_file())
    sources=[PROJECT_ROOT/'src/realman_rl_simtoreal'/p for p in (
        'envs/grasp_lift_env.py','tasks/grasp_lift.py','rl/grasp_lift.py',
        'simulation/grasp_controller.py','simulation/gripper.py','simulation/grasp_contact.py',
        'simulation/control.py','simulation/kinematics.py','simulation/model.py')]
    def hashes(paths):
        return {str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    return dict(assets=hashes(assets),sources=hashes(sources),
                runtime={name:version(name) for name in ('mujoco','numpy','gymnasium')},
                action_semantics=['world_dx','world_dy','world_dz','absolute_gripper_opening'],
                observation_dim=41,action_dim=4)


def rollout(env,seed,*,model=None,close=True,lift=True,tick=None):
    observation,info=env.reset(seed=int(seed))
    controller=GraspLiftController(info,action_scale=env.action_scale,control_period=env.control_period,
                                   close=close,lift=lift) if model is None else None
    records=[]
    total_reward=0.
    started=perf_counter()
    for _ in range(env.max_episode_steps):
        action=controller.predict(info) if model is None else model.predict(observation,deterministic=True)[0]
        env.recorder.phase=controller.phase if controller else 'ppo'
        observation,reward,terminated,truncated,info=env.step(action)
        total_reward+=reward
        records.append(dict(info,action=np.asarray(action).tolist(),reward=reward,
                            controller_phase=controller.phase if controller else 'ppo',
                            controller_centering_offset_m=controller.centering_offset.tolist() if controller else None))
        if tick is not None:tick(env,info)
        if terminated or truncated:break
    else:raise RuntimeError('Grasp episode exceeded its declared limit')
    try:
        statistics=env.gripper.statistics()
        statistics_error=None
    except (ValueError,RuntimeError) as exc:
        statistics=None
        statistics_error=str(exc)
    result=dict(seed=int(seed),success=info['is_success'],reason=info['termination_reason'],
        detail=info['detail'],is_safety_failure=info['is_safety_failure'],length=len(records),return_=total_reward,
        terminated=bool(terminated),truncated=bool(truncated),observation_valid=info['observation_valid'],
        final_support_clearance_m=info['support_clearance_m'],world_z_lift_m=info['world_z_lift_m'],
        hold_time_s=info['hold_time_s'],max_hold_time_s=info['max_hold_time_s'],
        actual_width_m=info['actual_width_m'],contact_forces_n=info['contact_forces_n'],
        initial_object_position_m=info['initial_object_position_m'],object_position_m=info['object_position_m'],
        gripper_statistics=statistics,gripper_statistics_error=statistics_error,wall_s=perf_counter()-started,records=records)
    if env.config['record_physics']:result['physics_records']=env.recorder.rows
    return result


def summarize(episodes):
    successful=[e for e in episodes if e['success']]
    return dict(episodes=len(episodes),successes=len(successful),success_rate=len(successful)/len(episodes) if episodes else None,
        termination_counts=dict(Counter(e['reason'] for e in episodes)),
        safety_failures=sum(e['is_safety_failure'] for e in episodes),
        mean_success_clearance_m=float(np.mean([e['final_support_clearance_m'] for e in successful])) if successful else None,
        mean_success_world_z_lift_m=float(np.mean([e['world_z_lift_m'] for e in successful])) if successful else None,
        mean_success_steps=float(np.mean([e['length'] for e in successful])) if successful else None)


def evaluate(config,seeds,*,model=None,close=True,lift=True,tick=None):
    with GraspLiftEnv(**config) as env:
        episodes=[rollout(env,seed,model=model,close=close,lift=lift,tick=tick) for seed in seeds]
        return dict(schema=1,task='grasp_lift',policy='staged_controller' if model is None else 'ppo',
                    environment=env.config,seeds=[e['seed'] for e in episodes],summary=summarize(episodes),episodes=episodes,
                    **experiment_metadata(),
                    success_criterion='Bilateral pad forces > 5mN, no support force, target clearance and low velocity/slip for the full hold duration')
