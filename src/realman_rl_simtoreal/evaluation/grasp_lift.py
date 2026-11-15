"""Independent four-action grasp episodes for caller-supplied policies or a controller."""
from time import perf_counter
import numpy as np

from ..envs.grasp_lift_env import GraspLiftEnv
from ..simulation.grasp_controller import GraspLiftController
from ..utils.artifacts import experiment_metadata
from .metrics import summarize_grasp_episodes as summarize


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


def evaluate(config,seeds,*,model=None,close=True,lift=True,tick=None):
    with GraspLiftEnv(**config) as env:
        episodes=[rollout(env,seed,model=model,close=close,lift=lift,tick=tick) for seed in seeds]
        return dict(schema=1,task='grasp_lift',policy='staged_controller' if model is None else 'ppo',
                    environment=env.config,seeds=[e['seed'] for e in episodes],summary=summarize(episodes),episodes=episodes,
                    **experiment_metadata(),
                    success_criterion='Bilateral pad forces > 5mN, no support force, target clearance and low velocity/slip for the full hold duration')
