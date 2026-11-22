"""运行局部夹取和抬升回合，生成独立的物理验收与实验来源记录。"""
from time import perf_counter
import numpy as np

from ..envs.grasp.grasp_lift_env import GraspLiftEnv
from ..simulation.motion.staged_grasp_controller import GraspLiftController
from ..paths import GRASP_CONTACT_SCENE, PROJECT_ROOT
from ..utils.artifacts import dependency_versions, file_hashes
from .metrics import summarize_grasp_episodes as summarize


def experiment_metadata():
    """记录局部抓取场景、资源、核心源码摘要及 SI 动作约定。"""
    assets={GRASP_CONTACT_SCENE}
    for folder in ('assets/robots/rml63','assets/grippers/eg2_4c2'):
        assets.update(p for p in (PROJECT_ROOT/folder).rglob('*') if p.is_file())
    sources=[PROJECT_ROOT/'src/realman_rl_simtoreal'/p for p in (
        'envs/grasp/grasp_lift_env.py','tasks/grasp/lift.py','evaluation/grasp_lift.py','utils/artifacts.py','evaluation/metrics.py',
        'simulation/motion/staged_grasp_controller.py','simulation/devices/mujoco_gripper.py','simulation/physics/grasp_contact.py',
        'simulation/physics/joint_control.py','simulation/motion/kinematics.py','simulation/physics/model.py')]
    return dict(assets=file_hashes(sorted(assets),root=PROJECT_ROOT),
                sources=file_hashes(sorted(sources),root=PROJECT_ROOT),
                runtime=dependency_versions(('mujoco','numpy','gymnasium')),
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


def evaluate(config,seeds,*,model=None,close=True,lift=True,tick=None):
    """按种子执行局部抓取回合并汇总结果，支持调用者提供确定性策略。"""
    with GraspLiftEnv(**config) as env:
        episodes=[rollout(env,seed,model=model,close=close,lift=lift,tick=tick) for seed in seeds]
        return dict(schema=1,task='grasp_lift',policy='staged_controller' if model is None else 'ppo',
                    environment=env.config,seeds=[e['seed'] for e in episodes],summary=summarize(episodes),episodes=episodes,
                    **experiment_metadata(),
                    success_criterion='Bilateral pad forces > 5mN, no support force, target clearance and low velocity/slip for the full hold duration')
