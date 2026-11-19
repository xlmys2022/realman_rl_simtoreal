"""Algorithm-independent Cartesian tabletop episodes and independent evaluation.

``predict(observation)`` returns deterministic, unclipped actions. Callers own
checkpoint loading and can supply a fingerprint of weights, optimizer and counters.
An external environment remains caller-owned; standalone evaluation simulates startup.
"""
import numpy as np
from ..envs.factory import make_tabletop_cartesian_env as TabletopEnv
from ..envs.tabletop_cartesian_env import action_contract
from ..envs.tabletop_cartesian_vision_env import VISION_CONTRACT
from ..tasks.tabletop_cartesian import CartesianTabletopConfig as Config
from ..simulation.observation_profiles import startup_contract
from ..simulation.grasp_metrics import CONTACT_MEASUREMENT


def rollout(env,predict,seed,tick=None):
    obs,initial=env.reset(seed=seed);records=[];total=0.
    if tick:tick(env.robot,'ppo')
    for _ in range(env.config.max_steps):
        policy_observation=obs.copy()
        raw=np.asarray(predict(obs)).reshape(env.action_space.shape)
        action=np.clip(raw,-1,1).astype(np.float32)
        obs,reward,t,tr,info=env.step(action);total+=reward
        records.append(dict(info,raw_action=raw.tolist(),action=action.tolist(),reward=reward))
        if env.config.observation_source=='rgbd_hsv':records[-1]['policy_observation']=policy_observation.tolist()
        if tick:tick(env.robot,'ppo')
        if t or tr:break
    return dict(seed=seed,initial_object_position_m=initial['object_position_m'],
                success=info['is_success'],reason=info['termination_reason'],detail=info['detail'],
                length=len(records),return_=total,clearance_m=info['clearance_m'],hold_seconds=info['hold_seconds'],
                initial_startup=initial['startup'],gripper_statistics=env.robot.gripper.statistics(),records=records)


def evaluate(predict,config,seeds,*,action_shape,full=False,tick=None,environment=None,
             observation_pose=None,object_radius=None,state_fingerprint=None,sources=None,algorithm='callable'):
    before=state_fingerprint() if state_fingerprint is not None else None
    if config.get('schema',1)<2 or 'startup_contract' not in config:
        raise ValueError('Evaluation requires the fixed observation pose startup contract')
    if config.get('contact_measurement')!=CONTACT_MEASUREMENT:
        raise ValueError('Evaluation requires the current inward grip-force observation contract')
    if config['startup_contract']!=startup_contract(Config(**config['environment']).observation_pose):
        raise ValueError('Checkpoint fixed observation pose contract does not match the configured preset')
    if environment is not None and (full or observation_pose is not None or object_radius is not None):
        raise ValueError('Full or overridden-pose evaluation must construct its own physical-startup environment')
    values=dict(config['environment']);values['cache_startup']=False
    if observation_pose is not None:values['observation_pose']=observation_pose
    if full:
        default=Config()
        for key in ('initial_width','success_height','lift_height','height_tolerance','hold_seconds','max_steps','object_radius','axis_scale'):
            values[key]=getattr(default,key)
    # Explicit evaluation randomization follows the full-task defaults. Otherwise
    # full=True silently restores radius zero and repeats the training position.
    if object_radius is not None:values['object_radius']=object_radius
    if tuple(action_shape)!=(action_contract(Config(**values))['shape'],):
        raise ValueError('The checkpoint cannot control this observation pose; use pose control to turn from a forward start')
    own=environment is None
    env=TabletopEnv(Config(**values),startup_tick=tick) if own else environment
    try:episodes=[rollout(env,predict,int(seed),tick=tick) for seed in seeds]
    finally:
        if own:env.close()
    if state_fingerprint is not None:
        assert state_fingerprint()==before, 'Evaluation changed the supplied policy state'
    return dict(task='tabletop_cartesian',algorithm=algorithm,full_task=full,environment=env.config.to_dict(),
                observation_source=env.config.observation_source,
                vision_contract=VISION_CONTRACT if env.config.observation_source=='rgbd_hsv' else None,
                startup_contract=startup_contract(env.config.observation_pose),
                parameters_unchanged=True if state_fingerprint is not None else None,scripted_task_actions=False,ik_joint_targets_in_policy=False,
                summary=dict(episodes=len(episodes),successes=sum(e['success'] for e in episodes),
                success_rate=float(np.mean([e['success'] for e in episodes])),
                safety_failures=sum(e['reason'] in ('safety_failure','numerical_failure') for e in episodes)),
                episodes=episodes,sources=sources() if sources is not None else {})
