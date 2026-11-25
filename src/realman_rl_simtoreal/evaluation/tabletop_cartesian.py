"""真值与视觉桌面抓取共用的评估循环，接收调用者提供的动作预测与状态指纹。"""
import numpy as np
from ..utils.artifacts import file_sha256
from ..envs.factory import make_tabletop_cartesian_env as TabletopEnv
from ..envs.grasp.tabletop_cartesian_env import action_contract
from ..envs.grasp.tabletop_cartesian_vision_env import VISION_CONTRACT
from ..tasks.grasp.tabletop_cartesian import CartesianTabletopConfig as Config
from ..simulation.runtime.observation_presets import startup_contract
from ..simulation.physics.grasp_metrics import CONTACT_MEASUREMENT
from ..tasks.grasp.tabletop_cartesian_reward import REWARD


class RolloutMetrics:
    """从真实仿真状态审计阶段结果。评估量独立于策略输入和奖励。"""

    def __init__(self, config):
        self.required_clearance = (config.lift_height if config.success_height is None
                                   else config.success_height) - config.height_tolerance
        self.reached = False
        self.lifted = False
        self.max_clearance = 0.
        self.slip = None
        self.grasp_anchor = None
        self.previous_action = None
        self.action_changes = []
        self.checks = dict(attempts=0, ik_failures=0, joint_limit_steps=0, collision_rejected_steps=0,
                           missing_vision_steps=0)

    def update(self, env, info, action):
        tcp_error = np.asarray(info['tcp_position_m']) - info['object_position_m']
        tcp_error[2] -= REWARD['grasp_offset_z_m']
        self.reached |= bool(np.linalg.norm(tcp_error) <= .006)
        bilateral = min(info['contact_forces_n'][:2]) > .005
        self.lifted |= bilateral and info['clearance_m'] >= self.required_clearance
        self.max_clearance = max(self.max_clearance, info['max_clearance_m'])
        row = env.robot.recorder.last
        if bilateral and row is not None:
            relative = np.asarray(row['object_relative_to_pads_m'])
            if self.grasp_anchor is None:
                self.grasp_anchor = relative
            self.slip = max(self.slip or 0., float(np.linalg.norm(relative - self.grasp_anchor)))
        elif self.grasp_anchor is not None:
            self.grasp_anchor = None
        if self.previous_action is not None:
            self.action_changes.append(float(np.linalg.norm(action - self.previous_action)))
        self.previous_action = action
        checks = env.controller.last_checks
        self.checks['attempts'] += checks.get('attempts', 0)
        self.checks['ik_failures'] += checks.get('ik_failures', 0)
        self.checks['joint_limit_steps'] += bool(checks.get('joint_limits', 0))
        self.checks['collision_rejected_steps'] += bool(checks.get('collision_rejections', 0))
        self.checks['missing_vision_steps'] += 'vision' in info and not info['vision']['detected']

    def finish(self, env, info):
        unexpected = {'robot_support', 'target_other', 'robot_internal_or_other'}
        contact = bool(env.robot.recorder._seen_categories & unexpected)
        contact |= any(str(info.get('detail', '')).startswith(kind + ':') for kind in unexpected)
        return dict(reach_success=bool(self.reached), lift_success=bool(self.lifted),
                    unexpected_collision=contact, max_clearance_m=float(self.max_clearance),
                    max_slip_while_bilateral_m=self.slip,
                    action_change_l2_sum=sum(self.action_changes),
                    action_change_count=len(self.action_changes), **self.checks)


def summarize_metrics(episodes, visual):
    """对同一组完整回合汇总阶段成功率、拒绝率与连续量。"""
    measures = [episode['metrics'] for episode in episodes]
    steps = sum(episode['length'] for episode in episodes)
    attempts = sum(row['attempts'] for row in measures)
    action_changes = sum(row['action_change_count'] for row in measures)
    slips = [row['max_slip_while_bilateral_m'] for row in measures if row['max_slip_while_bilateral_m'] is not None]
    return dict(
        reach_success_rate=float(np.mean([row['reach_success'] for row in measures])),
        lift_success_rate=float(np.mean([row['lift_success'] for row in measures])),
        mean_return=float(np.mean([episode['return_'] for episode in episodes])),
        mean_episode_steps=float(np.mean([episode['length'] for episode in episodes])),
        mean_success_steps=float(np.mean([e['length'] for e in episodes if e['success']]))
        if any(e['success'] for e in episodes) else None,
        collision_episode_rate=float(np.mean([row['unexpected_collision'] for row in measures])),
        collision_rejected_step_rate=sum(row['collision_rejected_steps'] for row in measures) / steps,
        ik_failure_rate=sum(row['ik_failures'] for row in measures) / attempts if attempts else None,
        joint_limit_step_rate=sum(row['joint_limit_steps'] for row in measures) / steps,
        action_mean_change_l2=sum(row['action_change_l2_sum'] for row in measures) / action_changes if action_changes else None,
        vision_lost_step_rate=sum(row['missing_vision_steps'] for row in measures) / steps if visual else None,
        vision_lost_episode_rate=sum(e['reason'] == 'vision_lost' for e in episodes) / len(episodes) if visual else None,
        mean_max_slip_m=float(np.mean(slips)) if slips else None,
        mean_max_clearance_m=float(np.mean([row['max_clearance_m'] for row in measures])),
        ik_attempts=attempts, measured_steps=steps,
    )


def rollout(env,predict,seed,tick=None,*,detailed_metrics=False,compact=False):
    """以 seed 重置环境并执行策略，记录原始动作、实际动作与完成结果。"""
    obs,initial=env.reset(seed=seed);records=[];total=0.
    metrics = RolloutMetrics(env.config) if detailed_metrics or compact else None
    if tick:tick(env.robot,'ppo')
    for step in range(1,env.config.max_steps+1):
        policy_observation=obs.copy()
        raw=np.asarray(predict(obs)).reshape(env.action_space.shape)
        action=np.clip(raw,-1,1).astype(np.float32)
        obs,reward,t,tr,info=env.step(action);total+=reward
        if not compact:
            records.append(dict(info,raw_action=raw.tolist(),action=action.tolist(),reward=reward))
            if env.config.observation_source=='rgbd_hsv':records[-1]['policy_observation']=policy_observation.tolist()
        if metrics is not None:
            metrics.update(env, info, action)
        if tick:tick(env.robot,'ppo')
        if t or tr:break
    report = dict(seed=seed,initial_object_position_m=initial['object_position_m'],
                success=info['is_success'],reason=info['termination_reason'],detail=info['detail'],
                length=step,return_=total,clearance_m=info['clearance_m'],hold_seconds=info['hold_seconds'],
                initial_startup=initial['startup'],gripper_statistics=env.robot.gripper.statistics(),records=records)
    if metrics is not None:
        report['metrics'] = metrics.finish(env, info)
    return report


def evaluate(predict,config,seeds,*,action_shape,full=False,tick=None,environment=None,
             observation_pose=None,object_radius=None,state_fingerprint=None,sources=None,algorithm='callable',
             scene_profile=None,hand_eye_file=None,detailed_metrics=False,compact=False):
    before=state_fingerprint() if state_fingerprint is not None else None
    if config.get('schema',1)<2 or 'startup_contract' not in config:
        raise ValueError('Evaluation requires the fixed observation pose startup contract')
    if config.get('contact_measurement')!=CONTACT_MEASUREMENT:
        raise ValueError('Evaluation requires the current inward grip-force observation contract')
    saved = Config(**config['environment'])
    if config['startup_contract']!=startup_contract(saved.observation_pose, saved.scene_profile):
        raise ValueError('Checkpoint fixed observation pose contract does not match the configured preset')
    if environment is not None and any((full, observation_pose is not None, object_radius is not None,
                                         scene_profile is not None, hand_eye_file is not None)):
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
    target = Config(**values)
    if scene_profile is not None or hand_eye_file is not None:
        target = target.with_scene(scene_profile or target.scene_profile, hand_eye_file=hand_eye_file)
    if tuple(action_shape)!=(action_contract(target)['shape'],):
        raise ValueError('The checkpoint cannot control this observation pose; use pose control to turn from a forward start')
    own=environment is None
    env=TabletopEnv(target,startup_tick=tick) if own else environment
    try:
        episodes=[rollout(env,predict,int(seed),tick=tick,detailed_metrics=detailed_metrics,compact=compact)
                  for seed in seeds]
    finally:
        if own:env.close()
    if state_fingerprint is not None:
        assert state_fingerprint()==before, 'Evaluation changed the supplied policy state'
    result = dict(task='tabletop_cartesian',algorithm=algorithm,full_task=full,environment=env.config.to_dict(),
                observation_source=env.config.observation_source,
                vision_contract=VISION_CONTRACT if env.config.observation_source=='rgbd_hsv' else None,
                startup_contract=startup_contract(env.config.observation_pose, env.config.scene_profile),
                parameters_unchanged=True if state_fingerprint is not None else None,scripted_task_actions=False,ik_joint_targets_in_policy=False,
                summary=dict(episodes=len(episodes),successes=sum(e['success'] for e in episodes),
                success_rate=float(np.mean([e['success'] for e in episodes])),
                safety_failures=sum(e['reason'] in ('safety_failure','numerical_failure') for e in episodes)),
                episodes=episodes,sources=sources() if sources is not None else {})
    if detailed_metrics or compact:
        result['summary'].update(summarize_metrics(episodes, env.config.observation_source == 'rgbd_hsv'))
        result.update(scene_profile=env.config.scene_profile, scene_sha256=file_sha256(env.robot.scene_path),
                      normalization='fixed_si_scaling_no_running_statistics')
    return result
