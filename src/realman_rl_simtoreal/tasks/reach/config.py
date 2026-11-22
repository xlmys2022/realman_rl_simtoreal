"""定义 Reach 目标分布、SI 动作与时间尺度，以及可保存的环境参数。"""
from math import radians

from .reward import progress_parameters


OBSERVATION_MODES = ('legacy', 'fetch_like_v1', 'goal_relative_v1', 'goal_relative_v2', 'goal_relative_v3')
ENV_CONFIG = dict(target_mode='random', reward_mode='safe_dense', action_scale=.001,
                  control_period=.05, max_episode_steps=200, success_threshold=.001,
                  gravity_compensation=True)


TASKS = {
    'fixed_reach': dict(target_mode='fixed',target_radius=.005),
    'easy_random_reach': dict(target_mode='random',target_radius=.003),
    'random_reach': dict(target_mode='random',target_radius=.005),
    'random_reach_20cm': dict(target_mode='random', target_radius=.2, target_min_radius=.005,
        workspace_radius=.22, max_home_joint_delta=radians(360), synchronized_joint_rate=True,
        max_orientation_error=radians(5),
        cartesian_rate_limit=True,
        max_episode_steps=600, action_scale=.001,
        reward_progress_parameters=dict(progress_scale=.005, distance_scale=.2,
            w_distance=.2, success_bonus=20., safety_penalty=50., rejection_cost=1.)),
    'axis_pair_reach': dict(target_mode='discrete',target_radius=.005,
                           target_offsets=((.005,0.,0.),(-.005,0.,0.))),
    'axis_six_reach': dict(target_mode='discrete',target_radius=.005,
                          target_offsets=((.005,0.,0.),(-.005,0.,0.),(0.,.005,0.),
                                          (0.,-.005,0.),(0.,0.,.005),(0.,0.,-.005))),
}

TASKS['random_start_multi_reach_20cm']=dict(TASKS['random_reach_20cm'],start_mode='random',
    goals_per_episode=5,min_goal_distance=.01,max_episode_steps=1000,control_ik_iterations=600,
    track_issued_command=True,control_ik_damping=.001,control_ik_seed_weight=1e-6,
    reward_progress_parameters=dict(TASKS['random_reach_20cm']['reward_progress_parameters'],distance_scale=.4))


def task_environment(task, base, *, reward_mode=None, action_scale=None, observation_mode=None,
                     reward_progress_parameters=None):
    """合成任务默认值与显式覆盖，保留不同观测与奖励版本的配置契约。"""
    if task not in TASKS:
        raise ValueError('Unknown pure RL task: '+str(task))
    result=dict(base,**TASKS[task])
    if result['target_mode']!='discrete':result.pop('target_offsets',None)
    if reward_mode is not None:result['reward_mode']=reward_mode
    if action_scale is not None:result['action_scale']=action_scale
    if observation_mode is not None:
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError('Unknown observation_mode: '+str(observation_mode))
        result['observation_mode']=observation_mode
    if result['reward_mode']=='reach_shaped_v1':
        result.setdefault('reward_sigma_m',.005)
        result.setdefault('reward_gamma',.99)
    if result['reward_mode']=='reach_progress_v1':
        result['reward_progress_parameters']=progress_parameters(
            result.get('reward_progress_parameters') if reward_progress_parameters is None else reward_progress_parameters)
    elif reward_progress_parameters is not None:
        raise ValueError('reward_progress_parameters requires reach_progress_v1')
    else:
        result.pop('reward_progress_parameters', None)
    return result


def ppo_environment_config(task='fixed_reach',reward_mode=None,action_scale=None,
                       observation_mode=None, reward_progress_parameters=None):
    """选择现有 PPO Reach 任务的默认观测、奖励和动作尺度。"""
    if observation_mode is None:
        observation_mode = ('goal_relative_v3' if TASKS.get(task,{}).get('target_radius',0)>.01 else
                            'legacy' if task == 'fixed_reach' else 'goal_relative_v1')
    if observation_mode not in OBSERVATION_MODES:
        raise ValueError('Unknown observation_mode: '+str(observation_mode))
    if reward_mode is None:
        reward_mode = 'reach_progress_v1' if observation_mode.startswith('goal_relative_') else 'reach_shaped_v1'
    if action_scale is None:
        action_scale = TASKS.get(task,{}).get('action_scale',.00025)
    return task_environment(task,dict(ENV_CONFIG,boundary_mode='terminate'),
                            reward_mode=reward_mode,action_scale=action_scale,
                            observation_mode=None if observation_mode=='legacy' else observation_mode,
                            reward_progress_parameters=reward_progress_parameters)
