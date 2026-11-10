"""Algorithm-independent Reach defaults and task distributions."""


ENV_CONFIG = dict(target_mode='random', reward_mode='safe_dense', action_scale=.001,
                  control_period=.05, max_episode_steps=200, success_threshold=.001,
                  gravity_compensation=True)


TASKS = {
    'fixed_reach': dict(target_mode='fixed',target_radius=.005),
    'easy_random_reach': dict(target_mode='random',target_radius=.003),
    'random_reach': dict(target_mode='random',target_radius=.005),
}


def task_environment(task, base, *, reward_mode=None, action_scale=None):
    if task not in TASKS:
        raise ValueError('Unknown pure RL task: '+str(task))
    result=dict(base,**TASKS[task])
    if reward_mode is not None:result['reward_mode']=reward_mode
    if action_scale is not None:result['action_scale']=action_scale
    if result['reward_mode']=='reach_shaped_v1':
        result.setdefault('reward_sigma_m',.005)
        result.setdefault('reward_gamma',.99)
    return result
