"""Local, contact-driven pick and lift with the EG2-4C2 gripper."""
import numpy as np

GRASP_LIFT_CONFIG = dict(
    mass=.01, friction=.7, object_radius=.002, approach_distance=.02,initial_width=.06,
    lift_height=.03, height_tolerance=.002, hold_seconds=2.,
    action_scale=.001, control_period=.05, physics_timestep=.001, max_episode_steps=700,
    workspace_radius=.06, max_home_joint_delta=.4363323129985824,
    gripper_reference_speed=.5, record_physics=False,
)


# Historical training reward and curriculum contracts; numerical definitions are unchanged.
REWARD=dict(name='grasp_goal_dense_v5',gamma=.995,reach_scale_m=.01,
    width_scale_m=.025,cube_width_m=.02,reach_weight=2.,aperture_weight=1.,
    grip_weight=2.,height_weight=8.,hold_weight=2.,step_cost=.01,
    success_bonus=1000.,safety_penalty=25.,drop_penalty=5.,dense_reach=.2,
    dense_aperture=.3,dense_grip=.1,dense_height=1.,dense_hold=.2,speed_cost=.1,
    goal_lateral_scale_m=.005,lateral_potential_weight=2.,dense_lateral=.5,
    pregrasp_width_m=.06,close_distance_m=.005,open_distance_m=.015)

CURRICULUM=dict(
    pinch=dict(approach_distance=0.,initial_width=.022,lift_height=.005,hold_seconds=.2,max_episode_steps=300),
    easy=dict(approach_distance=.005,initial_width=.04,lift_height=.005,hold_seconds=.2,max_episode_steps=300),
    precision=dict(approach_distance=.005,initial_width=.04,lift_height=.005,height_tolerance=.005/15,
                   hold_seconds=.2,max_episode_steps=500),
    lift=dict(approach_distance=.005,initial_width=.04,lift_height=.015,height_tolerance=.001,
              hold_seconds=.5,max_episode_steps=500),
    lift_full=dict(approach_distance=.005,initial_width=.04,lift_height=.03,hold_seconds=2.,max_episode_steps=700),
    approach=dict(approach_distance=.01,initial_width=.06,lift_height=.03,hold_seconds=2.,max_episode_steps=700),
    approach_mid=dict(approach_distance=.015,initial_width=.06,lift_height=.03,hold_seconds=2.,max_episode_steps=700),
    approach_easy=dict(approach_distance=.02,initial_width=.06,lift_height=.005,height_tolerance=.005/15,
                       hold_seconds=.2,max_episode_steps=500),
    medium=dict(approach_distance=.01,lift_height=.015,hold_seconds=.5,max_episode_steps=500),
    full=dict())

def goal_alignment(info,reward):
    if 'goal_lateral_scale_m' not in reward:return 1.,0.
    n=np.asarray(info['support_normal'])
    error=np.asarray(info['object_position_m'])-info['object_goal_m']
    ratio=float(np.linalg.norm(error-n*(error@n))/reward['goal_lateral_scale_m'])
    return float(np.exp(-ratio)),min(ratio,5.)

def aperture_score(info,reward):
    """Reward room to approach before rewarding a cube-sized aperture.

    This changes only the state reward. All four commands still come from PPO.
    Historical reward contracts retain their original near-weighted score.
    """
    near=1-np.tanh(info['grasp_distance_m']/reward['reach_scale_m'])
    desired=reward['cube_width_m']
    if 'pregrasp_width_m' in reward:
        fraction=np.clip((info['grasp_distance_m']-reward['close_distance_m'])/
                         (reward['open_distance_m']-reward['close_distance_m']),0,1)
        desired+=fraction*(reward['pregrasp_width_m']-desired)
        weight=1.
    else:weight=near
    return float(weight*np.exp(-((info['actual_width_m']-desired)/reward['width_scale_m'])**2))

def potential(info,config,reward=REWARD):
    near=1-np.tanh(info['grasp_distance_m']/reward['reach_scale_m'])
    aperture=aperture_score(info,reward)
    grip=float(np.clip(min(info['contact_forces_n'][:2])/.05,0,1))
    height=float(np.clip(info['support_clearance_m']/config['lift_height'],0,1))
    held=float(np.clip(info['hold_time_s']/config['hold_seconds'],0,1))
    alignment,lateral=goal_alignment(info,reward)
    return float(reward['reach_weight']*near+reward['aperture_weight']*aperture
                 +reward['grip_weight']*grip+reward['height_weight']*grip*height*alignment+reward['hold_weight']*held
                 -reward.get('lateral_potential_weight',0)*grip*lateral)

def shaped_reward(previous,info,terminated,config,reward=REWARD):
    next_phi=0. if terminated else potential(info,config,reward)
    terms=dict(potential_change=reward['gamma']*next_phi-previous,step_cost=-reward['step_cost'],
        success_bonus=reward['success_bonus']*info['is_success'],
        safety_penalty=-reward['safety_penalty']*info['is_safety_failure'],
        drop_penalty=-reward['drop_penalty']*(info['termination_reason'] in ('dropped','object_lost')))
    near=1-np.tanh(info['grasp_distance_m']/reward['reach_scale_m'])
    aperture=aperture_score(info,reward)
    grip=float(np.clip(min(info['contact_forces_n'][:2])/.05,0,1))
    height=float(np.clip(info['support_clearance_m']/config['lift_height'],0,1))
    held=float(np.clip(info['hold_time_s']/config['hold_seconds'],0,1))
    alignment,lateral=goal_alignment(info,reward)
    terms.update(dense_reach=reward.get('dense_reach',0)*near,
        dense_aperture=reward.get('dense_aperture',0)*aperture,
        dense_grip=reward.get('dense_grip',0)*grip,dense_height=reward.get('dense_height',0)*grip*height*alignment,
        lateral_penalty=-reward.get('dense_lateral',0)*grip*lateral,
        dense_hold=reward.get('dense_hold',0)*held,
        speed_cost=-reward.get('speed_cost',0)*grip*(height>.9)*min(np.linalg.norm(info['object_velocity_m_s'])/.01,1))
    terms={key:float(value*reward.get('scale',1.)) for key,value in terms.items()}
    result=float(sum(terms.values()))
    if not np.isfinite(result):raise ValueError('Non-finite PPO grasp reward')
    return result,terms
