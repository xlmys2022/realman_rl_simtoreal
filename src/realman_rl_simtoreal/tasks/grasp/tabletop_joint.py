"""关节空间桌面抓取的任务、课程和奖励定义，含接近子目标与夹爪开口正则。"""
import copy

import numpy as np
TABLETOP_PICK_CONFIG=dict(
    initial_q_deg=[0.,0.,0.,0.,0.,0.],
    initial_pose_name='zero_development',
    initial_pose_source='https://github.com/RealManRobot/ros2_rm_robot/blob/humble/rm_moveit2_config/rm_63_config/config/initial_positions.yaml',
    object_xy=[.4,0.],object_radius=0.,mass=.01,friction=.7,
    initial_width=.06,lift_height=.03,goal_height=None,height_tolerance=.002,hold_seconds=2.,
    joint_action_scale=.008726646259971648,control_period=.05,physics_timestep=.001,
    max_episode_steps=1200,gripper_reference_speed=.5,gripper_command_margin=.002,record_physics=False,
    reference_smoothing=1.,
    start_fraction=1.,pregrasp_height=.06,reject_unsafe_targets=True,
    approach_waypoint_height=0.,approach_waypoint_tolerance=.015,
)


# Historical training reward and curriculum contracts; numerical definitions are unchanged.
REWARD=dict(name='tabletop_pick_action_regularized_v6',gamma=.995,scale=.01,success=1000.,safety=50.,drop=10.,
    joint_potential=10.,reach_potential=1.,orientation_potential=.5,aperture_potential=1.,
    grip_potential=2.,height_potential=8.,hold_potential=2.,step=.01,rejection=.05,
    dense_joint=1.,dense_reach=.2,dense_aperture=.2,dense_grip=.5,dense_height=.8,dense_hold=.1,speed_cost=.1,
    binary_contact_joint_goal=True,pregrasp_joint_weight=.1,
    approach_joint_scale=.2,grasp_joint_scale=.03,action_target_weight=1.,action_target_scale_rad=.1,
    far_jaw_action_weight=.5)

CURRICULUM=dict(
    pinch=dict(start_fraction=0.,pregrasp_height=.003,initial_width=.022,lift_height=.005,
               height_tolerance=.0005,hold_seconds=.2,max_episode_steps=300,object_radius=0.),
    approach=dict(start_fraction=0.,pregrasp_height=.03,initial_width=.06,lift_height=.005,
                  height_tolerance=.0005,hold_seconds=.2,max_episode_steps=500,object_radius=0.),
    lift=dict(start_fraction=0.,pregrasp_height=.03,object_radius=0.,max_episode_steps=700),
    unfold25=dict(start_fraction=.25,object_radius=0.,max_episode_steps=900),
    unfold50=dict(start_fraction=.5,object_radius=0.,max_episode_steps=1000),
    unfold75=dict(start_fraction=.75,object_radius=0.,max_episode_steps=1100),
    full=dict(),full_random=dict(object_radius=.02))

def features(info,config,reward=REWARD):
    """从当前转移记录提取关节桌面任务所需的定位、夹持、净空和保持量。"""
    grip=float(np.clip(min(info['contact_forces_n'][:2])/.05,0,1))
    distance=info['grasp_distance_m']
    fraction=float(np.clip((distance-.005)/.035,0,1))
    approaching=not info.get('approach_complete',True)
    if approaching:fraction=1.
    if info['bilateral_contact']:fraction=0.
    desired=.02+.04*fraction
    aperture=float(np.exp(-((info['actual_width_m']-desired)/.02)**2))
    goal_height=config.get('goal_height') or config['lift_height']
    height=float(np.clip(info['support_clearance_m']/goal_height,0,1))
    held=float(np.clip(info['hold_time_s']/config['hold_seconds'],0,1))
    error=np.asarray(info['object_position_m'])-info['object_goal_m']
    alignment=float(np.exp(-np.linalg.norm(error[:2])/.005))
    if reward.get('binary_contact_joint_goal',False):
        joint=info['lift_joint_error_rad'] if info['bilateral_contact'] else info.get('reach_joint_error_rad',info['grasp_joint_error_rad'])
    else:joint=(1-grip)*info['grasp_joint_error_rad']+grip*info['lift_joint_error_rad']
    joint_weight=1. if info['bilateral_contact'] or approaching else reward.get('pregrasp_joint_weight',1.)
    return dict(joint=joint,reach=float(np.exp(-info.get('reach_goal_distance_m',distance)/.08)),
        orientation=float(np.exp(-(info['orientation_error_rad']/.5)**2)),
        aperture=aperture,grip=grip,height=height*alignment,hold=held,joint_weight=joint_weight,
        joint_scale=reward.get('approach_joint_scale',.2) if approaching else reward.get('grasp_joint_scale',.03))

def potential(info,config,reward=REWARD):
    f=features(info,config,reward)
    return float(-reward['joint_potential']*f['joint']+reward['reach_potential']*f['reach']
        +reward['orientation_potential']*f['orientation']+reward['aperture_potential']*f['aperture']
        +reward['grip_potential']*f['grip']+reward['height_potential']*f['grip']*f['height']
        +reward['hold_potential']*f['hold'])

def shaped_reward(previous,info,terminated,config,reward=REWARD,*,action=None,action_info=None):
    """计算势差及动作正则，使用动作执行前的子目标评估关节命令。"""
    f=features(info,config,reward)
    terms=dict(potential_change=reward['gamma']*(0 if terminated else potential(info,config,reward))-previous,
        step=-reward['step'],joint_distance=-.02*f['joint'],
        joint_alignment=reward.get('dense_joint',0)*f['joint_weight']*np.exp(-f['joint']/f['joint_scale']),
        reach=reward['dense_reach']*f['reach'],aperture=reward['dense_aperture']*f['aperture'],
        grip=reward['dense_grip']*f['grip'],height=reward['dense_height']*f['grip']*f['height'],
        hold=reward['dense_hold']*f['hold'],
        speed=-reward['speed_cost']*f['grip']*(f['height']>.9)*min(np.linalg.norm(info['object_velocity_m_s'])/.01,1),
        rejected=-reward['rejection']*info['action_rejected'],
        success=reward['success']*info['is_success'],safety=-reward['safety']*info['is_safety_failure'],
        dropped=-reward['drop']*(info['termination_reason'] in ('dropped','object_lost')))
    if reward.get('action_target_weight',0):
        if action is None or action_info is None:raise ValueError('Action regularization requires the observed pre-action goal')
        target=action_info['lift_goal_joint_rad'] if action_info['bilateral_contact'] else action_info['reach_goal_joint_rad']
        error=(np.deg2rad(np.asarray(action[:6],dtype=float))-target)/reward['action_target_scale_rad']
        terms['action_target']=-reward['action_target_weight']*float(error@error)
        if (not action_info['bilateral_contact'] and
                (not action_info.get('approach_complete',True) or action_info['grasp_distance_m']>.03)):
            from ...simulation.devices.mujoco_gripper import MAX_WIDTH
            open_action=2*config['initial_width']/MAX_WIDTH-1
            terms['far_jaw_action']=-reward.get('far_jaw_action_weight',0)*(float(action[6])-open_action)**2
    terms={k:float(v*reward['scale']) for k,v in terms.items()}
    result=float(sum(terms.values()))
    if not np.isfinite(result):raise ValueError('Non-finite tabletop reward')
    return result,terms


def environment_config(stage='full',**overrides):
    """按课程阶段构造环境配置，保留各阶段的原始覆盖规则。"""
    if stage not in CURRICULUM:raise ValueError('Unknown tabletop curriculum')
    result=copy.deepcopy(TABLETOP_PICK_CONFIG);result.update(CURRICULUM[stage]);result.update(overrides)
    return result
