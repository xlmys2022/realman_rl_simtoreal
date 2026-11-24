"""计算笛卡尔桌面的接近、围拢、夹持、抬升等奖励项，评估诊断可复用同一计算。"""
import numpy as np

REWARD=dict(name='cartesian_task_v10',gamma=.999,scale=.1,success=1000.,safety=20.,dropped=20.,
            step=.01,action_l2=.005,rejection=.02,aperture=.5,over_open_fraction=.03,
            dense_reach=.1,dense_cage=.1,dense_pinch=.1,dense_grip=.2,dense_height=.4,
            approach_clearance_m=.05,grasp_offset_z_m=.003,approach_xy_scale_m=.015,
            cage_xy_scale_m=.01,cage_z_scale_m=.015,pinch_xy_scale_m=.004,pinch_z_scale_m=.005)


def quantities(info,config):
    """将实际位姿、开口和接触力转换为各奖励项使用的有界特征。"""
    position=np.asarray(info['object_position_m']);goal=np.asarray(info['object_goal_m'])
    offset=np.asarray(info['tcp_position_m'])-position
    distance=float(np.linalg.norm(offset))
    grip=float(np.clip(min(info['contact_forces_n'][:2])/.12,0,1))
    required=config.lift_height if config.success_height is None else config.success_height
    height=float(np.clip(info['clearance_m']/required,0,1))
    xy=float(np.linalg.norm((position-goal)[:2]))
    alignment=float(np.exp(-xy/.01))
    hold=float(np.clip(info['hold_seconds']/config.hold_seconds,0,1))
    orientation=float(np.exp(-info['grasp_orientation_error_rad']/.5)) if config.uses_orientation_actions else 0.
    oriented=orientation if config.uses_orientation_actions else 1.
    lateral=float(np.linalg.norm(offset[:2]));height_error=float(offset[2]-REWARD['grasp_offset_z_m'])
    # 目标上方的期望间隙随横向对齐平滑减小，使策略先越过桌面再靠近夹取点。
    desired_height=REWARD['grasp_offset_z_m']+REWARD['approach_clearance_m']*(
        1-np.exp(-(lateral/REWARD['approach_xy_scale_m'])**2))
    approach_distance=float(np.hypot(lateral,offset[2]-desired_height))
    cage=float(np.exp(-(lateral/REWARD['cage_xy_scale_m'])**2-
                      (height_error/REWARD['cage_z_scale_m'])**2)*oriented)
    readiness=float(np.exp(-(lateral/REWARD['pinch_xy_scale_m'])**2-
                           (height_error/REWARD['pinch_z_scale_m'])**2)*oriented)
    desired=.06-.04*readiness
    width=info['gripper_width_m']
    # Premature closure is costly; an aligned open jaw can safely learn closure.
    aperture_error=float((max(desired-width,0)+REWARD['over_open_fraction']*max(width-desired,0))/.06)
    pinch=float(readiness*np.clip((.06-width)/.04,0,1))
    return dict(distance=distance,grip=grip,height=height,alignment=alignment,xy=xy,
                aperture_error=aperture_error,hold=hold,orientation=orientation,
                approach_distance=approach_distance,cage=cage,pinch=pinch,desired_width_m=desired)


def potential(info,config):
    """按任务进度合成单状态的非负势函数。"""
    f=quantities(info,config)
    # Bounded and nonnegative: discounting a far-away negative potential used
    # to create a positive reward for doing nothing at the observation pose.
    return float(8*(1-np.clip(f['approach_distance']/.6,0,1))+3*f['cage']+2*f['pinch']+2*f['grip']+
                 8*f['height']*f['alignment']*f['grip']+2*f['orientation'])


def calculate(previous,info,terminated,action,config):
    """给定上一步势值和本次转移，返回总奖励及明细，终止转移使用零后继势。"""
    f=quantities(info,config)
    terms=dict(potential=REWARD['gamma']*(0. if terminated else potential(info,config))-previous,
               reach=REWARD['dense_reach']*np.exp(-f['approach_distance']/.1),
               ready=REWARD['dense_cage']*f['cage'],
               pinch=REWARD['dense_pinch']*f['pinch'],
               grip=REWARD['dense_grip']*f['grip'],
               height=REWARD['dense_height']*f['grip']*f['height']*f['alignment'],
               alignment=-.05*f['grip']*min(f['xy']/.003,5),
               aperture=-REWARD['aperture']*f['aperture_error'],
               orientation=-.05*(1-f['orientation']) if config.uses_orientation_actions else 0.,
               hold=-.01*f['grip']*f['height']*(1-f['hold']),
               step=-REWARD['step'],action=-REWARD['action_l2']*float(np.asarray(action[:-1])@np.asarray(action[:-1])),
               rejected=-REWARD['rejection']*(info['ik_fraction']==0),
               speed=-.05*f['grip']*(f['height']>=1)*min(np.linalg.norm(info['object_velocity_m_s'])/.002,3),
               success=REWARD['success']*info['is_success'],safety=-REWARD['safety']*info['is_safety_failure'],
               dropped=-REWARD['dropped']*(info['termination_reason']=='dropped'))
    terms={k:float(v*REWARD['scale']) for k,v in terms.items()}
    result=float(sum(terms.values()))
    if not np.isfinite(result):raise ValueError('Nonfinite reward')
    return result,terms
