"""真值观测的桌面笛卡尔抓取环境，调用运行时执行物理步并计算任务完成状态。"""
import copy
import math
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from ...tasks.grasp.tabletop_cartesian import CartesianTabletopConfig as Config
from ...simulation.runtime.tabletop import TabletopRuntime as Runtime
from ...simulation.motion.cartesian_controller import CartesianController
from ...simulation.devices.mujoco_gripper import MAX_WIDTH
from ...tasks.grasp.tabletop_cartesian_reward import calculate,potential
from ...simulation.runtime.observation_presets import OBSERVATION_POSES
from ...simulation.physics.grasp_metrics import CONTACT_MEASUREMENT

OBSERVATION_FIELDS=(
    ('tcp_base_relative',3),('tcp_velocity',3),('object_tcp_relative',3),('goal_object_relative',3),
    ('object_velocity',3),('object_angular_velocity',3),('gripper_width_velocity_reference',3),
    ('pad_and_support_forces',3),('hold_progress',1),('cartesian_tracking_error',3),('ik_fraction',1))
ACTION_CONTRACT=dict(name='world_tcp_delta_and_absolute_jaw_v1',shape=4,
                     components=['delta_x','delta_y','delta_z','absolute_gripper_opening'])
POSE_OBSERVATION_FIELDS=OBSERVATION_FIELDS+(
    ('grasp_orientation_error',3),('tcp_angular_velocity',3),('angular_tracking_error',3))
POSE_ACTION_CONTRACT=dict(name='world_tcp_pose_delta_and_absolute_jaw_v1',shape=7,
    components=['delta_x','delta_y','delta_z','delta_rx','delta_ry','delta_rz','absolute_gripper_opening'])
DELTA_JAW_ACTION_CONTRACT=dict(name='world_tcp_delta_and_incremental_jaw_v1',shape=4,
    components=['delta_x','delta_y','delta_z','delta_gripper_opening'])
POSE_DELTA_JAW_ACTION_CONTRACT=dict(name='world_tcp_pose_delta_and_incremental_jaw_v1',shape=7,
    components=['delta_x','delta_y','delta_z','delta_rx','delta_ry','delta_rz','delta_gripper_opening'])
VISION_OBSERVATION_FIELDS=(
    ('tcp_base_relative',3),('tcp_velocity',3),('rgbd_object_tcp_relative',3),('rgbd_goal_object_relative',3),
    ('rgbd_object_velocity',3),('rgbd_surface_normal',3),('gripper_width_velocity_reference',3),
    ('visual_grip_confidence',1),('rgbd_detection_confidence',1),('rgbd_measurement_age',1),
    ('visual_hold_progress',1),('cartesian_tracking_error',3),('ik_fraction',1))


def action_contract(config):
    """从控制配置生成动作维度和语义，供环境、训练和 checkpoint 校验共用。"""
    if config.jaw_control=='delta':
        return POSE_DELTA_JAW_ACTION_CONTRACT if config.uses_orientation_actions else DELTA_JAW_ACTION_CONTRACT
    return POSE_ACTION_CONTRACT if config.uses_orientation_actions else ACTION_CONTRACT


def observation_fields(config):
    """返回当前模式按顺序排列的观测字段及维度。"""
    if config.observation_source=='rgbd_hsv':
        return VISION_OBSERVATION_FIELDS+POSE_OBSERVATION_FIELDS[-3:] if config.uses_orientation_actions else VISION_OBSERVATION_FIELDS
    return POSE_OBSERVATION_FIELDS if config.uses_orientation_actions else OBSERVATION_FIELDS


class TabletopCartesianEnv(gym.Env):
    """从固定观察位开始的笛卡尔桌面抓取环境。

    config 选定位置或位姿控制、夹爪模式和完成条件，startup_tick 可记录启动回放。
    reset 完成传统启动后返回观测。step 将归一化增量传给 CartesianController，
    并通过每物理步的双侧夹持力、支撑净空和稳定时间判断完成。
    """
    metadata={'render_modes':['ansi'],'render_fps':20}

    def __init__(self,config=None,*,startup_tick=None):
        self.config=Config() if config is None else config
        if self.config.observation_source=='rgbd_hsv' and type(self) is TabletopCartesianEnv:
            raise ValueError('Use make_tabletop_cartesian_env for the camera-only observation contract')
        if self.config.observation_pose=='forward' and not self.config.uses_orientation_actions:
            raise ValueError('A horizontal tabletop grasp start requires pose control so PPO can turn the gripper')
        self.robot=Runtime(self.config);self.startup_tick=startup_tick
        self.action_space=spaces.Box(-1.,1.,(action_contract(self.config)['shape'],),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,(sum(n for _,n in observation_fields(self.config)),),dtype=np.float32)
        self.grasp_orientation=np.array(OBSERVATION_POSES['table_down'].tool_rotation)
        self._done=True;self._closed=False

    def reset(self,*,seed=None,options=None):
        """采样平面目标、执行固定观察位启动，并初始化夹爪参考和保持计数。"""
        if self._closed:raise RuntimeError('Environment closed')
        if options:raise ValueError('Configure reset through the serialized Config')
        super().reset(seed=seed)
        angle=self.np_random.uniform(0,2*np.pi);radius=self.config.object_radius*np.sqrt(self.np_random.uniform())
        xy=np.asarray(self.config.object_xy)+radius*np.array([np.cos(angle),np.sin(angle)])
        self.startup,orientation=self.robot.prepare(xy,tick=self.startup_tick)
        self.controller=CartesianController(self.robot,orientation)
        self._jaw_reference=float(self.config.initial_width)
        self._initial_object=self.robot.data.qpos[self.robot.object_qadr:self.robot.object_qadr+3].copy()
        self.goal=self._initial_object+np.array([0,0,self.config.lift_height])
        self._policy_start_time=float(self.robot.data.time)
        self.steps=0;self._stable_ticks=0;self._maximum_hold=0;self._anchor=None;self._ever_lifted=False;self._lost_ticks=0
        self._max_clearance=0.;self._max_grip_force=0.
        self._min_distance=float(np.linalg.norm(self._initial_object-self.robot.tcp_pose()[0]))
        self._required_ticks=math.ceil(self.config.hold_seconds/self.config.physics_timestep)
        self._done=False;self._last_ik=dict(ik_fraction=1.,ik_reason=None,physics_steps=0)
        self.robot.recorder.phase='ppo'
        info=self._info();obs=self._observation(info);self.previous=potential(info,self.config)
        self._last_obs=obs.copy();self._last_info=copy.deepcopy(info)
        return obs,info

    def _hold_tick(self,row):
        # 逐物理步累计满足任务条件的稳定时间，避免策略步之间的滑落被遗漏。
        forces=np.array([row['left_grasp_n'],row['right_grasp_n'],row['support_normal_n']])
        bilateral=min(forces[:2])>.005
        self._max_clearance=max(self._max_clearance,float(row['support_clearance_m']))
        self._max_grip_force=max(self._max_grip_force,float(min(forces[:2])))
        self._ever_lifted|=bilateral and row['support_clearance_m']>.003
        self._lost_ticks=self._lost_ticks+1 if self._ever_lifted and not bilateral else 0
        position=np.asarray(row['object_position_m']);velocity=np.asarray(row['object_qvel'])
        required=self.config.lift_height if self.config.success_height is None else self.config.success_height
        good=(bilateral and forces[2]<.001 and row['support_clearance_m']>=required-self.config.height_tolerance
              and np.linalg.norm((position-self.goal)[:2])<.003 and np.linalg.norm(velocity[:3])<.002
              and np.linalg.norm(velocity[3:])<.1)
        relative=np.asarray(row['object_relative_to_pads_m'])
        if good and self._anchor is not None and np.linalg.norm(relative-self._anchor)>=.002:good=False
        if good:
            if self._anchor is None:self._anchor=relative.copy()
            self._stable_ticks+=1;self._maximum_hold=max(self._maximum_hold,self._stable_ticks)
        else:self._stable_ticks=0;self._anchor=None

    def _info(self):
        r=self.robot;data=r.data
        joints=r.get_joint_state()
        position=data.qpos[r.object_qadr:r.object_qadr+3].copy()
        object_rotation=data.xmat[r.object_body].reshape(3,3)
        bottom_radius=.01*np.sum(abs(object_rotation[2,:]))
        tcp,rotation=r.tcp_pose()
        orientation_error=r.rotation_error(self.grasp_orientation,rotation)
        tracking_error=r.rotation_error(self.controller.orientation,rotation)
        return dict(task='tabletop_cartesian',phase='ppo',policy_steps=self.steps,
            simulation_time_s=float(data.time),policy_time_s=float(data.time)-self._policy_start_time,
            tcp_position_m=tcp.tolist(),tcp_velocity_m_s=r.tcp_velocity().tolist(),
            object_position_m=position.tolist(),object_goal_m=self.goal.tolist(),
            object_velocity_m_s=data.qvel[r.object_vadr:r.object_vadr+3].tolist(),
            object_angular_velocity_rad_s=data.qvel[r.object_vadr+3:r.object_vadr+6].tolist(),
            clearance_m=float(position[2]-r.table_height-bottom_radius),
            world_z_lift_m=float(position[2]-self._initial_object[2]),
            contact_forces_n=r.forces().tolist(),gripper_width_m=r.gripper.get_width(),
            jaw_control=self.config.jaw_control,jaw_command_reference_m=self._jaw_reference,
            raw_contact_normal_forces_n=r.raw_forces().tolist(),contact_measurement=CONTACT_MEASUREMENT,
            hold_seconds=self._stable_ticks*.001,max_hold_seconds=self._maximum_hold*.001,
            max_clearance_m=self._max_clearance,max_bilateral_force_n=self._max_grip_force,
            min_tcp_object_distance_m=self._min_distance,
            orientation_error_rad=float(np.linalg.norm(tracking_error)),
            grasp_orientation_error_rad=float(np.linalg.norm(orientation_error)),
            grasp_orientation_error_vector=orientation_error.tolist(),
            angular_tracking_error=tracking_error.tolist(),tcp_angular_velocity_rad_s=r.tcp_angular_velocity().tolist(),
            joint_position_rad=joints['position_rad'].tolist(),
            joint_reference_rad=joints['reference_rad'].tolist(),camera=r.camera_frame(),
            startup=copy.deepcopy(self.startup),observation_valid=True,is_success=False,
            is_safety_failure=False,termination_reason=None,detail=None,**self._last_ik)

    def _observation(self,info):
        r=self.robot;tcp=np.asarray(info['tcp_position_m']);position=np.asarray(info['object_position_m'])
        gripper=r.gripper
        width_speed=gripper.get_velocity()
        value=np.concatenate([
            (tcp-r.base_position)/.5,np.asarray(info['tcp_velocity_m_s'])/.05,
            (position-tcp)/.05,(self.goal-position)/.05,
            np.asarray(info['object_velocity_m_s'])/.05,np.asarray(info['object_angular_velocity_rad_s'])/.1,
            [info['gripper_width_m']/MAX_WIDTH,width_speed/.04,
             (self._jaw_reference if self.config.jaw_control=='delta' else
              gripper.get_reference_width())/MAX_WIDTH],
            np.asarray(info['contact_forces_n'])/np.array([.2,.2,.1]),[self._stable_ticks/self._required_ticks],
            (self.controller.reference-tcp)/.01,[self.controller.last_fraction]]).astype(np.float32)
        # Fixed metric scaling; no IK-solved joint targets enter the observation.
        value[6:12]=np.arcsinh(value[6:12]*5)
        if self.config.uses_orientation_actions:
            value=np.r_[value,np.asarray(info['grasp_orientation_error_vector'])/np.pi,
                        np.asarray(info['tcp_angular_velocity_rad_s'])/.5,
                        np.asarray(info['angular_tracking_error'])/.05].astype(np.float32)
        if value.shape!=self.observation_space.shape or not np.isfinite(value).all():raise ValueError('Invalid observation')
        return value

    def step(self,action):
        if self._closed or self._done:raise RuntimeError('Reset an open environment before stepping')
        action=np.asarray(action,dtype=np.float32)
        if action.shape!=self.action_space.shape or not np.isfinite(action).all() or np.any(abs(action)>1):
            raise ValueError('A finite normalized Cartesian action matching the selected control mode is required')
        self.steps+=1;reason=detail=None;start=float(self.robot.data.time)
        self.robot.recorder.phase='ppo'
        if self.config.jaw_control=='delta':
            self._jaw_reference=float(np.clip(self._jaw_reference+action[-1]*self.config.jaw_delta_scale,
                self.config.jaw_margin,MAX_WIDTH-self.config.jaw_margin))
        else:self._jaw_reference=float((action[-1]+1)*MAX_WIDTH/2)
        try:
            self._last_ik=self.controller.execute(action[:3]*self.config.action_scale*np.asarray(self.config.axis_scale),
                                                 self._jaw_reference,observer=self._hold_tick,
                                                 delta_rotation=action[3:6]*self.config.rotation_action_scale
                                                 if self.config.uses_orientation_actions else None)
            p,r=self.robot.tcp_pose()
            if np.linalg.norm(self.robot.rotation_error(self.controller.orientation,r))>np.deg2rad(3):
                reason='safety_failure';detail='Actual TCP orientation exceeded the fixed-pose envelope'
        except (ValueError,RuntimeError,mujoco.FatalError) as exc:
            reason='safety_failure';detail=str(exc)
            self._last_ik.update(physics_steps=round((float(self.robot.data.time)-start)/.001))
        try:
            info=self._info();obs=self._observation(info)
            self._min_distance=min(self._min_distance,float(np.linalg.norm(np.array(info['object_position_m'])-info['tcp_position_m'])))
            info['min_tcp_object_distance_m']=self._min_distance
        except (ValueError,RuntimeError,mujoco.FatalError) as exc:
            reason='numerical_failure';detail=str(exc);info=copy.deepcopy(self._last_info)
            info.update(observation_valid=False);obs=self._last_obs.copy()
        if reason is None:
            if self._stable_ticks>=self._required_ticks:reason='success'
            elif self._lost_ticks*.001>=.5:reason='dropped'
            elif info['object_position_m'][2]<self.robot.table_height-.03:reason='dropped'
        terminated=reason is not None;truncated=not terminated and self.steps>=self.config.max_steps
        if truncated:reason='time_limit'
        info.update(is_success=reason=='success',termination_reason=reason,detail=detail,
                    is_safety_failure=reason not in (None,'success','time_limit','dropped'))
        reward,terms=calculate(self.previous,info,terminated,action,self.config)
        self.previous=potential(info,self.config);info['reward_components']=terms
        self._done=terminated or truncated;self._last_obs=obs.copy();self._last_info=copy.deepcopy(info)
        return obs,reward,terminated,truncated,info

    def close(self):self._closed=True
