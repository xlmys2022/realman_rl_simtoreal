"""使用腕部 RGB-D 和机器人本体状态构造桌面策略观测，并保留独立的物理验收。"""
import copy
import numpy as np
from .tabletop_cartesian_env import TabletopCartesianEnv
from ...perception.rgbd import OrangeCubeDetector,VisualTracker
from ...simulation.devices.mujoco_rgbd_camera import MujocoRgbdCamera
from ...simulation.devices.mujoco_gripper import MAX_WIDTH
from ...tasks.grasp.tabletop_cartesian_reward import calculate

VISION_CONTRACT=dict(name='wrist_rgbd_hsv_cube_v4',
    object_source='aligned_rgb_depth_and_calibration',mask='opencv_rgb_to_hsv_inrange',
    geometry_prior='upright_cube_with_configured_edge_length',
    object_pose_truth=False,object_velocity_truth=False,contact_force_observations=False,
    privileged_hold_progress=False,completion_source='camera_visual_hold_with_settling_margin',
    neural_control='zero_anchored_cartesian_network',
    privileged_reward_and_success_audit=True)


def visual_grip_and_hold(visual,offset,goal_error,*,width,jaw_reference,cube_size,
                         timeout,previous_grip,previous_hold,dt):
    """由显式的视觉估计和本体量更新夹持置信度与保持时间（秒）。

    offset 和 goal_error 为基座系位移（m）。检测失效、相对速度超限或夹持不足时
    清零累计保持时间。
    """
    enclosure=np.exp(-(np.linalg.norm(offset[:2])/.006)**2-(max(0.,abs(offset[2]+.003)-.012)/.01)**2)
    aperture=np.exp(-((width-cube_size)/.004)**2)
    compression=np.clip((width-jaw_reference-.010)/.010,0,1)
    measured_grip=float(enclosure*aperture*compression*visual['confidence'])
    alpha=1-np.exp(-dt/.08) if dt else 1.
    grip=previous_grip
    grip+=alpha*(measured_grip-grip)
    stable=(visual['visible'] and visual['age']<=timeout
        and np.linalg.norm(goal_error[:2])<.003 and abs(goal_error[2])<.001
        and np.linalg.norm(visual['velocity_m_s'])<.0015 and grip>.5)
    hold=previous_hold+dt if stable else 0.
    return grip,hold


def encode_visual_observation(tcp,tcp_velocity,offset,goal_error,visual,*,width,width_speed,
                              jaw_reference,grip,hold,timeout,hold_required,tracking_error,ik_fraction):
    """用固定尺度编码 29 个视觉控制通道，返回 float32 数组。"""
    return np.concatenate([tcp/.5,tcp_velocity/.05,
        np.arcsinh(offset/.01),np.arcsinh(goal_error/.01),visual['velocity_m_s']/.05,visual['normal'],
        [width/MAX_WIDTH,width_speed/.04,jaw_reference/MAX_WIDTH],
        [grip,visual['confidence'],min(visual['age']/timeout,2.)],
        [min(hold/hold_required,1.)],tracking_error/.01,[ik_fraction]]).astype(np.float32)


class VisionTabletopCartesianEnv(TabletopCartesianEnv):
    """在桌面物理环境上使用 RGB-D 检测与跟踪替代物体真值观测。

    config 提供相机、HSV 和超时参数。reset 清理跟踪历史，step 依据视觉保持和
    感知超时决定回合结束，再用物理结果验收。visual_state 保存基座系估计及新鲜度。
    """
    def __init__(self,config,*,startup_tick=None):
        if config.observation_source!='rgbd_hsv':raise ValueError('Vision environment requires rgbd_hsv')
        super().__init__(config,startup_tick=startup_tick)
        self.camera=MujocoRgbdCamera(self.robot,width=config.vision_width,height=config.vision_height)
        self.tracker=VisualTracker(OrangeCubeDetector(edge_m=config.vision_cube_size,
            hsv_low=config.vision_hsv_low,hsv_high=config.vision_hsv_high),timeout=config.vision_timeout)
        self._visual_goal=None;self._last_capture=-np.inf;self._last_observation_time=None
        self._visual_hold=0.;self._visual_grip=0.;self.visual_state=None

    def reset(self,*,seed=None,options=None):
        self.tracker.reset();self._visual_goal=None;self._last_capture=-np.inf
        self._last_observation_time=None;self._visual_hold=0.;self._visual_grip=0.;self.visual_state=None
        if hasattr(self.robot,'vision_debug'):del self.robot.vision_debug
        return super().reset(seed=seed,options=options)

    def _observation(self,info):
        # 用首次相机测量建立抬升目标。TCP、夹爪和参考量来自本体读数，物体估计来自检测与跟踪。
        robot=self.robot;now=float(robot.data.time)
        if now-self._last_capture>=self.config.vision_period-1e-8:
            self.tracker.update(self.camera.capture());self._last_capture=now
        visual=self.tracker.state(now);self.visual_state=visual
        if self._visual_goal is None:
            self._visual_goal=visual['position_m']+np.array([0.,0.,self.config.lift_height])
        base_position,base_rotation=self.camera.base_pose()
        tcp_world,_=robot.tcp_pose();tcp=base_rotation.T@(tcp_world-base_position)
        tcp_velocity=base_rotation.T@robot.tcp_velocity()
        position=visual['position_m'];offset=position-tcp;goal_error=self._visual_goal-position
        gripper=robot.get_gripper_state()
        width=gripper['width_m'];width_speed=gripper['velocity_m_s']
        dt=0. if self._last_observation_time is None else max(0.,now-self._last_observation_time)
        self._last_observation_time=now
        self._visual_grip,self._visual_hold=visual_grip_and_hold(visual,offset,goal_error,
            width=width,jaw_reference=self._jaw_reference,cube_size=self.config.vision_cube_size,
            timeout=self.config.vision_timeout,previous_grip=self._visual_grip,
            previous_hold=self._visual_hold,dt=dt)
        value=encode_visual_observation(tcp,tcp_velocity,offset,goal_error,visual,
            width=width,width_speed=width_speed,jaw_reference=self._jaw_reference,
            grip=self._visual_grip,hold=self._visual_hold,timeout=self.config.vision_timeout,
            hold_required=self.config.hold_seconds+self.config.vision_hold_margin,
            tracking_error=base_rotation.T@(self.controller.reference-tcp_world),
            ik_fraction=self.controller.last_fraction)
        if self.config.uses_orientation_actions:
            _,rotation=robot.tcp_pose()
            value=np.r_[value,robot.rotation_error(self.grasp_orientation,rotation)/np.pi,
                robot.tcp_angular_velocity()/.5,robot.rotation_error(self.controller.orientation,rotation)/.05].astype(np.float32)
        if value.shape!=self.observation_space.shape or not np.isfinite(value).all():raise ValueError('Invalid camera observation')
        info['vision']=dict(source=VISION_CONTRACT['name'],object_position_base_m=position.tolist(),
            object_velocity_base_m_s=visual['velocity_m_s'].tolist(),object_goal_base_m=self._visual_goal.tolist(),
            surface_normal_base=visual['normal'].tolist(),detected=visual['visible'],confidence=visual['confidence'],
            age_seconds=visual['age'],pixels=visual['pixels'],centroid_px=list(visual['centroid_px']),
            visual_grip_confidence=self._visual_grip,visual_hold_seconds=self._visual_hold,
            visual_hold_required_seconds=self.config.hold_seconds+self.config.vision_hold_margin,
            simulator_object_state_in_observation=False)
        robot.vision_debug=dict(frame=self.camera.frame,mask=self.tracker.detector.mask,measurement=info['vision'])
        return value

    def step(self,action):
        previous=self.previous
        observation,reward,terminated,truncated,info=super().step(action)
        physical_success=self._stable_ticks>=self._required_ticks
        info.update(completion_source=VISION_CONTRACT['completion_source'],physical_audit_success=bool(physical_success))
        changed=False
        if self.visual_state is not None and self.visual_state['expired']:
            terminated=True;truncated=False;reward=-2.
            info.update(is_success=False,termination_reason='vision_lost',
                detail='RGB-D target missing beyond the configured camera timeout',
                reward_components={'vision_lost':reward})
            changed=True
        elif info['termination_reason'] in (None,'success','time_limit'):
            ready=self._visual_hold+1e-9>=self.config.hold_seconds+self.config.vision_hold_margin
            if ready:
                terminated=True;truncated=False
                info.update(is_success=bool(physical_success),
                    termination_reason='success' if physical_success else 'visual_false_success',
                    detail=None if physical_success else 'Camera completion did not pass the independent physical audit')
                changed=True
            elif info['termination_reason']=='success':
                # A privileged success event cannot end visual execution.
                terminated=False;truncated=self.steps>=self.config.max_steps
                info.update(is_success=False,termination_reason='time_limit' if truncated else None,detail=None)
                changed=True
            if changed:
                reward,terms=calculate(previous,info,terminated,action,self.config)
                if info['termination_reason']=='visual_false_success':reward=-2.;terms={'visual_false_success':reward}
                info['reward_components']=terms
        if changed:
            self._done=terminated or truncated;self._last_info=copy.deepcopy(info)
        return observation,reward,terminated,truncated,info

    def close(self):
        self.camera.close();super().close()
