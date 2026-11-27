"""使用腕部 RGB-D 和机器人本体状态构造桌面策略观测，并保留独立的物理验收。"""
import copy
import numpy as np
from .tabletop_cartesian_env import TabletopCartesianEnv
from ...hardware.realman_config import REALMAN_TABLETOP
from ...perception.rgbd import OrangeCubeDetector, UprightCubeDetector, VisualTracker
from ...simulation.devices.mujoco_rgbd_camera import MujocoRgbdCamera
from ...tasks.grasp.tabletop_policy_io import (
    VISION_CONTRACT, VisualPolicyProprioception, advance_visual_observation, encode_visual_observation,
    vision_contract, visual_grip_and_hold,
)
from ...tasks.grasp.tabletop_cartesian_reward import calculate
from ...simulation.runtime.observation_presets import REALMAN_TRANSFER_PROFILE


class VisionTabletopCartesianEnv(TabletopCartesianEnv):
    """在桌面物理环境上使用 RGB-D 检测与跟踪替代物体真值观测。

    config 提供相机、HSV 和超时参数。reset 清理跟踪历史，step 依据视觉保持和
    感知超时决定回合结束，再用物理结果验收。visual_state 保存基座系估计及新鲜度。
    """
    def __init__(self,config,*,startup_tick=None):
        if config.observation_source!='rgbd_hsv':raise ValueError('Vision environment requires rgbd_hsv')
        super().__init__(config,startup_tick=startup_tick)
        transfer=config.scene_profile==REALMAN_TRANSFER_PROFILE
        self.camera=MujocoRgbdCamera(self.robot,width=config.vision_width,height=config.vision_height,
                                   depth_unit_m=REALMAN_TABLETOP.depth_unit_m if transfer else None)
        detector=UprightCubeDetector if transfer else OrangeCubeDetector
        self.tracker=VisualTracker(detector(edge_m=config.vision_cube_size,
            hsv_low=config.vision_hsv_low,hsv_high=config.vision_hsv_high),timeout=config.vision_timeout)
        self.vision_contract=vision_contract(config)
        self._visual_goal=None;self._last_capture=-np.inf;self._next_capture=-np.inf;self._last_observation_time=None
        self._visual_hold=0.;self._visual_grip=0.;self.visual_state=None

    def reset(self,*,seed=None,options=None):
        self.tracker.reset();self._visual_goal=None;self._last_capture=-np.inf
        self._last_observation_time=None;self._visual_hold=0.;self._visual_grip=0.;self.visual_state=None
        if hasattr(self.robot,'vision_debug'):del self.robot.vision_debug
        return super().reset(seed=seed,options=options)

    def _observation(self,info):
        # 用首次相机测量建立抬升目标。TCP、夹爪和参考量来自本体读数，物体估计来自检测与跟踪。
        robot=self.robot;now=float(robot.data.time)
        if self.config.scene_profile != "legacy" and not np.isfinite(self._last_capture):
            self._next_capture = now
        capture_due = (now-self._last_capture>=self.config.vision_period-1e-8
                       if self.config.scene_profile == "legacy" else now+1e-8 >= self._next_capture)
        if capture_due:
            self.tracker.update(self.camera.capture());self._last_capture=now
            # 新场景用相位截止时间，在 20 Hz 控制循环上维持约 15 Hz 的平均观测率。
            if self.config.scene_profile != "legacy":
                if self._next_capture < now - self.config.vision_period:
                    self._next_capture = now
                self._next_capture += self.config.vision_period
        visual=self.tracker.state(now);self.visual_state=visual
        base_position,base_rotation=self.camera.base_pose()
        tcp_world,_=robot.tcp_pose()
        gripper=robot.get_gripper_state()
        sample=VisualPolicyProprioception(timestamp_s=now,tcp_base_m=base_rotation.T@(tcp_world-base_position),
            tcp_velocity_base_m_s=base_rotation.T@robot.tcp_velocity(),gripper_width_m=gripper['width_m'],
            gripper_width_velocity_m_s=gripper['velocity_m_s'],jaw_reference_m=self._jaw_reference,
            tcp_reference_base_m=base_rotation.T@(self.controller.reference-base_position),
            ik_fraction=self.controller.last_fraction)
        value,self._visual_goal,self._visual_grip,self._visual_hold=advance_visual_observation(
            self.config,sample,visual,goal_base_m=self._visual_goal,previous_time_s=self._last_observation_time,
            grip=self._visual_grip,hold_s=self._visual_hold)
        self._last_observation_time=now
        if self.config.uses_orientation_actions:
            _,rotation=robot.tcp_pose()
            value=np.r_[value,robot.rotation_error(self.grasp_orientation,rotation)/np.pi,
                robot.tcp_angular_velocity()/.5,robot.rotation_error(self.controller.orientation,rotation)/.05].astype(np.float32)
        if value.shape!=self.observation_space.shape or not np.isfinite(value).all():raise ValueError('Invalid camera observation')
        info['vision']=dict(source=self.vision_contract['name'],object_position_base_m=visual['position_m'].tolist(),
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
