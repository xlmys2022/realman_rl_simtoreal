"""RGB-D policy observations, with privileged physics used only for reward/audit.

The observation builder ignores object fields in simulator debug info. Its
only object inputs are the RGB-D detector/tracker and a goal initialized from
the first camera measurement at the fixed observation pose.
"""
import copy
import numpy as np
from .tabletop_cartesian_env import TabletopCartesianEnv
from ..perception.rgbd import OrangeCubeDetector,VisualTracker
from ..simulation.rgbd_camera import MujocoRgbdCamera
from ..simulation.gripper import MAX_WIDTH
from ..tasks.tabletop_cartesian_reward import calculate

VISION_CONTRACT=dict(name='wrist_rgbd_hsv_cube_v4',
    object_source='aligned_rgb_depth_and_calibration',mask='opencv_rgb_to_hsv_inrange',
    geometry_prior='upright_cube_with_configured_edge_length',
    object_pose_truth=False,object_velocity_truth=False,contact_force_observations=False,
    privileged_hold_progress=False,completion_source='camera_visual_hold_with_settling_margin',
    neural_control='zero_anchored_cartesian_network',
    privileged_reward_and_success_audit=True)


class VisionTabletopCartesianEnv(TabletopCartesianEnv):
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
        # Do not read info['object_*'], true contact forces or true hold ticks.
        # Joint encoders / FK, commanded jaw width and robot tracking are
        # proprioception; all target state below comes from camera arrays.
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
        enclosure=np.exp(-(np.linalg.norm(offset[:2])/.006)**2-(max(0.,abs(offset[2]+.003)-.012)/.01)**2)
        aperture=np.exp(-((width-self.config.vision_cube_size)/.004)**2)
        compression=np.clip((width-self._jaw_reference-.010)/.010,0,1)
        measured_grip=float(enclosure*aperture*compression*visual['confidence'])
        dt=0. if self._last_observation_time is None else max(0.,now-self._last_observation_time)
        self._last_observation_time=now
        alpha=1-np.exp(-dt/.08) if dt else 1.
        self._visual_grip+=alpha*(measured_grip-self._visual_grip)
        visual_stable=(visual['visible'] and visual['age']<=self.config.vision_timeout
            and np.linalg.norm(goal_error[:2])<.003 and abs(goal_error[2])<.001
            and np.linalg.norm(visual['velocity_m_s'])<.0015 and self._visual_grip>.5)
        self._visual_hold=self._visual_hold+dt if visual_stable else 0.
        value=np.concatenate([tcp/.5,tcp_velocity/.05,
            np.arcsinh(offset/.01),np.arcsinh(goal_error/.01),visual['velocity_m_s']/.05,visual['normal'],
            [width/MAX_WIDTH,width_speed/.04,self._jaw_reference/MAX_WIDTH],
            [self._visual_grip,visual['confidence'],min(visual['age']/self.config.vision_timeout,2.)],
            [min(self._visual_hold/(self.config.hold_seconds+self.config.vision_hold_margin),1.)],
            (base_rotation.T@(self.controller.reference-tcp_world))/.01,[self.controller.last_fraction]]).astype(np.float32)
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
