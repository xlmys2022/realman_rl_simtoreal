"""定义桌面笛卡尔抓取的观察位、动作与传感器参数，以及课程完成门槛。"""
from dataclasses import asdict, dataclass, replace
import math
from ...simulation.runtime.observation_presets import OBSERVATION_POSES


@dataclass(frozen=True)
class CartesianTabletopConfig:
    """不可变的笛卡尔抓取配置，供运行时、环境和 checkpoint 重建使用。

    位置、开口和区间采用 m，旋转增量采用 rad。相机分辨率用像素，视场角用度，
    控制周期和保持时间用秒。to_dict 将自动控制模式展开成实际动作契约。
    """
    observation_pose: str = 'table_down'
    control_mode: str = 'auto'
    observation_source: str = 'state'
    vision_width: int = 640
    vision_height: int = 480
    vision_period: float = .1
    vision_timeout: float = .25
    vision_hold_margin: float = 1.
    vision_cube_size: float = .02
    vision_hsv_low: tuple = (3,175,50)
    vision_hsv_high: tuple = (25,255,255)
    # CAD camera_link has no calibrated optical frame. These are explicit
    # simulation conventions: optical +Z parallel to the fingers, local +X up.
    camera_forward_local: tuple = (0., 0., 1.)
    camera_up_local: tuple = (1., 0., 0.)
    camera_fovy_deg: float = 90.
    object_xy: tuple = (.4, 0.)
    object_radius: float = 0.
    lift_height: float = .03
    success_height: float | None = None
    height_tolerance: float = .002
    hold_seconds: float = 2.
    # Leave time to refine alignment, close, lift and hold after the long reach.
    max_steps: int = 1600
    action_scale: float = .003
    rotation_action_scale: float = math.pi / 60
    max_rotation_lag: float = math.pi / 60
    axis_scale: tuple = (1., 1., 1.)
    control_period: float = .05
    physics_timestep: float = .001
    joint_step: float = math.pi / 360
    cartesian_smoothing: float = .3
    max_reference_lag: float = .005
    initial_width: float = .06
    jaw_margin: float = .002
    jaw_control: str = 'absolute'
    jaw_delta_scale: float = .002
    cache_startup: bool = False
    # TCP clearance for the complete open/closed gripper envelope above the desk.
    workspace_low: tuple = (-.15, -.3, .755)
    workspace_high: tuple = (.65, .3, 1.5)

    def __post_init__(self):
        import numpy as np
        if self.observation_source not in ('state','rgbd_hsv'):raise ValueError('Select state or rgbd_hsv observations')
        if (not isinstance(self.vision_width,int) or not isinstance(self.vision_height,int)
                or not 160<=self.vision_width<=1280 or not 120<=self.vision_height<=960
                or not .05<=self.vision_period<=.2 or not self.vision_period<=self.vision_timeout<=.5
                or not .005<=self.vision_cube_size<=.1 or not 0<=self.vision_hold_margin<=2.):raise ValueError('Invalid RGB-D sensor configuration')
        low=np.asarray(self.vision_hsv_low);high=np.asarray(self.vision_hsv_high)
        if (low.shape!=(3,) or high.shape!=(3,) or np.any(low<0) or np.any(high>[179,255,255])
                or np.any(low>high) or np.any(low!=low.astype(int)) or np.any(high!=high.astype(int))):
            raise ValueError('Invalid HSV thresholds')
        if self.observation_pose not in OBSERVATION_POSES:
            raise ValueError('Select a fixed observation pose: table_down or forward')
        if self.control_mode not in ('auto','position','pose'):
            raise ValueError('control_mode must be auto, position or pose')
        if self.jaw_control not in ('absolute','delta') or not 0<self.jaw_delta_scale<=.003:
            raise ValueError('Select absolute or bounded incremental jaw control')
        if not (0 < self.rotation_action_scale <= math.pi/30 and 0 < self.max_rotation_lag <= math.pi/30):
            raise ValueError('Invalid Cartesian rotation bounds')
        if not (.001 <= self.action_scale <= .005 and 0 < self.cartesian_smoothing <= 1
                and .001 <= self.max_reference_lag <= .015):
            raise ValueError("Invalid Cartesian control profile")
        if self.control_period != .05 or self.physics_timestep != .001:
            raise ValueError("Protected control cycle must contain 50 one-ms physics steps")
        if not (0 < self.joint_step <= math.pi/360 and .021 <= self.initial_width <= .064
                and .001 <= self.jaw_margin <= .005):
            raise ValueError("Invalid actuator bounds")
        required=self.lift_height if self.success_height is None else self.success_height
        if not (.005 <= required <= self.lift_height <= .08 and 0 < self.height_tolerance < required/2
                and .1 <= self.hold_seconds <= 5 and 0 <= self.object_radius <= .05):
            raise ValueError("Invalid task parameters")
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise ValueError("A positive episode step budget is required")
        vectors = (self.observation_tcp,self.object_xy,self.camera_forward_local,self.camera_up_local,self.axis_scale,
                   self.workspace_low,self.workspace_high)
        if not all(np.isfinite(v).all() for v in vectors):raise ValueError("Finite geometry required")
        if (len(self.observation_tcp)!=3 or len(self.object_xy)!=2 or
                len(self.camera_forward_local)!=3 or len(self.camera_up_local)!=3 or
                len(self.workspace_low)!=3 or len(self.workspace_high)!=3 or len(self.axis_scale)!=3):
            raise ValueError("Invalid geometry dimensions")
        if np.any(np.asarray(self.axis_scale)<=0) or np.any(np.asarray(self.axis_scale)>1):
            raise ValueError("Cartesian curriculum axis scales must lie in (0,1]")
        low=np.asarray(self.workspace_low);high=np.asarray(self.workspace_high)
        if np.any(low>=high) or np.any(np.asarray(self.observation_tcp)<low) or np.any(np.asarray(self.observation_tcp)>high):
            raise ValueError("The observation TCP must be inside ordered workspace bounds")
        if not isinstance(self.cache_startup,bool):raise ValueError("cache_startup must be an explicit boolean")
        forward=np.asarray(self.camera_forward_local,dtype=float)
        up=np.asarray(self.camera_up_local,dtype=float)
        if not (np.isclose(np.linalg.norm(forward),1) and np.isclose(np.linalg.norm(up),1)
                and abs(forward@up)<1e-8):raise ValueError("Camera axes must be orthonormal")
        if not 30 <= self.camera_fovy_deg <= 120:raise ValueError("Invalid simulated camera field of view")

    @property
    def observation_tcp(self):
        return OBSERVATION_POSES[self.observation_pose].tcp_m

    @property
    def uses_orientation_actions(self):
        return self.control_mode == 'pose' or (self.control_mode == 'auto' and self.observation_pose == 'forward')

    def to_dict(self):
        values=asdict(self)
        values['control_mode']='pose' if self.uses_orientation_actions else 'position'
        return values


# Curriculum changes completion criteria only; startup and action authority are unchanged.
EASY=dict(success_height=.005,height_tolerance=.0005,hold_seconds=.2)
STAGES=dict(easy=dict(EASY),pinch=dict(EASY),approach=dict(EASY),lift=dict(),full=dict())


def environment_config(stage,*,observation_pose='table_down',control_mode='auto',cache_startup=False,
                       jaw_control='absolute',jaw_delta_scale=.002,observation_source='state'):
    """应用课程门槛及显式控制选项，校验观察位与动作模式是否兼容。"""
    cfg=replace(CartesianTabletopConfig(observation_pose=observation_pose,control_mode=control_mode,
                       jaw_control=jaw_control,jaw_delta_scale=jaw_delta_scale,observation_source=observation_source),
                **STAGES[stage],cache_startup=cache_startup)
    if cfg.observation_pose=='forward' and not cfg.uses_orientation_actions:
        raise ValueError('A forward tabletop grasp start requires pose control so PPO can turn the gripper')
    return cfg
