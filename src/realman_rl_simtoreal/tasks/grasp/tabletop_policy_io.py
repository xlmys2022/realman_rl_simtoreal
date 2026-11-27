"""桌面策略的无设备编解码。米制本体数据与 RGB-D 跟踪状态都由调用端提供。"""

from dataclasses import dataclass

import numpy as np

from ...deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from ...hardware.realman_config import REALMAN_TABLETOP
from ...simulation.motion.cartesian_request import CartesianActionRequest, limit_tcp_request
from ...simulation.runtime.observation_presets import REALMAN_TRANSFER_PROFILE


# v4 checkpoint 固定的开口归一化尺度，取自原仿真夹爪可达间距。
JAW_WIDTH_SCALE_M = .06621735660659232
ACTION_CONTRACT = dict(name='world_tcp_delta_and_absolute_jaw_v1', shape=4,
                       components=['delta_x', 'delta_y', 'delta_z', 'absolute_gripper_opening'])
POSE_ACTION_CONTRACT = dict(name='world_tcp_pose_delta_and_absolute_jaw_v1', shape=7,
                            components=['delta_x', 'delta_y', 'delta_z', 'delta_rx', 'delta_ry', 'delta_rz',
                                        'absolute_gripper_opening'])
DELTA_JAW_ACTION_CONTRACT = dict(name='world_tcp_delta_and_incremental_jaw_v1', shape=4,
                                components=['delta_x', 'delta_y', 'delta_z', 'delta_gripper_opening'])
POSE_DELTA_JAW_ACTION_CONTRACT = dict(name='world_tcp_pose_delta_and_incremental_jaw_v1', shape=7,
                                     components=['delta_x', 'delta_y', 'delta_z', 'delta_rx', 'delta_ry', 'delta_rz',
                                                 'delta_gripper_opening'])
VISION_OBSERVATION_FIELDS = (
    ('tcp_base_relative', 3), ('tcp_velocity', 3), ('rgbd_object_tcp_relative', 3), ('rgbd_goal_object_relative', 3),
    ('rgbd_object_velocity', 3), ('rgbd_surface_normal', 3), ('gripper_width_velocity_reference', 3),
    ('visual_grip_confidence', 1), ('rgbd_detection_confidence', 1), ('rgbd_measurement_age', 1),
    ('visual_hold_progress', 1), ('cartesian_tracking_error', 3), ('ik_fraction', 1),
)
VISION_CONTRACT = dict(name='wrist_rgbd_hsv_cube_v4',
    object_source='aligned_rgb_depth_and_calibration', mask='opencv_rgb_to_hsv_inrange',
    geometry_prior='upright_cube_with_configured_edge_length',
    object_pose_truth=False, object_velocity_truth=False, contact_force_observations=False,
    privileged_hold_progress=False, completion_source='camera_visual_hold_with_settling_margin',
    neural_control='zero_anchored_cartesian_network', privileged_reward_and_success_audit=True)


def vision_contract(config):
    if config.scene_profile != REALMAN_TRANSFER_PROFILE:
        return VISION_CONTRACT
    calibration = load_calibration(config.hand_eye_file or DEFAULT_CALIBRATION)
    return dict(VISION_CONTRACT, name='realman_d435_upright_cube_v1',
                encoder=VISION_CONTRACT['name'], estimator='upright_top_surface',
                simulated_depth_unit_m=REALMAN_TABLETOP.depth_unit_m, calibration_sha256=calibration.checksum())


def action_contract(config):
    """返回 checkpoint 使用的动作维度和语义。"""
    if config.jaw_control == 'delta':
        contract = POSE_DELTA_JAW_ACTION_CONTRACT if config.uses_orientation_actions else DELTA_JAW_ACTION_CONTRACT
    else:
        contract = POSE_ACTION_CONTRACT if config.uses_orientation_actions else ACTION_CONTRACT
    if config.scene_profile != REALMAN_TRANSFER_PROFILE:
        return contract
    return dict(contract, name=contract['name'].replace('world_', 'base_', 1),
                translation_frame='Base', control_period_s=config.control_period,
                max_request_speed_m_s=config.tcp_request_speed_limit_m_s,
                jaw_width_scale_m=JAW_WIDTH_SCALE_M)


def decode_cartesian_action(action, config, previous_jaw_reference_m):
    """解码归一化策略输出。增量夹爪模式要求提供上一开口参考的米值。"""
    action = np.asarray(action, dtype=np.float32)
    shape = (7 if config.uses_orientation_actions else 4,)
    if action.shape != shape or not np.isfinite(action).all() or np.any(abs(action) > 1):
        raise ValueError('A finite normalized Cartesian action matching the selected control mode is required')
    if config.jaw_control == 'delta':
        if previous_jaw_reference_m is None or not np.isfinite(previous_jaw_reference_m) or previous_jaw_reference_m < 0:
            raise ValueError('Incremental jaw control requires an explicit previous reference in metres')
        width = float(np.clip(previous_jaw_reference_m + action[-1] * config.jaw_delta_scale,
                              config.jaw_margin, JAW_WIDTH_SCALE_M - config.jaw_margin))
    else:
        width = float((action[-1] + 1) * JAW_WIDTH_SCALE_M / 2)
    nominal = action[:3] * config.action_scale * np.asarray(config.axis_scale)
    translation = limit_tcp_request(nominal, speed_limit_m_s=config.tcp_request_speed_limit_m_s,
                                    control_period_s=config.control_period)
    rotation = action[3:6] * config.rotation_action_scale if config.uses_orientation_actions else None
    return CartesianActionRequest(translation, width, rotation)


def visual_grip_and_hold(visual, offset, goal_error, *, width, jaw_reference, cube_size,
                         timeout, previous_grip, previous_hold, dt):
    """由视觉估计和本体量更新夹持置信度与保持时间（秒）。"""
    enclosure = np.exp(-(np.linalg.norm(offset[:2])/.006)**2-(max(0., abs(offset[2]+.003)-.012)/.01)**2)
    aperture = np.exp(-((width-cube_size)/.004)**2)
    compression = np.clip((width-jaw_reference-.010)/.010, 0, 1)
    measured_grip = float(enclosure*aperture*compression*visual['confidence'])
    alpha = 1-np.exp(-dt/.08) if dt else 1.
    grip = previous_grip
    grip += alpha*(measured_grip-grip)
    stable = (visual['visible'] and visual['age'] <= timeout
              and np.linalg.norm(goal_error[:2]) < .003 and abs(goal_error[2]) < .001
              and np.linalg.norm(visual['velocity_m_s']) < .0015 and grip > .5)
    hold = previous_hold+dt if stable else 0.
    return grip, hold


def encode_visual_observation(tcp, tcp_velocity, offset, goal_error, visual, *, width, width_speed,
                              jaw_reference, grip, hold, timeout, hold_required, tracking_error, ik_fraction):
    """按既有顺序和尺度编码 29 个视觉控制通道，输出 float32。"""
    return np.concatenate([tcp/.5, tcp_velocity/.05,
        np.arcsinh(offset/.01), np.arcsinh(goal_error/.01), visual['velocity_m_s']/.05, visual['normal'],
        [width/JAW_WIDTH_SCALE_M, width_speed/.04, jaw_reference/JAW_WIDTH_SCALE_M],
        [grip, visual['confidence'], min(visual['age']/timeout, 2.)],
        [min(hold/hold_required, 1.)], tracking_error/.01, [ik_fraction]]).astype(np.float32)


@dataclass(frozen=True)
class VisualPolicyProprioception:
    """同一控制时刻的实测 TCP、开口和控制器参考。相机与本体时间戳须同域。"""

    timestamp_s: float
    tcp_base_m: np.ndarray
    tcp_velocity_base_m_s: np.ndarray
    gripper_width_m: float
    gripper_width_velocity_m_s: float
    jaw_reference_m: float
    tcp_reference_base_m: np.ndarray
    ik_fraction: float

    def __post_init__(self):
        for name in ('tcp_base_m', 'tcp_velocity_base_m_s', 'tcp_reference_base_m'):
            vector = np.asarray(getattr(self, name), dtype=float)
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise ValueError(f'{name} requires a measured or controller-reported finite Base vector')
            object.__setattr__(self, name, vector)
        for name in ('timestamp_s', 'gripper_width_m', 'gripper_width_velocity_m_s', 'jaw_reference_m', 'ik_fraction'):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != () or not np.isfinite(value):
                raise ValueError(f'{name} requires a finite value in the agreed SI units')
            object.__setattr__(self, name, value.item())
        # 物理仿真可能读到受压的微小有符号间距，参考值仍必须是非负米值。
        if self.jaw_reference_m < 0 or not 0 <= self.ik_fraction <= 1:
            raise ValueError('Jaw reference must be non-negative and the IK fraction must be in [0, 1]')


class VisualPolicyObserver:
    """仿真与真机共用一次抓取的首帧目标、夹持置信度及视觉保持进度。"""

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.goal_base_m = None
        self.last_timestamp_s = None
        self.grip = 0.
        self.hold_s = 0.

    def observe(self, sample: VisualPolicyProprioception, visual):
        value, self.goal_base_m, self.grip, self.hold_s = advance_visual_observation(
            self.config, sample, visual, goal_base_m=self.goal_base_m, previous_time_s=self.last_timestamp_s,
            grip=self.grip, hold_s=self.hold_s)
        self.last_timestamp_s = sample.timestamp_s
        return value


def advance_visual_observation(config, sample, visual, *, goal_base_m, previous_time_s, grip, hold_s):
    """共用两端的首帧目标、视觉保持逻辑和编码，回传本周期的观测与状态。"""
    dt = 0. if previous_time_s is None else sample.timestamp_s - previous_time_s
    if dt < 0:
        raise ValueError('Policy observation timestamps must be monotonic')
    position = visual['position_m']
    if goal_base_m is None:
        goal_base_m = position + np.array([0., 0., config.lift_height])
    offset = position - sample.tcp_base_m
    goal_error = goal_base_m - position
    grip, hold_s = visual_grip_and_hold(visual, offset, goal_error,
        width=sample.gripper_width_m, jaw_reference=sample.jaw_reference_m,
        cube_size=config.vision_cube_size, timeout=config.vision_timeout,
        previous_grip=grip, previous_hold=hold_s, dt=dt)
    value = encode_visual_observation(sample.tcp_base_m, sample.tcp_velocity_base_m_s, offset, goal_error, visual,
        width=sample.gripper_width_m, width_speed=sample.gripper_width_velocity_m_s,
        jaw_reference=sample.jaw_reference_m, grip=grip, hold=hold_s, timeout=config.vision_timeout,
        hold_required=config.hold_seconds+config.vision_hold_margin,
        tracking_error=sample.tcp_reference_base_m-sample.tcp_base_m, ik_fraction=sample.ik_fraction)
    return value, goal_base_m, grip, hold_s
