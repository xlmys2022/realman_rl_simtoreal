"""固定法兰朝向的末端控制演示与对照实验，组织 IK 预检、插值和稳定性验收。"""

import mujoco
import numpy as np

from ..physics import joint_control as control
from .kinematics import rotation_vector

MAX_TRANSLATION = 0.010
MAX_ACTUAL_POSITION_ERROR = 0.005
MAX_ACTUAL_ORIENTATION_ERROR = np.deg2rad(1.0)

def check_motion(model, data, joint_ids, actuator_ids, kin, result, target, *,
                 max_translation=MAX_TRANSLATION, max_home_delta=np.deg2rad(5)):
    """在独立计算数据中预检 IK、关节限位和插值碰撞。

    target 为世界系位置（m），result.q 为 rad。返回目标 FK 位置、旋转和路径最大姿态偏差（rad）。
    """
    control.require(result.success and result.q is not None, f"IK failed; no motion: {result.reason}")
    target = np.asarray(target, dtype=float)
    control.require(target.shape == (3,) and np.isfinite(target).all(), "Expected a finite XYZ target")
    control.require(np.isfinite(max_translation) and max_translation > 0, "Invalid translation envelope")
    control.require(np.linalg.norm(target - kin.home_position) <= max_translation + 1e-12,
                    f"Demo translation exceeds {1000 * max_translation:g} mm from home")
    q = control.validate_target(model, joint_ids, kin.home_q, result.q, max_home_delta=max_home_delta)
    theoretical_position, theoretical_rotation = kin.fk(q)
    control.require(np.linalg.norm(theoretical_position - target) <= 1e-5
                    and np.linalg.norm(rotation_vector(kin.home_rotation, theoretical_rotation)) <= 1e-4,
                    "IK result does not satisfy the requested fixed-orientation pose")
    # 在独立数据中离散抽检关节插值，执行时仍需检查真实积分后的碰撞。
    probe = mujoco.MjData(model)
    probe.qpos[:] = data.qpos  # Preserve accessory configuration in combined models.
    start = data.qpos[kin.qadr].copy()
    max_rotation_drift = 0.0
    samples = max(51, int(np.ceil(np.max(np.abs(q - start)) / np.deg2rad(1))) + 1)
    for alpha in np.linspace(0, 1, samples):
        angles = start + alpha * (q - start)
        probe.qpos[kin.qadr] = angles
        mujoco.mj_forward(model, probe)
        for contact in probe.contact:
            control.require(contact.dist >= -0.0001, "Joint interpolation has an unexpected collision")
        _, rotation = kin.fk(angles)
        max_rotation_drift = max(max_rotation_drift, float(np.linalg.norm(rotation_vector(kin.home_rotation, rotation))))
    return theoretical_position, theoretical_rotation, max_rotation_drift


def execute_motion(model, data, joint_ids, actuator_ids, kin, result, target, tick=None, step_observer=None):
    """先预检再驱动关节位置执行器，返回理论与实际末端位姿及追踪误差。"""
    theory_p, theory_r, command_rotation_drift = check_motion(model, data, joint_ids, actuator_ids, kin, result, target)
    mujoco.mj_forward(model, data)
    initial_position = data.site_xpos[kin.site_id].copy()
    max_orientation_error, min_distance = 0.0, 0.0

    def sample():
        nonlocal max_orientation_error, min_distance
        # mj_step integrates qpos; refresh site poses/contacts at that new state.
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[kin.site_id].reshape(3, 3)
        angle = float(np.linalg.norm(rotation_vector(kin.home_rotation, rotation)))
        max_orientation_error = max(max_orientation_error, angle)
        control.require(angle <= MAX_ACTUAL_ORIENTATION_ERROR, "Actual orientation exceeded the 1 degree demo envelope")
        for contact in data.contact:
            min_distance = min(min_distance, float(contact.dist))
            control.require(contact.dist >= -0.0001, "Unexpected collision after integration")
        if tick is not None:
            tick()

    hold = control.run_segment(model, data, joint_ids, actuator_ids, kin.home_q, "settle home", ramp=0, hold=3, tick=sample, step_observer=step_observer)
    motion_start = data.site_xpos[kin.site_id].copy()
    motion = control.run_segment(model, data, joint_ids, actuator_ids, result.q, "end-effector target", ramp=2, hold=3, tick=sample, step_observer=step_observer)
    actual_position = data.site_xpos[kin.site_id].copy()
    actual_rotation = data.site_xmat[kin.site_id].reshape(3, 3).copy()
    position_error = float(np.linalg.norm(np.asarray(target) - actual_position))
    orientation_error = float(np.linalg.norm(rotation_vector(kin.home_rotation, actual_rotation)))
    control.require(position_error <= MAX_ACTUAL_POSITION_ERROR, "Actual end-effector error exceeds 5 mm")
    return dict(initial_position_m=initial_position.tolist(), motion_start_position_m=motion_start.tolist(),
                target_position_m=np.asarray(target).tolist(), target_rotation=kin.home_rotation.tolist(),
                theoretical_position_m=theory_p.tolist(), theoretical_rotation=theory_r.tolist(),
                actual_position_m=actual_position.tolist(), actual_rotation=actual_rotation.tolist(),
                actual_position_error_m=position_error, actual_orientation_error_rad=orientation_error,
                max_actual_orientation_error_rad=max_orientation_error,
                max_command_interpolation_rotation_error_rad=command_rotation_drift,
                min_contact_distance_m=min_distance, joint_target_rad=result.q.tolist(),
                joint_actual_rad=data.qpos[kin.qadr].tolist(), simulation_time_s=float(data.time),
                hold=hold, motion=motion)

