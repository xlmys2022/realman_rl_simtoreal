"""从真实接触中提取沿夹持方向的压力，用于桌面任务的双侧夹持和支撑判定。"""
import numpy as np
import mujoco

CONTACT_MEASUREMENT = 'inward_pad_normal_projection_v1'


def inward_normal(contact, normal_force, target_geom, inward_axis):
    """按目标几何体在接触对中的次序定向，再取沿夹指内向轴的正向法向分量（N）。"""
    sign = 1. if int(contact.geom[1]) == target_geom else -1.
    normal = np.asarray(contact.frame).reshape(3, 3)[0]
    return max(0., float(sign * normal_force * (normal @ inward_axis)))


def grasp_forces(model, data, gripper, policy, recorder):
    """返回左、右夹指向内压力和桌面支撑法向力，单位为 N。"""
    row = recorder.last
    result = np.array([0., 0., row['support_normal_n']])
    # 近乎闭合的夹指无法跨住立方体，顶部相碰时只保留支撑读数。
    if row['actual_width_m'] < .015:
        return result
    axis = recorder.probe.xmat[gripper.base].reshape(3, 3)[:, 0]
    for index, contact in enumerate(data.contact):
        category = policy.classify(*contact.geom)
        if category not in ('left_pad_target', 'right_pad_target'):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        side = 0 if category == 'left_pad_target' else 1
        result[side] += inward_normal(contact, force[0], policy.target, axis if side == 0 else -axis)
    return result
