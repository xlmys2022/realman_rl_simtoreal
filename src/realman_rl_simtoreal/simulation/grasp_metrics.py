"""Measure opposing inward pad pressure without treating top contact as grip."""
import numpy as np
import mujoco

CONTACT_MEASUREMENT = 'inward_pad_normal_projection_v1'


def inward_normal(contact, normal_force, target_geom, inward_axis):
    """Project normal force on the object onto one pad's inward jaw axis."""
    sign = 1. if int(contact.geom[1]) == target_geom else -1.
    normal = np.asarray(contact.frame).reshape(3, 3)[0]
    return max(0., float(sign * normal_force * (normal @ inward_axis)))


def grasp_forces(model, data, gripper, policy, recorder):
    row = recorder.last
    result = np.array([0., 0., row['support_normal_n']])
    # A nearly closed empty jaw touching the cube's top cannot straddle it.
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
