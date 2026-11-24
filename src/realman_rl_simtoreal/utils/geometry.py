"""SE(3) helpers shared by offline kinematics comparisons."""
import numpy as np


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def inverse(t):
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ t[:3, 3]
    return result


def modified_dh_transform(a_m, d_m, alpha_rad, theta_rad):
    """改进 DH 单节从子坐标系到父坐标系的变换。长度为米，角度为弧度。"""
    ca, sa = np.cos(alpha_rad), np.sin(alpha_rad)
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    # Tx(a) @ Rx(alpha) @ Rz(theta) @ Tz(d)。
    rotation = np.array([[ct, -st, 0.],
                         [ca * st, ca * ct, -sa],
                         [sa * st, sa * ct, ca]])
    position = np.array([a_m, 0., 0.]) + rotation[:, 2] * d_m
    return transform(position, rotation)
