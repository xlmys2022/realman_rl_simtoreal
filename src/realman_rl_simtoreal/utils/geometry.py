"""SE(3) helpers shared by offline kinematics comparisons."""
import numpy as np


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def validate_transform(matrix, label):
    """返回独立的刚体变换副本，平移使用米。"""
    try:
        result = np.array(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数值 4×4 齐次矩阵") from exc
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{label} 必须是有限的 4×4 齐次矩阵")
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"{label} 必须为右手刚体变换，最后一行应为 [0, 0, 0, 1]")
    result.setflags(write=False)
    return result


def inverse(t):
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ t[:3, 3]
    return result


def euler_matrix(rpy):
    """固定轴 XYZ 欧拉角（弧度），R = Rz @ Ry @ Rx。"""
    values = np.asarray(rpy, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected three finite Euler angles in radians")
    roll, pitch, yaw = values
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


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
