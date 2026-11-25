"""定义 MuJoCo 相机朝向的几何约定，供桌面运行时设置腕部相机姿态。"""
import numpy as np


OPTICAL_TO_MUJOCO = np.diag([1., -1., -1.])


def camera_rotation(forward, up):
    """输入刚体坐标系的前向和上向，返回相机到刚体的旋转矩阵，MuJoCo 视线沿局部 −Z。"""
    forward=np.asarray(forward,dtype=float)
    up=np.asarray(up,dtype=float)
    return np.column_stack([np.cross(forward,up),up,-forward])


def intrinsic_matrix(model, camera_id, width, height):
    """把 MJCF 传感器单位和主点偏移换成实际输出的 OpenCV 像素内参。"""
    sensor = model.cam_sensorsize[camera_id]
    if not np.all(sensor > 0):
        f = height / (2 * np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2))
        return np.array([[f, 0, (width - 1) / 2], [0, f, (height - 1) / 2], [0, 0, 1.]])
    fx, fy, px, py = model.cam_intrinsic[camera_id]
    sx, sy = np.array([width, height], dtype=float) / sensor
    return np.array([[fx * sx, 0, (width - 1) / 2 - px * sx],
                     [0, fy * sy, (height - 1) / 2 - py * sy],
                     [0, 0, 1.]])
