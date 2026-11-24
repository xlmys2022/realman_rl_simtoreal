"""定义 MuJoCo 相机朝向的几何约定，供桌面运行时设置腕部相机姿态。"""
import numpy as np


def camera_rotation(forward, up):
    """输入刚体坐标系的前向和上向，返回相机到刚体的旋转矩阵，MuJoCo 视线沿局部 −Z。"""
    forward=np.asarray(forward,dtype=float)
    up=np.asarray(up,dtype=float)
    return np.column_stack([np.cross(forward,up),up,-forward])
