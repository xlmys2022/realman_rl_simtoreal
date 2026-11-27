"""控制边界共用的米制笛卡尔请求及限幅。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianActionRequest:
    """当前周期的平移增量、开口目标和可选旋转增量。迁移场景的增量在 Base 系。"""

    tcp_delta_m: np.ndarray
    jaw_width_m: float
    rotation_delta_rad: np.ndarray | None


def limit_tcp_request(delta_m, *, speed_limit_m_s, control_period_s):
    """限制合成位移，防止三轴同时满幅时请求超过单周期上限。"""
    if speed_limit_m_s is None:
        return delta_m
    limit = speed_limit_m_s * control_period_s
    distance = np.linalg.norm(delta_m)
    return delta_m * (limit / distance) if distance > limit else delta_m
