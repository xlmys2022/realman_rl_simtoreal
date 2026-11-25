"""低速直线调试所需的现场安全边界。"""

from dataclasses import dataclass

import numpy as np

from ..utils.geometry import validate_transform
from . import HardwareError


class MotionSafetyError(HardwareError):
    """运动安全检查拦截或停止。"""


@dataclass(frozen=True)
class MotionLimits:
    """显式提供基座系 TCP 工作空间、速度、单次位移、时限及实测 TCP。"""

    workspace_min_m: tuple[float, float, float]
    workspace_max_m: tuple[float, float, float]
    T_flange_tcp: np.ndarray
    max_speed_pct: int
    max_linear_speed_m_s: float
    max_step_m: float
    completion_timeout_s: float

    def __post_init__(self):
        low = np.asarray(self.workspace_min_m, dtype=float)
        high = np.asarray(self.workspace_max_m, dtype=float)
        if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
                or np.any(low >= high)):
            raise ValueError("工作空间需要三轴有限且有序的基座系边界（m）")
        if type(self.max_speed_pct) is not int or not 1 <= self.max_speed_pct <= 100:
            raise ValueError("max_speed_pct 必须为 1–100 的整数")
        for name in ("max_linear_speed_m_s", "max_step_m", "completion_timeout_s"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        object.__setattr__(self, "workspace_min_m", tuple(low))
        object.__setattr__(self, "workspace_max_m", tuple(high))
        object.__setattr__(self, "T_flange_tcp", validate_transform(self.T_flange_tcp, "T_flange_tcp"))

    def require_in_workspace(self, position_m, label):
        xyz = np.array(position_m, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise MotionSafetyError(f"{label} 必须为有限的基座系 XYZ（m）")
        if np.any(xyz < self.workspace_min_m) or np.any(xyz > self.workspace_max_m):
            raise MotionSafetyError(f"{label} 超出 TCP 工作空间 · {xyz.tolist()}")
        return xyz

    def require_speed(self, speed_pct, controller_max_m_s):
        if type(speed_pct) is not int or not 1 <= speed_pct <= self.max_speed_pct:
            raise MotionSafetyError(f"速度百分比必须为 1–{self.max_speed_pct} 的整数")
        if not np.isfinite(controller_max_m_s) or controller_max_m_s <= 0:
            raise MotionSafetyError("控制器末端线速度上限无效")
        if controller_max_m_s * speed_pct / 100 > self.max_linear_speed_m_s:
            raise MotionSafetyError("按控制器读回的线速度上限换算后，指令超出现场限速")
