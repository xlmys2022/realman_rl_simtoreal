"""加载本地实测标定。变换 T_A_B 将 B 系的坐标变换到 A 系。"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np

from ..paths import PROJECT_ROOT
from ..utils.artifacts import file_sha256
from ..utils.geometry import validate_transform


EXAMPLE_CALIBRATION = PROJECT_ROOT / "assets/calibration/realman/calibration.example.json"
LEGACY_CALIBRATION = EXAMPLE_CALIBRATION.with_name("calibration.local.json")
DEFAULT_CALIBRATION = EXAMPLE_CALIBRATION.with_name("calibration.20261127.json")
HsvTriplet = tuple[int, int, int]
HsvRange = tuple[HsvTriplet, HsvTriplet]


class CalibrationError(ValueError):
    """文件缺失或标定参数无效。"""


@dataclass(frozen=True)
class RealCalibration:
    path: Path
    hsv_path: Path
    hsv_ranges: tuple[HsvRange, ...]
    camera_parent: Literal["base", "flange", "tcp"]
    T_parent_color: np.ndarray
    T_flange_tcp: np.ndarray
    cube_edge_m: float
    camera_serial: str | None
    base_leveling: np.ndarray | None = None

    def checksum(self):
        """返回当前标定文件的内容校验值。"""
        return file_sha256(self.path)

    def camera_to_flange(self):
        """返回相机的实测安装变换，供仿真等需使用法兰外参的场景调用。"""
        if self.camera_parent == "base":
            raise CalibrationError("固定相机没有法兰外参")
        if self.camera_parent == "tcp":
            return self.T_flange_tcp @ self.T_parent_color
        return self.T_parent_color

    def camera_to_base(self, T_base_flange=None):
        """将彩色光学系变换到基座系，按现场标定校正视觉重建的水平基准。"""
        if self.camera_parent == "base":
            optical = self.T_parent_color
        else:
            if T_base_flange is None:
                raise CalibrationError("腕部相机定位需要同一时段的实机 T_base_flange")
            optical = T_base_flange @ self.camera_to_flange()
        return optical if self.base_leveling is None else self.base_leveling @ optical


def _read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"无法读取标定文件 {path} · {exc}") from exc
    if not isinstance(data, dict):
        raise CalibrationError(f"{path} 的顶层必须是 JSON 对象")
    return data


def load_hsv(path):
    """读取现场 OpenCV HSV 区间并校验范围。"""
    path = Path(path)
    data = _read_json(path)
    if "hsv_ranges" in data:
        ranges = data["hsv_ranges"]
    else:
        ranges = [{"low": data.get("vision_hsv_low"), "high": data.get("vision_hsv_high")}]
    if not isinstance(ranges, list) or not ranges:
        raise CalibrationError(f"{path} 的 hsv_ranges 必须包含至少一个 HSV 区间")

    parsed = []
    for index, bounds in enumerate(ranges):
        if not isinstance(bounds, dict):
            raise CalibrationError(f"{path} 的 hsv_ranges[{index}] 必须包含 low 和 high")
        low, high = bounds.get("low"), bounds.get("high")
        if (not all(isinstance(v, list) and len(v) == 3 for v in (low, high))
                or not all(type(v) is int for v in low + high)
                or not all(0 <= a <= b <= limit for a, b, limit in zip(low, high, (179, 255, 255)))):
            raise CalibrationError(f"{path} 的 hsv_ranges[{index}] 需要有效的 OpenCV HSV low/high")
        parsed.append((tuple(low), tuple(high)))
    return tuple(parsed)


def load_calibration(path=DEFAULT_CALIBRATION):
    """加载 HSV、手眼、TCP 和方块尺寸，所有实测量均须显式提供。"""
    path = Path(path).expanduser().resolve()
    if path == DEFAULT_CALIBRATION and not path.is_file():
        raise CalibrationError(f"缺少真机标定配置 {path}。可复制模板 {EXAMPLE_CALIBRATION}")
    data = _read_json(path)
    hand_eye = data.get("hand_eye")
    if hand_eye is not None and not isinstance(hand_eye, dict):
        raise CalibrationError(f"{path} 的 hand_eye 必须是 JSON 对象")
    hand_eye = hand_eye or {}
    required = {
        "hsv_file": data.get("hsv_file"),
        "hand_eye.parent_frame": hand_eye.get("parent_frame"),
        "hand_eye.T_parent_color": hand_eye.get("T_parent_color"),
        "T_flange_tcp": data.get("T_flange_tcp"),
        "cube_edge_m": data.get("cube_edge_m"),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise CalibrationError(f"{path} 缺少真机标定参数 · {', '.join(missing)}")
    if not isinstance(data["hsv_file"], str) or not data["hsv_file"].strip():
        raise CalibrationError(f"{path} 的 hsv_file 必须是 HSV JSON 的路径")
    hsv_path = (path.parent / data["hsv_file"]).resolve()
    parent = hand_eye["parent_frame"]
    if parent not in ("base", "flange", "tcp"):
        raise CalibrationError(f"{path} 的 hand_eye.parent_frame 必须为 base、flange 或 tcp")
    edge = data["cube_edge_m"]
    if type(edge) not in (int, float) or not 0.005 <= edge <= 0.1:
        raise CalibrationError(f"{path} 的 cube_edge_m 必须为检测器支持的 0.005–0.100 m 实测边长")
    serial = data.get("camera_serial")
    if serial is not None and (not isinstance(serial, str) or not serial.strip()):
        raise CalibrationError(f"{path} 的 camera_serial 必须为相机序列号字符串")
    try:
        hand_eye_transform = validate_transform(hand_eye["T_parent_color"], "hand_eye.T_parent_color")
        tcp_transform = validate_transform(data["T_flange_tcp"], "T_flange_tcp")
        leveling = data.get("base_leveling")
        if leveling is not None:
            leveling = validate_transform(leveling, "base_leveling")
    except ValueError as exc:
        raise CalibrationError(f"{path} · {exc}") from exc
    return RealCalibration(path, hsv_path, load_hsv(hsv_path), parent,
                           hand_eye_transform, tcp_transform, float(edge), serial, leveling)
