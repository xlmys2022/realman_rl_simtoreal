"""复用 RGB-D 感知，以实测标定输出基座系方块中心。"""

from dataclasses import dataclass
from math import isfinite
from time import monotonic

import numpy as np

from ..hardware import HardwareError
from ..hardware.realsense import RgbdSample
from ..perception.rgbd import OrangeCubeDetector, UprightCubeDetector, VisualTracker
from .calibration import RealCalibration


@dataclass(frozen=True)
class CubeObservation:
    sample: RgbdSample
    T_base_color: np.ndarray
    mask: np.ndarray
    tracking: dict | None
    arm_timestamps_s: tuple[float, float] | None
    T_base_flange: np.ndarray | None = None
    T_base_tcp: np.ndarray | None = None
    controller_pose_m_rad: np.ndarray | None = None


class RealCubeLocalizer:
    """静止腕部或固定外置相机的只读定位。状态失效时清空跟踪。"""

    def __init__(self, calibration: RealCalibration, *, max_pose_window_s=0.25, timeout_s=0.25,
                 estimator="upright"):
        if not isfinite(max_pose_window_s) or max_pose_window_s <= 0:
            raise ValueError("机械臂与相机的最大配对时间窗必须为有限正数")
        if not isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("视觉时限必须为有限正数")
        self.calibration = calibration
        self.max_pose_window_s = max_pose_window_s
        if estimator not in ("upright", "legacy"):
            raise ValueError("定位算法必须为 upright 或 legacy")
        detector = UprightCubeDetector if estimator == "upright" else OrangeCubeDetector
        self.detector = detector(edge_m=calibration.cube_edge_m, hsv_ranges=calibration.hsv_ranges)
        self.tracker = VisualTracker(self.detector, timeout=timeout_s)

    def _capture_wrist(self, camera, arm):
        if arm is None:
            raise HardwareError("腕部定位需要只读连接机械臂")
        before = arm.read_state()
        if any(before.error_codes):
            raise HardwareError(f"采集前机械臂报错 · {before.error_codes}")
        sample = camera.capture(fresh=True)
        # 主机同域的曝光时间可用于检查 SDK 中途返回的旧帧。
        while (sample.timing.host_capture_interval_s is not None
               and sample.timing.host_capture_interval_s[0] < before.request_end_s
               and sample.timing.host_received_s - before.request_start_s < self.max_pose_window_s):
            sample = camera.capture()
        after = arm.read_state()
        if any(after.error_codes):
            raise HardwareError(f"采集后机械臂报错 · {after.error_codes}")
        interval = sample.timing.host_capture_interval_s
        if interval is None:
            # 设备硬件时钟无法与主机直接比较，此时按新帧收包时间夹取静止姿态。
            interval = (sample.timing.host_received_s,) * 2
        if (after.request_end_s - before.request_start_s > self.max_pose_window_s
                or not before.request_end_s <= interval[0] <= interval[1] <= after.request_start_s):
            raise HardwareError("腕部曝光与机械臂采样未落在允许的时间窗内")
        if np.max(np.abs(after.joint_deg - before.joint_deg)) > 0.05:
            raise HardwareError("腕部采样期间关节变化超过 0.05°，请待机械臂静止后复测")
        return sample, before, (before.timestamp_s, after.timestamp_s)

    def capture(self, camera, arm=None):
        try:
            serial = self.calibration.camera_serial
            if serial is not None and (camera.info is None or camera.info.get("serial") != serial):
                raise HardwareError(f"RealSense 序列号和标定不匹配 · 需要 {serial}")
            if self.calibration.camera_parent == "base":
                sample, arm_state, arm_times = camera.capture(), None, None
            else:
                sample, arm_state, arm_times = self._capture_wrist(camera, arm)
            flange = arm_state.T_base_flange if arm_state is not None else None
            extrinsics = self.calibration.camera_to_base(flange)
            frame = sample.to_frame(extrinsics)
            interval = sample.timing.host_capture_interval_s
            oldest = min(frame.timestamp, interval[0]) if interval is not None else frame.timestamp
            if interval is not None and interval[1] > frame.timestamp + 0.005:
                raise HardwareError("RealSense 曝光时钟超前于收包时间，请检查设备时间域")
            if monotonic() - oldest > self.tracker.timeout:
                raise HardwareError("RGB-D 帧超出视觉时限")
            found = self.tracker.update(frame)
            now = monotonic()
            tracking = self.tracker.state(now) if found is not None else None
            if now - oldest > self.tracker.timeout:
                self.tracker.reset()
                tracking = None
            return CubeObservation(sample, extrinsics, self.detector.mask, tracking, arm_times, flange,
                                   flange @ self.calibration.T_flange_tcp if flange is not None else None,
                                   arm_state.controller_pose_m_rad if arm_state is not None else None)
        except (HardwareError, ValueError):
            self.tracker.reset()
            raise
