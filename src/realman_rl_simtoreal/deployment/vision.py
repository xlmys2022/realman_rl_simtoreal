"""复用 RGB-D 感知，以实测标定输出基座系方块中心。"""

from dataclasses import dataclass
from math import isfinite
from time import monotonic

import numpy as np

from ..hardware import HardwareError
from ..hardware.realsense import RgbdSample
from ..perception.rgbd import OrangeCubeDetector, UprightCubeDetector, VisualTracker
from .calibration import RealCalibration


def cube_outline_yaw(frame, mask, top_z_m, edge_m):
    """将 HSV 外轮廓投到实测顶面，用四边共同确定固定俯视的对夹方向。"""
    import cv2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    # 平直边上的小角点会被栅格噪声分成多段，拟合后同时核对轮廓面积与米制尺寸。
    polygon = cv2.approxPolyDP(contour, .035 * cv2.arcLength(contour, True), True)
    if len(polygon) != 4 or not cv2.isContourConvex(polygon):
        return None
    area = cv2.contourArea(contour)
    if not .8 * area <= cv2.contourArea(polygon) <= 1.2 * area:
        return None

    pixels = polygon.astype(np.float64)
    distortion = np.asarray(frame.distortion) if frame.distortion else None
    rays = cv2.undistortPoints(pixels, frame.intrinsics, distortion).reshape(-1, 2)
    optical = frame.camera_to_base
    rays_base = np.c_[rays, np.ones(len(rays))] @ optical[:3, :3].T
    if not np.isfinite(top_z_m) or top_z_m >= optical[2, 3] or np.any(rays_base[:, 2] >= -1e-6):
        return None
    distance = (top_z_m - optical[2, 3]) / rays_base[:, 2]
    xy = optical[:2, 3] + distance[:, None] * rays_base[:, :2]
    sides = np.roll(xy, -1, axis=0) - xy
    lengths = np.linalg.norm(sides, axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths < .70 * edge_m) or np.any(lengths > 1.30 * edge_m):
        return None
    # 四条边相隔约 90°，乘四后做圆均值可避开角度折返。边方向分散时拒绝这一帧。
    angles = np.arctan2(sides[:, 1], sides[:, 0])
    direction = np.mean(np.exp(4j * angles))
    if abs(direction) < .92:
        return None
    return float(np.angle(direction) / 4)


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
    yaw_rad: float | None = None


@dataclass(frozen=True)
class StationaryCube:
    """一批静止腕部图像的方块中心、各帧边方向和采样姿态。"""

    center_m: np.ndarray
    yaw_rad: tuple[float, ...]
    T_base_tcp: np.ndarray
    detections: int
    frames: int
    spread_m: float


class RealCubeLocalizer:
    """静止腕部或固定外置相机的只读定位。状态失效时清空跟踪。"""

    def __init__(self, calibration: RealCalibration, *, max_pose_window_s=0.25, timeout_s=0.25,
                 estimator="upright", outline_yaw=False):
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
        self.outline_yaw = outline_yaw

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
            yaw = found.yaw_rad if tracking is not None else None
            if tracking is not None and self.outline_yaw:
                edge = self.calibration.cube_edge_m
                yaw = cube_outline_yaw(frame, self.detector.mask, found.position_m[2] + edge / 2, edge)
            return CubeObservation(sample, extrinsics, self.detector.mask, tracking, arm_times, flange,
                                   flange @ self.calibration.T_flange_tcp if flange is not None else None,
                                   arm_state.controller_pose_m_rad if arm_state is not None else None,
                                   yaw)
        except (HardwareError, ValueError):
            self.tracker.reset()
            raise

    def estimate_stationary(self, camera, arm, *, max_spread_m, frames=8, min_detections=5, require_yaw=True):
        """重新采集静止腕部样本。可见性或一致性不足时不使用上一次跟踪值。"""
        if (type(frames) is not int or type(min_detections) is not int or not 3 <= min_detections <= frames <= 100
                or not isfinite(max_spread_m) or max_spread_m <= 0):
            raise ValueError("静止定位需要至少三帧检测和有限正数的中心离散限值")
        self.tracker.reset()
        observations = [self.capture(camera, arm) for _ in range(frames)]
        anchor = observations[0].T_base_tcp
        if anchor is None or any(row.T_base_tcp is None
                                 or np.linalg.norm(row.T_base_tcp[:3, 3] - anchor[:3, 3]) > 0.001
                                 or not np.allclose(row.T_base_tcp[:3, :3], anchor[:3, :3], atol=0.003, rtol=0)
                                 for row in observations):
            raise HardwareError("批量定位期间 TCP 不静止")
        accepted = [row for row in observations
                    if row.tracking is not None and (row.yaw_rad is not None or not require_yaw)]
        if len(accepted) < min_detections:
            fields = "中心和边方向" if require_yaw else "中心"
            raise HardwareError(f"静止批次仅有 {len(accepted)}/{frames} 帧提供{fields}")
        positions = np.asarray([row.tracking["position_m"] for row in accepted])
        center = np.median(positions, axis=0)
        spread = float(np.max(np.linalg.norm(positions - center, axis=1)))
        if not np.isfinite(spread) or spread > max_spread_m:
            raise HardwareError(f"静止方块定位离散程度超限 · {spread * 1000:.2f} mm")
        return StationaryCube(center, tuple(row.yaw_rad for row in accepted if row.yaw_rad is not None),
                              observations[-1].T_base_tcp, len(accepted), frames, spread)
