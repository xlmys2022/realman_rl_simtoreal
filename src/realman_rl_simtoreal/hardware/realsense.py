"""从 RealSense 采集对齐的 RGB、彩色光轴深度与设备时间戳。"""

from dataclasses import dataclass
from importlib import import_module
from math import ceil, isfinite
from time import monotonic, time

import numpy as np

from ..perception.rgbd import RgbdFrame
from . import HardwareError, HardwareTimeout


@dataclass(frozen=True)
class FrameTiming:
    host_received_s: float
    color_timestamp_ms: float
    depth_timestamp_ms: float
    color_clock: str
    depth_clock: str
    color_frame_number: int
    depth_frame_number: int
    host_capture_interval_s: tuple[float, float] | None


@dataclass(frozen=True)
class RgbdSample:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: np.ndarray
    timing: FrameTiming

    def to_frame(self, T_base_color):
        """RGB 和深度已校正成同一针孔图像，外参由部署层显式传入。"""
        return RgbdFrame(self.rgb, self.depth_m, self.intrinsics, T_base_color,
                         self.timing.host_received_s)


def _rectification_map(rs, intr, ray_x, ray_y):
    brown = (rs.distortion.brown_conrady, rs.distortion.modified_brown_conrady,
             rs.distortion.inverse_brown_conrady)
    if intr.model == rs.distortion.none or (intr.model in brown and not any(intr.coeffs)):
        return None
    # 用 SDK 的实际畸变模型构建针孔像素到原图的映射，避免套用 OpenCV 其他模型的系数。
    pixels = np.fromiter(
        (coord for x, y in zip(ray_x.flat, ray_y.flat)
         for coord in rs.rs2_project_point_to_pixel(intr, (float(x), float(y), 1.0))),
        dtype=np.float32, count=2 * ray_x.size,
    ).reshape((*ray_x.shape, 2))
    if not np.isfinite(pixels).all():
        raise HardwareError("RealSense 彩色畸变校正映射无效")
    return pixels


class RealSenseCamera:
    """以米制深度和 RGB 顺序输出对齐帧。with 块负责 SDK 数据流的启停。"""

    def __init__(self, *, serial=None, width=640, height=480, fps=30, timeout_ms=2000):
        for name, value in (("width", width), ("height", height), ("fps", fps),
                            ("timeout_ms", timeout_ms)):
            if type(value) is not int or not 1 <= value <= 60_000:
                raise ValueError(f"RealSense {name} 需要 1–60000 的整数")
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_ms = timeout_ms
        self.info = None
        self.intrinsics = None
        self._pipeline = None

    def __enter__(self):
        if self._pipeline is not None:
            raise HardwareError("RealSense 数据流已打开")
        try:
            rs = import_module("pyrealsense2")
        except (ImportError, OSError) as exc:
            raise HardwareError(f"无法加载 RealSense SDK，请安装 requirements-hardware.txt · {exc}") from exc
        try:
            context = rs.context()
            devices = {dev.get_info(rs.camera_info.serial_number): dev for dev in context.query_devices()}
            if not devices:
                raise HardwareError("未发现可访问的 RealSense，请检查 USB 连接和设备权限")
            if self.serial is None and len(devices) != 1:
                raise HardwareError(f"发现多个 RealSense，请通过 --serial 选择 · {', '.join(devices)}")
            serial = self.serial if self.serial is not None else next(iter(devices))
            if serial not in devices:
                raise HardwareError(f"未发现序列号为 {serial} 的 RealSense · {', '.join(devices)}")
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            pipeline = rs.pipeline(context)
            profile = pipeline.start(config)
            self._pipeline = pipeline
            self._align = rs.align(rs.stream.color)
            self._last_numbers = None
            self._configure_geometry(rs, profile)
            self.info.update(serial=serial, device=devices[serial].get_info(rs.camera_info.name))
        except BaseException as exc:
            self.__exit__(type(exc), exc, exc.__traceback__)
            if isinstance(exc, RuntimeError) and not isinstance(exc, HardwareError):
                raise HardwareError(f"无法启动 RealSense RGB-D 数据流 · {exc}") from exc
            raise
        return self

    def _configure_geometry(self, rs, profile):
        color = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = color.get_intrinsics()
        if (not np.isfinite([intr.fx, intr.fy, intr.ppx, intr.ppy, *intr.coeffs]).all()
                or min(intr.fx, intr.fy, intr.width, intr.height) <= 0):
            raise HardwareError("RealSense 彩色内参无效")
        self.intrinsics = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
        self.intrinsics.setflags(write=False)
        self._shape = (intr.height, intr.width)
        v, u = np.indices(self._shape, dtype=np.float32)
        ray_x, ray_y = (u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy
        self._map = _rectification_map(rs, intr, ray_x, ray_y)

        # rs.align 改变像素索引，但写入的 Z16 仍沿原深度相机光轴。
        # p_depth = R_depth_color @ p_color + t_depth_color，逐像素求彩色光轴 Z。
        extr = color.get_extrinsics_to(depth)
        rotation = np.asarray(extr.rotation).reshape(3, 3, order="F")
        if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)
                or not np.isfinite(extr.translation).all()):
            raise HardwareError("RealSense 流间外参无效")
        self._depth_offset = extr.translation[2]
        self._depth_factor = rotation[2, 0] * ray_x + rotation[2, 1] * ray_y + rotation[2, 2]
        scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        if np.any(self._depth_factor <= 0) or not isfinite(scale) or scale <= 0:
            raise HardwareError("RealSense 光轴换算或深度比例无效")
        self._max_pair_skew_s = 1 / min(color.fps(), depth.fps()) + 0.005
        self.info = dict(width=intr.width, height=intr.height, color_fps=color.fps(),
                         depth_fps=depth.fps(), intrinsics=self.intrinsics.tolist(),
                         color_distortion_model=intr.model.name,
                         color_distortion_coefficients=list(intr.coeffs),
                         rectified=self._map is not None, nominal_z16_units_m=scale)

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as close_error:
            if exc is None:
                raise
            exc.add_note(str(close_error))

    def close(self):
        if self._pipeline is not None:
            pipeline, self._pipeline = self._pipeline, None
            pipeline.stop()

    def _wait_for_pair(self, fresh):
        if fresh:
            self._pipeline.poll_for_frames()
        deadline = monotonic() + self.timeout_ms / 1000
        last_problem = ""
        while (remaining := deadline - monotonic()) > 0:
            ok, frames = self._pipeline.try_wait_for_frames(ceil(remaining * 1000))
            received = monotonic()
            received_wall = time()
            if not ok:
                break
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            if not color or not depth:
                continue
            numbers = (color.get_frame_number(), depth.get_frame_number())
            if self._last_numbers is not None and any(a <= b for a, b in zip(numbers, self._last_numbers)):
                last_problem = "彩色或深度帧号未更新"
                continue
            timing = self._timing(color, depth, numbers, received, received_wall)
            interval = timing.host_capture_interval_s
            if interval is not None and interval[1] - interval[0] > self._max_pair_skew_s:
                # 流启动时 SDK 可能用旧深度反复配对新彩色，等待两路进入稳定配对。
                skew_ms = 1000 * (interval[1] - interval[0])
                last_problem = f"彩色与深度时间差 {skew_ms:.1f} ms，超过一个帧周期"
                continue
            return frames, color, depth, numbers, timing
        detail = f" · {last_problem}" if last_problem else ""
        raise HardwareTimeout(f"等待新的 RealSense RGB-D 配对帧超时（{self.timeout_ms} ms）{detail}")

    def _timing(self, color, depth, numbers, received, received_wall):
        timestamps = (color.get_timestamp(), depth.get_timestamp())
        if not all(isfinite(stamp) and stamp >= 0 for stamp in timestamps):
            raise HardwareError("RealSense 帧时间戳无效")
        clocks = (color.get_frame_timestamp_domain().name, depth.get_frame_timestamp_domain().name)
        interval = None
        if all(clock in ("global_time", "system_time") for clock in clocks):
            # 仅能将与主机墙钟同域的曝光时间换算到主机单调时钟。
            wall_to_monotonic = received - received_wall
            stamps = sorted(stamp / 1000 + wall_to_monotonic for stamp in timestamps)
            interval = (stamps[0], stamps[1])
        return FrameTiming(received, *timestamps, *clocks, *numbers, interval)

    def capture(self, *, fresh=False):
        """读取同步帧。fresh 先丢弃 SDK 中已排队的帧，便于对应腕部位姿。"""
        if self._pipeline is None:
            raise HardwareError("RealSense 数据流未打开")
        try:
            frames, color, source_depth, numbers, timing = self._wait_for_pair(fresh)
            aligned_depth = self._align.process(frames).get_depth_frame()
            if not aligned_depth:
                raise HardwareError("RealSense 深度对齐未返回 Z16 帧")
            units = float(source_depth.get_units())
            rgb = np.asarray(color.get_data())
            z16 = np.asarray(aligned_depth.get_data())
            if (rgb.shape != (*self._shape, 3) or rgb.dtype != np.uint8
                    or z16.shape != self._shape or z16.dtype != np.uint16
                    or not isfinite(units) or units <= 0):
                raise HardwareError("RealSense 返回了无效的对齐图像、像素格式或深度比例")
            if self._map is not None:
                import cv2
                rgb = cv2.remap(rgb, self._map, None, cv2.INTER_LINEAR)
                z16 = cv2.remap(z16, self._map, None, cv2.INTER_NEAREST)
            else:
                rgb = rgb.copy()
            depth_m = (z16.astype(np.float32) * units - self._depth_offset) / self._depth_factor
            depth_m[(z16 == 0) | (depth_m <= 0)] = 0
            self._last_numbers = numbers
            return RgbdSample(rgb, depth_m, self.intrinsics, timing)
        except HardwareError:
            raise
        except RuntimeError as exc:
            raise HardwareError(f"RealSense 采集失败 · {exc}") from exc
