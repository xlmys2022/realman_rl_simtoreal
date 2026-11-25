"""设备检查与定位演示使用的只读命令行选项。"""

from dataclasses import asdict
from itertools import count
import json
from math import isfinite

import numpy as np

from ...hardware.realsense import RealSenseCamera


def add_arm_arguments(parser, *, required):
    parser.add_argument("--ip", required=required, help="实际机械臂的 IPv4 地址")
    parser.add_argument("--port", type=int, default=8080, help="控制器端口（默认 8080）")
    parser.add_argument("--arm-timeout-ms", type=int, default=500, help="控制器回包超时（ms）")


def add_camera_arguments(parser):
    parser.add_argument("--serial", help="RealSense 序列号。多台设备时选择其中一台")
    parser.add_argument("--width", type=int, default=640, help="RGB 和深度流宽度（像素）")
    parser.add_argument("--height", type=int, default=480, help="RGB 和深度流高度（像素）")
    parser.add_argument("--fps", type=int, default=30, help="RGB 和深度流帧率")
    parser.add_argument("--camera-timeout-ms", type=int, default=2000, help="配对帧等待超时（ms）")
    parser.add_argument("--frames", type=int, default=0, help="采集帧数。0 表示持续运行")
    parser.add_argument("--no-preview", action="store_true", help="输出逐帧 JSONL，适用于无桌面会话")
    parser.add_argument("--depth-max-m", type=float, default=1.5, help="深度预览色条的上限（m）")


def camera_from_args(args, *, calibrated_serial=None):
    if args.frames < 0:
        raise ValueError("--frames 必须为非负整数")
    if not isfinite(args.depth_max_m) or args.depth_max_m <= 0:
        raise ValueError("--depth-max-m 必须为有限正数")
    if calibrated_serial is not None and args.serial not in (None, calibrated_serial):
        raise ValueError("--serial 与真机标定中的 camera_serial 不一致")
    return RealSenseCamera(serial=calibrated_serial or args.serial, width=args.width,
                           height=args.height, fps=args.fps, timeout_ms=args.camera_timeout_ms)


def print_json(data):
    print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)


def run_rgbd_view(args, camera, *, localizer=None, arm=None):
    preview = None
    print_json({"camera": camera.info})
    try:
        if not args.no_preview:
            try:
                from ..visualization.real_rgbd import RealRgbdPreview
                preview = RealRgbdPreview(localization=localizer is not None, depth_max_m=args.depth_max_m)
            except (ImportError, OSError) as exc:
                raise RuntimeError("预览需要 Tkinter，终端采集可使用 --no-preview") from exc
        for index in count(1):
            observation = localizer.capture(camera, arm) if localizer is not None else None
            sample = observation.sample if observation is not None else camera.capture()
            valid = sample.depth_m[np.isfinite(sample.depth_m) & (sample.depth_m > 0)]
            fraction = valid.size / sample.depth_m.size
            data = {
                "frame": index,
                "timing": asdict(sample.timing),
                "valid_depth_fraction": fraction,
                "median_depth_m": float(np.median(valid)) if valid.size else None,
            }
            status = (f"帧 {index}    有效深度 {fraction:.1%}"
                      f"    预览范围 0–{args.depth_max_m:g} m")
            mask, centroid = None, None
            if observation is not None:
                tracking, mask = observation.tracking, observation.mask
                data.update(cube_center_base_m=None, confidence=0.0,
                            T_base_color=observation.T_base_color.tolist(),
                            arm_timestamps_s=observation.arm_timestamps_s)
                if tracking is None:
                    status += "\n未检出有效方块"
                else:
                    coordinates = tracking["position_m"].tolist()
                    centroid = tracking["centroid_px"]
                    confidence = tracking["confidence"]
                    data.update(cube_center_base_m=coordinates, confidence=confidence)
                    x, y, z = coordinates
                    status += (f"\n基座系方块中心  X {x:+.4f}   Y {y:+.4f}   Z {z:+.4f} m"
                               f"    置信度 {confidence:.2f}")
            if preview is None:
                print_json(data)
            else:
                preview.render(sample, status, mask=mask, centroid=centroid)
                if not preview.open:
                    break
            if args.frames and index >= args.frames:
                break
    finally:
        if preview is not None:
            preview.close()
