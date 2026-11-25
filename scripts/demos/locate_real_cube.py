#!/usr/bin/env python3
"""只读显示实物方块在机械臂基座系的位置。"""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

from realman_rl_simtoreal.apps.cli.real_hardware import (
    add_arm_arguments, add_camera_arguments, camera_from_args, print_json, run_rgbd_view,
)
from realman_rl_simtoreal.deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from realman_rl_simtoreal.deployment.vision import RealCubeLocalizer
from realman_rl_simtoreal.hardware.realman import RealManArm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                        help=f"本地实测标定 JSON（默认 {DEFAULT_CALIBRATION}）")
    parser.add_argument("--validate-calibration", action="store_true", help="仅验证本地标定文件")
    parser.add_argument("--pose-window-ms", type=float, default=250,
                        help="腕部图像和前后关节查询的最大配对时间窗（ms）")
    parser.add_argument("--estimator", choices=("upright", "legacy"), default="upright",
                        help="选择顶面拟合或原中心算法")
    add_arm_arguments(parser, required=False)
    add_camera_arguments(parser)
    args = parser.parse_args()
    try:
        calibration = load_calibration(args.calibration)
        print_json({"calibration": str(calibration.path), "hsv_file": str(calibration.hsv_path),
                    "camera_parent": calibration.camera_parent, "cube_edge_m": calibration.cube_edge_m,
                    "camera_serial": calibration.camera_serial})
        if args.validate_calibration:
            return 0
        localizer = RealCubeLocalizer(calibration, max_pose_window_s=args.pose_window_ms / 1000,
                                     estimator=args.estimator)
        camera = camera_from_args(args, calibrated_serial=calibration.camera_serial)
        with ExitStack() as stack:
            arm = None
            if calibration.camera_parent != "base":
                if args.ip is None:
                    parser.error("腕部相机定位需要 --ip 指定实际机械臂")
                arm = stack.enter_context(RealManArm(args.ip, port=args.port,
                                                    timeout_ms=args.arm_timeout_ms))
            stack.enter_context(camera)
            if arm is not None:
                # 先等 RGB-D 启动配对完成，再开始 250 ms 的腕部采样窗口。
                camera.capture()
            run_rgbd_view(args, camera, localizer=localizer, arm=arm)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"真机视觉定位失败 · {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
