#!/usr/bin/env python3
"""独立检查 RealSense 内参、对齐彩色与深度、帧号及时间戳。"""

import argparse
import sys

from realman_rl_simtoreal.apps.cli.real_hardware import (
    add_camera_arguments, camera_from_args, run_rgbd_view,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_camera_arguments(parser)
    args = parser.parse_args()
    try:
        with camera_from_args(args) as camera:
            run_rgbd_view(args, camera)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"相机检查失败 · {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
