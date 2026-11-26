#!/usr/bin/env python3
"""只读检查 RML63II-6FB 型号、关节、位姿和可选的夹爪状态。"""

import argparse
from dataclasses import asdict
import sys

from realman_rl_simtoreal.apps.cli.real_hardware import add_arm_arguments, print_json
from realman_rl_simtoreal.hardware.realman import RealManArm, RealManGripper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arm_arguments(parser, required=True)
    parser.add_argument("--gripper", action="store_true", help="查询控制器内置 EG2-4C2 状态")
    parser.add_argument("--kinematics", action="store_true", help="同时记录控制器 DH 和关节限位")
    parser.add_argument("--collision", action="store_true", help="查询动力学碰撞等级与末端线速度上限")
    args = parser.parse_args()
    try:
        with RealManArm(args.ip, port=args.port, timeout_ms=args.arm_timeout_ms) as arm:
            state = arm.read_state()
            data = {
                "identity": arm.identity,
                "controller_frames": arm.read_frames(),
                "joint_deg": state.joint_deg.tolist(),
                "controller_pose_m_rad": state.controller_pose_m_rad.tolist(),
                "T_base_flange": state.T_base_flange.tolist(),
                "error_codes": state.error_codes,
                "timestamp_s": state.timestamp_s,
                "request_interval_s": [state.request_start_s, state.request_end_s],
            }
            healthy = not any(state.error_codes)
            if args.kinematics:
                data["kinematics"] = arm.read_kinematic_parameters()
            if args.collision:
                data["collision_stage"] = arm.read_collision_stage()
                data["max_line_speed_m_s"] = arm.read_max_line_speed()
                healthy = healthy and data["collision_stage"] > 0
            if args.gripper:
                gripper = RealManGripper(arm).read_state()
                data["gripper"] = asdict(gripper)
                healthy = healthy and gripper.online and not gripper.error
            print_json(data)
            return 0 if healthy else 1
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"机械臂检查失败 · {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
