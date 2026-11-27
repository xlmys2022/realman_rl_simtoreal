#!/usr/bin/env python3
"""只读检查 RML63II-6FB 型号、关节、位姿和可选的夹爪状态。"""

import argparse
from dataclasses import asdict
import sys
from time import sleep

import numpy as np

from realman_rl_simtoreal.apps.cli.real_hardware import add_arm_arguments, print_json
from realman_rl_simtoreal.hardware.realman import RealManArm, RealManGripper


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arm_arguments(parser, required=True)
    parser.add_argument("--gripper", action="store_true", help="查询控制器内置 EG2-4C2 状态")
    parser.add_argument("--gripper-samples", type=int, default=1, help="静止读回次数 1–50，间隔 50 ms")
    parser.add_argument("--kinematics", action="store_true", help="同时记录控制器 DH 和关节限位")
    parser.add_argument("--collision", action="store_true", help="查询动力学碰撞等级与末端线速度上限")
    args = parser.parse_args(argv)
    if not 1 <= args.gripper_samples <= 50 or (args.gripper_samples > 1 and not args.gripper):
        parser.error("--gripper-samples 需要 1–50，多次采样须同时指定 --gripper")
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
                gripper = RealManGripper(arm)
                states = [gripper.read_state()]
                for _ in range(args.gripper_samples - 1):
                    sleep(0.05)
                    states.append(gripper.read_state())
                data["gripper"] = asdict(states[-1])
                healthy = healthy and all(row.online and not row.error for row in states)
                if len(states) > 1:
                    after = arm.read_state()
                    drift = float(np.max(np.abs(after.joint_deg - state.joint_deg)))
                    data["gripper_observed"] = dict(
                        arm_endpoint_joint_drift_deg=drift,
                        opening_raw=[min(row.opening_raw for row in states), max(row.opening_raw for row in states)],
                        force_g=[min(row.force_g for row in states), max(row.force_g for row in states)],
                        modes=sorted({row.mode for row in states}), samples=[asdict(row) for row in states])
                    healthy = healthy and drift <= 0.05 and not any(after.error_codes)
            print_json(data)
            return 0 if healthy else 1
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"机械臂检查失败 · {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
