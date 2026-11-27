#!/usr/bin/env python3
"""逐段验收 RealMan 直线、夹爪和视觉抓取。默认只读。"""

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from realman_rl_simtoreal.apps.cli.real_hardware import add_arm_arguments, print_json
from realman_rl_simtoreal.deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from realman_rl_simtoreal.deployment.grasp_config import grasp_site, gripper_opening_limits, motion_limits, read_site
from realman_rl_simtoreal.deployment.real_grasp import GRASP_STAGES, check_tcp_path, execute_grasp, plan_grasp
from realman_rl_simtoreal.deployment.vision import RealCubeLocalizer
from realman_rl_simtoreal.hardware.realman import RealManArm, RealManGripper
from realman_rl_simtoreal.hardware.realsense import RealSenseCamera


def confirm_motion(action, details):
    """每个 SDK 动作单独显示参数，经现场交互终端输入同名阶段才可发出。"""
    if not sys.stdin.isatty():
        return False
    print_json({"request_confirmation": action, **details})
    token = action.upper()
    return input(f"现场确认区域安全并核对动作参数后输入 {token}，回车取消 > ").strip() == token


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    move = commands.add_parser("move", help="核对单条固定姿态的基座系 TCP 直线")
    move.add_argument("--tcp", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    move.add_argument("--speed-pct", type=int, required=True)
    open_jaw = commands.add_parser("open", help="验收夹爪低速张满及现场开口读回")
    grasp = commands.add_parser("grasp", help="定位后逐段验收传统抓取和抬升")
    grasp.add_argument("--until", choices=GRASP_STAGES, default="lift")
    grasp.add_argument("--camera-timeout-ms", type=int, default=2000)
    for command in (move, open_jaw, grasp):
        add_arm_arguments(command, required=False)
        command.add_argument("--site", type=Path, required=True, help="经现场验收的抓取 JSON")
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
        command.add_argument("--validate-site", action="store_true", help="仅校验文件，离线退出")
        command.add_argument("--execute", action="store_true", help="启用动作提示，仍须逐条人工确认")

    args = parser.parse_args(argv)
    if args.execute and (args.validate_site or not sys.stdin.isatty()):
        parser.error("--execute 需要现场交互终端，且不能与 --validate-site 同时使用")
    if not args.validate_site and not args.ip:
        parser.error("真机只读检查或动作验收需要 --ip")

    try:
        calibration = load_calibration(args.calibration)
        data = read_site(args.site)
        if args.command == "grasp":
            site = grasp_site(data, calibration)
            motion, jaw = site.motion, site.gripper
        else:
            motion = motion_limits(data, calibration)
            jaw = gripper_opening_limits(data) if args.command == "open" else None
        print_json({"command": args.command, "site": str(args.site), "calibration": str(calibration.path),
                    "calibration_sha256": calibration.checksum(), "motion_enabled": args.execute})
        if args.validate_site:
            return 0

        with RealManArm(args.ip, port=args.port, timeout_ms=args.arm_timeout_ms, limits=motion,
                        enable_motion=args.execute, confirm=confirm_motion if args.execute else None) as arm:
            state = arm.check_motion_ready()
            tcp = state.T_base_flange @ motion.T_flange_tcp
            print_json({"controller": arm.identity, "frames": arm.read_frames(),
                        "collision_stage": arm.read_collision_stage(), "tcp_m": tcp[:3, 3].tolist()})
            if args.command == "move":
                motion.require_speed(args.speed_pct, arm.read_max_line_speed())
                check_tcp_path(state, motion, [args.tcp], joint_bounds_deg=arm.read_joint_bounds())
                print_json({"status": "plan_checked", "target_tcp_m": args.tcp, "speed_pct": args.speed_pct})
                if args.execute:
                    def verify_move():
                        check_tcp_path(arm.check_motion_ready(), motion, [args.tcp],
                                       joint_bounds_deg=arm.read_joint_bounds())

                    end = arm.move_linear_tcp(args.tcp, speed_pct=args.speed_pct, preflight=verify_move)
                    print_json({"status": "move_completed",
                                "tcp_m": (end.T_base_flange @ motion.T_flange_tcp)[:3, 3].tolist()})
                return 0

            gripper = RealManGripper(arm, limits=jaw)
            print_json({"gripper": asdict(gripper.read_state())})
            if args.command == "open":
                if args.execute:
                    print_json({"status": "open_confirmed", "gripper": asdict(gripper.open_max())})
                return 0

            motion.require_speed(site.geometry.speed_pct, arm.read_max_line_speed())
            with RealSenseCamera(serial=calibration.camera_serial, timeout_ms=args.camera_timeout_ms) as camera:
                camera.capture()  # 等待 RGB-D 首组配对，随后每次定位重新取静止图像。
                localizer = RealCubeLocalizer(calibration)

                def observe():
                    return localizer.estimate_stationary(camera, arm,
                                                         max_spread_m=site.geometry.max_target_shift_m)

                estimate = observe()
                print_json({"cube_center_m": estimate.center_m.tolist(), "detections": estimate.detections,
                            "frames": estimate.frames, "spread_m": estimate.spread_m})
                plan = plan_grasp(estimate, site, calibration.cube_edge_m)
                check_tcp_path(arm.check_motion_ready(), motion, plan.waypoints.values(),
                               joint_bounds_deg=arm.read_joint_bounds())
                print_json(dict(status="plan_checked", **plan.report()))
                if not args.execute:
                    return 0
                result = execute_grasp(plan, site, calibration.cube_edge_m, arm, gripper, observe,
                                       until=args.until, report=print_json)
                print_json(result)
                return 2 if result["status"] == "lift_needs_visual_confirmation" else 0
    except (KeyboardInterrupt, EOFError):
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"本次验收中止 · {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
