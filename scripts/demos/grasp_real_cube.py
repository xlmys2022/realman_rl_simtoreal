#!/usr/bin/env python3
"""逐段验收 RealMan 直线、夹爪和视觉抓取。默认只读。"""

import argparse
from dataclasses import asdict
from math import isfinite
from pathlib import Path
import sys

from realman_rl_simtoreal.apps.cli.real_hardware import add_arm_arguments, print_json
from realman_rl_simtoreal.deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from realman_rl_simtoreal.deployment.grasp_config import grasp_site, gripper_opening_limits, motion_limits, read_site
from realman_rl_simtoreal.deployment.real_grasp import (
    GRASP_STAGES, check_tcp_path, execute_grasp, inspect_grasp_observation, plan_grasp, requested_grasp_stages,
)
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


def warm_up_vision(camera):
    # 近桌时开流颜色收敛更慢。先丢弃约四秒的配对帧，再用原有门限定位。
    for _ in range(4 * camera.info["color_fps"]):
        camera.capture()


def survey(args, calibration, data):
    """现场限值未填完整时仍可只读核验方块与观察姿态。"""
    geometry = data.get("grasp")
    table = geometry.get("table_z_base_m") if isinstance(geometry, dict) else None
    if type(table) not in (int, float) or not isfinite(table):
        raise ValueError("实测 grasp.table_z_base_m 未填写")
    if not isfinite(args.max_spread_mm) or args.max_spread_mm <= 0:
        raise ValueError("--max-spread-mm 须为有限正数")
    with RealManArm(args.ip, port=args.port, timeout_ms=args.arm_timeout_ms) as arm:
        state = arm.read_state()
        print_json({"controller": arm.identity, "frames": arm.read_frames(),
                    "collision_stage": arm.read_collision_stage(), "joint_deg": state.joint_deg.tolist(),
                    "controller_pose_m_rad": state.controller_pose_m_rad.tolist(),
                    "gripper": asdict(RealManGripper(arm).read_state())})
        with RealSenseCamera(serial=calibration.camera_serial, timeout_ms=args.camera_timeout_ms) as camera:
            warm_up_vision(camera)
            localizer = RealCubeLocalizer(calibration, outline_yaw=True)
            estimate = localizer.estimate_stationary(camera, arm, max_spread_m=args.max_spread_mm / 1000)
    print_json(dict(status="survey_complete", **inspect_grasp_observation(estimate, table, calibration.cube_edge_m)))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    move = commands.add_parser("move", help="核对单条固定姿态的基座系 TCP 直线")
    move.add_argument("--tcp", nargs=3, type=float, required=True, metavar=("X", "Y", "Z"))
    move.add_argument("--speed-pct", type=int, required=True)
    open_jaw = commands.add_parser("open", help="验收夹爪低速张满及现场开口读回")
    grasp = commands.add_parser("grasp", help="定位后逐段验收传统抓取和抬升")
    grasp.add_argument("--until", choices=GRASP_STAGES, default="lift")
    grasp.add_argument("--from", dest="checkpoint", choices=("pregrasp", "descend", "close"),
                       help="从经现场核验的停稳检查点续行，保持当前 XY 与腕姿")
    survey_pose = commands.add_parser("survey", help="只读采集固定俯视观察位的可见性和偏差")
    survey_pose.set_defaults(execute=False, validate_site=False)
    survey_pose.add_argument("--max-spread-mm", type=float, default=3., help="只读采样的中心离散限值（mm）")
    for vision_command in (grasp, survey_pose):
        vision_command.add_argument("--camera-timeout-ms", type=int, default=2000)
    for command in (move, open_jaw, grasp, survey_pose):
        add_arm_arguments(command, required=command is survey_pose)
        command.add_argument("--site", type=Path, required=True, help="经现场验收的抓取 JSON")
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    for command in (move, open_jaw, grasp):
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
        if args.command == "survey":
            print_json({"command": "survey", "calibration": str(calibration.path),
                        "calibration_sha256": calibration.checksum(), "motion_enabled": False})
            return survey(args, calibration, data)
        if args.command == "grasp":
            requested_grasp_stages(args.until, args.checkpoint)
            site = grasp_site(data, calibration, until=args.until)
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
            jaw_state = gripper.read_state()
            print_json({"gripper": asdict(jaw_state)})
            gripper.verify_ready(jaw_state)
            if args.command == "open":
                if args.execute:
                    print_json({"status": "open_confirmed", "gripper": asdict(gripper.open_max())})
                return 0

            motion.require_speed(site.geometry.speed_pct, arm.read_max_line_speed())
            if args.checkpoint == "close":
                gripper.require_holding()
            elif args.checkpoint is not None:
                gripper.require_open()
            with RealSenseCamera(serial=calibration.camera_serial, timeout_ms=args.camera_timeout_ms) as camera:
                warm_up_vision(camera)
                localizer = RealCubeLocalizer(calibration, outline_yaw=True)

                def observe():
                    return localizer.estimate_stationary(camera, arm,
                                                         max_spread_m=site.geometry.max_target_shift_m,
                                                         require_yaw=args.checkpoint != "close")

                def observe_lift():
                    # 抬升验收比较 XYZ 位移。夹指挡住部分边时仍可检测中心，保留同样的质量门限。
                    return localizer.estimate_stationary(camera, arm, max_spread_m=site.geometry.max_target_shift_m,
                                                         require_yaw=False)

                estimate = observe()
                print_json(inspect_grasp_observation(estimate, site.geometry.table_z_base_m, calibration.cube_edge_m))
                plan = plan_grasp(estimate, site, calibration.cube_edge_m, until=args.until, start_after=args.checkpoint)
                check_tcp_path(arm.check_motion_ready(), motion, plan.waypoints.values(),
                               joint_bounds_deg=arm.read_joint_bounds())
                print_json(dict(status="plan_checked", start_after=args.checkpoint, until=args.until, **plan.report()))
                if not args.execute:
                    return 0
                result = execute_grasp(plan, site, calibration.cube_edge_m, arm, gripper, observe,
                                       until=args.until, report=print_json, start_after=args.checkpoint,
                                       observe_lift=observe_lift)
                print_json(result)
                return 2 if result["status"] == "lift_needs_visual_confirmation" else 0
    except (KeyboardInterrupt, EOFError):
        return 130
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"本次验收中止 · {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
