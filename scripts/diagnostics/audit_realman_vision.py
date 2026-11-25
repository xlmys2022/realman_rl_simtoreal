#!/usr/bin/env python3
"""只读保存静止 RGB-D 采样，离线比较同一方块的坐标链和估计方法。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from realman_rl_simtoreal.deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from realman_rl_simtoreal.deployment.vision import RealCubeLocalizer
from realman_rl_simtoreal.hardware.realman import RealManArm, RealManGripper
from realman_rl_simtoreal.hardware.realman_config import REALMAN_TABLETOP
from realman_rl_simtoreal.hardware.realsense import RealSenseCamera
from realman_rl_simtoreal.perception.rgbd import OrangeCubeDetector, RgbdFrame, UprightCubeDetector
from realman_rl_simtoreal.utils.artifacts import file_sha256, write_json
from realman_rl_simtoreal.utils.geometry import euler_matrix, inverse, transform


def capture(args, calibration):
    """保存原相机帧及同时段的机械臂只读回包，供离线重算。"""
    with RealManArm(args.ip) as arm:
        before = arm.read_frames()
        robot = dict(identity=arm.identity, kinematics=arm.read_kinematic_parameters(),
                     gripper=asdict(RealManGripper(arm).read_state()), active_frames=before)
        with RealSenseCamera(serial=calibration.camera_serial, timeout_ms=4000) as camera:
            camera.capture()
            localizer = RealCubeLocalizer(calibration, estimator="legacy")
            observations = [localizer.capture(camera, arm) for _ in range(args.frames)]
            robot['camera'] = camera.info
        if arm.read_frames() != before:
            raise RuntimeError("采样期间控制器活动坐标系变化")
    robot['calibration_sha256'] = file_sha256(calibration.path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        np.savez_compressed(
            stream, rgb=np.stack([row.sample.rgb for row in observations]),
            depth_m=np.stack([row.sample.depth_m for row in observations]),
            K=observations[0].sample.intrinsics,
            T_base_flange=np.stack([row.T_base_flange for row in observations]),
            controller_pose_m_rad=np.stack([row.controller_pose_m_rad for row in observations]),
            host_received_s=[row.sample.timing.host_received_s for row in observations],
            metadata=json.dumps(robot))
    data = dict(output=str(args.output), frames=len(observations), identity=robot['identity'],
                active_frames=before, kinematics=robot['kinematics'], gripper=robot['gripper'])
    print(json.dumps(data, ensure_ascii=False))


def pose_matrix(values):
    values = np.asarray(values)
    return transform(values[:3], euler_matrix(values[3:]))


def rotation_difference_deg(a, b):
    cosine = np.clip((np.trace(a.T @ b) - 1.) / 2., -1., 1.)
    return float(np.rad2deg(np.arccos(cosine)))


def summarize_centers(centers, camera_centers, expected_z):
    if not centers:
        return dict(detections=0, median_base_m=None, median_color_m=None)
    xyz = np.asarray(centers)
    median = np.median(xyz, axis=0)
    return dict(detections=len(xyz), median_base_m=median.tolist(),
                median_color_m=np.median(camera_centers, axis=0).tolist(),
                std_base_mm=(1000 * xyz.std(axis=0)).tolist(),
                center_z_error_mm=1000 * float(median[2] - expected_z))


def analyze_capture(path, calibration):
    with np.load(path, allow_pickle=False) as source:
        rgb, depth, K, flanges = (source[k] for k in ("rgb", "depth_m", "K", "T_base_flange"))
        stamps = source['host_received_s']
        metadata = json.loads(source['metadata'].item()) if 'metadata' in source else {}
        reported = source['controller_pose_m_rad'] if 'controller_pose_m_rad' in source else None
    if metadata.get('calibration_sha256') not in (None, file_sha256(calibration.path)):
        raise ValueError(f"{path} 的标定文件校验值已变化")
    H = calibration.T_parent_color
    tcp = calibration.T_flange_tcp
    alternatives = dict(flange=H, tcp=tcp @ H, flange_inverse=inverse(H), tcp_inverse=tcp @ inverse(H))
    expected_z = -REALMAN_TABLETOP.base_above_table_m + calibration.cube_edge_m / 2
    report = dict(file=str(path), frames=len(stamps), expected_cube_center_z_base_m=expected_z,
                  mean_flange_position_base_m=flanges[:, :3, 3].mean(axis=0).tolist(),
                  T_base_flange=flanges[0].tolist(), T_base_tcp=(flanges[0] @ tcp).tolist(),
                  median_rate_hz=float(1 / np.median(np.diff(stamps))) if len(stamps) > 1 else None,
                  tool_down_tilt_deg=float(np.rad2deg(np.arccos(np.clip(-flanges[0, 2, 2], -1., 1.)))),
                  candidates={})
    if reported is not None:
        work = pose_matrix(metadata['active_frames']['work']['pose'])
        tool = pose_matrix(metadata['active_frames']['tool']['pose'])
        controller = np.stack([work @ pose_matrix(row) for row in reported])
        fk = flanges @ tool
        report['controller_check'] = dict(
            active_frames=metadata['active_frames'],
            T_base_active_tool_reported=controller[0].tolist(),
            T_flange_active_tool=tool.tolist(),
            max_xyz_error_mm=float(1000 * np.max(np.linalg.norm(fk[:, :3, 3] - controller[:, :3, 3], axis=1))),
            max_rotation_error_deg=max(rotation_difference_deg(a[:3, :3], b[:3, :3]) for a, b in zip(fk, controller)))
    for name, attachment in alternatives.items():
        T = flanges @ attachment
        candidate = dict(T_base_color=T[0].tolist())
        for detector_type, estimator in ((OrangeCubeDetector, "legacy"), (UprightCubeDetector, "upright")):
            detector = detector_type(edge_m=calibration.cube_edge_m, hsv_ranges=calibration.hsv_ranges)
            base_points, color_points = [], []
            for pixels, distances, optical, stamp in zip(rgb, depth, T, stamps):
                measurement = detector.detect(RgbdFrame(pixels, distances, K, optical, float(stamp)))
                if measurement is not None:
                    p = measurement.position_m
                    base_points.append(p)
                    color_points.append(optical[:3, :3].T @ (p - optical[:3, 3]))
            candidate[estimator] = summarize_centers(base_points, color_points, expected_z)
        report['candidates'][name] = candidate
    return report


def compare(args, calibration):
    runs = [analyze_capture(path, calibration) for path in args.samples]
    pairs = []
    for other in runs[1:]:
        first = runs[0]
        arm_delta = np.asarray(other['mean_flange_position_base_m']) - first['mean_flange_position_base_m']
        angle = rotation_difference_deg(np.asarray(first['T_base_flange'])[:3, :3],
                                        np.asarray(other['T_base_flange'])[:3, :3])
        delta = {}
        for parent, candidates in first['candidates'].items():
            for name in ('legacy', 'upright'):
                p, q = candidates[name]['median_base_m'], other['candidates'][parent][name]['median_base_m']
                delta[f"{parent}/{name}"] = None if p is None or q is None else float(
                    1000 * np.linalg.norm(np.asarray(q) - p))
        pairs.append(dict(first=first['file'], second=other['file'], flange_translation_mm=(1000 * arm_delta).tolist(),
                          flange_rotation_delta_deg=angle, cube_center_delta_mm=delta))
    report = dict(calibration=str(calibration.path), configured_parent=calibration.camera_parent,
                  calibration_sha256=file_sha256(calibration.path), samples=runs, comparisons=pairs)
    write_json(args.output, report, ensure_ascii=False)
    print(json.dumps(dict(output=str(args.output), comparisons=pairs), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("capture", help="只读采样静止姿态")
    live.add_argument("--ip", required=True)
    live.add_argument("--frames", type=int, default=8)
    audit = commands.add_parser("compare", help="离线比较已有 RGB-D 采样")
    audit.add_argument("samples", type=Path, nargs="+")
    for command in (live, audit):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    if args.output.exists() or (args.command == "capture" and not 2 <= args.frames <= 100):
        parser.error("需要新的输出路径，采样帧数为 2–100")
    calibration = load_calibration(args.calibration)
    if calibration.camera_parent not in ("flange", "tcp"):
        parser.error("当前诊断针对眼在手上相机")
    if args.command == "capture":
        capture(args, calibration)
    else:
        compare(args, calibration)


if __name__ == "__main__":
    main()
