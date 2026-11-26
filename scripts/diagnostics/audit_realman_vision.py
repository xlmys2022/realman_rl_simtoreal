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
from realman_rl_simtoreal.perception.rgbd import OrangeCubeDetector, RgbdFrame, UprightCubeDetector, backproject
from realman_rl_simtoreal.utils.artifacts import write_json
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
    robot['calibration_sha256'] = calibration.checksum()
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


def measure_local_support(frame, mask, box, edge_m, expected_table_z):
    """拟合方块周围点云，仅诊断 Z 斜率与偏差，不在抓取或仿真中使用拟合面。"""
    import cv2

    u0, v0 = np.rint(box.centroid_px).astype(int)
    height, width = frame.depth_m.shape
    v, u = np.mgrid[max(0, v0 - 65):min(height, v0 + 66):3, max(0, u0 - 65):min(width, u0 + 66):3]
    depth = frame.depth_m[v, u]
    excluded = cv2.dilate(mask, np.ones((19, 19), np.uint8))[v, u] != 0
    valid = (depth > 0.02) & (depth < 3.0) & ~excluded
    if valid.sum() < 80:
        return None
    cloud = backproject(frame, np.c_[u[valid], v[valid]])
    z = cloud[:, 2]
    center_z = box.position_m[2]
    nearby = cloud[(z < center_z - 0.2 * edge_m) & (z > center_z - 2 * edge_m)]
    if len(nearby) < 80:
        return None
    xy = np.c_[nearby[:, :2] - box.position_m[:2], np.ones(len(nearby))]
    point_z = nearby[:, 2]
    selected = np.ones(len(point_z), dtype=bool)
    for _ in range(4):
        coeff, _, rank, _ = np.linalg.lstsq(xy[selected], point_z[selected], rcond=None)
        if rank != 3:
            return None
        error = xy @ coeff - point_z
        selected = abs(error) < 0.002
        if selected.sum() < max(80, 0.7 * len(point_z)):
            return None
    support_z = coeff[2]
    return dict(z_at_cube_m=float(support_z), support_z_error_mm=1000 * float(support_z - expected_table_z),
                cube_center_above_support_mm=1000 * float(center_z - support_z),
                apparent_dz_dx=float(coeff[0]), apparent_dz_dy=float(coeff[1]),
                inlier_residual_mm=1000 * float(np.sqrt(np.mean(error[selected] ** 2))))


def analyze_capture(path, calibration, *, capture_calibration=None, table_z_base_m=None):
    with np.load(path, allow_pickle=False) as source:
        rgb, depth, K, flanges = (source[k] for k in ("rgb", "depth_m", "K", "T_base_flange"))
        stamps = source['host_received_s']
        metadata = json.loads(source['metadata'].item()) if 'metadata' in source else {}
        reported = source['controller_pose_m_rad'] if 'controller_pose_m_rad' in source else None
    # 离线替换外参时仍按采样原配置验证哈希，报告同时记录实际回放的配置。
    original = capture_calibration or calibration
    if metadata.get('calibration_sha256') not in (None, original.checksum()):
        raise ValueError(f"{path} 与采样标定文件校验值不符，使用 --capture-calibration 指定原配置")
    serial = metadata.get('camera', {}).get('serial')
    if calibration.camera_serial is not None and serial not in (None, calibration.camera_serial):
        raise ValueError(f"{path} 的采样相机序列号 {serial} 与回放标定不符")
    H = calibration.T_parent_color
    tcp = calibration.T_flange_tcp
    alternatives = dict(flange=H, tcp=tcp @ H, flange_inverse=inverse(H), tcp_inverse=tcp @ inverse(H))
    table_z = -REALMAN_TABLETOP.base_above_table_m if table_z_base_m is None else table_z_base_m
    expected_z = table_z + calibration.cube_edge_m / 2
    report = dict(file=str(path), frames=len(stamps), expected_table_z_base_m=table_z,
                  expected_cube_center_z_base_m=expected_z,
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
            base_points, color_points, support = [], [], []
            audit_height = table_z_base_m is not None and name == calibration.camera_parent and estimator == "upright"
            for pixels, distances, optical, stamp in zip(rgb, depth, T, stamps):
                frame = RgbdFrame(pixels, distances, K, optical, float(stamp))
                measurement = detector.detect(frame)
                if measurement is not None:
                    p = measurement.position_m
                    base_points.append(p)
                    color_points.append(optical[:3, :3].T @ (p - optical[:3, 3]))
                    if audit_height:
                        fitted = measure_local_support(frame, detector.mask, measurement, calibration.cube_edge_m, table_z)
                        if fitted is not None:
                            support.append(fitted)
            candidate[estimator] = summarize_centers(base_points, color_points, expected_z)
            if audit_height:
                candidate[estimator]["support_frames"] = len(support)
                candidate[estimator]["support_median"] = (
                    {key: float(np.median([record[key] for record in support])) for key in support[0]}
                    if support else None)
        report['candidates'][name] = candidate
    return report


def compare(args, calibration):
    original = load_calibration(args.capture_calibration) if args.capture_calibration else calibration
    runs = [analyze_capture(path, calibration, capture_calibration=original, table_z_base_m=args.table_z_base_m)
            for path in args.samples]
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
                  calibration_sha256=calibration.checksum(), capture_calibration=str(original.path),
                  capture_calibration_sha256=original.checksum(), samples=runs, comparisons=pairs)
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
    audit.add_argument("--capture-calibration", type=Path,
                       help="采样时使用的原标定 JSON，用于核对保存的校验值")
    audit.add_argument("--table-z-base-m", type=float, help="实测水平桌面的基座 Z（m），用于离线诊断")
    for command in (live, audit):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    if args.output.exists() or (args.command == "capture" and not 2 <= args.frames <= 100):
        parser.error("需要新的输出路径，采样帧数为 2–100")
    if args.command == "compare" and args.table_z_base_m is not None and not np.isfinite(args.table_z_base_m):
        parser.error("--table-z-base-m 必须为有限数值")
    calibration = load_calibration(args.calibration)
    if calibration.camera_parent not in ("flange", "tcp"):
        parser.error("当前诊断针对眼在手上相机")
    if args.command == "capture":
        capture(args, calibration)
    else:
        compare(args, calibration)


if __name__ == "__main__":
    main()
