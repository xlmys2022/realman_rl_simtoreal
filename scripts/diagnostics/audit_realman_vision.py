#!/usr/bin/env python3
"""只读保存静止 RGB-D 采样，离线核对方块坐标链和实测棋盘高度。"""

import argparse
from dataclasses import asdict, replace
from itertools import combinations
import json
import os
from pathlib import Path

import numpy as np

from realman_rl_simtoreal.deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from realman_rl_simtoreal.deployment.rgbd_recording import load_recording
from realman_rl_simtoreal.deployment.vision import RealCubeLocalizer
from realman_rl_simtoreal.deployment.z_reference import (
    BoardReference, board_corners, board_pose, fit_base_leveling, measure_board, upward_normal,
)
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
            for _ in range(args.warmup_frames):
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
    original = capture_calibration or calibration
    rgb, depth, K, flanges, stamps, metadata, reported = load_recording(path, calibration, original)
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
        T = calibration.camera_to_base(flanges) if name == calibration.camera_parent else flanges @ attachment
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


def analyze_board_capture(path, calibration, original, reference):
    rgb, depth, K, flanges, stamps, metadata, _ = load_recording(path, calibration, original)
    if (not calibration.camera_serial or metadata.get('camera', {}).get('serial') != calibration.camera_serial
            or metadata.get('calibration_sha256') != original.checksum()):
        raise ValueError(f"{path} 缺少与回放匹配的相机序列号或采集标定校验值")
    count = len(stamps)
    if count < 3 or any(len(data) != count for data in (rgb, depth, flanges)):
        raise ValueError(f"{path} 至少需要三组配对的静止姿态和彩色深度帧")
    drift_m = float(np.max(np.linalg.norm(flanges[:, :3, 3] - flanges[0, :3, 3], axis=1)))
    drift_deg = max(rotation_difference_deg(flanges[0, :3, :3], tf[:3, :3]) for tf in flanges)
    if drift_m > 0.001 or drift_deg > 0.2:
        raise ValueError(f"{path} 采样批次未保持静止 · {1000 * drift_m:.2f} mm · {drift_deg:.3f}°")
    accepted, rejected = [], []
    for index, (pixels, distances, flange, stamp) in enumerate(zip(rgb, depth, flanges, stamps)):
        optical = calibration.camera_to_base(flange)
        frame = RgbdFrame(pixels, distances, K, optical, float(stamp))
        try:
            measurement = measure_board(frame, reference)
        except ValueError as exc:
            rejected.append(dict(frame=index, reason=str(exc)))
        else:
            accepted.append(dict(frame=index, **measurement))
    summary = ({key: float(np.median([row[key] for row in accepted])) for key in accepted[0] if key != "frame"}
               if accepted else None)
    return dict(file=str(path), frames=count, usable_frames=len(accepted), enough_frames=len(accepted) >= 3,
                flange_drift_mm=1000 * drift_m, flange_drift_deg=drift_deg, medians=summary,
                measurements=accepted, rejected_frames=rejected)


def compare_reference(args, calibration):
    original = load_calibration(args.capture_calibration) if args.capture_calibration else calibration
    reference = reference_from_args(args)
    if args.write_leveling_calibration:
        calibration = replace(calibration, base_leveling=None)
    runs = [analyze_board_capture(path, calibration, original, reference) for path in args.samples]
    report = dict(reference=asdict(reference), expected_top_z_base_m=reference.top_z_base_m,
                  calibration=str(calibration.path), calibration_sha256=calibration.checksum(),
                  camera_serial=calibration.camera_serial, capture_calibration_sha256=original.checksum(), samples=runs)
    enough_frames = all(row['enough_frames'] for row in runs)
    if args.write_leveling_calibration and enough_frames:
        report['base_leveling_fit'] = write_leveling_calibration(args.write_leveling_calibration,
                                                                calibration, original, reference, runs, args.samples)
    write_json(args.output, report, ensure_ascii=False)
    print(json.dumps(dict(output=str(args.output), samples=[{key: row[key] for key in
                     ("file", "usable_frames", "frames", "medians")} for row in runs]), ensure_ascii=False))
    return 0 if enough_frames else 2


def write_leveling_calibration(path, calibration, original, reference, runs, samples):
    """从棋盘 RGB 拟合水平校正，并记录整组留出误差及校正后 RGB-D 残差。"""
    top_z = reference.top_z_base_m
    medians = [row['medians'] for row in runs]
    centers = np.array([[m['rgb_board_center_x_m'], m['rgb_board_center_y_m'],
                         top_z + m['rgb_board_center_base_z_error_mm'] / 1000] for m in medians])
    normals = np.array([[m[f'rgb_normal_{axis}_base'] for axis in 'xyz'] for m in medians])
    leveling = fit_base_leveling(centers, normals, top_z)
    held_errors = []
    for i, row in enumerate(runs):
        train = np.arange(len(runs)) != i
        held_fit = fit_base_leveling(centers[train], normals[train], top_z)
        held_z = held_fit[2, :3] @ centers[i] + held_fit[2, 3]
        held_errors.append(dict(file=row['file'], rgb_board_center_z_error_mm=1000 * float(held_z - top_z)))

    corrected = replace(calibration, base_leveling=leveling)
    evaluated = [analyze_board_capture(sample, corrected, original, reference) for sample in samples]
    data = json.loads(calibration.path.read_text(encoding='utf-8-sig'))
    data['hsv_file'] = os.path.relpath(calibration.hsv_path, start=path.resolve().parent)
    data['base_leveling'] = leveling.tolist()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return dict(calibration=str(path), calibration_sha256=load_calibration(path).checksum(),
                base_leveling=leveling.tolist(), rotation_deg=rotation_difference_deg(np.eye(3), leveling[:3, :3]),
                leave_one_batch_out=held_errors, corrected_samples=evaluated)


def reference_from_args(args):
    return BoardReference(*args.board_corners, args.board_spans_mm[0] / 1000, args.board_spans_mm[1] / 1000,
                          args.board_top_mm / 1000, args.table_z_base_m)


def summarize_board_poses(name, targets, color_board, reference):
    center = np.median(targets[:, :3, 3], axis=0)
    normals = [upward_normal(row[:3, 2]) for row in targets]
    normal = upward_normal(np.median(normals, axis=0))
    return dict(sample=name, frames=len(targets), center_base_m=center.tolist(),
                center_base_z_error_mm=1000 * float(center[2] - reference.top_z_base_m),
                center_optical_z_m=float(np.median(color_board[:, 2, 3])), up_normal_base=normal.tolist(),
                normal_tilt_deg=float(np.rad2deg(np.arccos(np.clip(normal[2], -1, 1)))))


def pairwise_board_differences(samples):
    comparisons = []
    for a, b in combinations(samples, 2):
        shift = 1000 * (np.asarray(b["center_base_m"]) - a["center_base_m"])
        cosine = np.dot(a["up_normal_base"], b["up_normal_base"])
        comparisons.append(dict(from_sample=a["sample"], to_sample=b["sample"],
                                center_xyz_shift_mm=shift.tolist(), center_xy_shift_mm=float(np.linalg.norm(shift[:2])),
                                center_3d_shift_mm=float(np.linalg.norm(shift)),
                                normals_angle_deg=float(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))))))
    return comparisons


def compare_hand_eye(args, calibration):
    """报告原数据残差以及独立棋盘 RGB-D 检查，候选矩阵只存在于诊断报告中。"""
    import cv2

    from realman_rl_simtoreal.deployment.hand_eye_audit import audit_hand_eye, read_calibration_views

    if calibration.camera_parent != "tcp":
        raise ValueError("Grip_plus 原始标定位姿需要以 TCP 为父系的部署文件")
    if args.cube_samples and not args.cube_capture_calibration:
        raise ValueError("回放方块采样需要提供 --cube-capture-calibration")
    if not callable(getattr(cv2, "calibrateHandEye", None)):
        raise ValueError("当前 OpenCV 没有 calibrateHandEye。请使用原 handeye 环境的 OpenCV 4.x")

    reference = reference_from_args(args)
    historical_tcp, history_pixels, history_size = read_calibration_views(args.legacy_dir, reference)
    records = []
    for path in args.samples:
        reference_result = analyze_board_capture(path, calibration, calibration, reference)
        if not reference_result["enough_frames"]:
            raise ValueError(f"{path} 未通过至少三组独立棋盘 RGB-D 的质量校核")
        rgb, _, K, flanges, _, metadata, controller_pose = load_recording(path, calibration, calibration)
        size = (rgb.shape[2], rgb.shape[1])
        camera = metadata["camera"]
        if (camera.get("color_distortion_model") != "inverse_brown_conrady"
                or camera.get("color_distortion_coefficients") != [0] * 5):
            raise ValueError(f"{path} 缺少零畸变 D435 RGB 配置的实测元数据")
        if records and (size != records[0]["rgb_size"] or not np.array_equal(K, records[0]["K"])):
            raise ValueError("独立 RGB-D 样本的相机内参与分辨率必须一致")
        idx = [entry["frame"] for entry in reference_result["measurements"]]
        pixels = [board_corners(rgb[i], reference) for i in idx]
        tcp = flanges[idx] @ calibration.T_flange_tcp
        frames = metadata.get("active_frames", {})
        if controller_pose is None or frames.get("work", {}).get("name") != "Base" or frames.get("tool", {}).get("name") != "Grip_plus":
            raise ValueError(f"{path} 缺少 Base / Grip_plus 活动坐标系和 SDK 位姿记录")
        via_sdk = np.stack([pose_matrix(controller_pose[i]) for i in idx])
        reference_result.update(
            nominal_z16_units_m=camera.get("nominal_z16_units_m"),
            tcp_flange_fk_sdk_max_xyz_mm=1000 * float(np.linalg.norm(tcp[:, :3, 3] - via_sdk[:, :3, 3], axis=1).max()),
            tcp_flange_fk_sdk_max_rotation_deg=max(rotation_difference_deg(a[:3, :3], b[:3, :3]) for a, b in zip(tcp, via_sdk)))
        records.append(dict(name=path.stem, pixels=pixels, tcp=tcp, K=K, rgb_size=size, reference=reference_result))

    points = reference.points_m()
    history, candidates = audit_hand_eye(historical_tcp, history_pixels, history_size, points, records[0]["K"],
                                         records[0]["rgb_size"], calibration.T_parent_color)
    for variant, (K, d, alternatives) in candidates.items():
        paths = []
        for sample in records:
            fitted = [board_pose(points, uv, K, d) for uv in sample["pixels"]]
            paths.append((sample, np.stack([pose for pose, _ in fitted]),
                          float(np.sqrt(np.mean([error ** 2 for _, error in fitted])))))
        for method, hand_eye in alternatives.items():
            board_poses = [dict(**summarize_board_poses(sample["name"], sample["tcp"] @ hand_eye @ color_board,
                                                       color_board, reference), image_reprojection_rms_px=error)
                           for sample, color_board, error in paths]
            history[variant]["models"][method]["held_out_rgbd_reference_rgb_only"] = dict(
                samples=board_poses, pairs=pairwise_board_differences(board_poses))

    cube_support = []
    if args.cube_samples:
        prior = load_calibration(args.cube_capture_calibration)
        for path in args.cube_samples:
            cube = analyze_capture(path, calibration, capture_calibration=prior, table_z_base_m=reference.table_z_base_m)
            estimate = cube["candidates"]["tcp"]["upright"]
            support = estimate["support_median"]
            if support is None:
                raise ValueError(f"{path} 没有可用的方块附近支撑面")
            slope = np.array([support["apparent_dz_dx"], support["apparent_dz_dy"]])
            n = upward_normal(np.r_[-slope, 1.])
            slope_height_m = float(slope @ estimate["median_base_m"][:2])
            cube_support.append(dict(sample=path.stem, support_frames=estimate["support_frames"],
                                     up_normal_base=n.tolist(), ground_tilt_deg=float(np.rad2deg(np.arccos(n[2]))),
                                     cube_center_m=estimate["median_base_m"],
                                     cube_center_z_error_mm=estimate["center_z_error_mm"],
                                     diagnostic_slope_height_at_cube_mm=1000 * slope_height_m,
                                     extrapolated_support_z_at_base_xy_zero_m=support["z_at_cube_m"] - slope_height_m,
                                     **support))

    report = dict(opencv_version=cv2.__version__, calibration=str(calibration.path), calibration_sha256=calibration.checksum(),
                  camera_serial=calibration.camera_serial, reference=asdict(reference), expected_top_z_base_m=reference.top_z_base_m,
                  historical_folder=str(args.legacy_dir), historical_images=len(history_pixels), historical_size=history_size,
                  native_rgb_size=records[0]["rgb_size"], intrinsics_and_hand_eye=history,
                  held_out_board_depth=[sample["reference"] for sample in records], cube_nearby_support=cube_support)
    write_json(args.output, report, ensure_ascii=False)
    print(json.dumps(dict(output=str(args.output), historical_views=len(history_pixels),
                          independent_rgbd_views=sum(sample["reference"]["usable_frames"] for sample in records)), ensure_ascii=False))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    live = commands.add_parser("capture", help="只读采样静止姿态")
    live.add_argument("--ip", required=True)
    live.add_argument("--frames", type=int, default=8)
    live.add_argument("--warmup-frames", type=int, default=1, help="丢弃指定数量的开流配对帧（1–300）")
    audit = commands.add_parser("compare", help="离线比较已有 RGB-D 采样")
    audit.add_argument("samples", type=Path, nargs="+")
    audit.add_argument("--capture-calibration", type=Path,
                       help="采样时使用的原标定 JSON，用于核对保存的校验值")
    audit.add_argument("--table-z-base-m", type=float, help="实测水平桌面的基座 Z（m），用于离线诊断")
    board = commands.add_parser("reference", help="用实测棋盘格区分光轴深度与基座 Z 残差")
    board.add_argument("--capture-calibration", type=Path, help="采样时使用的原标定 JSON")
    board.add_argument("--write-leveling-calibration", type=Path, metavar="JSON",
                       help="使用多组实测水平棋盘的 RGB 重建结果生成新的基座水平校正标定")
    hand_eye = commands.add_parser("hand-eye", help="离线比较存档手眼算法、内参与独立 RGB-D 法向")
    hand_eye.add_argument("--legacy-dir", type=Path, required=True, help="原始 poses.txt 与 1.jpg 至 N.jpg 所在目录")
    hand_eye.add_argument("--cube-samples", type=Path, nargs="+", help="可选，用旧方块采样复查实际桌面支撑法向")
    hand_eye.add_argument("--cube-capture-calibration", type=Path, help="采集旧方块采样时的标定 JSON")
    for command in (board, hand_eye):
        command.add_argument("samples", type=Path, nargs="+")
        command.add_argument("--board-corners", type=int, nargs=2, required=True, metavar=("COLS", "ROWS"))
        command.add_argument("--board-spans-mm", type=float, nargs=2, required=True, metavar=("X", "Y"),
                             help="首末内角点横向和竖向的实测跨度，单位 mm")
        command.add_argument("--board-top-mm", type=float, required=True, help="图案表面高出桌面的实测高度，单位 mm")
        command.add_argument("--table-z-base-m", type=float, required=True, help="水平桌面的基座 Z，单位 m")
    for command in (live, audit, board, hand_eye):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, required=command in (board, hand_eye))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("需要新的输出路径")
    if args.command == "capture" and (not 2 <= args.frames <= 100 or not 1 <= args.warmup_frames <= 300):
        parser.error("采样帧数须为 2–100，开流丢弃帧数须为 1–300")
    if args.command == "compare" and args.table_z_base_m is not None and not np.isfinite(args.table_z_base_m):
        parser.error("--table-z-base-m 必须为有限数值")
    if args.command == "reference" and args.write_leveling_calibration:
        if len({path.resolve() for path in args.samples}) != len(args.samples) or len(args.samples) < 2:
            parser.error("基座水平校正需要至少两组不同位置或姿态的独立录像")
        if args.write_leveling_calibration.exists() or args.write_leveling_calibration.resolve() == args.output.resolve():
            parser.error("水平校正标定需要单独的新输出路径")
    calibration = load_calibration(args.calibration)
    if calibration.camera_parent not in ("flange", "tcp"):
        parser.error("当前诊断针对眼在手上相机")
    if args.command == "capture":
        return capture(args, calibration)
    if args.command == "reference":
        return compare_reference(args, calibration)
    if args.command == "hand-eye":
        return compare_hand_eye(args, calibration)
    return compare(args, calibration)


if __name__ == "__main__":
    raise SystemExit(main())
