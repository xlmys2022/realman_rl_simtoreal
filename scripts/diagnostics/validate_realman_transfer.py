"""离线回放现场 RGB-D，并在同一 TCP 位姿对照两种 MuJoCo 夹爪装配。"""

import argparse
import os
from pathlib import Path

import numpy as np

from realman_rl_simtoreal.deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from realman_rl_simtoreal.deployment.policy_io import RealVisualPolicyIO
from realman_rl_simtoreal.deployment.rgbd_recording import load_recording
from realman_rl_simtoreal.deployment.vision import cube_outline_yaw
from realman_rl_simtoreal.hardware.realman_config import REALMAN_TABLETOP
from realman_rl_simtoreal.perception.rgbd import RgbdFrame, UprightCubeDetector
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig
from realman_rl_simtoreal.tasks.grasp.tabletop_policy_io import action_contract, vision_contract
from realman_rl_simtoreal.utils.artifacts import write_json


def replay(path, config, calibration):
    rgb, depth, K, flanges, stamps, metadata, _ = load_recording(path, calibration, calibration)
    checksum = calibration.checksum()
    work, tool = (metadata.get('active_frames', {}).get(name, {}) for name in ('work', 'tool'))
    installation = REALMAN_TABLETOP
    if (metadata.get('calibration_sha256') != checksum or not calibration.camera_serial
            or metadata.get('camera', {}).get('serial') != calibration.camera_serial):
        raise ValueError(f'{path} requires the recorded calibration checksum and camera serial')
    if (work.get('name') != 'Base' or tool.get('name') != installation.tcp_name
            or not np.allclose(work['pose'], np.zeros(6), atol=1e-6, rtol=0)
            or not np.allclose(tool['pose'], [0, 0, installation.flange_tcp_m, 0, 0, 0], atol=1e-6, rtol=0)):
        raise ValueError(f'{path} requires the recorded Base / {installation.tcp_name} frames')
    if len(stamps) < 2:
        raise ValueError(f'{path} requires at least two frames to measure cadence')
    interface = RealVisualPolicyIO(config)
    positions, counts = [], []
    outlines = 0
    for pixels, distances, flange, stamp in zip(rgb, depth, flanges, stamps, strict=True):
        frame = RgbdFrame(pixels, distances, K, calibration.camera_to_base(flange), float(stamp))
        measurement = interface.ingest_frame(frame)
        if measurement is None:
            continue
        positions.append(measurement.position_m)
        counts.append(measurement.pixels)
        top_z = measurement.position_m[2] + calibration.cube_edge_m / 2
        outlines += cube_outline_yaw(frame, interface.tracker.detector.mask, top_z, calibration.cube_edge_m) is not None

    # 支撑真值只用于离线误差统计。真实位姿保持采样时的刚体旋转，使用离批次中央最近的一帧。
    T_base_tcp = flanges[len(stamps)//2] @ calibration.T_flange_tcp
    center = np.median(positions, axis=0) if positions else None
    expected_z = -installation.base_above_table_m + calibration.cube_edge_m / 2
    return T_base_tcp, dict(recording=str(path), frames=len(stamps), detected_frames=len(positions), outline_frames=outlines,
                T_base_tcp=T_base_tcp.tolist(), median_sample_interval_s=float(np.median(np.diff(stamps))),
                observed_hz=float((len(stamps)-1)/(stamps[-1]-stamps[0])),
                median_detected_pixels=float(np.median(counts)) if positions else None,
                center_base_m=center.tolist() if positions else None,
                center_z_error_m=float(center[2]-expected_z) if positions else None,
                max_center_spread_m=float(np.max(np.linalg.norm(positions-center, axis=1))) if positions else None,
                raw_gripper_snapshot=metadata.get('gripper'), complete_policy_proprioception=False)


def simulate_poses(config, poses, reference_xy, seed_joints_deg):
    import mujoco
    from realman_rl_simtoreal.simulation.devices.mujoco_rgbd_camera import MujocoRgbdCamera
    from realman_rl_simtoreal.simulation.runtime.tabletop import TabletopRuntime

    robot = TabletopRuntime(config)
    camera = MujocoRgbdCamera(robot, width=config.vision_width, height=config.vision_height,
                             depth_unit_m=REALMAN_TABLETOP.depth_unit_m)
    detector = UprightCubeDetector(edge_m=config.vision_cube_size,
                                   hsv_low=config.vision_hsv_low, hsv_high=config.vision_hsv_high)
    results = []
    try:
        for pose in poses:
            robot.reset_zero(reference_xy)
            base_position, base_rotation = camera.base_pose()
            result = robot.kinematics.solve_ik(base_position + base_rotation @ pose[:3, 3],
                                               np.deg2rad(seed_joints_deg), base_rotation @ pose[:3, :3],
                                               max_iterations=200, max_step=.05, position_tolerance=2e-6)
            if not result.success:
                results.append(dict(ik_success=False, reason=result.reason))
                continue
            safe, reason = robot.reference_is_safe(result.q)
            if not safe:
                results.append(dict(ik_success=True, pose_safe=False, reason=reason))
                continue
            robot.data.qpos[robot.kinematics.qadr] = result.q
            mujoco.mj_forward(robot.model, robot.data)
            left, right = robot.gripper.pad_positions()
            _, R = robot.tcp_pose()
            pad_axis = R.T @ ((right-left)/np.linalg.norm(right-left))
            measurement = detector.detect(camera.capture())
            center = measurement.position_m if measurement is not None else None
            expected_center = [*reference_xy, robot.table_height-robot.base_position[2]+robot.cube_half_edge]
            results.append(dict(ik_success=True, pose_safe=True, jaw_axis_tcp=pad_axis.tolist(),
                                center_base_m=center.tolist() if center is not None else None,
                                center_error_base_m=(center-expected_center).tolist() if center is not None else None,
                                detected_pixels=measurement.pixels if measurement is not None else 0))
    finally:
        camera.close()
    return results


def main():
    os.environ.setdefault('MUJOCO_GL', 'egl')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', action='append', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--reference-xy-m', nargs=2, type=float, required=True, metavar=('X', 'Y'),
                        help='配对渲染中静止方块的同一 Base XY。可取无遮挡位置的可靠估计。')
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists() or not np.isfinite(args.reference_xy_m).all():
        parser.error('需要新的输出文件和有限的方块参考 XY')
    calibration = load_calibration(args.calibration)
    config = CartesianTabletopConfig(scene_profile='realman_transfer', hand_eye_file=str(calibration.path))
    pairs = [replay(path, config, calibration) for path in args.recording]
    poses, recorded = zip(*pairs)
    from realman_rl_simtoreal.simulation.runtime.observation_presets import observation_preset

    seeds = observation_preset('table_down', config.scene_profile).joints_deg
    comparison = config.with_scene('realman_hand_eye', hand_eye_file=calibration.path)
    original = simulate_poses(comparison, poses, args.reference_xy_m, seeds)
    updated = simulate_poses(config, poses, args.reference_xy_m, seeds)
    rows = [dict(real=real, original_jaw_mount=before, transfer=after)
            for real, before, after in zip(recorded, original, updated, strict=True)]
    report = dict(vision_contract=vision_contract(config), action_contract=action_contract(config),
                  calibration=str(calibration.path), ground_reference_xy_m=args.reference_xy_m,
                  measured_closing_axis_tcp=REALMAN_TABLETOP.gripper_closing_axis_tcp,
                  ground_reference_xy_source='caller_supplied_comparison_reference',
                  real_missing_proprioception=['gripper_width_m and velocity away from measured endpoints',
                      'jaw_reference_m', 'controller TCP reference and IK fraction', 'synchronized dynamic TCP velocity'],
                  same_detector_calibration_and_depth_quantization_for_sim_comparison=True, poses=rows)
    write_json(args.json, report, ensure_ascii=False)
    for row in rows:
        real, before, after = row['real'], row['original_jaw_mount'], row['transfer']
        print(Path(real['recording']).stem, f"RGB-D {real['detected_frames']}/{real['frames']}",
              f"轮廓 {real['outline_frames']}/{real['frames']}",
              f"夹爪调整前后像素 {before.get('detected_pixels')} → {after.get('detected_pixels')}")
    return int(not all(row['transfer'].get('pose_safe') for row in rows))


if __name__ == '__main__':
    raise SystemExit(main())
