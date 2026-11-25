#!/usr/bin/env python3
"""验收独立安装场景的 TCP、随机台面点及传统分段抓取。"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from realman_rl_simtoreal.evaluation.tabletop_cartesian import rollout, summarize_metrics
from realman_rl_simtoreal.envs.factory import make_tabletop_cartesian_env
from realman_rl_simtoreal.hardware.realman_config import REALMAN_TABLETOP
from realman_rl_simtoreal.simulation.devices.mujoco_gripper import MAX_WIDTH, SIGNS, width_to_angle
from realman_rl_simtoreal.simulation.runtime.tabletop import TabletopRuntime
from realman_rl_simtoreal.simulation.motion.staged_grasp_controller import GraspLiftController
from realman_rl_simtoreal.simulation.runtime.observation_presets import observation_preset
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig
from realman_rl_simtoreal.utils.artifacts import write_json
from realman_rl_simtoreal.utils.geometry import euler_matrix, transform


class TabletopRule:
    """传统控制先在桌面上空对齐，再复用现有夹取、预紧和抬升分段控制。"""

    def __init__(self, env):
        self.env = env
        self.waypoints = None
        self.grasp = None

    def __call__(self, _observation):
        env = self.env
        runtime = env.robot
        if self.waypoints is None:
            initial = env.goal - [0., 0., env.config.lift_height]
            pregrasp = initial + [0., 0., .043]
            grasp = initial + [0., 0., .003]
            lift = grasp + [0., 0., env.config.lift_height]
            self.waypoints = [np.r_[initial[:2], env.config.observation_tcp[2]], pregrasp]
            self.grasp = GraspLiftController(
                dict(pregrasp_reference_m=pregrasp, grasp_reference_m=grasp,
                     lift_reference_m=lift, support_normal=[0., 0., 1.]),
                action_scale=env.config.action_scale, control_period=env.config.control_period)
        tcp, _ = runtime.tcp_pose()
        if self.waypoints and np.linalg.norm(tcp - self.waypoints[0]) <= .0008:
            self.waypoints.pop(0)
        if self.waypoints:
            delta = (self.waypoints[0] - env.controller.reference) / env.config.action_scale
            return np.r_[np.clip(delta, -1, 1), 2 * env.config.initial_width / MAX_WIDTH - 1]
        object_xyz = runtime.data.qpos[runtime.object_qadr:runtime.object_qadr + 3]
        row = runtime.recorder.last
        return self.grasp.predict(
            dict(command_position_m=env.controller.reference,
                 bilateral_contact=min(runtime.forces()[:2]) > .005,
                 object_position_m=object_xyz, grasp_position_m=tcp if row is None else
                 object_xyz - runtime.data.xmat[runtime.gripper.base].reshape(3, 3)
                 @ row['object_relative_to_pads_m']))


def attachment_inertia(runtime):
    """在 20 mm 开口下将 MJCF 原附件惯量合成到法兰系。"""
    runtime.reset_zero(runtime.config.object_xy)
    model, data = runtime.model, runtime.data
    for name, sign in SIGNS.items():
        data.qpos[model.joint(name).qposadr[0]] = sign * width_to_angle(.02)
    mujoco.mj_kinematics(model, data)
    flange = model.body('link_6').id
    link_names = [model.body(i).name for i in range(model.nbody)]
    bodies = np.array([i for i, name in enumerate(link_names)
                       if name.startswith('4C2_') or name == 'eg2_experimental_adapter'])
    flange_rotation = data.xmat[flange].reshape(3, 3)
    positions = (data.xipos[bodies] - data.xpos[flange]) @ flange_rotation
    masses = model.body_mass[bodies]
    center = np.average(positions, axis=0, weights=masses)
    tensor = np.zeros((3, 3))
    for i, mass, delta in zip(bodies, masses, positions - center):
        rotation = flange_rotation.T @ data.ximat[i].reshape(3, 3)
        own = rotation @ np.diag(model.body_inertia[i]) @ rotation.T
        tensor += own + mass * ((delta @ delta) * np.eye(3) - np.outer(delta, delta))
    return dict(total_mass_kg=float(masses.sum()), center_of_mass_flange_m=center.tolist(),
                inertia_at_assembly_centroid_flange_kg_m2=tensor.tolist())


def geometry_report(runtime):
    legacy = TabletopRuntime(CartesianTabletopConfig())
    actual = REALMAN_TABLETOP
    return dict(
        world_base_m=runtime.base_position.tolist(), world_table_z_m=runtime.table_height,
        base_table_z_m=runtime.table_height - runtime.base_position[2],
        base_cube_center_z_m=runtime.table_height + runtime.cube_half_edge - runtime.base_position[2],
        assembly_comparison=dict(legacy=attachment_inertia(legacy), realman_nominal=attachment_inertia(runtime),
                                 pendant=dict(payload_kg=actual.tool_payload_kg,
                                              reported_center_m=actual.tool_center_of_mass_m)),
    )


def controller_pose_check(runtime, readback):
    """用同一组实机关节角交叉检查 MuJoCo 法兰、Grip_plus 和活动工具回包。"""
    q = np.deg2rad(readback['joint_deg'])
    flange_p, flange_R = runtime.kinematics.body_pose(q, 'link_6')
    tcp_p, tcp_R = runtime.kinematics.fk(q)
    T_base_flange = transform(flange_p - runtime.base_position, flange_R)
    T_base_tcp = transform(tcp_p - runtime.base_position, tcp_R)
    work = np.asarray(readback['controller_frames']['work']['pose'])
    reported = np.asarray(readback['controller_pose_m_rad'])
    T_base_reported = transform(work[:3], euler_matrix(work[3:])) @ transform(
        reported[:3], euler_matrix(reported[3:]))
    return dict(joint_deg=readback['joint_deg'],
                T_base_flange_mujoco=T_base_flange.tolist(), T_base_tcp_mujoco=T_base_tcp.tolist(),
                T_base_active_tool_reported=T_base_reported.tolist(),
                flange_xyz_error_mm=float(1000*np.linalg.norm(T_base_flange[:3, 3] -
                                              np.asarray(readback['T_base_flange'])[:3, 3])),
                active_tool=readback['controller_frames']['tool']['name'],
                tcp_xyz_error_mm=float(1000*np.linalg.norm(T_base_tcp[:3, 3] - T_base_reported[:3, 3])),
                tcp_rotation_error_deg=float(np.rad2deg(np.linalg.norm(
                    runtime.rotation_error(T_base_tcp[:3, :3], T_base_reported[:3, :3])))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=997100)
    parser.add_argument("--object-radius", type=float, default=0.)
    parser.add_argument("--geometry-only", action="store_true", help="仅输出高度和附件动力学审计")
    parser.add_argument("--controller-readback", type=Path, help="比较已有的控制器状态，仿真程序保持离线")
    args = parser.parse_args()
    if args.json.exists() or args.episodes < 1:
        parser.error("需要新的 JSON 路径和正数回合")
    config = CartesianTabletopConfig(scene_profile="realman_nominal", object_radius=args.object_radius)
    env = make_tabletop_cartesian_env(config)
    try:
        geometry = geometry_report(env.robot)
        if args.controller_readback:
            import json
            sample = json.loads(args.controller_readback.read_text())
            geometry['live_controller_pose_comparison'] = controller_pose_check(env.robot, sample)
        if args.geometry_only:
            write_json(args.json, geometry)
            return 0
        episodes = [rollout(env, TabletopRule(env), seed, detailed_metrics=True, compact=True)
                    for seed in range(args.seed, args.seed + args.episodes)]
        summary = summarize_metrics(episodes, visual=False)
        summary.update(successes=sum(row["success"] for row in episodes), episodes=len(episodes))
        # 在桌面半径 50 mm 的边界用同一夹取姿态做 TCP 逆解和碰撞预检。
        runtime = env.robot
        current = np.deg2rad(observation_preset(config.observation_pose, config.scene_profile).joints_deg)
        height = runtime.table_height + runtime.cube_half_edge + .003
        targets = []
        for angle in np.arange(8) * (np.pi / 4):
            xy = np.asarray(config.object_xy) + .05 * np.array([np.cos(angle), np.sin(angle)])
            runtime.reset_zero(xy)
            position = np.r_[xy, height]
            solution = runtime.kinematics.solve_ik(position, current, env.grasp_orientation)
            safe, reason = runtime.reference_is_safe(solution.q, held=False) if solution.success else (False, solution.reason)
            targets.append(dict(target_world_m=position.tolist(), ik=solution.success, safe=safe, reason=reason))
        write_json(args.json, dict(geometry=geometry, summary=summary, episodes=episodes, random_boundary_ik=targets))
        print(summary, flush=True)
        return int(summary["successes"] != args.episodes or not all(t["safe"] for t in targets))
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
