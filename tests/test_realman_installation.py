"""实机安装场景的运动学、碰撞、相机投影和跨姿态定位回归。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import mujoco
import numpy as np

from realman_rl_simtoreal.hardware.realman_config import REALMAN_HARDWARE, REALMAN_TABLETOP
from realman_rl_simtoreal.paths import TABLETOP_CARTESIAN_SCENE
from realman_rl_simtoreal.perception.rgbd import UprightCubeDetector, backproject
from realman_rl_simtoreal.simulation.devices.mujoco_rgbd_camera import MujocoRgbdCamera
from realman_rl_simtoreal.simulation.motion.cartesian_controller import rotation_matrix
from realman_rl_simtoreal.simulation.runtime.observation_presets import observation_preset, startup_contract
from realman_rl_simtoreal.simulation.runtime.observation_startup import observation_solution
from realman_rl_simtoreal.simulation.runtime.tabletop import TabletopRuntime
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig
from realman_rl_simtoreal.utils.geometry import inverse, modified_dh_transform, transform


class InstallationTests(unittest.TestCase):
    def setUp(self):
        self.config = CartesianTabletopConfig(scene_profile="realman_nominal", observation_source="rgbd_hsv")
        self.robot = TabletopRuntime(self.config)
        self.robot.reset_zero(self.config.object_xy)

    def test_measured_flange_fk_tcp_and_joint_axes(self):
        runtime = self.robot
        dh = REALMAN_HARDWARE.controller_dh
        joint_bounds_deg = np.column_stack([REALMAN_HARDWARE.joint_min_deg, REALMAN_HARDWARE.joint_max_deg])
        np.testing.assert_allclose(np.rad2deg(runtime.kinematics.limits), joint_bounds_deg, atol=1e-10)
        for q in (np.zeros(6), np.deg2rad([-30, 1, 90, 18, 82, 182]), np.deg2rad([40, -55, 80, 33, -20, 12])):
            controller = np.eye(4)
            for a, d, alpha, offset, angle in zip(dh.a_m, dh.d_m, np.deg2rad(dh.alpha_deg),
                                                   np.deg2rad(dh.offset_deg), q):
                controller = controller @ modified_dh_transform(a, d, alpha, offset + angle)
            world_flange, rotation = runtime.kinematics.body_pose(q, "link_6")
            tcp, tcp_rotation = runtime.kinematics.fk(q)
            np.testing.assert_allclose(transform(world_flange - runtime.base_position, rotation), controller, atol=1e-8)
            np.testing.assert_allclose(rotation.T @ (tcp - world_flange), [0., 0., .256], atol=1e-10)
            np.testing.assert_allclose(tcp_rotation, rotation, atol=1e-10)
        q, _ = observation_solution(runtime)
        dp, _ = runtime.kinematics.jacobian(q)
        epsilon = 1e-6
        numeric = np.column_stack([
            (runtime.kinematics.fk(q + dq)[0] - runtime.kinematics.fk(q - dq)[0]) / (2 * epsilon)
            for dq in np.eye(6) * epsilon
        ])
        np.testing.assert_allclose(dp, numeric, atol=2e-8)

    def test_desk_cube_spacer_and_legacy_profile(self):
        runtime = self.robot
        m = runtime.model
        self.assertAlmostEqual(runtime.table_height - runtime.base_position[2], -.04)
        self.assertAlmostEqual(runtime.data.qpos[runtime.object_qadr + 2] - runtime.base_position[2], -.03)
        adapter = m.geom('eg2_adapter_proxy')
        self.assertAlmostEqual(adapter.size[0], .0285)
        self.assertAlmostEqual(2 * adapter.size[1], REALMAN_TABLETOP.adapter_length_m)
        self.assertGreater(adapter.contype[0], 0)
        self.assertAlmostEqual(m.body('4C2_baselink').pos[2], REALMAN_TABLETOP.adapter_length_m)
        legacy = TabletopRuntime(CartesianTabletopConfig())
        self.assertEqual(legacy.scene_path, TABLETOP_CARTESIAN_SCENE)
        self.assertAlmostEqual(legacy.base_position[2], legacy.table_height)
        self.assertEqual(startup_contract('table_down')['version'], 3)
        self.assertEqual(startup_contract('table_down', self.config.scene_profile)['version'], 4)
        for label in ('link_6', 'eg2_experimental_adapter', '4C2_baselink'):
            a, b = m.body(label), legacy.model.body(label)
            np.testing.assert_array_equal(a.mass, b.mass)
            np.testing.assert_array_equal(a.inertia, b.inertia)

    def test_reachable_tcp_and_collision_precheck(self):
        nominal, rotation = observation_solution(self.robot)
        for x, y in ((.35, 0), (.45, 0), (.4, -.05), (.4, .05)):
            target = [x, y, self.robot.table_height + self.robot.cube_half_edge + .003]
            result = self.robot.kinematics.solve_ik(target, nominal, rotation)
            self.assertTrue(result.success, result.reason)
            self.assertEqual(self.robot.reference_is_safe(result.q, held=False), (True, None))
        config = CartesianTabletopConfig(scene_profile="realman_nominal", observation_pose="forward")
        other = TabletopRuntime(config)
        other.reset_zero(config.object_xy)
        q, _ = observation_solution(other)
        self.assertTrue(other.path_is_safe(np.zeros(6), q))

    def test_candidate_hand_eye_converts_mujoco_axes_and_tcp_once(self):
        nominal_rotation = rotation_matrix([0., 0., .35])
        flange_optical = transform([-.08, .035, .024], nominal_rotation)
        flange_tcp = transform([0., 0., .256], np.eye(3))
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "hsv.json").write_text(
                '{"hsv_ranges":[{"low":[49,80,66],"high":[84,255,255]}]}')
            path = folder / "candidate.json"
            path.write_text(json.dumps(dict(hsv_file="hsv.json", cube_edge_m=.02, T_flange_tcp=flange_tcp.tolist(),
                                            hand_eye=dict(parent_frame="tcp", T_parent_color=
                                                          (inverse(flange_tcp) @ flange_optical).tolist()))))
            config = CartesianTabletopConfig(scene_profile="realman_hand_eye", hand_eye_file=str(path))
            robot = TabletopRuntime(config)
            robot.reset_zero(config.object_xy)
            flange = robot.kinematics.body_pose(np.zeros(6), 'link_6')
            base_flange = transform(flange[0] - robot.base_position, flange[1])
            camera_id = robot.model.camera('wrist_rgb').id
            camera_rotation = robot.data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1., -1., -1.])
            camera_position = robot.data.cam_xpos[camera_id] - robot.base_position
            np.testing.assert_allclose(transform(camera_position, camera_rotation), base_flange @ flange_optical,
                                       atol=1e-9)

    def test_rgbd_projection_and_fixed_cube_across_three_poses(self):
        robot = self.robot
        q, down = observation_solution(robot)
        detector = UprightCubeDetector(hsv_low=self.config.vision_hsv_low, hsv_high=self.config.vision_hsv_high)
        camera = MujocoRgbdCamera(robot, width=480, height=360)
        seen = []
        try:
            cases = (([.19, 0., 1.1], down), ([.25, -.04, 1.1], down),
                     ([.29, -.05, 1.06], rotation_matrix([0., -np.deg2rad(15), 0.]) @ down))
            for target, rotation in cases:
                solution = robot.kinematics.solve_ik(target, q, rotation)
                self.assertTrue(solution.success, solution.reason)
                self.assertEqual(robot.reference_is_safe(solution.q, held=False), (True, None))
                robot.data.qpos[robot.kinematics.qadr] = solution.q
                mujoco.mj_forward(robot.model, robot.data)
                frame = camera.capture()
                upper = np.array([.4, 0., -.02])
                T_camera_base = inverse(frame.camera_to_base)
                point = T_camera_base[:3, :3] @ upper + T_camera_base[:3, 3]
                uv = np.rint((frame.intrinsics @ (point / point[2]))[:2])
                np.testing.assert_allclose(backproject(frame, [uv])[0], upper, atol=.0015)
                found = detector.detect(frame)
                self.assertIsNotNone(found)
                np.testing.assert_allclose(found.position_m, [.4, 0., -.03], atol=.0018)
                seen.append(found.position_m)
            self.assertLess(max(np.linalg.norm(p - seen[0]) for p in seen[1:]), .002)
        finally:
            camera.close()


if __name__ == "__main__":
    unittest.main()
