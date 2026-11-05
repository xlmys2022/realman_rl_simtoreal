"""FK, full-pose IK, execution gating and actual end-effector dynamics tests."""

from dataclasses import replace
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from realman_rl_simtoreal.simulation import control, model as model_api
from realman_rl_simtoreal.simulation import ee_control as ee
from realman_rl_simtoreal.simulation.kinematics import Kinematics, rotation_vector


class KinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.jids, cls.aids = model_api.load_model()

    def setUp(self):
        self.kin = Kinematics(self.model)

    def test_a_fk_home_and_site_definition(self):
        kin, model = self.kin, self.model
        p, rotation = kin.fk(kin.home_q)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(rotation).all())
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(rotation), 1)
        sid = model.site("ee_site").id
        self.assertEqual(model.site_bodyid[sid], model.body("link_6").id)
        np.testing.assert_array_equal(model.site_pos[sid], [0, 0, 0])
        np.testing.assert_array_equal(model.site_quat[sid], [1, 0, 0, 0])
        d = model_api.initialize(model)
        np.testing.assert_allclose(p, d.xpos[model.body("link_6").id], atol=1e-14)
        p[:] = 99  # Return values do not alias the private MuJoCo buffers.
        self.assertLess(np.max(np.abs(kin.fk(kin.home_q)[0])), 2)

    def test_site_survives_model_regeneration(self):
        path = ROOT / "assets/robots/rml63/rml63_dynamics.xml"
        before = path.read_bytes()
        p = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/build_dynamics_model.py")],
                           cwd="/tmp", text=True, capture_output=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(path.read_bytes(), before)
        rebuilt, _, _ = model_api.load_model()
        self.assertEqual(rebuilt.nsite, 1)
        np.testing.assert_array_equal(rebuilt.body_mass, self.model.body_mass)
        np.testing.assert_array_equal(rebuilt.body_inertia, self.model.body_inertia)

    def test_fk_ik_do_not_mutate_live_data_or_model(self):
        d = model_api.initialize(self.model)
        snapshots = {name: getattr(d, name).copy() for name in ("qpos", "qvel", "ctrl", "qacc", "site_xpos")}
        model_mass = self.model.body_mass.copy()
        target = self.kin.home_position + [.005, 0, 0]
        self.kin.jacobian(self.kin.home_q)
        self.assertTrue(self.kin.solve_ik(target, self.kin.home_q).success)
        for name, value in snapshots.items():
            np.testing.assert_array_equal(getattr(d, name), value)
        np.testing.assert_array_equal(self.model.body_mass, model_mass)
        self.assertEqual(d.time, 0)

    def test_jacobian_matches_translation_and_rotation_finite_differences(self):
        kin = self.kin
        q = kin.home_q + np.deg2rad([1, -1, 2, -1, 1, -2])
        jp, jr = kin.jacobian(q)
        epsilon = 1e-6
        for i in range(6):
            dq = np.eye(6)[i] * epsilon
            pp, rp = kin.fk(q + dq)
            pm, rm = kin.fk(q - dq)
            np.testing.assert_allclose(jp[:, i], (pp - pm) / (2 * epsilon), atol=2e-9, rtol=0)
            np.testing.assert_allclose(jr[:, i], rotation_vector(rp, rm) / (2 * epsilon), atol=2e-9, rtol=0)

    def test_rotation_error_near_zero_and_pi(self):
        for angle in (1e-8, 0.4, np.pi - 1e-8, np.pi):
            quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0, 0])
            matrix = np.empty(9)
            mujoco.mju_quat2Mat(matrix, quat)
            error = rotation_vector(matrix.reshape(3, 3), np.eye(3))
            self.assertTrue(np.isfinite(error).all())
            self.assertAlmostEqual(np.linalg.norm(error), angle, places=7)

    def test_b_fk_ik_full_pose_roundtrip(self):
        kin = self.kin
        for offsets in ([1, -1, 1, 1, -1, 1], [-1, 1, -2, 1, 2, -1], [2, 0, 0, -1, 0, 1]):
            with self.subTest(offsets=offsets):
                goal_q = kin.home_q + np.deg2rad(offsets)
                target, rotation = kin.fk(goal_q)
                result = kin.solve_ik(target, kin.home_q, rotation)
                self.assertTrue(result.success, result)
                actual, actual_rotation = kin.fk(result.q)
                self.assertLessEqual(np.linalg.norm(actual - target), 1e-5)
                self.assertLessEqual(np.linalg.norm(rotation_vector(rotation, actual_rotation)), 1e-4)
                self.assertLess(np.max(np.abs(result.q - kin.home_q)), np.deg2rad(5))

    def test_default_ik_really_constrains_home_orientation(self):
        kin = self.kin
        for delta in ([.005, 0, 0], [-.005, 0, 0], [0, 0, .005]):
            result = kin.solve_ik(kin.home_position + delta, kin.home_q)
            self.assertTrue(result.success, result)
            p, rotation = kin.fk(result.q)
            self.assertLessEqual(np.linalg.norm(rotation_vector(kin.home_rotation, rotation)), 1e-4)
            self.assertLessEqual(np.linalg.norm(p - kin.home_position - delta), 1e-5)

    def test_d_unreachable_target_returns_failure_without_angles(self):
        result = self.kin.solve_ik([10, 10, 10], self.kin.home_q)
        self.assertFalse(result.success)
        self.assertIsNone(result.q)
        self.assertIn("reach bound", result.reason)
        self.assertEqual(result.iterations, 0)

    def test_iteration_budget_and_near_singularity(self):
        kin = self.kin
        r = kin.solve_ik(kin.home_position + [.005, 0, 0], kin.home_q, max_iterations=1)
        self.assertFalse(r.success)
        self.assertIsNone(r.q)
        self.assertEqual(r.iterations, 1)
        singular = kin.home_q.copy()
        singular[4] = 0
        q = singular + np.deg2rad([0, .1, -.1, 0, 0, 0])
        p, rotation = kin.fk(q)
        r = kin.solve_ik(p, singular, rotation, max_iterations=40)
        self.assertLessEqual(r.iterations, 40)
        if r.success:
            self.assertTrue(np.isfinite(r.q).all())
        else:
            self.assertIsNone(r.q)
            self.assertTrue(r.reason)

    def test_step_and_joint_bounds_are_respected(self):
        kin = self.kin
        original = kin._forward
        visited = []
        def bounded(q):
            self.assertTrue(np.all(q >= kin.limits[:, 0]))
            self.assertTrue(np.all(q <= kin.limits[:, 1]))
            visited.append(q.copy())
            return original(q)
        with patch.object(kin, "_forward", side_effect=bounded):
            r = kin.solve_ik(kin.home_position + [.005, 0, 0], kin.home_q, max_step=.001)
        self.assertTrue(r.success, r)
        self.assertLessEqual(np.max(np.abs(np.diff(visited, axis=0))), .001 + 1e-12)
        # An initial value outside the hard bounds is rejected, not silently clamped.
        bad = kin.home_q.copy()
        bad[0] = kin.limits[0, 1] + 1e-6
        self.assertFalse(kin.solve_ik(kin.home_position, bad).success)
        with self.assertRaises(ValueError):
            kin.fk(bad)

    def test_invalid_inputs_have_structured_failure(self):
        kin = self.kin
        for target in ([np.nan, 0, 0], [np.inf, 0, 0], [0, 0], [1e300, 0, 0]):
            result = kin.solve_ik(target, kin.home_q)
            self.assertFalse(result.success)
            self.assertIsNone(result.q)
        for rotation in (np.zeros((3, 3)), np.diag([-1, 1, 1]), np.eye(2)):
            self.assertFalse(kin.solve_ik(kin.home_position, kin.home_q, rotation).success)
        for kw in (dict(damping=0), dict(max_iterations=1001), dict(max_step=np.nan)):
            self.assertFalse(kin.solve_ik(kin.home_position, kin.home_q, **kw).success)

    def test_iteration_at_a_joint_limit_stays_bounded(self):
        kin = self.kin
        seed = kin.home_q.copy()
        seed[0] = kin.limits[0, 1] - 1e-4
        position, rotation = kin.fk(seed)
        angle = 0.05
        rz = np.array([[np.cos(angle), -np.sin(angle), 0],
                       [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        original, visited = kin._forward, []
        def bounded(q):
            self.assertTrue(np.all(q >= kin.limits[:, 0]))
            self.assertTrue(np.all(q <= kin.limits[:, 1]))
            visited.append(q.copy())
            return original(q)
        with patch.object(kin, "_forward", side_effect=bounded):
            result = kin.solve_ik(rz @ position, seed, rz @ rotation, max_iterations=30)
        self.assertLessEqual(result.iterations, 30)
        self.assertTrue(any(abs(q[0] - kin.limits[0, 1]) < 1e-12 for q in visited))
        if not result.success:
            self.assertIsNone(result.q)
            self.assertTrue(result.reason)

    def test_e_failed_or_excessive_goal_never_steps(self):
        kin = self.kin
        data = model_api.initialize(self.model)
        q, ctrl = data.qpos.copy(), data.ctrl.copy()
        target = np.array([10, 10, 10])
        failed = kin.solve_ik(target, kin.home_q)
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("Must not step")):
            with self.assertRaisesRegex(ValueError, "IK failed"):
                ee.execute_motion(self.model, data, self.jids, self.aids, kin, failed, target)
            # Full-pose IK can succeed in +Y, but its J4 change exceeds this demo's 5 degree guard.
            target = kin.home_position + [0, .005, 0]
            solution = kin.solve_ik(target, kin.home_q)
            self.assertTrue(solution.success)
            with self.assertRaisesRegex(ValueError, "5 degrees"):
                ee.execute_motion(self.model, data, self.jids, self.aids, kin, solution, target)
        np.testing.assert_array_equal(data.qpos, q)
        np.testing.assert_array_equal(data.ctrl, ctrl)
        self.assertEqual(data.time, 0)

    def test_c_dynamics_uses_actuators_without_live_qpos_overwrites(self):
        kin = self.kin
        data = model_api.initialize(self.model)
        target = kin.home_position + [.005, 0, 0]
        solution = kin.solve_ik(target, kin.home_q)
        step = mujoco.mj_step
        previous, calls = None, 0
        def recorded_step(model, state):
            nonlocal previous, calls
            if previous is not None:
                np.testing.assert_array_equal(state.qpos, previous)
            step(model, state)
            previous, calls = state.qpos.copy(), calls + 1
        with patch.object(mujoco, "mj_step", side_effect=recorded_step):
            report = ee.execute_motion(self.model, data, self.jids, self.aids, kin, solution, target)
        self.assertEqual(calls, 4000)
        self.assertAlmostEqual(data.time, 8)
        self.assertLess(report["actual_position_error_m"], .005)
        self.assertLess(report["actual_orientation_error_rad"], np.deg2rad(1))
        self.assertGreater(report["actual_position_error_m"], solution.position_error * 100)
        self.assertEqual(report["motion"]["max_contacts"], 0)
        np.testing.assert_array_equal(data.qpos, previous)
        displacement = np.array(report["actual_position_m"]) - report["motion_start_position_m"]
        self.assertGreater(displacement[0], .004)

    def test_collision_path_is_rejected_before_execution(self):
        spec = mujoco.MjSpec.from_file(str(model_api.SCENE))
        home = model_api.initialize(self.model)
        spec.worldbody.add_geom(name="obstacle", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                               size=[.02, 0, 0], pos=home.xipos[self.model.body("link_2").id])
        model = spec.compile()
        data, kin = model_api.initialize(model), Kinematics(model)
        target = kin.home_position + [.005, 0, 0]
        result = kin.solve_ik(target, kin.home_q)
        with patch.object(mujoco, "mj_step", side_effect=AssertionError("Must not step")):
            with self.assertRaisesRegex(ValueError, "collision"):
                ee.execute_motion(model, data, self.jids, self.aids, kin, result, target)
        self.assertEqual(data.time, 0)

    def test_headless_cli_unreachable_target(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            p = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/control_ee.py"),
                                "--target", "10", "10", "10", "--json", str(output)],
                               cwd="/tmp", capture_output=True, text=True, timeout=15)
            self.assertEqual(p.returncode, 1, p.stdout + p.stderr)
            import json
            result = json.loads(output.read_text())
            self.assertFalse(result["motion_executed"])
            self.assertFalse(result["ik"]["success"])
            self.assertIsNone(result["ik"]["q"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
