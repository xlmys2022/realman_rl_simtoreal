"""Offline FK contract tests. A measured model mismatch is NOT an equivalence pass."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_fk as comparison
import control_joints as control
from kinematics import Kinematics
import realman_fk as adapter


class FKComparisonMathTests(unittest.TestCase):
    def test_quaternion_order_normalization_and_sign(self):
        q = np.array([np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)])
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        for value in (q, -q, q * (1+1e-7)):
            r = adapter.quaternion_matrix(value)
            np.testing.assert_allclose(r, expected, atol=1e-14)
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(r), 1)
        for value in ([0]*4, [2, 0, 0, 0], [np.nan, 0, 0, 0], [1, 0, 0], [np.inf]*4):
            with self.assertRaises(ValueError):
                adapter.quaternion_matrix(value)

    def test_euler_order_and_so3_error(self):
        rx = adapter.euler_matrix([.3, 0, 0])
        ry = adapter.euler_matrix([0, -.4, 0])
        rz = adapter.euler_matrix([0, 0, .5])
        np.testing.assert_allclose(adapter.euler_matrix([.3, -.4, .5]), rz @ ry @ rx, atol=1e-15)
        a = comparison.transform(np.zeros(3), adapter.euler_matrix([0, 0, np.pi-1e-8]))
        b = comparison.transform(np.zeros(3), np.eye(3))
        self.assertAlmostEqual(comparison.errors(a, b)["rotation_error_rad"], np.pi-1e-8)
        self.assertFalse(comparison.errors(a, b)["within_tolerance"])

    def test_mdh_convention_is_modified_dh(self):
        dh = dict(a=[.2]*6, d=[.3]*6, alpha=[90]*6, offset=[30]*6)
        t = comparison.mdh_local(dh, 0, np.deg2rad(15))
        np.testing.assert_allclose(t[:3, 3], [.2, -.3, 0], atol=1e-15)
        np.testing.assert_allclose(t[:3, :3], adapter.euler_matrix([np.pi/2, 0, 0]) @
                                   adapter.euler_matrix([0, 0, np.pi/4]), atol=1e-15)

    def test_frame_inverse_and_relative_pose(self):
        a = comparison.transform([1, 2, 3], adapter.euler_matrix([.2, .3, .4]))
        delta = comparison.transform([.01, -.02, .03], adapter.euler_matrix([.1, -.1, .2]))
        np.testing.assert_allclose(comparison.inverse(a) @ a, np.eye(4), atol=1e-15)
        np.testing.assert_allclose(comparison.inverse(a) @ (a @ delta), delta, atol=1e-15)

    def test_missing_sdk_has_clear_error_and_no_mandatory_import(self):
        with patch.object(adapter, "import_module", side_effect=ModuleNotFoundError("Robotic_Arm")):
            with self.assertRaisesRegex(adapter.SDKUnavailable, "requirements-vendor.txt"):
                adapter.RealManFK()
        # Normal simulation FK remains usable when the optional SDK import fails.
        model, _, _ = control.load_model()
        kin = Kinematics(model)
        self.assertTrue(np.isfinite(kin.fk(kin.home_q)[0]).all())

    def test_samples_are_within_limits_and_not_only_singular(self):
        model, _, _ = control.load_model()
        kin = Kinematics(model)
        samples = comparison.sample_poses(kin.home_q)
        self.assertEqual(len(samples), 9)
        self.assertEqual([name for name, _ in samples[1:7]], [f"J{i}_plus_3deg" for i in range(1, 7)])
        for _, q in samples:
            kin.fk(q)  # Named-joint bound check.
            jp, jr = kin.jacobian(q)
            self.assertGreater(np.linalg.svd(np.vstack([jp, .2*jr]), compute_uv=False)[-1], .01)


@unittest.skipUnless(importlib.util.find_spec("Robotic_Arm") is not None,
                     "Optional Robotic_Arm SDK not installed; simulation tests remain independent")
class VendorFKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = comparison.run_comparison("all", "all")

    def test_offline_native_fk_and_representation_audits(self):
        self.assertEqual(len(self.report["configurations"]), 12)
        json.dumps(self.report, allow_nan=False)
        for config in self.report["configurations"]:
            info = config["sdk"]
            self.assertEqual(info["handle_id"], 0)
            self.assertFalse(info["sensor_mapping_confirmed"])
            for row in config["rows"]:
                # Tight conversion audit, independent of the robot comparison criterion.
                for key in ("sdk_mdh_audit", "sdk_euler_audit"):
                    self.assertLess(row[key]["position_error_m"], 1e-6)
                    self.assertLess(row[key]["rotation_error_rad"], 1e-6)
                for key in ("mujoco_base", "vendor_base"):
                    r = np.array(row[key])[:3, :3]
                    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
                    self.assertAlmostEqual(np.linalg.det(r), 1)

    def test_retrieved_sensor_and_generation_parameters(self):
        expected_d6 = dict(B=.1151, ZF=.1241, SF=.1436, ISF=.1323, BV=.1379, ISFV=.1551)
        for c in self.report["configurations"]:
            info = c["sdk"]
            gen = "III" if "III" in info["model"] else "II"
            sensor = info["sensor"].removeprefix("RM_MODEL_RM_").removesuffix("_E")
            self.assertAlmostEqual(info["dh"]["d"][0], .172 if gen == "II" else .1625, places=7)
            self.assertAlmostEqual(info["dh"]["d"][5], expected_d6[sensor], places=7)
            self.assertEqual(info["dh"]["offset"], [0, -90, 90, 0, 180, 180])

    def test_actual_mismatch_is_reported_without_relaxing_criterion(self):
        # Fixture for this pinned SDK / unchanged II-6F-V URDF. This asserts that
        # the diagnostic detects the discrepancy, NOT that the models agree.
        self.assertFalse(self.report["all_numerically_consistent"])
        self.assertFalse(self.report["same_robot_model_confirmed"])
        sf = self.report["configurations"][2]
        self.assertFalse(sf["summary"]["numerically_consistent"])
        self.assertGreater(sf["rows"][0]["absolute"]["position_error_m"], .02)
        self.assertGreater(sf["summary"]["absolute"]["max_rotation_error_rad"], .001)
        self.assertEqual(self.report["position_tolerance_m"], 1e-5)
        self.assertAlmostEqual(self.report["rotation_tolerance_rad"], np.deg2rad(.001))

    def test_wrist_reference_and_joint_zero_differences(self):
        c = self.report["configurations"][2]
        model, _, _ = control.load_model()
        data = control.initialize(model)
        kin = Kinematics(model)
        for row in c["rows"]:
            data.qpos[kin.qadr] = np.deg2rad(row["joint_deg"])  # Calculation-only data.
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(row["wrist_mujoco_base_m"], data.xpos[model.body("link_5").id], atol=1e-14)
        self.assertLess(c["summary"]["common_wrist"]["max_position_error_m"], .0003)
        self.assertFalse(c["summary"]["common_wrist"]["all_within_tolerance"])
        self.assertGreater(c["joint_zero_frames"][1]["difference"]["rotation_error_rad"], .0015)
        self.assertAlmostEqual(c["joint_zero_frames"][5]["difference"]["position_error_m"], .023, places=7)

    def test_home_transform_is_not_accepted_from_one_pose(self):
        for c in self.report["configurations"]:
            for key in ("home_left_validation", "home_right_validation"):
                self.assertFalse(c["summary"][key]["all_within_tolerance"])
            # Rigid relative home transforms are recorded, not Euler subtraction.
            home = c["rows"][0]
            np.testing.assert_allclose(home["mujoco_relative_home"], np.eye(4), atol=1e-14)
            np.testing.assert_allclose(home["vendor_relative_home"], np.eye(4), atol=1e-14)

    def test_comparison_never_connects_steps_or_mutates_model(self):
        from Robotic_Arm import rm_robot_interface as sdk
        model, _, _ = control.load_model()
        kin = Kinematics(model)
        live = control.initialize(model)
        before = {key: getattr(live, key).copy() for key in ("qpos", "qvel", "ctrl", "site_xpos")}
        mass, inertia = model.body_mass.copy(), model.body_inertia.copy()
        with (patch.object(sdk, "rm_init", side_effect=AssertionError("No communications initialization")),
              patch.object(sdk, "rm_create_robot_arm", side_effect=AssertionError("No robot connection")),
              patch.object(sdk.RoboticArm, "__init__", side_effect=AssertionError("Algo only")),
              patch.object(mujoco, "mj_step", side_effect=AssertionError("FK only"))):
            baseline = comparison.measure_configuration(kin)
            # A changed world placement must be removed using the known base
            # transform, never fitted from a robot pose. Calculation-only model.
            shifted, _, _ = control.load_model()
            base = shifted.body("base_link").id
            shifted.body_pos[base] = [.2, -.1, .3]
            angle = np.deg2rad(25)
            shifted.body_quat[base] = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
            moved = comparison.measure_configuration(Kinematics(shifted))
            for original, placed in zip(baseline["rows"], moved["rows"]):
                np.testing.assert_allclose(original["mujoco_base"], placed["mujoco_base"], atol=1e-14)
                self.assertGreater(np.linalg.norm(np.array(original["mujoco_world"])[:3, 3] -
                                                 np.array(placed["mujoco_world"])[:3, 3]), .1)
        self.assertEqual(live.time, 0)
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(live, key), value)
        np.testing.assert_array_equal(model.body_mass, mass)
        np.testing.assert_array_equal(model.body_inertia, inertia)

    def test_sdk_global_configuration_cannot_silently_leak(self):
        a = adapter.RealManFK("II", "SF")
        expected = a.fk(np.zeros(6))
        b = adapter.RealManFK("III", "ISFV")
        with self.assertRaisesRegex(RuntimeError, "process-global"):
            a.fk(np.zeros(6))
        b.fk(np.zeros(6))
        c = adapter.RealManFK("II", "SF")
        np.testing.assert_array_equal(c.fk(np.zeros(6))[0], expected[0])

    def test_sdk_frames_and_units_with_offline_perturbation(self):
        a = adapter.RealManFK("II", "SF")
        q = np.deg2rad([12, -23, 35, 14, 26, -17])
        p, r = a.fk(q)
        # Only algorithm frames change; restore explicitly. This audit confirms
        # tool translation is flange-local and work translation is base-local.
        sdk = a._sdk
        a._algo.rm_algo_set_toolframe(sdk.rm_frame_t("probe", [0, 0, .01, 0, 0, 0]))
        pt, rt = a.fk(q)
        np.testing.assert_allclose(pt, p + r @ [0, 0, .01], atol=2e-7)
        np.testing.assert_allclose(rt, r, atol=1e-7)
        a._algo.rm_algo_set_toolframe(sdk.rm_frame_t("zero", [0]*6))
        a._algo.rm_algo_set_workframe(sdk.rm_frame_t("probe", [.01, 0, 0, 0, 0, 0]))
        pw, _ = a.fk(q)
        np.testing.assert_allclose(pw, p - [.01, 0, 0], atol=2e-7)
        a._algo.rm_algo_set_workframe(sdk.rm_frame_t("zero", [0]*6))
        for value in ([np.nan]*6, [np.inf]*6, [0]*5):
            with self.assertRaises(ValueError):
                a.fk(value)

    def test_cli_from_other_directory_reports_mismatch_and_matrices(self):
        with tempfile.TemporaryDirectory(prefix="realman-fk-") as directory:
            path = Path(directory) / "comparison.json"
            result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/compare_fk.py"), "--json", str(path)],
                                    cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("[MISMATCH]", result.stdout)
            report = json.loads(path.read_text())
            self.assertTrue(report["completed"])
            self.assertFalse(report["motion_executed"])
            self.assertEqual(report["simulation_steps"], 0)
            self.assertFalse(report["all_numerically_consistent"])
            self.assertEqual(np.asarray(report["configurations"][0]["rows"][0]["vendor_base"]).shape, (4, 4))


if __name__ == "__main__":
    unittest.main(verbosity=2)
