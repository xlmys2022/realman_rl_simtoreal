"""Offline IK contracts, native branch validation and unchanged MuJoCo regressions."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("_test_compare_ik", ROOT / "scripts/compare_ik.py")
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)
from realman_rl_simtoreal.simulation import control, model as model_api
from realman_rl_simtoreal.simulation.kinematics import Kinematics
from realman_rl_simtoreal.vendor import realman_fk
from realman_rl_simtoreal.vendor.realman_ik import (RealManIK, quaternion_pose, periodic_equivalent,
                       POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD)


class IKEncodingTests(unittest.TestCase):
    def test_target_quaternion_wxyz_and_roundtrip_near_pi(self):
        for rpy in ([0, 0, 0], [.3, -.4, .5], [np.pi-1e-8, 0, 0], [0, 0, np.pi]):
            r = realman_fk.euler_matrix(rpy)
            pose = quaternion_pose([.1, -.2, .3], r)
            np.testing.assert_array_equal(pose[:3], [.1, -.2, .3])
            self.assertAlmostEqual(np.linalg.norm(pose[3:]), 1)
            np.testing.assert_allclose(realman_fk.quaternion_matrix(pose[3:]), r, atol=1e-14)
        np.testing.assert_allclose(quaternion_pose([0]*3, np.eye(3))[3:], [1, 0, 0, 0], atol=1e-15)

    def test_illegal_pose_rejected_before_native_encoding(self):
        for p, r in (([0, 0], np.eye(3)), ([np.nan, 0, 0], np.eye(3)),
                     ([np.inf]*3, np.eye(3)), ([1e100, 0, 0], np.eye(3)),
                     ([0]*3, np.ones((3, 3))), ([0]*3, np.diag([-1, 1, 1])),
                     ([0]*3, np.full((3, 3), np.nan))):
            with self.assertRaises(ValueError):
                quaternion_pose(p, r)

    def test_periodic_conversion_is_integer_turns_never_clipping(self):
        limits = np.deg2rad([[-178, 178]]*5 + [[-359.8, 359.8]])
        raw = np.deg2rad([205, 0, 0, 199, 0, 350])
        seed = np.deg2rad([-150, 0, 0, -160, 0, 340])
        q, turns = periodic_equivalent(raw, seed, limits)
        np.testing.assert_allclose(np.rad2deg(q), [-155, 0, 0, -161, 0, 350], atol=1e-12)
        np.testing.assert_allclose(q-raw, 2*np.pi*np.array(turns), atol=1e-15)
        bad = raw.copy()
        bad[0] = np.pi
        self.assertEqual(periodic_equivalent(bad, seed, limits), (None, None))
        integer_input, _ = periodic_equivalent([7, 0, 0, 0, 0, 0], [0]*6, limits)
        self.assertAlmostEqual(integer_input[0], 7-2*np.pi)  # No integer truncation.

    def test_optional_sdk_and_invalid_model_ranges(self):
        for limits in (np.zeros((6, 2)), np.zeros((5, 2)), np.full((6, 2), np.nan)):
            with self.assertRaises(ValueError):
                RealManIK(limits)
        with patch.object(realman_fk, "import_module", side_effect=ModuleNotFoundError("Robotic_Arm")):
            with self.assertRaises(realman_fk.SDKUnavailable):
                RealManIK(np.array([[-1., 1.]]*6))


@unittest.skipUnless(importlib.util.find_spec("Robotic_Arm") is not None,
                     "Optional Robotic_Arm SDK missing; native IK tests skipped")
class VendorIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, _, _ = model_api.load_model()
        cls.kin = Kinematics(cls.model)
        cls.report = comparison.run_comparison(repeats=2)

    def new_vendor(self, sensor="SF"):
        return RealManIK(self.kin.limits, "II", sensor)

    def assert_verified(self, result):
        self.assertTrue(result["success"], result.get("reason"))
        q = np.asarray(result["q"])
        self.assertEqual(q.shape, (6,))
        self.assertTrue(np.isfinite(q).all())
        self.assertTrue(np.all(q >= self.kin.limits[:, 0]))
        self.assertTrue(np.all(q <= self.kin.limits[:, 1]))
        self.assertLessEqual(result["position_error_m"], POSITION_TOLERANCE_M)
        self.assertLessEqual(result["orientation_error_rad"], ORIENTATION_TOLERANCE_RAD)

    def test_native_self_fk_ik_fk_all_main_poses_and_seeds(self):
        for c in self.report["configurations"]:
            self.assertEqual(c["internal"]["summary"]["successes"], 36)
            for row in c["internal"]["rows"]:
                self.assert_verified(row["result"])
                self.assertEqual(row["result"]["sdk_code"], 0)
                self.assertTrue(row["result"]["within_vendor_limits"])
                self.assertTrue(row["result"]["within_model_limits"])
                self.assertTrue(row["independent_fk_audit"]["passed"])

    def test_mujoco_local_seed_sensitivity_is_bounded_and_visible(self):
        mj = self.report["mujoco_internal"]
        for seed in ("home", "near", "other_elbow"):
            self.assertEqual(mj["by_seed"][seed]["successes"], 9)
        failed = [r for r in mj["rows"] if not r["result"]["success"]]
        self.assertTrue(failed)  # This pinned local solver does not globally search branches.
        for row in mj["rows"]:
            self.assertLessEqual(row["result"]["iterations"], 150)
            if row["result"]["success"]:
                self.assertTrue(row["independent_fk_audit"]["passed"])
            else:
                self.assertIsNone(row["result"]["q"])
                self.assertTrue(row["result"]["reason"])

    def test_targets_are_generated_by_each_solvers_own_fk(self):
        mj_rows = self.report["mujoco_internal"]["rows"]
        for c in self.report["configurations"]:
            vendor = self.new_vendor(c["sdk"]["sensor"].removeprefix("RM_MODEL_RM_").removesuffix("_E"))
            for mj, native in zip(mj_rows, c["internal"]["rows"]):
                np.testing.assert_array_equal(mj["source_q_rad"], native["source_q_rad"])
                p, r = vendor.fk(native["source_q_rad"])
                np.testing.assert_allclose(native["target_position_m"], p, atol=1e-14)
                np.testing.assert_allclose(native["target_rotation"], r, atol=1e-14)
                self.assertGreater(np.linalg.norm(np.array(mj["target_position_m"])-p), .01)

    def test_all_branches_checked_and_only_valid_periodic_solutions_exposed(self):
        for c in self.report["configurations"]:
            sensor = c["sdk"]["sensor"].removeprefix("RM_MODEL_RM_").removesuffix("_E")
            vendor = self.new_vendor(sensor)
            for g in c["multisolutions"]:
                result = g["all_result"]
                self.assertEqual(result["sdk_code"], 0)
                self.assertEqual(result["reported_count"], 8)
                target = next(r for r in c["internal"]["rows"] if r["pose"] == g["pose"])
                accepted = [r for r in result["candidates"] if r["success"]]
                self.assertEqual(len(accepted), 2 if g["pose"] == "home" else 8)
                for row in result["candidates"]:
                    if row["success"]:
                        self.assert_verified(row)
                        p, r = vendor.fk(row["q"])
                        e = comparison.pose_errors(target["target_position_m"], target["target_rotation"], p, r)
                        self.assertLessEqual(e["position_error_m"], POSITION_TOLERANCE_M)
                        self.assertLessEqual(e["orientation_error_rad"], ORIENTATION_TOLERANCE_RAD)
                    else:
                        self.assertIsNone(row["q"])
                        self.assertIn("limits", row["reason"])
                if g["pose"] != "home":
                    # Native raw angle representatives depend on q_in. Home seed
                    # already returns legal representatives; far seed exposes
                    # equivalent 205/199 degree values needing integer turns.
                    far_candidates = [r for r in g["all_far_seed"]["candidates"] if r["success"]]
                    self.assertEqual(len(far_candidates), 8)
                    self.assertTrue(any(not r["raw_within_vendor_limits"] and any(r["periodic_turns"]) for r in far_candidates))
                    for candidate in far_candidates:
                        self.assert_verified(candidate)
                        p, r = vendor.fk(candidate["q"])
                        e = comparison.pose_errors(target["target_position_m"], target["target_rotation"], p, r)
                        self.assertLessEqual(e["position_error_m"], POSITION_TOLERANCE_M)
                        self.assertLessEqual(e["orientation_error_rad"], ORIENTATION_TOLERANCE_RAD)

    def test_multiple_solutions_and_seed_selection(self):
        for c in self.report["configurations"]:
            for g in c["multisolutions"]:
                self.assertGreaterEqual(len(g["distinct_single_solutions"]), 2)
                self.assertTrue(all(s["nearest_enumerated_l2"] for s in g["seed_selection"]))
                q1, q2 = np.array(g["distinct_single_solutions"][:2])
                self.assertGreater(np.max(np.abs(np.arctan2(np.sin(q1-q2), np.cos(q1-q2)))), .1)

    def test_nearby_sequence_has_no_large_joint_branch_jumps(self):
        for c in self.report["configurations"]:
            seq = c["continuity"]
            self.assertEqual(seq["summary"]["successes"], 8)
            self.assertLess(seq["max_joint_jump_rad"], np.deg2rad(5))

    def test_unreachable_native_and_mujoco_never_expose_angles(self):
        self.assertIsNone(self.report["mujoco_unreachable"]["result"]["q"])
        for c in self.report["configurations"]:
            r = c["unreachable"]["result"]
            self.assertFalse(r["success"])
            self.assertEqual(r["sdk_code"], 1)
            self.assertIsNone(r["q"])
            self.assertFalse(c["unreachable_all"]["success"])
            self.assertEqual(c["unreachable_all"]["candidates"], [])

    def test_nonzero_sdk_codes_discard_even_plausible_output_buffers(self):
        v = self.new_vendor()
        p, r = v.fk(self.kin.home_q)
        for code in (1, -1, -2, 99):
            with (patch.object(v._algo, "rm_algo_inverse_kinematics", return_value=(code, [0]*6)),
                  patch.object(v, "fk", side_effect=AssertionError("Failed buffers must not be used"))):
                result = v.solve_ik(p, self.kin.home_q, r)
            self.assertFalse(result.success)
            self.assertEqual(result.sdk_code, code)
            self.assertIsNone(result.q)

    def test_success_code_still_requires_count_finite_values_limits_and_fk(self):
        v = self.new_vendor()
        p, r = v.fk(self.kin.home_q)
        for raw in ([0]*5, [np.nan]*6, [np.inf]*6, [0]*6):
            with patch.object(v._algo, "rm_algo_inverse_kinematics", return_value=(0, raw)):
                result = v.solve_ik(p, self.kin.home_q, r)
            self.assertEqual(result.sdk_code, 0)
            self.assertFalse(result.success)
            self.assertIsNone(result.q)
        for count in (0, 9):
            bad = SimpleNamespace(result=0, num=count)
            with patch.object(v._algo, "rm_algo_inverse_kinematics_all", return_value=bad):
                result = v.solve_all(p, self.kin.home_q, r)
            self.assertFalse(result.success)
            self.assertEqual(result.candidates, ())

    def test_illegal_input_never_calls_native_ik(self):
        v = self.new_vendor()
        p, r = v.fk(self.kin.home_q)
        bad_seed = self.kin.home_q.copy()
        bad_seed[0] = self.kin.limits[0, 1] + .01
        inputs = ((p, [np.nan]*6, r), (p, [np.inf]*6, r), (p, [0]*5, r), (p, bad_seed, r),
                  ([np.inf]*3, self.kin.home_q, r), (p, self.kin.home_q, np.zeros((3, 3))))
        with (patch.object(v._algo, "rm_algo_inverse_kinematics", side_effect=AssertionError("Unsafe input")),
              patch.object(v._algo, "rm_algo_inverse_kinematics_all", side_effect=AssertionError("Unsafe input"))):
            for position, seed, rotation in inputs:
                result = v.solve_ik(position, seed, rotation)
                self.assertFalse(result.success)
                self.assertIsNone(result.sdk_code)
                self.assertIsNone(result.q)
                self.assertFalse(v.solve_all(position, seed, rotation).success)

    def test_sdk_joint_units_quaternion_flag_and_model_range_gate(self):
        v = self.new_vendor()
        q = self.kin.home_q.copy()
        q[0] = np.deg2rad(177.99)  # Legal vendor FK input, beyond the URDF rounded bound.
        p, r = v.fk(q)
        seed = self.kin.home_q.copy()
        seed[0] = np.deg2rad(177.9)
        params, _ = v._params(p, r, seed)
        self.assertEqual(params.flag, 0)
        np.testing.assert_allclose(list(params.q_in)[:6], np.rad2deg(seed), atol=1e-5)
        self.assertEqual(params.q_in[6], 0)
        self.assertAlmostEqual(np.linalg.norm([params.q_pose.quaternion.w, params.q_pose.quaternion.x,
                                              params.q_pose.quaternion.y, params.q_pose.quaternion.z]), 1, places=6)
        result = v.solve_ik(p, seed, r)
        self.assertEqual(result.sdk_code, 0)
        self.assertFalse(result.success)
        self.assertTrue(result.within_vendor_limits)
        self.assertFalse(result.within_model_limits)
        self.assertIsNone(result.q)
        self.assertIn("limits", result.reason)

    def test_boundary_failure_retained_despite_successful_other_probes(self):
        self.assertEqual(self.report["mujoco_boundaries"]["summary"]["successes"], 12)
        for c in self.report["configurations"]:
            self.assertEqual(c["boundaries"]["summary"]["successes"], 11)
            self.assertEqual(len(c["boundary_failure_probes"]), 1)
            probe = c["boundary_failure_probes"][0]
            self.assertEqual(probe["pose"], "J5_lower_inside_0.05deg")
            self.assertEqual(probe["seed_at_target"]["result"]["sdk_code"], 1)
            self.assertIsNone(probe["seed_at_target"]["result"]["q"])
            self.assertEqual(probe["seed_home"]["result"]["sdk_code"], 0)
            self.assertTrue(probe["all_original_seed"]["success"])
            self.assertEqual(sum(r["success"] for r in probe["all_original_seed"]["candidates"]), 2)
            sensor = c["sdk"]["sensor"].removeprefix("RM_MODEL_RM_").removesuffix("_E")
            v = self.new_vendor(sensor)
            row = next(r for r in c["boundaries"]["rows"] if r["pose"] == probe["pose"])
            params, _ = v._params(row["target_position_m"], row["target_rotation"], row["seed_rad"])
            for name in ("w", "x", "y", "z"):
                setattr(params.q_pose.quaternion, name, -getattr(params.q_pose.quaternion, name))
            code, _ = v._algo.rm_algo_inverse_kinematics(params)
            self.assertEqual(code, 1)  # Negating the same unit quaternion does not resolve it.
            try:
                v._algo.rm_algo_set_redundant_parameter_traversal_mode(False)
                self.assertEqual(v.solve_ik(row["target_position_m"], row["seed_rad"],
                                           row["target_rotation"]).sdk_code, 1)
            finally:
                v._algo.rm_algo_set_redundant_parameter_traversal_mode(True)

    def test_global_configuration_switch_rejects_stale_ik_without_dh_edits(self):
        a = self.new_vendor("SF")
        p, r = a.fk(self.kin.home_q)
        expected = a.solve_ik(p, self.kin.home_q, r).q
        b = self.new_vendor("ISFV")
        failed = a.solve_ik(p, self.kin.home_q, r)
        self.assertFalse(failed.success)
        self.assertIsNone(failed.q)
        self.assertIn("process-global", failed.reason)
        self.assertFalse(a.solve_all(p, self.kin.home_q, r).success)
        bp, br = b.fk(self.kin.home_q)
        self.assertTrue(b.solve_ik(bp, self.kin.home_q, br).success)
        for key, value in b.info["dh"].items():
            np.testing.assert_array_equal(b._algo.rm_algo_get_dh()[key], value)
        restored = self.new_vendor("SF")
        np.testing.assert_array_equal(restored.solve_ik(p, self.kin.home_q, r).q, expected)

    def test_cross_use_is_explicit_model_diagnostic_not_equivalence(self):
        self.assertFalse(self.report["same_robot_model_confirmed"])
        for c in self.report["configurations"]:
            self.assertIsNone(c["cross"]["tcp_transform"])
            self.assertIn("NOT a fair IK accuracy comparison", c["cross"]["caveat"])
            for row in c["cross"]["rows"]:
                self.assertTrue(row["result"]["success"])
                self.assertLessEqual(row["own_model_error"]["position_error_m"], POSITION_TOLERANCE_M)
                self.assertGreater(row["other_model_error"]["position_error_m"], .01)
                self.assertGreater(row["other_model_error"]["orientation_error_rad"], .001)

    def test_repeats_and_timings_are_recorded_without_nan(self):
        json.dumps(self.report, allow_nan=False)
        self.assertTrue(self.report["safety_audits_passed"])
        self.assertEqual(self.report["integrity_anomalies"], [])
        experiments = [self.report["mujoco_internal"]] + [c["internal"] for c in self.report["configurations"]]
        for experiment in experiments:
            self.assertTrue(experiment["summary"]["repeat_consistent"])
            for row in experiment["rows"]:
                self.assertEqual(len(row["public_call_times_s"]), 2)
                self.assertTrue(all(t > 0 for t in row["public_call_times_s"]))

    def test_full_comparison_never_connects_steps_or_commands(self):
        from Robotic_Arm import rm_robot_interface as sdk
        data = model_api.initialize(self.model)
        before = {k: getattr(data, k).copy() for k in ("qpos", "qvel", "ctrl", "site_xpos")}
        assets = [ROOT/"assets/robots/rml63"/name for name in ("scene.xml", "rml63_dynamics.xml", "rml63_mujoco.urdf")]
        original = [p.read_bytes() for p in assets]
        with (patch.object(sdk, "rm_init", side_effect=AssertionError("Offline only")),
              patch.object(sdk, "rm_create_robot_arm", side_effect=AssertionError("No robot connection")),
              patch.object(sdk.RoboticArm, "__init__", side_effect=AssertionError("Algo only")),
              patch.object(sdk, "rm_movej", side_effect=AssertionError("No motion API")),
              patch.object(mujoco, "mj_step", side_effect=AssertionError("No dynamics")),
              patch.object(control, "run_segment", side_effect=AssertionError("No actuator commands")),
              patch.object(sdk.Algo, "rm_algo_set_dh", side_effect=AssertionError("No DH edits"))):
            result = comparison.run_comparison(repeats=1)
        self.assertFalse(result["motion_executed"])
        self.assertEqual(result["simulation_steps"], 0)
        for key, snapshot in before.items():
            np.testing.assert_array_equal(getattr(data, key), snapshot)
        self.assertEqual([p.read_bytes() for p in assets], original)

    def test_cli_from_other_directory(self):
        with tempfile.TemporaryDirectory(prefix="realman-ik-") as directory:
            output = Path(directory)/"result.json"
            result = subprocess.run([sys.executable, "-I", str(ROOT/"scripts/compare_ik.py"),
                                     "--repeats", "1", "--json", str(output)],
                                    cwd=directory, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["completed"])
            self.assertIn("BOUNDARY FAILURE", result.stdout)
            self.assertFalse(report["same_robot_model_confirmed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
