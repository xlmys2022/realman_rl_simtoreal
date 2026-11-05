"""Matched A/B dynamics, gravity diagnosis, force routing and total-drive limits."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import control_joints as control
import control_ee as ee
import compare_controllers as compare
from kinematics import Kinematics


class GravityCompensationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = compare.run_comparison()

    def test_only_compensation_configuration_changes(self):
        baseline, jids, aids = control.load_model()
        enabled, _, _ = control.load_model(gravity_compensation=True)
        for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat",
                     "body_weldid", "jnt_range", "dof_damping", "actuator_gainprm", "actuator_biasprm",
                     "actuator_gear", "actuator_forcerange", "jnt_actfrcrange", "jnt_actfrclimited",
                     "geom_contype", "geom_conaffinity", "geom_size", "site_pos", "exclude_signature",
                     "key_qpos", "key_ctrl"):
            np.testing.assert_array_equal(getattr(baseline, name), getattr(enabled, name), err_msg=name)
        np.testing.assert_array_equal(baseline.opt.gravity, enabled.opt.gravity)
        self.assertEqual(baseline.opt.timestep, enabled.opt.timestep)
        self.assertEqual(baseline.opt.integrator, enabled.opt.integrator)
        self.assertEqual(baseline.opt.disableflags, enabled.opt.disableflags)
        self.assertEqual(baseline.ngravcomp, 0)
        self.assertEqual(enabled.ngravcomp, 8)
        self.assertTrue(np.all(enabled.jnt_actgravcomp[jids]))
        self.assertTrue(np.all(enabled.body_gravcomp[enabled.body_weldid != 0] == 1))
        self.assertFalse(np.any(baseline.body_gravcomp))
        off_again, _, _ = control.load_model(gravity_compensation=False)
        self.assertEqual(off_again.ngravcomp, 0)
        self.assertFalse(np.any(off_again.body_gravcomp))

    def test_zero_velocity_gravity_matches_com_jacobians(self):
        m, _, _ = control.load_model()
        data = control.initialize(m)
        for offset in (np.zeros(6), np.deg2rad([2, -2, 2, -2, 2, -2])):
            data.qpos[:] = m.key_qpos[0] + offset  # Calculation-only data, no physical integration.
            mujoco.mj_forward(m, data)
            g = np.zeros(m.nv)
            for bid in range(1, m.nbody):
                jp = np.zeros((3, m.nv))
                mujoco.mj_jacBodyCom(m, data, jp, None, bid)
                g += jp.T @ (-m.body_mass[bid] * m.opt.gravity)
            np.testing.assert_allclose(compare.gravity_torque(m, data.qpos), g, atol=1e-11, rtol=0)

    def test_moving_bias_is_not_pure_gravity_and_probe_is_isolated(self):
        m, _, _ = control.load_model(gravity_compensation=True)
        data = control.initialize(m)
        data.qvel[:] = [.6, -.5, .4, -.3, .2, -.1]
        mujoco.mj_forward(m, data)
        before = {name: getattr(data, name).copy() for name in ("qpos", "qvel", "ctrl", "qfrc_bias")}
        g = compare.gravity_torque(m, data.qpos)
        self.assertGreater(np.linalg.norm(data.qfrc_bias - g), 1e-4)
        np.testing.assert_allclose(data.qfrc_gravcomp, g, atol=1e-11, rtol=0)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(data, name), value)

    def test_static_compensation_is_in_actuator_not_passive_channel(self):
        m, jids, aids = control.load_model(gravity_compensation=True)
        data = control.initialize(m)
        g = compare.gravity_torque(m, data.qpos)
        np.testing.assert_allclose(data.qfrc_actuator, g, atol=1e-12, rtol=0)
        np.testing.assert_allclose(data.qfrc_gravcomp, g, atol=1e-12, rtol=0)
        np.testing.assert_allclose(data.actuator_force, 0, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(data.qfrc_passive, np.zeros(6))
        compare.ForceMonitor(m, jids, aids).observe(data)

    def test_servo_plus_compensation_saturates_at_joint_total_bounds(self):
        m, jids, aids = control.load_model(gravity_compensation=True)
        for sign in (-1, 1):
            data = control.initialize(m)
            data.ctrl[aids] = m.actuator_ctrlrange[aids, 0 if sign < 0 else 1]
            mujoco.mj_forward(m, data)  # Deliberate saturation probe; no large motion.
            requested = data.actuator_force[aids] + data.qfrc_gravcomp
            limits = m.jnt_actfrcrange[jids]
            self.assertGreater(np.max(np.abs(requested - data.qfrc_actuator)), 0.01)
            np.testing.assert_allclose(data.qfrc_actuator, np.clip(requested, limits[:, 0], limits[:, 1]), atol=1e-12)
            compare.ForceMonitor(m, jids, aids).observe(data)

    def test_multiple_actuators_and_compensation_share_one_joint_limit(self):
        original, _, _ = control.load_model(gravity_compensation=True)
        spec = mujoco.MjSpec.from_file(str(control.SCENE))
        for bid in range(1, original.nbody):
            spec.body(original.body(bid).name).gravcomp = float(original.body_gravcomp[bid])
        for name in control.JOINTS:
            spec.joint(name).actgravcomp = True
        motor = spec.add_actuator(name="extra_test_motor", target="joint_2", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        motor.gainprm[0], motor.gear[0] = 1, 1
        motor.forcelimited, motor.forcerange = True, [-60, 60]
        spec.key("home").ctrl = np.r_[spec.key("home").ctrl, 0]
        m = spec.compile()
        for sign in (-1, 1):
            data = control.initialize(m)
            pd, extra = m.actuator("position_2").id, m.actuator("extra_test_motor").id
            data.ctrl[pd] = m.actuator_ctrlrange[pd, 0 if sign < 0 else 1]
            data.ctrl[extra] = sign * 60
            mujoco.mj_forward(m, data)
            requested = data.actuator_force[pd] + data.actuator_force[extra] + data.qfrc_gravcomp[1]
            self.assertGreater(abs(requested), 100)
            self.assertAlmostEqual(data.qfrc_actuator[1], sign * 60)
            self.assertTrue(np.all(data.qfrc_actuator >= m.jnt_actfrcrange[:, 0]))
            self.assertTrue(np.all(data.qfrc_actuator <= m.jnt_actfrcrange[:, 1]))

    def test_passive_bypass_configuration_is_detected_and_rejected(self):
        m, jids, aids = control.load_model(gravity_compensation=True)
        m.jnt_actgravcomp[:] = False  # Deliberate unsafe counterexample, never integrated.
        data = control.initialize(m)
        data.ctrl[aids] = m.actuator_ctrlrange[aids, 1]
        mujoco.mj_forward(m, data)
        self.assertGreater(data.qfrc_actuator[1] + data.qfrc_passive[1], 60)
        with self.assertRaisesRegex(ValueError, "routed"):
            compare.ForceMonitor(m, jids, aids)

    def test_missing_joint_limits_and_external_injection_rejected(self):
        m, jids, aids = control.load_model(gravity_compensation=True)
        m.jnt_actfrclimited[0] = False
        with self.assertRaisesRegex(ValueError, "limits"):
            compare.ForceMonitor(m, jids, aids)
        m.jnt_actfrclimited[0] = True
        monitor = compare.ForceMonitor(m, jids, aids)
        data = control.initialize(m)
        data.qfrc_applied[0] = 1
        mujoco.mj_forward(m, data)
        with self.assertRaisesRegex(ValueError, "External"):
            monitor.observe(data)

    def test_home_gravity_equilibrium_and_improvement(self):
        pd = self.report["controllers"]["pd"]["cases"]["home"]
        gc = self.report["controllers"]["gravity"]["cases"]["home"]
        for case in (pd, gc):
            d = case["diagnostics"]
            self.assertLess(np.max(np.abs(d["force_balance_residual_nm"])), 1e-8)
            np.testing.assert_allclose(d["pd_generated_nm"], d["pd_formula_nm"], atol=1e-10)
            self.assertLess(np.max(np.abs(d["velocity_rad_s"])), 1e-10)
            self.assertLess(np.max(np.abs(d["velocity_bias_nm"])), 1e-10)
        np.testing.assert_allclose(pd["diagnostics"]["pd_generated_nm"], pd["diagnostics"]["pure_gravity_nm"], atol=1e-8)
        self.assertGreater(pd["metrics"]["home_offset_norm_mm"], 2)
        self.assertLess(gc["metrics"]["home_offset_norm_mm"], 1e-6)
        self.assertLess(np.max(np.abs(gc["diagnostics"]["joint_error_deg"])), 1e-8)
        self.assertLess(np.max(np.abs(gc["diagnostics"]["pd_generated_nm"])), 1e-8)

    def test_error_definitions_are_vector_consistent(self):
        for mode in self.report["controllers"].values():
            metrics = [case["metrics"] for case in mode["cases"].values()]
            metrics += [phase["metrics"] for phase in mode["joint_sequence"]["phases"]]
            for result in metrics:
                np.testing.assert_allclose(np.array(result["home_offset_mm"]) + result["relative_error_mm"],
                                           result["absolute_error_mm"], atol=1e-12, rtol=0)
        case = self.report["controllers"]["pd"]["cases"]["x_plus_5mm"]["metrics"]
        self.assertAlmostEqual(case["home_offset_norm_mm"], 2.2102383598, places=6)
        self.assertAlmostEqual(case["relative_error_norm_mm"], .210909, places=5)
        self.assertAlmostEqual(case["absolute_error_norm_mm"], 2.3915444026, places=6)

    def test_matched_ik_targets_and_below_one_mm_exploratory_goal(self):
        a, b = [self.report["controllers"][mode]["cases"] for mode in ("pd", "gravity")]
        for name in ("x_plus_5mm", "x_minus_5mm", "z_plus_5mm"):
            np.testing.assert_array_equal(a[name]["ik"]["target_joint_rad"], b[name]["ik"]["target_joint_rad"])
            np.testing.assert_array_equal(a[name]["metrics"]["target_position_m"], b[name]["metrics"]["target_position_m"])
            self.assertEqual(a[name]["dynamics"]["simulation_time_s"], b[name]["dynamics"]["simulation_time_s"])
            self.assertLess(b[name]["metrics"]["absolute_error_norm_mm"], 1)
            self.assertLess(b[name]["metrics"]["absolute_error_norm_mm"], a[name]["metrics"]["absolute_error_norm_mm"] / 100)
            self.assertLess(b[name]["metrics"]["orientation_error_deg"], .001)

    def test_all_normal_steps_are_bounded_quiet_and_collision_free(self):
        for mode in self.report["controllers"].values():
            cases = mode["cases"]
            blocks = [case["forces"] for case in cases.values()] + [mode["joint_sequence"]["forces"]]
            for f in blocks:
                limits = np.max(np.abs(f["joint_total_limits_nm"]), axis=1)
                self.assertTrue(np.all(np.array(f["peak_total_drive_nm"]) <= limits + 1e-9))
                self.assertEqual(f["saturated_steps"], 0)
            self.assertEqual(cases["home"]["forces"]["steps"], 1500)
            self.assertEqual(cases["x_plus_5mm"]["forces"]["steps"], 4000)
            self.assertEqual(mode["joint_sequence"]["forces"]["steps"], 19500)
            phases = [p["joint_stats"] for p in mode["joint_sequence"]["phases"]]
            phases += [cases["home"]["joint_stats"]]
            for name in ("x_plus_5mm", "x_minus_5mm", "z_plus_5mm"):
                phases.extend([cases[name]["dynamics"]["hold"], cases[name]["dynamics"]["motion"]])
            for p in phases:
                self.assertEqual(p["max_contacts"], 0)
                self.assertLessEqual(max(p["tail_range_deg"]), control.MAX_TAIL_RANGE_DEG)
                self.assertLessEqual(max(p["tail_speed_deg_s"]), control.MAX_TAIL_SPEED_DEG_S)

    def test_compensated_execution_never_rewrites_live_qpos(self):
        m, jids, aids = control.load_model(gravity_compensation=True)
        data, kin = control.initialize(m), Kinematics(m)
        target = kin.home_position + [.005, 0, 0]
        result = kin.solve_ik(target, kin.home_q)
        original_step = mujoco.mj_step
        previous, count = None, 0
        def checked(model, state):
            nonlocal previous, count
            if previous is not None:
                np.testing.assert_array_equal(state.qpos, previous)
            original_step(model, state)
            previous, count = state.qpos.copy(), count + 1
        with patch.object(mujoco, "mj_step", side_effect=checked):
            report = ee.execute_motion(m, data, jids, aids, kin, result, target)
        self.assertEqual(count, 4000)
        self.assertLess(report["actual_position_error_m"], .001)
        np.testing.assert_array_equal(data.qpos, previous)


if __name__ == "__main__":
    unittest.main(verbosity=2)
