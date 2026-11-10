"""Dynamic contact evidence, strict failure controls and one shared time base."""
from importlib import util
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from realman_rl_simtoreal.paths import PROJECT_ROOT, GRIPPER_SCENE, GRIPPER_ASSETS
from realman_rl_simtoreal.simulation import control
from realman_rl_simtoreal.simulation.gripper import Gripper
from realman_rl_simtoreal.simulation.grasp_contact import (
    SCENE, load_experiment, ContactRecorder, contact_force_on_target, groove_probe)
from realman_rl_simtoreal.simulation.model import load_model, initialize


def script_module(filename, name):
    spec = util.spec_from_file_location(name, PROJECT_ROOT/'scripts'/filename)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stl_ray_x(side, y, z):
    """Independent triangle intersections: do not confuse rear face with groove."""
    link = '4C2_Link5' if side == 'left' else '4C2_Link6'
    raw = (GRIPPER_ASSETS/f'upstream/meshes/{link}.STL').read_bytes()
    t = np.frombuffer(raw, dtype=np.dtype([('n', '<f4', (3,)), ('v', '<f4', (3, 3)),
                                         ('a', '<u2')]), offset=84)['v'].astype(float)
    a = t[:, 0, 1:]
    u, v = t[:, 1, 1:]-a, t[:, 2, 1:]-a
    det = u[:, 0]*v[:, 1]-u[:, 1]*v[:, 0]
    valid = abs(det) > 1e-14
    t, a, u, v, det = t[valid], a[valid], u[valid], v[valid], det[valid]
    w = np.array([y, z])-a
    b = (w[:, 0]*v[:, 1]-w[:, 1]*v[:, 0])/det
    c = (u[:, 0]*w[:, 1]-u[:, 1]*w[:, 0])/det
    valid = (b >= -1e-8) & (c >= -1e-8) & (b+c <= 1+1e-8)
    x = t[:, 0, 0]+b*(t[:, 1, 0]-t[:, 0, 0])+c*(t[:, 2, 0]-t[:, 0, 0])
    return np.unique(np.round(x[valid], 9))


class ContactModelTests(unittest.TestCase):
    def setUp(self):
        self.m, self.d, self.j, self.a, self.g, self.p = load_experiment()

    def test_free_object_mass_inertia_and_no_object_constraint(self):
        m = self.m
        joint, body = m.joint('target_free').id, m.body('target_object').id
        self.assertEqual((m.nq, m.nv, m.nu, m.neq), (19, 18, 7, 5))
        self.assertEqual(m.jnt_type[joint], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.body_parentid[body], 0)
        self.assertEqual(m.body_gravcomp[body], 0)
        self.assertAlmostEqual(m.body_mass[body], .01)
        np.testing.assert_allclose(m.body_inertia[body], [.01*.02**2/6]*3)
        np.testing.assert_array_equal(m.geom_size[m.geom('target_box').id], [.01]*3)
        self.assertTrue(np.all(m.eq_type == mujoco.mjtEq.mjEQ_JOINT))
        self.assertNotIn(joint, m.eq_obj1id)
        self.assertNotIn(joint, m.eq_obj2id)
        self.assertTrue(np.isfinite(self.d.qpos).all())

    def test_old_model_mapping_gains_and_inertias_preserved(self):
        old, j, a = load_model(GRIPPER_SCENE, gravity_compensation=True)
        np.testing.assert_array_equal(self.j, j)
        np.testing.assert_array_equal(self.a, a)
        for name in ('actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange'):
            np.testing.assert_array_equal(getattr(self.m, name), getattr(old, name))
        for bid in range(1, old.nbody):
            new = self.m.body(old.body(bid).name).id
            for attr in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat'):
                np.testing.assert_array_equal(getattr(self.m, attr)[new], getattr(old, attr)[bid])

    def test_scene_regeneration_identical_and_original_assets_untouched(self):
        def hashes():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in ('robots', 'grippers') for p in (PROJECT_ROOT/'assets'/folder).rglob('*') if p.is_file()}
        original, scene = hashes(), SCENE.read_bytes()
        script_module('models/build_grasp_scene.py', 'grasp_builder').build()
        self.assertEqual(SCENE.read_bytes(), scene)
        self.assertEqual(hashes(), original)

    def test_pad_layers_are_measured_inward_rectangles(self):
        builder = script_module('models/build_grasp_scene.py', 'pad_builder')
        for side, sign in [('left', 1), ('right', -1)]:
            rects = builder.pad_rectangles(side)
            self.assertEqual(len(rects), 12)
            for index, (center, size) in enumerate(rects):
                geom = self.m.geom(f'{side}_pad_{index}').id
                np.testing.assert_allclose(self.m.geom_pos[geom], center, atol=1e-15)
                np.testing.assert_allclose(self.m.geom_size[geom], size, atol=1e-15)
                self.assertAlmostEqual(size[0], .00025)
                self.assertAlmostEqual(sign*center[0]+size[0], .0177 if index < 10 else .0173, places=7)
                # Sample each layer inside the original STL solid. In
                # particular the old x=10mm rear surface is NOT a pad trough.
                for y in (center[1]-.8*size[1], center[1], center[1]+.8*size[1]):
                    for z in (center[2]-.8*size[2], center[2]+.8*size[2]):
                        intersections = stl_ray_x(side, y, z)
                        self.assertEqual(np.count_nonzero(intersections > center[0]) % 2, 1)

    def test_original_hull_fills_real_tooth_gap_proxy_does_not(self):
        original = groove_probe(hull=True)
        replacement = groove_probe()
        self.assertTrue(original)
        self.assertLess(min(c['distance_m'] for c in original), -.00024)
        self.assertEqual(replacement, [])
        for y in (.00595, .006, .00605):
            # Actual STL front is ~17.3mm in this gap. The 0.1mm
            # probe's innermost X=17.45mm is outside the REAL solid.
            self.assertLess(max(stl_ray_x('left', y, .018)), .01745)

    def test_collision_masks_keep_target_pads_robot_and_support(self):
        m = self.m
        def collides(one, two):
            i, j = m.geom(one).id, m.geom(two).id
            return bool((m.geom_contype[i]&m.geom_conaffinity[j]) or (m.geom_contype[j]&m.geom_conaffinity[i]))
        self.assertTrue(collides('left_pad_0', 'target_box'))
        self.assertTrue(collides('right_pad_0', 'target_box'))
        self.assertTrue(collides('4C2_Link5_collision', 'support_top'))
        self.assertTrue(collides('target_box', 'support_top'))
        self.assertFalse(collides('4C2_Link5_collision', 'target_box'))
        self.assertFalse(collides('left_pad_0', 'support_top'))
        self.assertFalse(m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)

    def test_interface_friction_really_changes_without_support_change(self):
        low, *_ = load_experiment(friction=.05, mass=.03)
        for i in range(low.npair):
            expected = 1 if low.pair(i).name == 'support_target' else .05
            np.testing.assert_array_equal(low.pair_friction[i, :2], [expected]*2)
        self.assertEqual(low.body_mass[low.body('target_object').id], .03)
        np.testing.assert_allclose(low.body_inertia[low.body('target_object').id], [.03*.02**2/6]*3)

    def fake_contact(self, one, two, distance):
        return SimpleNamespace(geom=np.array([self.m.geom(one).id, self.m.geom(two).id]),
                               dist=distance, pos=np.zeros(3), frame=np.eye(3).ravel())

    def test_small_allowed_penetration_and_severe_contact_rejected(self):
        c = self.fake_contact('left_pad_0', 'target_box', -.0002)
        d = SimpleNamespace(qvel=self.d.qvel.copy(), contact=[c])
        self.p.check(d)
        c.dist = -.00031
        with self.assertRaisesRegex(ValueError, 'left_pad_target'):
            self.p.check(d)

    def test_support_contact_has_own_bound(self):
        c = self.fake_contact('support_top', 'target_box', -.00015)
        d = SimpleNamespace(qvel=self.d.qvel.copy(), contact=[c])
        self.p.check(d)
        c.dist = -.00021
        with self.assertRaisesRegex(ValueError, 'target_support'):
            self.p.check(d)

    def test_unexpected_robot_table_and_internal_contacts_still_strict(self):
        for a, b, category in [('support_top', '4C2_Link5_collision', 'robot_support'),
                                ('4C2_Link5_collision', '4C2_Link6_collision', 'robot_internal')]:
            c = self.fake_contact(a, b, -.00011)
            with self.assertRaisesRegex(ValueError, category):
                self.p.check(SimpleNamespace(qvel=self.d.qvel.copy(), contact=[c]))

    def test_default_gripper_empty_contact_policy_remains_strict(self):
        m, *_ = load_model(GRIPPER_SCENE, gravity_compensation=True)
        d = initialize(m)
        g = Gripper(m, d)
        self.assertIsNone(g.contact_policy)
        g.set_width(.06)
        g.step(25)
        self.assertAlmostEqual(d.time, .05)

    def test_nonfinite_and_object_velocity_protected(self):
        for value in (np.nan, np.inf, 1.01):
            d = SimpleNamespace(qvel=self.d.qvel.copy(), contact=[])
            d.qvel[self.p.object_dof] = value
            with self.assertRaises(ValueError):
                self.p.check(d)
        self.d.ctrl[self.g.actuator_id] = np.nan
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            self.g.step()

    def test_force_frame_conversion_and_geom_order(self):
        # Contact normal world +Z, tangent1 +X, tangent2 +Y.
        c = SimpleNamespace(geom=[10, 20], frame=np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).ravel())
        force = [2., .3, .4, 0, 0, 0]
        np.testing.assert_allclose(contact_force_on_target(c, force, 20), [.3, .4, 2])
        np.testing.assert_allclose(contact_force_on_target(c, force, 10), [-.3, -.4, -2])
        with self.assertRaises(ValueError):
            contact_force_on_target(c, force, 30)

    def test_shared_cycle_integrates_25_not_50_and_never_teleports(self):
        m, d, j, a, g = self.m, self.d, self.j, self.a, self.g
        g.set_width(.06)
        target = d.ctrl[a].copy()
        target[0] += np.deg2rad(.1)
        previous = d.qpos.copy()
        original_step = mujoco.mj_step
        taken = 0
        def audited_step(model, data):
            nonlocal previous, taken
            # Between integrator calls all controllers may change only ctrl.
            np.testing.assert_array_equal(data.qpos, previous)
            original_step(model, data)
            previous = data.qpos.copy()
            taken += 1
        with patch.object(mujoco, 'mj_step', side_effect=audited_step):
            summary = control.advance_synchronized_cycle(m, d, j, a, target, g)
        self.assertEqual(taken, 25)
        self.assertEqual(g.steps, 25)
        self.assertAlmostEqual(summary['simulated_elapsed_s'], .05)
        np.testing.assert_array_equal(d.qpos, previous)
        self.assertNotEqual(float(d.qpos[m.jnt_qposadr[j[0]]]), 0.)
        self.assertNotEqual(float(d.ctrl[g.actuator_id]), .82)

    def test_reference_update_has_no_physics_and_no_arm_writes(self):
        before_q, before_a, before_t = self.d.qpos.copy(), self.d.ctrl[self.a].copy(), self.d.time
        self.g.close()
        self.g.update_control()
        np.testing.assert_array_equal(self.d.qpos, before_q)
        np.testing.assert_array_equal(self.d.ctrl[self.a], before_a)
        self.assertEqual(self.d.time, before_t)

    def test_invalid_arm_target_rejected_before_motion(self):
        target = self.d.ctrl[self.a].copy()
        target[0] += np.deg2rad(6)
        with self.assertRaises(ValueError):
            control.advance_synchronized_cycle(self.m, self.d, self.j, self.a, target, self.g)
        self.assertEqual(self.d.time, 0)

    def test_force_observer_reads_step_cache_before_refresh(self):
        rec = ContactRecorder(self.m, self.d, self.g, self.p)
        control.advance_synchronized_cycle(self.m, self.d, self.j, self.a, self.d.ctrl[self.a].copy(),
                                           self.g, physics_steps=25, step_observer=rec)
        self.assertEqual(len(rec.rows), 25)
        self.assertTrue(any(r['support_normal_n'] > 0 for r in rec.rows))
        self.assertTrue(np.isfinite(rec.rows[-1]['object_qvel']).all())
        self.assertEqual(self.g.arm_monitor.steps, 25)

    def test_injected_force_and_primary_torque_excess_rejected(self):
        self.d.qfrc_applied[0] = .01
        with self.assertRaisesRegex(ValueError, 'External force'):
            self.g.step()
        self.d.qfrc_applied[:] = 0
        self.d.qfrc_actuator[self.m.jnt_dofadr[self.g.primary]] = .21
        with self.assertRaisesRegex(ValueError, 'Gripper total drive'):
            self.g.observe_step()


class GraspExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        demo = script_module('demos/run_grasp_contact_cases.py', 'contact_experiment')
        cls.results = {name: demo.run_case(name, **demo.CASES[name])
                       for name in ('normal', 'no_close', 'release', 'low_friction', 'heavy')}

    def test_bilateral_physical_contact_and_airborne_hold(self):
        r = self.results['normal']
        self.assertTrue(r['completed'], r['safety_failure'])
        self.assertTrue(r['airborne_hold_passed'])
        hold = r['phases']['air_hold']
        self.assertGreater(hold['tail_bilateral_fraction'], .95)
        self.assertGreater(hold['tail_airborne_fraction'], .95)
        self.assertLess(hold['tail_relative_slip_m'], .002)
        self.assertGreater(hold['final']['left_normal_n'], .005)
        self.assertGreater(hold['final']['right_normal_n'], .005)
        self.assertGreater(hold['final']['support_clearance_m'], .004)
        self.assertLess(hold['final']['support_normal_n'], .001)
        self.assertGreater(hold['final']['actual_width_m']-hold['final']['command_width_m'], .019)

    def test_no_close_is_table_supported_not_grasp_success(self):
        r = self.results['no_close']
        self.assertTrue(r['completed'], r['safety_failure'])
        self.assertFalse(r['airborne_hold_passed'])
        final = r['phases']['air_hold']['final']
        self.assertEqual(final['left_normal_n']+final['right_normal_n'], 0)
        self.assertGreater(final['support_normal_n'], .05)
        self.assertLess(final['support_clearance_m'], .0001)

    def test_world_contact_resultant_balances_object_weight(self):
        rows = [r for r in self.results['normal']['records']
                if r['phase'] == 'air_hold' and r['contacts_sampled']]
        contact = rows[-1]['contacts']
        forces = [c['force_on_target_world_n'] for c in contact if c['force_on_target_world_n'] is not None]
        np.testing.assert_allclose(np.sum(forces, axis=0), [0, 0, .01*9.81], atol=.002, rtol=0)

    def test_release_returns_cube_to_support_without_constraint(self):
        r = self.results['release']
        self.assertTrue(r['completed'], r['safety_failure'])
        self.assertTrue(r['airborne_hold_passed'])
        final = r['phases']['after_release']['final']
        self.assertEqual(final['left_normal_n']+final['right_normal_n'], 0)
        self.assertGreater(final['support_normal_n'], .05)
        self.assertLess(final['support_clearance_m'], .0001)
        self.assertAlmostEqual(final['actual_width_m'], .06, delta=.0002)

    def test_adverse_friction_and_mass_failures_are_retained_and_stop(self):
        # These are safety-abort checks, NOT assertions of successful grasping.
        for name in ('low_friction', 'heavy'):
            r = self.results[name]
            self.assertFalse(r['completed'])
            self.assertFalse(r['airborne_hold_passed'])
            self.assertIsNotNone(r['safety_failure'])
            self.assertLess(r['simulation_time_s'], 12.5)
            self.assertGreater(max(row['right_normal_n'] for row in r['records']), .01)

    def test_completed_motion_has_no_numerical_warning_or_force_bypass(self):
        for name in ('normal', 'no_close', 'release'):
            r = self.results[name]
            self.assertTrue(r['completed'], r['safety_failure'])
            self.assertLess(r['gripper_statistics']['max_mimic_error_rad'], .001)
            self.assertLessEqual(r['gripper_statistics']['peak_primary_total_drive_nm'], .2)
            self.assertLessEqual(r['gripper_statistics']['peak_joint_speed_rad_s'], 1.001)
            self.assertLess(np.max(np.abs(r['final_arm_error_deg'])), .5)
            for row in r['records']:
                self.assertTrue(np.isfinite(row['object_position_m']+row['object_qvel']).all())
                self.assertLessEqual(abs(row['primary_total_drive_nm']), .2)


if __name__ == '__main__':
    unittest.main()
