"""Exact-name asset provenance, physical fingers, arm compatibility and contact."""

import hashlib
from importlib import util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from realman_rl_simtoreal.paths import GRIPPER_ASSETS, GRIPPER_SCENE, PROJECT_ROOT
from realman_rl_simtoreal.simulation import control
from realman_rl_simtoreal.simulation.gripper import Gripper, SIGNS, MAX_WIDTH, angle_to_width, width_to_angle
from realman_rl_simtoreal.simulation.kinematics import Kinematics
from realman_rl_simtoreal.simulation.model import load_model, initialize, gravity_torque


class GripperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.jids, cls.aids = load_model(GRIPPER_SCENE, gravity_compensation=True)
        cls.original, cls.ojids, cls.oaids = load_model(gravity_compensation=True)

    def setup_gripper(self, model=None):
        model = self.model if model is None else model
        data = initialize(model)
        return data, Gripper(model, data)

    def test_source_commit_and_all_original_bytes_are_auditable(self):
        manifest = json.loads((GRIPPER_ASSETS/'upstream_manifest.json').read_text())
        self.assertEqual(manifest['commit'], '776c1a7dd668ba67bfb03636015c59c8ef6e1b43')
        self.assertEqual(len(manifest['files']), 13)
        for row in manifest['files']:
            data = (GRIPPER_ASSETS/'upstream'/row['local_path']).read_bytes()
            self.assertEqual(len(data), row['size'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), row['sha256'])
            self.assertEqual(hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(), row['git_blob_sha1'])
            self.assertIn(manifest['commit'], row['url'])
        urdf = ET.parse(GRIPPER_ASSETS/'upstream/EG2-4C2.urdf').getroot()
        self.assertEqual(urdf.get('name'), 'EG2-4C2')
        self.assertEqual(len(urdf.findall('link')), 7)
        self.assertEqual(len(urdf.findall('joint')), 6)
        self.assertEqual(len(urdf.findall('.//mimic')), 5)
        self.assertTrue((GRIPPER_ASSETS/'SOURCE.md').is_file())

    def test_mass_and_stroke_discrepancies_are_not_hidden(self):
        raw = ET.parse(GRIPPER_ASSETS/'upstream/EG2-4C2.urdf').getroot()
        mass = sum(float(link.find('inertial/mass').get('value')) for link in raw.findall('link'))
        self.assertAlmostEqual(mass, .0942992, places=12)
        self.assertNotAlmostEqual(mass, .231, places=3)
        self.assertAlmostEqual(MAX_WIDTH, .06621735660659232, places=14)
        with self.assertRaisesRegex(ValueError, 'does not reach 70'):
            width_to_angle(.070)

    def test_combined_and_standalone_structure(self):
        m = self.model
        self.assertEqual((m.nq, m.nv, m.nu, m.neq), (12, 12, 7, 5))
        data, g = self.setup_gripper()
        self.assertEqual(len(g.joint_ids), 6)
        self.assertEqual(len(g.equality_ids), 5)
        self.assertNotIn(g.actuator_id, self.aids)
        self.assertEqual(m.actuator_trnid[g.actuator_id, 0], g.primary)
        standalone = mujoco.MjModel.from_xml_path(str(GRIPPER_ASSETS/'eg2_4c2.xml'))
        self.assertEqual((standalone.nq, standalone.nv, standalone.nu, standalone.neq), (6, 6, 1, 5))
        _, single = self.setup_gripper(standalone)
        single.close()
        single.step(1200)
        self.assertLess(abs(single.get_width()), .0002)
        self.assertEqual(data.time, 0)

    def test_all_existing_arm_body_joint_actuator_parameters_are_preserved(self):
        for i in range(1, self.original.nbody):
            old = self.original.body(i)
            new = self.model.body(old.name)
            for attr in ('mass', 'inertia', 'ipos', 'iquat', 'pos', 'quat'):
                np.testing.assert_array_equal(getattr(old, attr), getattr(new, attr))
        np.testing.assert_array_equal(self.jids, self.ojids)
        np.testing.assert_array_equal(self.aids, self.oaids)
        for attr in ('jnt_range', 'jnt_axis', 'jnt_actfrcrange'):
            np.testing.assert_array_equal(getattr(self.model, attr)[self.jids], getattr(self.original, attr)[self.ojids])
        for attr in ('actuator_gainprm', 'actuator_biasprm', 'actuator_ctrlrange', 'actuator_forcerange'):
            np.testing.assert_array_equal(getattr(self.model, attr)[self.aids], getattr(self.original, attr)[self.oaids])
        self.assertAlmostEqual(self.model.body_mass.sum()-self.original.body_mass.sum(), .1052992, places=12)

    def test_mount_camera_and_nominal_grasp_frames(self):
        d, g = self.setup_gripper()
        ee = self.model.site('ee_site').id
        mount = self.model.site('gripper_mount').id
        grasp = self.model.site('grasp_site').id
        rotation = d.site_xmat[ee].reshape(3, 3)
        np.testing.assert_allclose((d.site_xpos[mount]-d.site_xpos[ee]) @ rotation, [0, 0, .008], atol=1e-15)
        z = .079+.0223*np.sin(.41)+.035591*np.cos(.41)+.027
        np.testing.assert_allclose((d.site_xpos[grasp]-d.site_xpos[ee]) @ rotation, [0, 0, .008+z], atol=1e-15)
        np.testing.assert_array_equal(d.site_xmat[ee], d.site_xmat[mount])
        for name in ['camera_rolink', 'camera_link', '4C2_baselink', 'eg2_experimental_adapter']:
            self.assertEqual(self.model.body_weldid[self.model.body(name).id], self.model.body_weldid[self.model.body('link_6').id])
        original_data = initialize(self.original)
        for name in ['camera_rolink', 'camera_link']:
            np.testing.assert_array_equal(d.body(name).xpos, original_data.body(name).xpos)

    def test_width_inverse_and_actual_urdf_mechanism_agree(self):
        model = mujoco.MjModel.from_xml_path(str(GRIPPER_ASSETS/'eg2_4c2.xml'))
        data, g = self.setup_gripper(model)
        # Calculation-only configurations; this is not normal control.
        for angle in np.linspace(0, .82, 11):
            for name, sign in SIGNS.items():
                data.qpos[model.joint(name).qposadr] = sign*angle
            width = angle_to_width(angle)
            self.assertAlmostEqual(width_to_angle(width), angle, places=14)
            self.assertAlmostEqual(g.get_width(), width, places=14)

    def test_setpoint_calls_do_not_change_live_state_or_other_controls(self):
        data, g = self.setup_gripper()
        q, ctrl = data.qpos.copy(), data.ctrl.copy()
        for call in [lambda: g.set_width(.02), g.close, g.open]:
            call()
            np.testing.assert_array_equal(data.qpos, q)
            np.testing.assert_array_equal(data.ctrl, ctrl)
        for width in [-.001, .07, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                g.set_width(width)
        self.assertEqual(g.target_width, MAX_WIDTH)

    def test_actual_close_midpoint_and_open_are_stable(self):
        data, g = self.setup_gripper()
        for width in [0, .03, MAX_WIDTH]:
            g.set_width(width)
            samples = [g.step(25) for _ in range(60)]
            self.assertLess(abs(samples[-1]-width), .0002)
            self.assertLess(np.ptp(samples[-10:]), .0001)
            self.assertGreaterEqual(min(samples), -.0001)
            self.assertLessEqual(max(samples), MAX_WIDTH+.0001)
        self.assertLess(g.statistics()['peak_joint_speed_rad_s'], 1)
        self.assertLess(g.statistics()['max_mimic_error_rad'], .001)
        self.assertTrue(np.isfinite(data.qpos).all())

    def test_qpos_changes_only_inside_mj_step(self):
        data, g = self.setup_gripper()
        original = mujoco.mj_step
        previous = data.qpos.copy()
        calls = 0
        def observed(model, current):
            nonlocal previous, calls
            self.assertIs(current, data)
            np.testing.assert_array_equal(current.qpos, previous)
            original(model, current)
            previous = current.qpos.copy()
            calls += 1
        g.close()
        with patch.object(mujoco, 'mj_step', side_effect=observed):
            g.step(1200)
        np.testing.assert_array_equal(data.qpos, previous)
        self.assertEqual(calls, 1200)
        self.assertLess(g.get_width(), .0002)

    def test_width_measurement_does_not_refresh_live_force_cache(self):
        data, g = self.setup_gripper()
        before = {name: getattr(data, name).copy() for name in ['qpos', 'qvel', 'ctrl', 'qfrc_actuator', 'xpos']}
        g.get_width(); g.pad_positions()
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(data, name), value)

    def test_gripper_control_keeps_arm_ctrl_unchanged(self):
        data, g = self.setup_gripper()
        original = data.ctrl[self.aids].copy()
        g.close(); g.step(1000)
        np.testing.assert_array_equal(data.ctrl[self.aids], original)
        self.assertLess(np.max(np.abs(data.qpos[self.model.jnt_qposadr[self.jids]]-original)), np.deg2rad(.05))

    def test_arm_controller_only_assigns_arm_channels(self):
        data, g = self.setup_gripper()
        grip_ctrl = float(data.ctrl[g.actuator_id])
        target = data.ctrl[self.aids].copy()+np.deg2rad([1, 0, 0, 0, 0, 0])
        result = control.run_segment(self.model, data, self.jids, self.aids, target, 'combined J1',
                                     ramp=.5, hold=1, step_observer=g.arm_monitor.observe, tick=g._check_state)
        self.assertLess(np.max(np.abs(result['error_deg'])), .05)
        self.assertEqual(data.ctrl[g.actuator_id], grip_ctrl)
        self.assertLess(abs(g.get_width()-MAX_WIDTH), .0002)

    def test_gravity_load_hold_and_total_force_limits(self):
        data, g = self.setup_gripper()
        self.assertEqual(self.model.ngravcomp, 16)
        self.assertTrue(np.all(self.model.jnt_actgravcomp))
        payload_torque = gravity_torque(self.model, data.qpos)[self.model.jnt_dofadr[self.jids]]
        old = initialize(self.original)
        baseline = gravity_torque(self.original, old.qpos)
        self.assertGreater(np.max(np.abs(payload_torque-baseline)), .05)
        g.step(1000)
        np.testing.assert_allclose(data.qpos[self.model.jnt_qposadr[self.jids]], self.model.key_qpos[0, self.model.jnt_qposadr[self.jids]], atol=1e-7)
        stats = g.statistics()
        self.assertEqual(stats['arm_forces']['steps'], 1000)
        self.assertTrue(np.all(np.array(stats['arm_forces']['peak_total_drive_nm']) <= [60, 60, 30, 30, 10, 10]))
        self.assertLess(stats['peak_primary_total_drive_nm'], .2+1e-9)

    def test_unrelated_world_payload_is_not_gravity_compensated(self):
        spec = mujoco.MjSpec.from_file(str(GRIPPER_SCENE))
        spec.worldbody.add_body(name='unrelated').add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.01, 0, 0], mass=.1)
        # Exercise the production loader rather than copying its traversal.
        with tempfile.TemporaryDirectory() as directory:
            text = spec.to_xml()
            file = Path(directory)/'test.xml'
            # Export with resource paths fixed for a different working directory.
            tree = ET.fromstring(text)
            compiler = tree.find('compiler')
            if compiler is not None:
                compiler.attrib.pop('meshdir', None)
            for mesh in tree.findall('asset/mesh'):
                name = Path(mesh.get('file')).name
                base = GRIPPER_ASSETS/'upstream/meshes' if name.startswith('4C2_') else PROJECT_ROOT/'assets/robots/rml63/upstream/meshes'
                mesh.set('file', str(base/name))
            file.write_text(ET.tostring(tree, encoding='unicode'))
            model, _, _ = load_model(file, gravity_compensation=True)
        self.assertEqual(model.body_gravcomp[model.body('unrelated').id], 0)
        self.assertEqual(model.body_gravcomp[model.body('4C2_Link5').id], 1)

    def test_both_fingers_detect_external_contacts_and_contact_forces(self):
        data, g = self.setup_gripper()
        spec = mujoco.MjSpec.from_file(str(GRIPPER_SCENE))
        left, right = g.pad_positions()
        axis = data.xmat[g.base].reshape(3, 3)[:, 0]
        for name, point, sign in [('test_left', left, -1), ('test_right', right, 1)]:
            spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                    size=[.0015, 0, 0], pos=point+sign*.001*axis)
        model = spec.compile()
        current = initialize(model)
        pairs, forces = set(), []
        for index, contact in enumerate(current.contact):
            pairs.add(frozenset(model.geom(int(i)).name for i in contact.geom))
            force = np.zeros(6)
            mujoco.mj_contactForce(model, current, index, force)
            forces.append(force[0])
        self.assertIn(frozenset(['test_left', '4C2_Link5_collision']), pairs)
        self.assertIn(frozenset(['test_right', '4C2_Link6_collision']), pairs)
        self.assertGreater(max(forces), 0)
        with self.assertRaisesRegex(ValueError, 'Unexpected collision'):
            Gripper(model, current).step()
        self.assertEqual(current.time, 0)

    def test_collision_geometry_and_nonadjacent_arm_contacts_remain(self):
        model = self.model
        self.assertEqual(model.nexclude, 9)
        for name in ['4C2_baselink', *[f'4C2_Link{i}' for i in range(1, 7)]]:
            self.assertEqual(model.geom(name+'_collision').contype[0], 1)
            self.assertEqual(model.geom(name+'_visual').contype[0], 0)
        data = initialize(model)
        # Forward-only folded-arm counterexample, not a commanded movement.
        data.qpos[model.jnt_qposadr[self.jids]] = np.deg2rad([17.10682648, 140.16436566, -91.55096729, 1.5055794, -85.31318548, -188.64332841])
        mujoco.mj_forward(model, data)
        pairs = {frozenset(model.geom(int(i)).name for i in c.geom) for c in data.contact if c.dist < -.001}
        self.assertIn(frozenset(['base_link_collision', 'link_2_collision']), pairs)

    def test_invalid_state_and_disabled_coupling_are_rejected(self):
        for case in ['nan', 'inactive', 'bad_ctrl']:
            data, g = self.setup_gripper()
            if case == 'nan':
                data.qvel[g.vadr[0]] = np.nan
            elif case == 'inactive':
                data.eq_active[g.equality_ids[0]] = 0
            else:
                data.ctrl[g.actuator_id] = 100
            with self.subTest(case=case), self.assertRaises(ValueError):
                g.step()
            self.assertEqual(data.time, 0)

    def test_six_angle_fk_is_isolated_and_consistent_on_both_models(self):
        a, b = Kinematics(self.original), Kinematics(self.model)
        pa, ra = a.fk(a.home_q)
        pb, rb = b.fk(b.home_q)
        np.testing.assert_array_equal(pa, pb)
        np.testing.assert_array_equal(ra, rb)
        self.assertEqual(b.jacobian(b.home_q)[0].shape, (3, 6))
        data, g = self.setup_gripper()
        before = data.qpos.copy()
        self.assertTrue(b.solve_ik(pb+[.001, 0, 0], b.home_q).success)
        np.testing.assert_array_equal(data.qpos, before)
        self.assertEqual((self.original.nq, self.original.nv, self.original.nu), (6, 6, 6))

    def test_generator_is_byte_reproducible_and_headless_demo_works(self):
        files = [GRIPPER_ASSETS/name for name in ['eg2_4c2_mujoco.urdf', 'eg2_4c2.xml', 'rml63_eg2_4c2.xml']]
        before = [p.read_bytes() for p in files]
        result = subprocess.run([sys.executable, '-I', str(PROJECT_ROOT/'scripts/models/build_gripper_model.py')],
                                cwd='/tmp', capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual([p.read_bytes() for p in files], before)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            result = subprocess.run([sys.executable, '-I', str(PROJECT_ROOT/'scripts/demos/demo_gripper.py'),
                                     '--cycles', '1', '--json', str(output)], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report['completed'])
            self.assertEqual(len(report['phases']), 3)
            self.assertLess(max(p['error_m'] for p in report['phases']), .0002)


if __name__ == '__main__':
    unittest.main(verbosity=2)
