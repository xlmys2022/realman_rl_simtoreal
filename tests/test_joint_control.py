"""Physical invariants and integration tests; run with the realman_rl interpreter."""

from pathlib import Path
import hashlib
import subprocess
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import control_joints as control


class JointControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.jids, cls.aids = control.load_model()

    def test_original_assets_and_static_inspection(self):
        assets = ROOT / "assets/robots/rml63"
        for line in (assets / "SHA256SUMS").read_text().splitlines():
            checksum, path = line.split(maxsplit=1)
            self.assertEqual(hashlib.sha256((assets / path).read_bytes()).hexdigest(), checksum)
        result = subprocess.run([sys.executable, "-I", str(ROOT / "scripts/inspect_model.py")],
                                cwd="/tmp", text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nq=7, nv=7, nu=0", result.stdout)

    def test_conversion_preserves_geometry_inertia_and_limits(self):
        original = mujoco.MjModel.from_xml_path(str(ROOT / "assets/robots/rml63/rml63_mujoco.urdf"))
        old_data, data = mujoco.MjData(original), control.initialize(self.model)
        old_data.qpos[:6] = data.qpos
        mujoco.mj_forward(original, old_data)
        for bid in range(1, original.nbody):
            old, new = original.body(bid), self.model.body(original.body(bid).name)
            for field in ("mass", "ipos", "iquat", "inertia", "pos", "quat"):
                np.testing.assert_allclose(getattr(old, field), getattr(new, field), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(old_data.xpos[bid], data.xpos[new.id], rtol=0, atol=1e-13)
            np.testing.assert_allclose(old_data.xmat[bid], data.xmat[new.id], rtol=0, atol=1e-13)
        # Resolve by names; the scene compiler can place the world floor before robot geoms.
        for gid in range(original.ngeom):
            body = original.body(int(original.geom_bodyid[gid])).name
            suffix = "_visual" if original.geom_group[gid] == 1 else "_collision"
            new_gid = self.model.geom(body + suffix).id
            np.testing.assert_allclose(old_data.geom_xpos[gid], data.geom_xpos[new_gid], atol=1e-13, rtol=0)
            np.testing.assert_allclose(old_data.geom_xmat[gid], data.geom_xmat[new_gid], atol=1e-13, rtol=0)
        for name in control.JOINTS:
            np.testing.assert_array_equal(original.joint(name).range, self.model.joint(name).range)
        self.assertEqual([self.model.joint(i).name for i in range(self.model.njnt)], list(control.JOINTS))

    def test_position_actuator_force_mapping_and_clamps(self):
        for index in range(6):
            data = control.initialize(self.model)
            aid = self.aids[index]
            data.ctrl[aid] += 0.01
            mujoco.mj_forward(self.model, data)
            expected = np.zeros(6)
            expected[index] = self.model.actuator_gainprm[aid, 0] * 0.01
            np.testing.assert_allclose(data.qfrc_actuator, expected, atol=1e-11)
            data.ctrl[:] = 1e6  # Deliberately invalid input: verify model force clamping without stepping.
            mujoco.mj_forward(self.model, data)
            self.assertTrue(np.all(np.abs(data.qfrc_actuator) <= [60, 60, 30, 30, 10, 10]))

    def test_hold_base_camera_and_uncommanded_joints(self):
        data = control.initialize(self.model)
        base = self.model.body("base_link").id
        tip = self.model.body("link_6").id
        camera = self.model.body("camera_rolink").id
        initial_base = data.xpos[base].copy()
        relative = data.xmat[tip].reshape(3, 3).T @ (data.xpos[camera] - data.xpos[tip])
        home = data.ctrl[self.aids].copy()
        result = control.run_segment(self.model, data, self.jids, self.aids, home, "20s hold", ramp=0, hold=20)
        self.assertEqual(result["max_contacts"], 0)
        np.testing.assert_array_equal(data.xpos[base], initial_base)
        np.testing.assert_allclose(data.xmat[tip].reshape(3, 3).T @ (data.xpos[camera] - data.xpos[tip]), relative, atol=1e-13)
        self.assertEqual(self.model.body_jntnum[camera], 0)

    def test_each_joint_positive_and_negative_small_motion(self):
        for sign in (1, -1):
            for index in range(6):
                with self.subTest(sign=sign, joint=index + 1):
                    data = control.initialize(self.model)
                    home = data.ctrl[self.aids].copy()
                    control.run_segment(self.model, data, self.jids, self.aids, home, "settle", ramp=0, hold=2)
                    before = data.qpos.copy()
                    target = home.copy()
                    target[index] += np.deg2rad(sign * 3)
                    result = control.run_segment(self.model, data, self.jids, self.aids, target, "single joint")
                    self.assertEqual(result["max_contacts"], 0)
                    others = np.arange(6) != index
                    self.assertLess(np.max(np.abs(np.rad2deg(data.qpos[others] - before[others]))), 0.1)

    def test_half_timestep_and_combined_small_motion(self):
        model, jids, aids = control.load_model()
        model.opt.timestep = 0.001
        data = control.initialize(model)
        target = data.ctrl[aids] + np.deg2rad([2, -2, 2, -2, 2, -2])
        result = control.run_segment(model, data, jids, aids, target, "combined at 1ms")
        self.assertEqual(result["max_contacts"], 0)

    def test_base_assembly_exclusion_is_specific(self):
        data = control.initialize(self.model)
        for degrees in (-90, -10, 0, 10, 90):
            data.qpos[0] = np.deg2rad(degrees)
            mujoco.mj_forward(self.model, data)
            self.assertEqual(data.ncon, 0)
        self.assertEqual(self.model.nexclude, 1)

    def test_nonadjacent_self_collision_is_still_detected(self):
        data = control.initialize(self.model)
        # Forward-only collision fixture, deliberately outside the demo motion region.
        data.qpos[:] = np.deg2rad([17.10682648, 140.16436566, -91.55096729,
                                  1.5055794, -85.31318548, -188.64332841])
        mujoco.mj_forward(self.model, data)
        pairs = {frozenset((self.model.geom(int(c.geom1)).name, self.model.geom(int(c.geom2)).name))
                 for c in data.contact if c.dist < -0.001}
        self.assertIn(frozenset(("base_link_collision", "link_2_collision")), pairs)

    def test_environment_collision_detected_and_rejected(self):
        home_data = control.initialize(self.model)
        spec = mujoco.MjSpec.from_file(str(control.SCENE))
        spec.worldbody.add_geom(name="test_obstacle", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                               size=[0.02, 0, 0], pos=home_data.xipos[self.model.body("link_2").id])
        model = spec.compile()
        data = control.initialize(model)
        self.assertTrue(any(c.dist < -0.001 for c in data.contact))
        with self.assertRaisesRegex(ValueError, "Unexpected collision"):
            control.run_segment(model, data, self.jids, self.aids, data.ctrl.copy(), "obstacle")

    def test_invalid_commands_and_bad_mapping_rejected(self):
        data = control.initialize(self.model)
        home = data.ctrl.copy()
        for target in ([0] * 5, [np.nan] * 6, home + 1):
            with self.assertRaises(ValueError):
                control.validate_target(self.model, self.jids, home, target)
        spec = mujoco.MjSpec.from_file(str(control.SCENE))
        spec.actuator("position_1").target = "joint_2"
        # Test the actual loader against a real, incorrectly wired MJCF.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong.xml"
            xml = spec.to_xml()
            xml = xml.replace('meshdir="upstream/meshes"', f'meshdir="{ROOT / "assets/robots/rml63/upstream/meshes"}"')
            path.write_text(xml)
            with self.assertRaisesRegex(ValueError, "mapping"):
                control.load_model(path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
