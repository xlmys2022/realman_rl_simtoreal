"""迁移场景的装配、视觉时钟和策略数据契约回归。"""

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')

from realman_rl_simtoreal.deployment.calibration import load_calibration
from realman_rl_simtoreal.deployment.policy_io import RealVisualPolicyIO
from realman_rl_simtoreal.envs.factory import make_tabletop_cartesian_env
from realman_rl_simtoreal.evaluation.tabletop_cartesian import evaluate
from realman_rl_simtoreal.hardware.realman_config import REALMAN_TABLETOP
from realman_rl_simtoreal.simulation.devices.mujoco_gripper import MAX_WIDTH
from realman_rl_simtoreal.simulation.physics.grasp_metrics import CONTACT_MEASUREMENT
from realman_rl_simtoreal.simulation.runtime.observation_presets import startup_contract
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig
from realman_rl_simtoreal.tasks.grasp.tabletop_policy_io import (
    JAW_WIDTH_SCALE_M, VisualPolicyProprioception, action_contract, decode_cartesian_action,
)


class TestPolicyAction(unittest.TestCase):
    def test_randomization_stays_in_the_audited_d435_startup_area(self):
        self.assertEqual(CartesianTabletopConfig(scene_profile='realman_transfer', object_radius=.02).object_radius, .02)
        with self.assertRaisesRegex(ValueError, 'fixed D435 observation pose'):
            CartesianTabletopConfig(scene_profile='realman_transfer', object_radius=.03)
        self.assertEqual(CartesianTabletopConfig(scene_profile='realman_nominal', object_radius=.05).object_radius, .05)

    def test_cross_scene_evaluation_checks_the_requested_sampling_area(self):
        settings = [('realman_transfer', .02, 'legacy', .05),
                    ('legacy', .05, 'realman_transfer', .02)]
        for source, source_radius, target, radius in settings:
            with self.subTest(source=source, target=target):
                config = CartesianTabletopConfig(scene_profile=source, object_radius=source_radius,
                                                 observation_source='rgbd_hsv')
                saved = dict(schema=2, environment=config.to_dict(), contact_measurement=CONTACT_MEASUREMENT,
                             startup_contract=startup_contract(config.observation_pose, source))
                with patch('realman_rl_simtoreal.evaluation.tabletop_cartesian.TabletopEnv',
                           side_effect=RuntimeError('target config captured')) as factory:
                    with self.assertRaisesRegex(RuntimeError, 'target config captured'):
                        evaluate(None, saved, [997100], action_shape=(4,), scene_profile=target, object_radius=radius)
                requested = factory.call_args.args[0]
                self.assertEqual((requested.scene_profile, requested.object_radius), (target, radius))

    def test_legacy_action_scaling_is_unchanged(self):
        action = np.array([.2, -.3, .4, .5], dtype=np.float32)
        self.assertEqual(JAW_WIDTH_SCALE_M, MAX_WIDTH)
        for mode in ('absolute', 'delta'):
            config = CartesianTabletopConfig(jaw_control=mode, axis_scale=(1, .5, .25))
            result = decode_cartesian_action(action, config, .06)
            np.testing.assert_array_equal(result.tcp_delta_m, action[:3] * .003 * np.asarray(config.axis_scale))
            width = (action[-1]+1)*MAX_WIDTH/2 if mode == 'absolute' else .06+action[-1]*config.jaw_delta_scale
            self.assertEqual(result.jaw_width_m, float(width))
            self.assertIsNone(result.rotation_delta_rad)

    def test_request_is_bounded_in_base_and_uses_metric_jaw_reference(self):
        config = CartesianTabletopConfig(scene_profile='realman_transfer', jaw_control='delta')
        interface = RealVisualPolicyIO(config)
        action = np.array([1., -1., .8, -.5], dtype=np.float32)
        request = interface.decode_action(action, .06)
        simulated = decode_cartesian_action(action, config, .06)
        np.testing.assert_array_equal(simulated.tcp_delta_m, request.tcp_delta_m)
        self.assertEqual(simulated.jaw_width_m, request.jaw_width_m)
        self.assertAlmostEqual(np.linalg.norm(request.tcp_delta_m), .00025, places=12)
        self.assertEqual(action_contract(config)['translation_frame'], 'Base')
        self.assertAlmostEqual(request.jaw_width_m, .059)
        for missing_or_raw in (None, 1001):
            with self.assertRaisesRegex(ValueError, 'metric reference'):
                interface.decode_action(action, missing_or_raw)
        with self.assertRaises(ValueError):
            interface.decode_action([1, -1, np.nan, 0], .06)


class TestPolicySources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = CartesianTabletopConfig(scene_profile='realman_transfer', cache_startup=True, jaw_control='delta')
        cls.env = make_tabletop_cartesian_env(cls.config)
        cls.addClassCleanup(cls.env.close)

    def setUp(self):
        self.obs, self.info = self.env.reset(seed=997100)
        self.interface = RealVisualPolicyIO(self.config)

    def sample(self):
        env = self.env
        position, rotation = env.camera.base_pose()
        gripper = env.robot.get_gripper_state()
        return VisualPolicyProprioception(env.robot.data.time, rotation.T @ (env.robot.tcp_pose()[0]-position),
            rotation.T @ env.robot.tcp_velocity(), gripper['width_m'], gripper['velocity_m_s'], env._jaw_reference,
            rotation.T @ (env.controller.reference-position), env.controller.last_fraction)

    def test_tool_geometry_and_camera_keep_the_measured_frames(self):
        robot = self.env.robot
        calibration = load_calibration()
        base, _ = self.env.camera.base_pose()
        flange = robot.data.site('ee_site')
        tcp, rotation = robot.tcp_pose()
        T_base_flange = np.eye(4)
        T_base_flange[:3, :3] = flange.xmat.reshape(3, 3)
        T_base_flange[:3, 3] = flange.xpos-base
        np.testing.assert_allclose(tcp-base, (T_base_flange @ calibration.T_flange_tcp)[:3, 3], atol=1e-9)
        np.testing.assert_allclose(rotation, T_base_flange[:3, :3], atol=1e-9)
        left, right = robot.gripper.pad_positions()
        np.testing.assert_allclose(rotation.T @ ((right-left)/np.linalg.norm(right-left)), [0, 1, 0], atol=1e-7)
        self.assertAlmostEqual(robot.table_height-base[2], -.040)
        self.assertAlmostEqual(self.info['object_position_m'][2]-base[2], -.030, places=4)
        frame = self.env.camera.frame
        np.testing.assert_allclose(frame.camera_to_base, T_base_flange @ calibration.camera_to_flange(), atol=1e-8)
        np.testing.assert_allclose(frame.intrinsics, REALMAN_TABLETOP.camera_intrinsics, atol=.001, rtol=0)
        self.assertTrue(self.info['vision']['detected'])
        depth_ticks = frame.depth_m[200:400, 250:450] / REALMAN_TABLETOP.depth_unit_m
        np.testing.assert_allclose(depth_ticks, np.rint(depth_ticks), atol=.001, rtol=0)

    def test_simulation_and_replay_encode_the_same_inputs_bit_for_bit(self):
        self.interface.ingest_frame(self.env.camera.frame)
        np.testing.assert_array_equal(self.interface.observation(self.sample()), self.obs)
        instantaneous_frames = 0
        previous_stamp = self.env.camera.frame.timestamp
        for _ in range(20):
            action = np.array([.7, -.2, -.5, .05], dtype=np.float32)
            request = self.interface.decode_action(action, self.env._jaw_reference)
            reference = self.env.controller.reference.copy()
            obs, _, terminated, truncated, info = self.env.step(action)
            self.assertFalse(terminated or truncated, msg=str(info['detail']))
            np.testing.assert_allclose(np.array(info['requested_tcp_m'])-reference, request.tcp_delta_m, atol=1e-12)
            self.assertEqual(self.env._jaw_reference, request.jaw_width_m)
            if self.env.camera.frame.timestamp > previous_stamp:
                self.interface.ingest_frame(self.env.camera.frame)
                instantaneous_frames += 1
                previous_stamp = self.env.camera.frame.timestamp
            np.testing.assert_array_equal(self.interface.observation(self.sample()), obs)
        self.assertEqual(instantaneous_frames, 15)

    def test_real_observation_refuses_missing_telemetry_and_incorrect_times(self):
        sample = self.sample()
        frame = self.env.camera.frame
        self.interface.ingest_frame(frame)
        with self.assertRaisesRegex(ValueError, 'increase'):
            self.interface.ingest_frame(frame)
        with self.assertRaisesRegex(ValueError, 'Complete SI'):
            self.interface.observation({'opening_raw': 1001})
        with self.assertRaisesRegex(ValueError, 'gap range'):
            self.interface.observation(replace(sample, gripper_width_m=1001))
        with self.assertRaisesRegex(ValueError, 'ik_fraction'):
            replace(sample, ik_fraction=None)
        with self.assertRaisesRegex(ValueError, 'same clock'):
            self.interface.observation(replace(sample, timestamp_s=sample.timestamp_s-.01))
        with self.assertRaisesRegex(ValueError, 'expired'):
            self.interface.observation(replace(sample, timestamp_s=sample.timestamp_s+self.config.vision_timeout+.01))


if __name__ == '__main__':
    unittest.main()
