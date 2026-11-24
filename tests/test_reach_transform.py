"""Shared Reach environment and fixed observation transform contracts."""
import json
import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from realman_rl_simtoreal.envs.reach.reach_env import ReachEnv
from realman_rl_simtoreal.evaluation.reach import evaluate
from realman_rl_simtoreal.rl.common.observation import ReachTransform, ScaledReachObservation
from realman_rl_simtoreal.tasks.reach.config import ENV_CONFIG


class ReachTransformTests(unittest.TestCase):
    def setUp(self):
        with ReachEnv(**ENV_CONFIG) as raw:
            self.transform = ReachTransform.from_env(raw)

    def test_raw_and_normalized_gym_interfaces_and_six_axis_scene(self):
        with ReachEnv(**ENV_CONFIG) as raw:
            check_env(raw, skip_render_check=True)
            self.assertEqual((raw.model.nq, raw.model.nv, raw.model.nu), (6, 6, 6))
            with ScaledReachObservation(ReachEnv(**ENV_CONFIG)) as scaled:
                check_env(scaled, skip_render_check=True)
                obs, _ = scaled.reset(seed=73)
                self.assertTrue(scaled.observation_space.contains(obs))
                self.assertEqual(obs['observation'].dtype, np.float32)
                self.assertEqual(obs['desired_goal'].dtype, np.float64)
                self.assertEqual(obs['observation'].shape, (27,))

    def test_transform_is_fixed_reproducible_and_does_not_change_physical_units(self):
        with ReachEnv(**ENV_CONFIG) as raw:
            obs, _ = raw.reset(seed=73)
            before = {k: v.copy() for k, v in obs.items()}
            scaled = self.transform.encode(obs)
            repeated = ReachTransform(json.loads(json.dumps(self.transform.config))).encode(obs)
            for key in obs:
                np.testing.assert_array_equal(obs[key], before[key])
                np.testing.assert_array_equal(repeated[key], scaled[key])
            np.testing.assert_allclose(scaled['desired_goal']-scaled['achieved_goal'],
                                       .01*scaled['observation'][15:18], atol=1e-9)
            self.assertGreater(np.linalg.norm(obs['desired_goal']-obs['achieved_goal']), .001)
            self.assertLess(np.linalg.norm(scaled['observation'][15:18]), .501)
            wrapped = ScaledReachObservation(raw)
            _, reward, _, _, info = wrapped.step(np.zeros(3, dtype=np.float32))
            actual = raw.data.site_xpos[raw.kinematics.site_id]
            expected = raw.compute_reward(actual, obs['desired_goal'], info)
            self.assertAlmostEqual(reward, expected, places=14)

    def test_bad_transform_and_missing_preprocessing_fail_clearly(self):
        config = json.loads(json.dumps(self.transform.config))
        config['scale']['desired_goal'][0] = 0
        with self.assertRaises(ValueError):
            ReachTransform(config)
        with self.assertRaisesRegex(ValueError, 'saved observation transform'):
            evaluate('ppo', [1], model=object())
        config = json.loads(json.dumps(self.transform.config))
        config['offset']['desired_goal'][0] = .01
        with self.assertRaisesRegex(ValueError, 'SI coordinates'):
            ReachTransform(config)
