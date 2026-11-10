"""Seeded local goals, reward shortcut prevention and untrained policy audits."""

from contextlib import ExitStack
from importlib import util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.simulation.control import validate_target
from realman_rl_simtoreal.simulation.ee_control import check_motion
from realman_rl_simtoreal.simulation.kinematics import IKResult, rotation_vector

_spec = util.spec_from_file_location('_test_reach_baselines', PROJECT_ROOT/'scripts/evaluate/evaluate_reach_baselines.py')
baselines = util.module_from_spec(_spec)
_spec.loader.exec_module(baselines)


class ReachTaskTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = ReachEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def assert_observation(self, env, obs):
        self.assertTrue(env.observation_space.contains(obs))
        for value in obs.values():
            self.assertTrue(np.isfinite(value).all())
        np.testing.assert_array_equal(obs['observation'][12:15], obs['achieved_goal'])
        np.testing.assert_array_equal(obs['observation'][15:18], obs['desired_goal']-obs['achieved_goal'])

    def test_fixed_mode_preserves_original_transitions_and_rewards(self):
        default, explicit = self.make_env(), self.make_env(target_mode='fixed')
        for env in (default, explicit):
            env.reset(seed=7)
        rewards = []
        for index in range(6):
            action = [1, 0, 0] if index < 5 else [0, 0, 0]
            a, b = default.step(action), explicit.step(action)
            for key in a[0]:
                np.testing.assert_array_equal(a[0][key], b[0][key])
            self.assertEqual(a[1:4], b[1:4])
            self.assertEqual(a[1], -a[4]['distance_m'])
            rewards.append(a[1])
        self.assertTrue(a[2] and not a[3])
        self.assertAlmostEqual(sum(rewards), -.01853473770549964, places=14)
        self.assertEqual(default.reward_mode, 'distance')

    def test_random_goals_pass_original_pose_limits_and_collision_checks(self):
        env = self.make_env(target_mode='random')
        for seed in range(30):
            obs, info = env.reset(seed=seed)
            self.assert_observation(env, obs)
            distance = np.linalg.norm(obs['desired_goal']-obs['achieved_goal'])
            self.assertGreater(distance, env.success_threshold)
            self.assertGreaterEqual(distance, .003-1e-14)
            self.assertLessEqual(distance, .005+1e-14)
            result = env.kinematics.solve_ik(obs['desired_goal'], env.kinematics.home_q)
            self.assertTrue(result.success, result.reason)
            validate_target(env.model, env.joint_ids, env.kinematics.home_q, result.q)
            p, r, _ = check_motion(env.model, env.data, env.joint_ids, env.actuator_ids,
                                    env.kinematics, result, obs['desired_goal'])
            self.assertLessEqual(np.linalg.norm(p-obs['desired_goal']), 1e-5)
            self.assertLessEqual(np.linalg.norm(rotation_vector(env.kinematics.home_rotation, r)), 1e-4)
            np.testing.assert_array_equal(env.data.qpos, env.kinematics.home_q)
            np.testing.assert_array_equal(env.data.ctrl, env.kinematics.home_q)
            self.assertEqual(env.data.time, 0)
            self.assertFalse(info['is_success'])

    def test_same_seed_reproduces_a_sequence_not_just_one_goal(self):
        env = self.make_env(target_mode='random')
        sequences = []
        for _ in range(2):
            sequences.append([env.reset(seed=42 if index == 0 else None)[0]['desired_goal'] for index in range(6)])
        np.testing.assert_array_equal(*sequences)
        self.assertEqual(len(np.unique(sequences[0], axis=0)), 6)

    def test_different_seeds_produce_three_dimensional_diversity(self):
        env = self.make_env(target_mode='random')
        goals = np.array([env.reset(seed=i)[0]['desired_goal'] for i in range(20)])
        self.assertEqual(len(np.unique(goals, axis=0)), 20)
        self.assertTrue(np.all(np.std(goals, axis=0) > .001))

    def test_sampling_exhaustion_is_bounded_and_cannot_step(self):
        env = self.make_env(target_mode='random', max_target_attempts=3)
        failed = IKResult(False, None, 150, .01, .1, 'Maximum iterations reached')
        with patch.object(env.kinematics, 'solve_ik', return_value=failed) as solver:
            with self.assertRaisesRegex(RuntimeError, 'after 3 attempts'):
                env.reset(seed=11)
        self.assertEqual(solver.call_count, 3)
        self.assertEqual(env.data.time, 0)
        np.testing.assert_array_equal(env.data.qpos, env.kinematics.home_q)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step([0, 0, 0])
        env.reset(seed=11)  # The failure does not permanently poison a new reset.

    def test_rejected_collision_candidates_are_retried_and_recorded(self):
        env = self.make_env(target_mode='random', max_target_attempts=20)
        calls = 0
        def screen(*args):
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise ValueError('Joint interpolation has an unexpected collision')
            return check_motion(*args)
        with patch('realman_rl_simtoreal.envs.reach_env.check_motion', side_effect=screen):
            _, info = env.reset(seed=10)
        self.assertGreaterEqual(info['goal_sampling']['attempts'], 3)
        self.assertEqual(info['goal_sampling']['rejections']['collision'], 2)
        self.assertEqual(env.data.time, 0)

    def test_invalid_task_configuration_is_rejected(self):
        for kwargs in (dict(target_mode='unknown'), dict(reward_mode='unknown'), dict(target_radius=.02),
                       dict(target_radius=.001), dict(target_radius=np.inf), dict(max_target_attempts=0),
                       dict(max_target_attempts=1001), dict(max_target_attempts=True)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReachEnv(**kwargs)

    def test_training_reward_is_default_for_random_and_opt_in_for_fixed(self):
        self.assertEqual(self.make_env(target_mode='random').reward_mode, 'safe_dense')
        self.assertEqual(self.make_env().reward_mode, 'distance')
        env = self.make_env(target_mode='fixed', reward_mode='safe_dense')
        obs, _ = env.reset()
        self.assertAlmostEqual(env.compute_reward(obs['achieved_goal'], obs['desired_goal'],
                                                 {'is_safety_failure': False}), -.25)

    def test_normalized_distance_reward_and_failure_return_bound(self):
        env = self.make_env(target_mode='random', max_episode_steps=20)
        normal = {'is_safety_failure': False}
        failed = {'is_safety_failure': True}
        rewards = [env.compute_reward([d, 0, 0], [0, 0, 0], normal) for d in [0, .001, .005, .02, .1]]
        np.testing.assert_array_equal(rewards, [0, -.05, -.25, -1, -1])
        self.assertGreater(rewards[1], rewards[2])
        # Any <=20-step safe sum is >=-20; even a zero-distance failure is -21.
        self.assertLess(env.compute_reward([0, 0, 0], [0, 0, 0], failed), -20)

    def test_real_shortcut_counterexample_is_reversed_by_safe_reward(self):
        rows = baselines.reward_comparison()
        returns = {(r['reward_mode'], r['case']): r['return_'] for r in rows}
        self.assertGreater(returns['distance', 'deliberate_boundary_failure'], returns['distance', 'reach'])
        self.assertLess(returns['safe_dense', 'deliberate_boundary_failure'], returns['safe_dense', 'reach'])

    def test_goal_relabeling_uses_new_distance_and_goal_independent_safety_only(self):
        env = self.make_env(target_mode='random')
        old_info = {'is_safety_failure': False, 'is_success': True, 'termination_reason': 'success'}
        self.assertEqual(env.compute_reward([0, 0, 0], [.01, 0, 0], old_info), -.5)
        self.assertEqual(env.compute_reward([0, 0, 0], [0, 0, 0], old_info), 0)
        old_info['is_safety_failure'] = True
        self.assertEqual(env.compute_reward([0, 0, 0], [0, 0, 0], old_info), -201)

    def test_batched_rewards_accept_info_arrays_and_reject_missing_flags(self):
        env = self.make_env(target_mode='random')
        achieved = np.array([[.005, 0, 0], [0, 0, 0]])
        records = [{'is_safety_failure': False}, {'is_safety_failure': True}]
        a = env.compute_reward(achieved, [0, 0, 0], records)
        b = env.compute_reward(achieved, [0, 0, 0], {'is_safety_failure': np.array([False, True])})
        np.testing.assert_array_equal(a, [-.25, -201])
        np.testing.assert_array_equal(a, b)
        for info in (None, {}, [{'is_success': False}], {'is_safety_failure': 1}):
            with self.subTest(info=info), self.assertRaises(ValueError):
                env.compute_reward([0, 0, 0], [0, 0, 0], info)

    def test_step_rewards_and_goal_fields_consistent_through_success(self):
        env = self.make_env(target_mode='random')
        obs, _ = env.reset(seed=1)
        for _ in range(80):
            action = baselines.rule_action(obs, env.action_scale)
            self.assertTrue(env.action_space.contains(action))
            obs, reward, t, tr, info = env.step(action)
            self.assert_observation(env, obs)
            self.assertEqual(reward, env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info))
            if t or tr:
                break
        self.assertTrue(t and not tr)
        self.assertTrue(info['is_success'])
        self.assertFalse(info['is_safety_failure'])

    def test_safety_reasons_stay_terminal_and_receive_the_same_penalty_rule(self):
        for case in ('workspace_limit', 'ik_failure', 'joint_limit', 'collision', 'numerical_failure', 'drive_failure'):
            with self.subTest(case=case), ExitStack() as stack:
                env = self.make_env(target_mode='random', action_scale=.01)
                env.reset(seed=2)
                action = [0, 0, 0]
                if case == 'workspace_limit':
                    action = [1, 1, 1]
                elif case in ('ik_failure', 'joint_limit'):
                    result = (IKResult(False, None, 150, None, None, 'Maximum iterations reached')
                              if case == 'ik_failure' else IKResult(True, np.full(6, 100.), 1, 0, 0))
                    stack.enter_context(patch.object(env.kinematics, 'solve_ik', return_value=result))
                elif case == 'collision':
                    stack.enter_context(patch('realman_rl_simtoreal.envs.reach_env.check_motion',
                                              side_effect=ValueError('Unexpected collision')))
                elif case == 'numerical_failure':
                    env.data.qvel[0] = np.nan  # terminal fault fixture
                else:
                    env.data.qfrc_applied[0] = 1
                obs, reward, t, tr, info = env.step(action)
                self.assert_observation(env, obs)
                self.assertTrue(t and not tr)
                self.assertTrue(info['is_safety_failure'])
                self.assertEqual(info['termination_reason'], case)
                self.assertLessEqual(reward, -201)
                self.assertEqual(reward, env.compute_reward(obs['achieved_goal'], obs['desired_goal'], info))

    def test_timeout_is_truncation_without_safety_penalty(self):
        env = self.make_env(target_mode='random', max_episode_steps=1)
        env.reset(seed=1)
        obs, reward, t, tr, info = env.step([0, 0, 0])
        self.assert_observation(env, obs)
        self.assertFalse(t)
        self.assertTrue(tr)
        self.assertEqual(info['termination_reason'], 'time_limit')
        self.assertFalse(info['is_safety_failure'])
        self.assertGreaterEqual(reward, -1)

    def test_rule_responds_to_goal_direction_and_pending_command(self):
        outputs = []
        for goal in ([.004, 0, 0], [0, -.004, 0], [0, 0, .004]):
            obs = dict(achieved_goal=np.zeros(3), desired_goal=np.array(goal), observation=np.zeros(27))
            action = baselines.rule_action(obs, .001)
            self.assertEqual(action.dtype, np.float32)
            self.assertLessEqual(np.linalg.norm(action), .25+1e-8)
            outputs.append(action)
            obs['observation'][18:21] = goal  # command arrived, actual still lags
            np.testing.assert_allclose(baselines.rule_action(obs, .001), 0, atol=1e-14)
        self.assertGreater(outputs[0][0], 0)
        self.assertLess(outputs[1][1], 0)
        self.assertGreater(outputs[2][2], 0)

    def test_rule_uses_only_standard_steps_on_diverse_random_goals(self):
        env = self.make_env(target_mode='random')
        for seed in range(10):
            obs, _ = env.reset(seed=seed)
            with patch.object(env, 'step', wraps=env.step) as step:
                for _ in range(80):
                    obs, _, t, tr, info = step(baselines.rule_action(obs, env.action_scale))
                    if t or tr:
                        break
            self.assertGreater(step.call_count, 0)
            self.assertTrue(info['is_success'], (seed, info['termination_reason']))

    def test_reward_mode_does_not_change_goals_or_dynamics(self):
        a = self.make_env(target_mode='random', reward_mode='distance')
        b = self.make_env(target_mode='random', reward_mode='safe_dense')
        obs, _ = a.reset(seed=7)
        b.reset(seed=7)
        for _ in range(80):
            action = baselines.rule_action(obs, a.action_scale)
            ra, rb = a.step(action), b.step(action)
            for key in ra[0]:
                np.testing.assert_array_equal(ra[0][key], rb[0][key])
            self.assertEqual(ra[2:4], rb[2:4])
            obs = ra[0]
            if ra[2] or ra[3]:
                break

    def test_paired_evaluation_reports_success_separately_from_audit(self):
        a = baselines.evaluate_policy('random', [0, 1], max_episode_steps=20)
        b = baselines.evaluate_policy('rule', [0, 1], max_episode_steps=20)
        self.assertEqual([e['target_position_m'] for e in a['episodes']], [e['target_position_m'] for e in b['episodes']])
        for result in (a, b):
            successful = sum(e['success'] for e in result['episodes'])
            self.assertEqual(result['summary']['success_rate'], successful/2)
            self.assertGreater(result['timing']['step']['p95_s'], 0)
            self.assertGreater(result['timing']['interaction_steps_per_s'], 0)
            self.assertEqual(sum(result['summary']['termination_counts'].values()), 2)

    def test_official_checker_with_seeded_random_goals(self):
        env = self.make_env(target_mode='random', render_mode='ansi')
        with warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter('always')
            check_env(env)
        for note in notes:
            self.assertTrue('infinity' in str(note.message) or 'not having a spec' in str(note.message), note.message)

    def test_random_mode_without_sdk_in_an_isolated_process(self):
        code = '''
import importlib.abc,sys
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='Robotic_Arm' or fullname.startswith('Robotic_Arm.'):
            raise ModuleNotFoundError('SDK intentionally unavailable')
sys.meta_path.insert(0,BlockSDK())
from realman_rl_simtoreal.envs.reach_env import ReachEnv
with ReachEnv(target_mode='random') as env:
    obs,info=env.reset(seed=10)
    obs,r,t,tr,info=env.step([0,0,0])
    assert env.observation_space.contains(obs) and info['physics_steps']==25
assert not any(n.startswith('realman_rl_simtoreal.vendor') for n in sys.modules)
'''
        result = subprocess.run([sys.executable, '-I', '-c', code], cwd='/tmp', capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_evaluation_cli_from_other_directory_and_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            result = subprocess.run([sys.executable, '-I', str(PROJECT_ROOT/'scripts/evaluate/evaluate_reach_baselines.py'),
                                     '--episodes', '2', '--profile', '--json', str(output)], cwd=directory,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report['paired_goals_identical'])
            self.assertEqual(report['results']['rule']['summary']['episodes'], 2)
            self.assertGreater(report['profile']['stages']['native_mj_step']['calls'], 0)
            self.assertTrue(output.with_suffix('.profile.txt').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
