"""Gymnasium API, physical actuation and failure regression for the local reach task."""

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
import mujoco
import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.simulation import control
from realman_rl_simtoreal.simulation.kinematics import IKResult


class ReachEnvTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = ReachEnv(**kwargs)
        self.addCleanup(env.close)
        env.reset(seed=4)
        return env

    def assert_obs(self, env, obs):
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(set(obs), {"observation", "achieved_goal", "desired_goal"})
        for key, value in obs.items():
            self.assertEqual(value.dtype, np.float64)
            self.assertEqual(value.shape, (27,) if key == "observation" else (3,))
            self.assertTrue(np.isfinite(value).all())

    def reach(self, env):
        for index in range(40):
            result = env.step(np.array([1, 0, 0] if index < 5 else [0, 0, 0], dtype=np.float32))
            if result[2] or result[3]:
                return index+1, result
        self.fail("Fixed reachable target did not finish within 40 cycles")

    def test_reset_spaces_and_fixed_goal(self):
        env = self.make_env()
        obs, info = env.reset(seed=11)
        self.assert_obs(env, obs)
        self.assertEqual(env.action_space.shape, (3,))
        self.assertEqual(env.action_space.dtype, np.float32)
        np.testing.assert_array_equal(env.action_space.low, [-1]*3)
        np.testing.assert_array_equal(env.action_space.high, [1]*3)
        np.testing.assert_array_equal(env.data.qpos, env.kinematics.home_q)
        np.testing.assert_array_equal(env.data.ctrl, env.kinematics.home_q)
        np.testing.assert_allclose(obs['desired_goal']-obs['achieved_goal'], [.005, 0, 0], atol=1e-15)
        self.assertAlmostEqual(info['distance_m'], .005)
        self.assertEqual(env.data.time, 0)
        self.assertFalse(info['is_success'])

    def test_step_advances_exact_physics_count_and_reward(self):
        env = self.make_env()
        obs, reward, terminated, truncated, info = env.step([1, 0, 0])
        self.assert_obs(env, obs)
        self.assertIsInstance(reward, float)
        self.assertIs(type(terminated), bool)
        self.assertIs(type(truncated), bool)
        self.assertEqual(info['physics_steps'], 25)
        self.assertAlmostEqual(env.data.time, .05, places=12)
        self.assertEqual(info['control']['forces']['steps'], 25)
        self.assertAlmostEqual(reward, -info['distance_m'])
        self.assertFalse(terminated or truncated)

    def test_seeded_transition_is_reproducible(self):
        env = self.make_env()
        transitions = []
        for _ in range(2):
            env.reset(seed=123)
            run = []
            for action in ([.4, 0, 0], [.4, 0, 0], [0, 0, 0]):
                obs, r, t, tr, info = env.step(action)
                run.append((obs, r, t, tr, info['control']))
            transitions.append(run)
        for first, second in zip(*transitions):
            for key in first[0]:
                np.testing.assert_array_equal(first[0][key], second[0][key])
            self.assertEqual(first[1:], second[1:])

    def test_reset_clears_physics_references_and_returns_copies(self):
        env = self.make_env()
        obs, info = env.reset()
        obs['desired_goal'][:] = 10
        info['command_position_m'][:] = 20
        env.step([1, 0, 0])
        obs, info = env.reset()
        self.assertEqual(env.data.time, 0)
        self.assertEqual(info['elapsed_steps'], 0)
        np.testing.assert_array_equal(obs['achieved_goal'], env.kinematics.home_position)
        np.testing.assert_array_equal(info['command_position_m'], env.kinematics.home_position)
        self.assertAlmostEqual(info['distance_m'], .005)

    def test_movement_is_only_integrated_by_mj_step(self):
        env = self.make_env()
        initial = env.data.qpos.copy()
        last = initial.copy()
        original = mujoco.mj_step
        count = 0
        def observed_step(model, data):
            nonlocal last, count
            self.assertIs(data, env.data)
            np.testing.assert_array_equal(data.qpos, last)
            original(model, data)
            last = data.qpos.copy()
            count += 1
        with patch.object(control.mujoco, 'mj_step', side_effect=observed_step):
            steps, result = self.reach(env)
        np.testing.assert_array_equal(env.data.qpos, last)
        self.assertEqual(count, steps*25)
        self.assertGreater(np.linalg.norm(last-initial), 1e-3)
        self.assertGreater(result[0]['achieved_goal'][0]-env.kinematics.home_position[0], .004)
        self.assertTrue(result[4]['is_success'])

    def test_cartesian_reference_does_not_accumulate_tracking_error(self):
        env = self.make_env()
        first = env.step([1, 0, 0])
        measured = first[0]['achieved_goal'].copy()
        second = env.step([1, 0, 0])
        reference = second[4]['command_position_m']
        np.testing.assert_allclose(reference, env.kinematics.home_position+[.002, 0, 0], atol=1e-15)
        self.assertGreater(np.linalg.norm(reference-(measured+[.001, 0, 0])), .0005)

    def test_ik_seed_is_recent_legal_joint_command(self):
        env = self.make_env()
        env.step([1, 0, 0])
        previous = env.data.ctrl.copy()
        with patch.object(env.kinematics, 'solve_ik', wraps=env.kinematics.solve_ik) as solve:
            env.step([1, 0, 0])
        np.testing.assert_array_equal(solve.call_args.args[1], previous)
        np.testing.assert_array_equal(solve.call_args.args[2], env.kinematics.home_rotation)

    def test_ik_failure_stops_without_motion_or_reference_commit(self):
        env = self.make_env()
        q, ctrl = env.data.qpos.copy(), env.data.ctrl.copy()
        result = IKResult(False, None, 150, .01, .02, 'Maximum iterations reached')
        with (patch.object(env.kinematics, 'solve_ik', return_value=result),
              patch.object(control.mujoco, 'mj_step', side_effect=AssertionError('Must not step'))):
            obs, _, terminated, truncated, info = env.step([1, 0, 0])
        self.assert_obs(env, obs)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'], 'ik_failure')
        self.assertEqual(info['physics_steps'], 0)
        self.assertEqual(env.data.time, 0)
        np.testing.assert_array_equal(env.data.qpos, q)
        np.testing.assert_array_equal(env.data.ctrl, ctrl)
        np.testing.assert_array_equal(info['command_position_m'], env.kinematics.home_position)
        with self.assertRaises(gym.error.ResetNeeded):
            env.step([0, 0, 0])

    def test_illegal_successful_ik_solutions_are_rejected_without_clipping(self):
        for invalid in (np.full(6, np.nan), np.full(6, np.inf), np.full(6, 100.), np.zeros(5)):
            with self.subTest(q=invalid):
                env = self.make_env()
                result = IKResult(True, invalid, 1, 0, 0)
                with patch.object(env.kinematics, 'solve_ik', return_value=result):
                    _, _, t, tr, info = env.step([1, 0, 0])
                self.assertTrue(t and not tr)
                self.assertEqual(info['termination_reason'], 'joint_limit')
                self.assertEqual(env.data.time, 0)
                np.testing.assert_array_equal(env.data.ctrl, env.kinematics.home_q)

    def test_workspace_guard_is_not_reported_as_global_ik_failure(self):
        env = self.make_env(action_scale=.01)
        with patch.object(env.kinematics, 'solve_ik', side_effect=AssertionError('Must reject before IK')):
            _, _, t, tr, info = env.step([1, 1, 1])
        self.assertTrue(t and not tr)
        self.assertEqual(info['termination_reason'], 'workspace_limit')
        self.assertEqual(info['physics_steps'], 0)

    def test_proven_geometric_unreachability_has_distinct_reason(self):
        env = self.make_env()
        result = IKResult(False, None, 0, None, None, 'Target exceeds conservative reach bound')
        with patch.object(env.kinematics, 'solve_ik', return_value=result):
            _, _, t, tr, info = env.step([1, 0, 0])
        self.assertTrue(t and not tr)
        self.assertEqual(info['termination_reason'], 'unreachable_target')
        self.assertEqual(env.data.time, 0)

    def test_goal_success_uses_actual_position(self):
        env = self.make_env()
        # Five commands reach the theoretical goal; real motion still lags.
        for _ in range(5):
            obs, _, t, tr, info = env.step([1, 0, 0])
        np.testing.assert_allclose(info['command_position_m'], obs['desired_goal'], atol=1e-15)
        self.assertFalse(t or tr)
        self.assertGreater(info['distance_m'], env.success_threshold)
        obs, _, t, tr, info = env.step([0, 0, 0])
        self.assertTrue(t and not tr)
        self.assertTrue(info['is_success'])
        self.assertEqual(info['termination_reason'], 'success')
        self.assertLess(info['orientation_error_rad'], np.deg2rad(1))
        self.assertLessEqual(np.linalg.norm(obs['achieved_goal']-obs['desired_goal']), env.success_threshold)

    def test_timeout_truncates_without_task_termination(self):
        env = self.make_env(max_episode_steps=1)
        _, _, t, tr, info = env.step([0, 0, 0])
        self.assertFalse(t)
        self.assertTrue(tr)
        self.assertFalse(info['is_success'])
        self.assertEqual(info['termination_reason'], 'time_limit')
        self.assertEqual(info['physics_steps'], 25)

    def test_success_on_last_allowed_step_takes_priority(self):
        env = self.make_env(max_episode_steps=6)
        steps, result = self.reach(env)
        self.assertEqual(steps, 6)
        self.assertTrue(result[2])
        self.assertFalse(result[3])

    def test_invalid_actions_are_rejected_before_state_change(self):
        env = self.make_env()
        for action in ([1, 0], [2, 0, 0], [np.nan, 0, 0], [0, np.inf, 0]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env.data.time, 0)
        self.assertEqual(env._elapsed_steps, 0)

    def test_actual_collision_stops_a_partial_cycle(self):
        env = self.make_env()
        original = mujoco.mj_step
        count = 0
        def inject_obstacle(model, data):
            nonlocal count
            original(model, data)
            count += 1
            if count == 5:
                model.geom('floor').pos[2] = .5  # test fixture only; no asset mutation
                # Static-world transforms are cached in MjModel. Recompute in
                # scratch data, so this fixture does not reset live qpos/time.
                mujoco.mj_setConst(model, mujoco.MjData(model))
        with patch.object(control.mujoco, 'mj_step', side_effect=inject_obstacle):
            obs, _, t, tr, info = env.step([1, 0, 0])
        self.assert_obs(env, obs)
        self.assertTrue(t and not tr)
        self.assertEqual(info['termination_reason'], 'collision')
        self.assertEqual(info['physics_steps'], 5)
        self.assertAlmostEqual(env.data.time, .01)

    def test_path_collision_is_rejected_before_physics(self):
        env = self.make_env()
        env.model.geom('floor').pos[2] = .5
        mujoco.mj_setConst(env.model, mujoco.MjData(env.model))
        mujoco.mj_forward(env.model, env.data)
        _, _, t, tr, info = env.step([1, 0, 0])
        self.assertTrue(t and not tr)
        self.assertEqual(info['termination_reason'], 'collision')
        self.assertEqual(info['physics_steps'], 0)

    def test_nonfinite_state_returns_explicit_stale_terminal_snapshot(self):
        env = self.make_env()
        original = mujoco.mj_step
        def corrupt_velocity(model, data):
            original(model, data)
            data.qvel[0] = np.nan  # deliberate numerical failure fixture
        with patch.object(control.mujoco, 'mj_step', side_effect=corrupt_velocity):
            obs, reward, t, tr, info = env.step([1, 0, 0])
        self.assert_obs(env, obs)
        self.assertTrue(np.isfinite(reward))
        self.assertTrue(t and not tr)
        self.assertFalse(info['observation_valid'])
        self.assertEqual(info['termination_reason'], 'numerical_failure')
        self.assertEqual(info['physics_steps'], 1)
        self.assertTrue(np.isnan(env.data.qvel[0]))  # no repair/teleport in step

    def test_external_force_bypass_is_rejected(self):
        env = self.make_env()
        env.data.qfrc_applied[0] = 1
        _, _, t, tr, info = env.step([0, 0, 0])
        self.assertTrue(t and not tr)
        self.assertEqual(info['termination_reason'], 'drive_failure')
        self.assertEqual(info['physics_steps'], 0)

    def test_actual_joint_limit_failure_is_distinct_from_ik_failure(self):
        env = self.make_env()
        env.data.qpos[0] = env.kinematics.limits[0, 1]+.001  # fault fixture only
        with patch.object(env.kinematics, 'solve_ik', side_effect=AssertionError('Must not solve')):
            obs, _, t, tr, info = env.step([0, 0, 0])
        self.assert_obs(env, obs)
        self.assertTrue(t and not tr)
        self.assertEqual(info['termination_reason'], 'joint_limit')
        self.assertEqual(info['physics_steps'], 0)

    def test_fixed_cycle_rate_limit_smoothing_and_force_audit_order(self):
        env = self.make_env()
        start = env.data.ctrl.copy()
        target = start+np.deg2rad([3, 0, 0, 0, 0, 0])
        events, commands = [], []
        step, forward, observe = mujoco.mj_step, mujoco.mj_forward, control.ForceMonitor.observe
        def track_step(model, data):
            events.append('step'); commands.append(data.ctrl.copy()); step(model, data)
        def track_forward(model, data):
            events.append('forward'); forward(model, data)
        def track_observe(monitor, data):
            events.append('observe'); observe(monitor, data)
        with (patch.object(control.mujoco, 'mj_step', side_effect=track_step),
              patch.object(control.mujoco, 'mj_forward', side_effect=track_forward),
              patch.object(control.ForceMonitor, 'observe', new=track_observe)):
            result = control.advance_control_cycle(env.model, env.data, env.joint_ids, env.actuator_ids, target)
        self.assertEqual(events, ['step', 'observe', 'forward']*25)
        self.assertTrue(result['command_rate_limited'])
        for index, command in enumerate(commands):
            a = (index+1)/25
            expected = start+np.deg2rad([.5, 0, 0, 0, 0, 0])*a*a*(3-2*a)
            np.testing.assert_allclose(command, expected, atol=1e-15, rtol=0)
        peak = np.array(result['forces']['peak_total_drive_nm'])
        self.assertTrue(np.all(peak <= [60, 60, 30, 30, 10, 10]))
        self.assertEqual(result['forces']['steps'], 25)

    def test_control_cycle_rejects_illegal_target_and_speed(self):
        env = self.make_env()
        for target in ([100]*6, [0]*5, [np.nan]*6):
            with self.subTest(target=target), self.assertRaises(control.ControlCycleError) as error:
                control.advance_control_cycle(env.model, env.data, env.joint_ids, env.actuator_ids, target)
            self.assertEqual(error.exception.reason, 'joint_limit')
        self.assertEqual(env.data.time, 0)
        env.data.qvel[0] = 1  # rad/s > experimental 30deg/s
        with self.assertRaises(control.ControlCycleError) as error:
            control.advance_control_cycle(env.model, env.data, env.joint_ids, env.actuator_ids, env.data.ctrl)
        self.assertEqual(error.exception.reason, 'velocity_limit')
        self.assertEqual(env.data.time, 0)

    def test_control_period_configuration_and_invalid_parameters(self):
        env = self.make_env(control_period=.02)
        env.step([0, 0, 0])
        self.assertEqual(env.physics_steps, 10)
        self.assertAlmostEqual(env.data.time, .02)
        for kwargs in (dict(control_period=.051), dict(control_period=0), dict(action_scale=np.nan),
                       dict(action_scale=0), dict(max_episode_steps=0), dict(success_threshold=.005),
                       dict(render_mode='human')):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReachEnv(**kwargs)

    def test_official_checker_render_and_close(self):
        env = self.make_env(render_mode='ansi')
        with warnings.catch_warnings(record=True) as notes:
            warnings.simplefilter('always')
            check_env(env)  # all default API checks; no checker options disabled
        for note in notes:
            text = str(note.message)
            self.assertTrue('infinity' in text or 'not having a spec' in text, text)
        env.reset()
        self.assertIn('ee_world_m', env.render())
        env.close(); env.close()
        with self.assertRaises(RuntimeError):
            env.reset()
        with self.assertRaises(RuntimeError):
            env.step([0, 0, 0])

    def test_reward_batches(self):
        env = self.make_env()
        rewards = env.compute_reward([[0, 0, 0], [.003, .004, 0]], [0, 0, 0])
        np.testing.assert_array_equal(rewards, [0, -.005])

    def test_without_sdk_or_display_from_other_directory(self):
        code = '''
import importlib.abc,sys
class BlockSDK(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='Robotic_Arm' or fullname.startswith('Robotic_Arm.'):
            raise ModuleNotFoundError('SDK intentionally unavailable')
sys.meta_path.insert(0,BlockSDK())
from realman_rl_simtoreal.envs.reach_env import ReachEnv
with ReachEnv() as env:
    env.reset(seed=0)
    obs,r,t,tr,info=env.step([1,0,0])
    assert info['physics_steps']==25 and not t and not tr
    assert env.render() is None
assert 'Robotic_Arm' not in sys.modules and 'mujoco.viewer' not in sys.modules
assert not any(n.startswith('realman_rl_simtoreal.vendor') for n in sys.modules)
'''
        import os
        process_env = {k: v for k, v in os.environ.items() if k not in ('DISPLAY', 'WAYLAND_DISPLAY')}
        result = subprocess.run([sys.executable, '-I', '-c', code], cwd='/tmp', env=process_env,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

    def test_headless_smoke_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            result = subprocess.run([sys.executable, '-I', str(PROJECT_ROOT/'scripts/smoke_reach_env.py'),
                                     '--json', str(output)], cwd=directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            rows = json.loads(output.read_text())['steps']
            self.assertEqual(len(rows), 6)
            self.assertTrue(rows[-1]['info']['is_success'])
            self.assertEqual(rows[-1]['info']['physics_steps'], 25)


if __name__ == '__main__':
    unittest.main(verbosity=2)
