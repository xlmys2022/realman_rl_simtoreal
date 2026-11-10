"""Optional CPU SB3 integration: API, actual updates, persistence and inference.

Tiny test-only network/batch/buffer overrides are not training hyperparameters.
No short-run success rate requirement is imposed.
"""
import gzip
from importlib import util
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

AVAILABLE = util.find_spec('stable_baselines3') is not None and util.find_spec('torch') is not None
if AVAILABLE:
    import mujoco
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_checker import check_env
    from realman_rl_simtoreal.envs.reach_env import ReachEnv
    from realman_rl_simtoreal.tasks.reach import ENV_CONFIG
    from realman_rl_simtoreal.rl.common.observation import ReachTransform, ScaledReachObservation
    from realman_rl_simtoreal.rl.common.env_factory import training_env
    from realman_rl_simtoreal.rl.sac import make_model, save_checkpoint
    from realman_rl_simtoreal.rl.sac_diagnostics import q_diagnostic
    from realman_rl_simtoreal.rl.common.evaluation import evaluate
    from realman_rl_simtoreal.rl.common.artifacts import write_json
    from realman_rl_simtoreal.paths import PROJECT_ROOT


@unittest.skipUnless(AVAILABLE, 'Optional SB3/PyTorch CPU dependencies are not installed')
class SACReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        cls.temporary = tempfile.TemporaryDirectory(prefix='sac-reach-test-')
        cls.folder = Path(cls.temporary.name)
        cls.env, cls.transform = training_env(ENV_CONFIG)
        cls.model = make_model(cls.env, 73, overrides=dict(learning_starts=8, batch_size=16,
                               buffer_size=256, policy_kwargs=dict(net_arch=[32, 32])))
        cls.actor_before = [p.detach().clone() for p in cls.model.actor.parameters()]
        cls.critic_before = [p.detach().clone() for p in cls.model.critic.parameters()]
        cls.model.learn(total_timesteps=64)
        save_checkpoint(cls.model, cls.folder, replay=True)
        write_json(cls.folder/'observation_transform.json', cls.transform.config)
        cls.loaded = SAC.load(cls.folder/'model.zip', device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls.env.close()
        cls.temporary.cleanup()

    def test_raw_and_normalized_gym_interfaces_and_six_axis_scene(self):
        with ReachEnv(**ENV_CONFIG) as raw:
            check_env(raw, warn=True)
            self.assertEqual((raw.model.nq, raw.model.nv, raw.model.nu), (6, 6, 6))
            with ScaledReachObservation(ReachEnv(**ENV_CONFIG)) as scaled:
                check_env(scaled, warn=True)
                obs, _ = scaled.reset(seed=73)
                self.assertTrue(scaled.observation_space.contains(obs))
                self.assertEqual(obs['observation'].dtype, np.float32)
                self.assertEqual(obs['desired_goal'].dtype, np.float64)
                self.assertEqual(obs['observation'].shape, (27,))
                self.assertEqual(self.model.actor.features_extractor.features_dim, 33)

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

    def test_actor_outputs_legal_actions_and_sampling_populates_official_buffer(self):
        with ReachEnv(**ENV_CONFIG) as raw:
            obs, _ = raw.reset(seed=73)
            action, _ = self.model.predict(self.transform.encode(obs), deterministic=True)
            self.assertTrue(raw.action_space.contains(action))
            next_obs, reward, _, _, _ = raw.step(action)
            self.assertTrue(all(np.isfinite(v).all() for v in next_obs.values()))
            self.assertTrue(np.isfinite(reward))
        self.assertEqual(type(self.model.replay_buffer).__module__, 'stable_baselines3.common.buffers')
        self.assertEqual(self.model.replay_buffer.size(), 64)

    def test_official_actor_and_critic_both_really_updated(self):
        self.assertGreater(self.model._n_updates, 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(self.actor_before, self.model.actor.parameters())))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(self.critic_before, self.model.critic.parameters())))
        self.assertEqual(len(self.model.critic.q_networks), 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.policy.parameters()))

    def test_save_reload_inference_and_replay_restore(self):
        with ReachEnv(**ENV_CONFIG) as raw:
            obs, _ = raw.reset(seed=74)
            a, _ = self.model.predict(self.transform.encode(obs), deterministic=True)
            b, _ = self.loaded.predict(self.transform.encode(obs), deterministic=True)
            np.testing.assert_array_equal(a, b)
        with gzip.open(self.folder/'replay_buffer.pkl.gz', 'rb') as file:
            self.loaded.load_replay_buffer(io.BytesIO(file.read()))
        self.assertEqual(self.loaded.replay_buffer.size(), 64)
        self.assertEqual(self.loaded.num_timesteps, 64)
        self.assertEqual(self.loaded._n_updates, self.model._n_updates)

    def test_fixed_seed_evaluation_is_reproducible_and_does_not_train(self):
        config = dict(ENV_CONFIG, max_episode_steps=5)
        before = self.model.num_timesteps, self.model._n_updates, self.model.replay_buffer.size()
        a = evaluate('sac', [100, 101], model=self.model, transform=self.transform, config=config)
        b = evaluate('sac', [100, 101], model=self.model, transform=self.transform, config=config)
        for one, two in zip(a['episodes'], b['episodes']):
            for key in ('target_position_m', 'return_', 'reason', 'length', 'final_position_m', 'actions'):
                self.assertEqual(one[key], two[key])
        self.assertEqual(before, (self.model.num_timesteps, self.model._n_updates, self.model.replay_buffer.size()))

    def test_random_rule_sac_receive_identical_goals_and_keep_reward(self):
        config = dict(ENV_CONFIG, max_episode_steps=4)
        cases = [evaluate(name, [102, 103], model=self.model, transform=self.transform, config=config)
                 for name in ('random', 'rule', 'sac')]
        targets = [[e['target_position_m'] for e in c['episodes']] for c in cases]
        self.assertEqual(targets[0], targets[1])
        self.assertEqual(targets[1], targets[2])
        self.assertTrue(all(c['summary']['episodes'] == 2 for c in cases))

    def test_diagnostics_finite_and_do_not_consume_training_rng(self):
        numpy_state, torch_state = np.random.get_state(), torch.get_rng_state().clone()
        diagnostic = q_diagnostic(self.model)
        after = np.random.get_state()
        self.assertEqual(numpy_state[0], after[0])
        np.testing.assert_array_equal(numpy_state[1], after[1])
        self.assertEqual(numpy_state[2:], after[2:])
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertTrue(diagnostic['parameters_finite'])
        self.assertTrue(np.isfinite([diagnostic['q_min'], diagnostic['q_max']]).all())

    def test_training_sampling_does_not_teleport_actual_qpos(self):
        vector, _ = training_env(ENV_CONFIG)
        raw = vector.venv.envs[0].unwrapped
        normal_step, physics_step = raw.step, mujoco.mj_step
        last, calls = None, 0
        def checked_physics(model, data):
            nonlocal last, calls
            if data is raw.data:
                np.testing.assert_array_equal(data.qpos, last)
                physics_step(model, data)
                last, calls = data.qpos.copy(), calls+1
            else:
                physics_step(model, data)
        def checked_step(action):
            nonlocal last
            last = raw.data.qpos.copy()  # resets between episodes are allowed
            result = normal_step(action)
            np.testing.assert_array_equal(raw.data.qpos, last)
            return result
        try:
            with patch.object(raw, 'step', side_effect=checked_step), patch.object(mujoco, 'mj_step', side_effect=checked_physics):
                model = make_model(vector, 74, overrides=dict(learning_starts=4, batch_size=8, buffer_size=64,
                                                             policy_kwargs=dict(net_arch=[16, 16])))
                model.learn(total_timesteps=12)
            self.assertGreater(calls, 0)
            self.assertGreater(model._n_updates, 0)
        finally:
            vector.close()

    def test_saved_model_independent_process_without_vendor_sdk(self):
        code = f'''
import sys,importlib.abc,json
class BlockVendor(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.startswith(('Robotic_Arm','realman_rl_simtoreal.vendor')):raise ImportError('SDK blocked')
sys.meta_path.insert(0,BlockVendor())
import torch
from stable_baselines3 import SAC
from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.rl.common.observation import ReachTransform
from realman_rl_simtoreal.tasks.reach import ENV_CONFIG
from pathlib import Path
torch.set_num_threads(1)
m=SAC.load({str(self.folder/'model.zip')!r},device='cpu')
t=ReachTransform(json.loads(Path({str(self.folder/'observation_transform.json')!r}).read_text()))
with ReachEnv(**ENV_CONFIG) as e:
 o,_=e.reset(seed=123);a,_=m.predict(t.encode(o),deterministic=True)
 assert e.action_space.contains(a)
 e.step(a)
 assert not any(x.startswith('Robotic_Arm') for x in sys.modules)
print('independent inference OK')
'''
        p = subprocess.run([sys.executable, '-I', '-c', code], cwd='/tmp', capture_output=True, text=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stdout+p.stderr)
        self.assertIn('independent inference OK', p.stdout)

    def test_bad_transform_and_missing_preprocessing_fail_clearly(self):
        config = json.loads(json.dumps(self.transform.config))
        config['scale']['desired_goal'][0] = 0
        with self.assertRaises(ValueError):
            ReachTransform(config)
        with self.assertRaisesRegex(ValueError, 'saved observation transform'):
            evaluate('sac', [1], model=self.model)
        config = json.loads(json.dumps(self.transform.config))
        config['offset']['desired_goal'][0] = .01
        with self.assertRaisesRegex(ValueError, 'SI coordinates'):
            ReachTransform(config)

    def test_cli_training_chunks_preserve_total_steps_buffer_and_updates(self):
        spec = util.spec_from_file_location('sac_training_entry', PROJECT_ROOT/'scripts/train/train_sac_reach.py')
        driver = util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        result = driver.train(self.folder/'driver', timesteps=128, seed=74, eval_every=64,
                              eval_episodes=1, label='unit-test')
        self.assertEqual(result['status'], 'completed', result['error'])
        self.assertEqual(result['total_timesteps'], 128)
        self.assertEqual(result['buffer_size'], 128)
        self.assertEqual(result['n_updates'], 28)
        self.assertEqual([c['step'] for c in result['checkpoints']], [64, 128])
        self.assertTrue(result['actor_updated'] and result['critic_updated'])


if __name__ == '__main__':
    unittest.main()
