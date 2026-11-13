"""Raw physics, stock policy input and versioned model contracts for both modes.

All PPO training checks share one default 512-step rollout. Set
REALMAN_FETCH_PPO_TEST_ARTIFACTS to retain/reinspect it without retraining.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.tasks.reach import ENV_CONFIG, task_environment
from realman_rl_simtoreal.rl.common.observation import (
    FEATURE_KEYS, ReachTransform, ScaledReachObservation, features,
    observation_metadata, checkpoint_transform,
)


def script(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, PROJECT_ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native(value):
    if isinstance(value, np.ndarray):return value.tolist()
    if isinstance(value, np.generic):return value.item()
    if isinstance(value, dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):return [native(v) for v in value]
    return value


class ObservationPhysicsTests(unittest.TestCase):
    def test_a_shapes_spaces_defaults_and_all_task_distributions(self):
        for task in ('fixed_reach', 'axis_pair_reach', 'easy_random_reach', 'random_reach'):
            base = task_environment(task, ENV_CONFIG)
            for mode, size in [('legacy', 27), ('fetch_like_v1', 6)]:
                with self.subTest(task=task, mode=mode), ReachEnv(**base, observation_mode=mode) as env:
                    obs, _ = env.reset(seed=2026)
                    self.assertEqual(set(obs), set(FEATURE_KEYS))
                    self.assertEqual({k:v.shape for k,v in obs.items()},
                                     dict(observation=(size,), achieved_goal=(3,), desired_goal=(3,)))
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(sum(v.size for v in obs.values()), size+6)
                    self.assertTrue(all(v.dtype == np.float64 for v in obs.values()))
                    self.assertTrue(all(np.isfinite(v).all() for v in obs.values()))
        with ReachEnv() as env:self.assertEqual(env.observation_mode, 'legacy')
        with self.assertRaisesRegex(ValueError, 'observation_mode'):ReachEnv(observation_mode='fetch')

    def test_a_official_gym_checker_raw_and_wrapped(self):
        for mode in ('legacy', 'fetch_like_v1'):
            with ReachEnv(observation_mode=mode) as env:check_env(env, skip_render_check=True)
            with ScaledReachObservation(ReachEnv(observation_mode=mode)) as env:
                obs, _ = env.reset(seed=2026)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertEqual(obs['observation'].dtype, np.float32)

    def test_b_real_site_position_goal_and_motion_world_metres_per_period(self):
        # Two periods catch a missing/doubled time scaling. Nonzero real motion
        # and the rotated tool frame catch local/world frame confusion.
        for period in (.05, .1):
            with ReachEnv(observation_mode='fetch_like_v1', control_period=period) as env:
                env.reset(seed=2)
                for action in ([.25, -.1, .15], [-.1, .15, -.1], [0, 0, 0]):
                    obs, _, _, _, _ = env.step(action)
                    site = env.kinematics.site_id
                    np.testing.assert_array_equal(obs['achieved_goal'], env.data.site_xpos[site])
                    np.testing.assert_array_equal(obs['observation'][:3], obs['achieved_goal'])
                    np.testing.assert_array_equal(obs['desired_goal'], env._goal)
                    jac = np.empty((3, env.model.nv))
                    mujoco.mj_jacSite(env.model, env.data, jac, None, site)
                    velocity = jac @ env.data.qvel
                    self.assertGreater(np.linalg.norm(velocity), 1e-7)
                    np.testing.assert_allclose(obs['observation'][3:], velocity*period, rtol=0, atol=1e-15)
                    independent = np.empty(6)
                    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_SITE, site, independent, 0)
                    np.testing.assert_allclose(velocity, independent[3:], rtol=0, atol=1e-12)
                    local = np.empty(6)
                    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_SITE, site, local, 1)
                    np.testing.assert_allclose(env.data.site_xmat[site].reshape(3, 3) @ local[3:], velocity, atol=1e-12)
                    self.assertFalse(np.allclose(local[3:], velocity, rtol=0, atol=1e-8))
                    # Independent FK finite difference on private MjData only.
                    ends = []
                    for dt in (-1e-6, 1e-6):
                        private = mujoco.MjData(env.model)
                        private.qpos[:] = env.data.qpos
                        mujoco.mj_integratePos(env.model, private.qpos, env.data.qvel, dt)
                        mujoco.mj_forward(env.model, private)
                        ends.append(private.site_xpos[site].copy())
                    np.testing.assert_allclose((ends[1]-ends[0])/2e-6, velocity, rtol=0, atol=2e-9)
                    self.assertTrue(all(np.isfinite(v).all() for v in obs.values()))

    def test_b_full_nv_qvel_pairing_with_noncontiguous_arm_dofs(self):
        # An isolated model has an extra hinge and a free body (nq != nv).
        # Its site depends on DOFs outside the six arm indices; slicing qvel
        # to the arm or using qpos indices would produce a different result.
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <body><joint name="extra" axis="0 1 0"/><geom type="sphere" size=".1" mass="1"/>
            <body pos=".3 0 .2"><joint name="arm" axis="0 0 1"/>
              <geom type="sphere" size=".1" mass="1"/><site name="tip" pos=".2 .1 .1"/>
            </body></body>
          <body pos="2 0 0"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        data.qvel[:] = np.arange(1, model.nv+1)*.03
        mujoco.mj_forward(model, data)
        from types import SimpleNamespace
        stub = SimpleNamespace(model=model, data=data, observation_mode='fetch_like_v1', control_period=.05,
                               kinematics=SimpleNamespace(site_id=model.site('tip').id, vadr=np.array([1])),
                               _goal=np.array([.5, .1, .3]))
        obs = ReachEnv._observation(stub)
        expected = np.empty(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, model.site('tip').id, expected, 0)
        np.testing.assert_allclose(obs['observation'][3:], expected[3:]*.05, atol=1e-15)

    def test_b_observation_is_read_only_and_does_not_advance_physics(self):
        with ReachEnv(observation_mode='fetch_like_v1') as env:
            env.reset(seed=1);env.step([.1, .1, -.1])
            spec = mujoco.mjtState.mjSTATE_INTEGRATION
            before = np.empty(mujoco.mj_stateSize(env.model, spec))
            mujoco.mj_getState(env.model, env.data, before, spec)
            positions = env.data.site_xpos.copy()
            with patch.object(mujoco, 'mj_step', side_effect=AssertionError('Observation cannot step')), \
                 patch.object(mujoco, 'mj_forward', side_effect=AssertionError('Observation cannot forward')):
                a = env._observation();b = env._observation()
            after = np.empty_like(before)
            mujoco.mj_getState(env.model, env.data, after, spec)
            np.testing.assert_array_equal(before, after)
            np.testing.assert_array_equal(positions, env.data.site_xpos)
            for key in a:np.testing.assert_array_equal(a[key], b[key])
            for value in a.values():value[:] = 999
            self.assertFalse(np.any(env.data.site_xpos == 999))

    def test_b_fixed_normalization_and_goals_keep_si_semantics(self):
        with ReachEnv(observation_mode='fetch_like_v1', reward_mode='reach_shaped_v1') as env:
            env.reset(seed=1)
            obs, reward, _, _, info = env.step([.2, 0, -.1])
            transform = ReachTransform.from_env(env)
            encoded = transform.encode(obs)
            np.testing.assert_array_equal(encoded['observation'][:3],
                ((obs['observation'][:3]-env.kinematics.home_position)/.01).astype(np.float32))
            np.testing.assert_array_equal(encoded['observation'][3:], (obs['observation'][3:]/.01).astype(np.float32))
            for key in ('achieved_goal', 'desired_goal'):
                np.testing.assert_array_equal(encoded[key], obs[key]);self.assertEqual(encoded[key].dtype, np.float64)
            self.assertEqual(reward, env.compute_reward(encoded['achieved_goal'], encoded['desired_goal'], info))
            far = copy.deepcopy(obs);far['observation'][0] += .1
            self.assertGreater(transform.encode(far)['observation'][0], 9.)
            self.assertFalse(transform.config['clipping']);self.assertFalse(transform.config['reward_normalization'])
            self.assertEqual(transform.config['schema'], 3)

    def test_d_plus_minus_x_world_goals_same_home_and_distinct_inputs(self):
        base = task_environment('axis_pair_reach', ENV_CONFIG, observation_mode='fetch_like_v1')
        with ReachEnv(**base) as env:
            a, _ = env.reset(seed=1, options={'goal_index':0})
            b, _ = env.reset(seed=1, options={'goal_index':1})
            np.testing.assert_array_equal(a['achieved_goal'], b['achieved_goal'])
            np.testing.assert_array_equal(a['observation'], b['observation'])
            for obs, sign in ((a, 1), (b, -1)):
                np.testing.assert_array_equal(obs['desired_goal'], env.kinematics.home_position+[sign*.005, 0, 0])
            encoded_a = features(ReachTransform.from_env(env).encode(a))
            encoded_b = features(ReachTransform.from_env(env).encode(b))
            self.assertEqual(np.flatnonzero(encoded_a != encoded_b).tolist(), [3])

    def compare_physics(self, config, actions):
        with ReachEnv(**config, observation_mode='legacy') as old, \
             ReachEnv(**config, observation_mode='fetch_like_v1') as new:
            a, ai = old.reset(seed=83001);b, bi = new.reset(seed=83001)
            self.assertEqual(native(ai), native(bi))
            np.testing.assert_array_equal(a['desired_goal'], b['desired_goal'])
            active = {id(e.data):e.data.qpos.copy() for e in (old, new)}
            counts = {id(e.data):0 for e in (old, new)}
            physical = mujoco.mj_step
            def checked(m, d):
                if id(d) in active:np.testing.assert_array_equal(d.qpos, active[id(d)])
                physical(m, d)
                if id(d) in active:
                    active[id(d)] = d.qpos.copy();counts[id(d)] += 1
            with patch.object(mujoco, 'mj_step', side_effect=checked):
                for action in actions:
                    a, ar, at, atr, ai = old.step(action)
                    b, br, bt, btr, bi = new.step(action)
                    for field in ('qpos', 'qvel', 'ctrl', 'qacc', 'qfrc_actuator'):
                        np.testing.assert_array_equal(getattr(old.data, field), getattr(new.data, field))
                    np.testing.assert_array_equal(a['achieved_goal'], b['achieved_goal'])
                    self.assertEqual((ar, at, atr, old.data.time), (br, bt, btr, new.data.time))
                    self.assertEqual(native(ai), native(bi))
                    self.assertEqual(counts[id(old.data)], counts[id(new.data)])
                    if at or atr:break
            for env in (old, new):np.testing.assert_array_equal(env.data.qpos, active[id(env.data)])
            return ai

    def test_e_trajectories_rewards_safety_termination_and_physics_steps_identical(self):
        for task in ('fixed_reach', 'axis_pair_reach', 'easy_random_reach', 'random_reach'):
            for reward in ('distance', 'safe_dense', 'reach_shaped_v1'):
                for boundary in ('terminate', 'reject_and_hold'):
                    with self.subTest(task=task, reward=reward, boundary=boundary):
                        config = task_environment(task, dict(ENV_CONFIG, max_episode_steps=6, boundary_mode=boundary),
                                                  reward_mode=reward)
                        self.compare_physics(config, [[.1, -.1, .1], [0, 0, 0], [-.1, .1, 0]]*2)
                        self.compare_physics(dict(config, action_scale=.01), [[1, 1, 1]]*6)

    def test_e_success_transition_identical(self):
        info = self.compare_physics(dict(max_episode_steps=80), [[.25, 0, 0]]*20+[[0, 0, 0]]*60)
        self.assertEqual(info['termination_reason'], 'success')


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional SB3 absent')
class ObservationPolicyTests(unittest.TestCase):
    def test_c_actual_stock_combined_extractor_actor_critic_and_valid_action(self):
        import torch
        from torch import nn
        from stable_baselines3.common.policies import MultiInputActorCriticPolicy
        from stable_baselines3.common.torch_layers import CombinedExtractor
        from realman_rl_simtoreal.rl.common.env_factory import training_env
        from realman_rl_simtoreal.rl.ppo import make_ppo, PPO_CONFIG
        torch.set_num_threads(1)
        for mode, size in [('legacy', 33), ('fetch_like_v1', 12)]:
            env, transform = training_env(dict(observation_mode=mode))
            self.addCleanup(env.close)
            model, parameters = make_ppo(env, 2026)
            self.assertEqual(parameters, PPO_CONFIG)
            policy = model.policy
            self.assertIsInstance(policy, MultiInputActorCriticPolicy)
            self.assertIs(type(policy.features_extractor), CombinedExtractor)
            self.assertTrue(all(type(x) is nn.Flatten for x in policy.features_extractor.extractors.values()))
            self.assertEqual(tuple(policy.features_extractor.extractors), FEATURE_KEYS)
            self.assertEqual(policy.features_extractor.features_dim, size)
            self.assertIs(policy.pi_features_extractor, policy.vf_features_extractor)
            obs = env.reset();tensors, _ = policy.obs_to_tensor(obs)
            with torch.no_grad():
                actual = policy.extract_features(tensors)
                np.testing.assert_array_equal(actual.numpy(), features(obs))
                calls = []
                handles = [net[0].register_forward_pre_hook(lambda m, args:calls.append(args[0].clone()))
                           for net in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net)]
                try:
                    action, value, logp = policy(tensors, deterministic=True)
                finally:
                    for handle in handles:handle.remove()
            self.assertEqual(len(calls), 2)
            for network_input in calls:np.testing.assert_array_equal(network_input.numpy(), actual.numpy())
            self.assertEqual(tuple(action.shape), (1, 3));self.assertEqual(tuple(value.shape), (1, 1))
            self.assertTrue(torch.isfinite(action).all() and torch.isfinite(value).all() and torch.isfinite(logp).all())
            predicted, _ = model.predict(obs, deterministic=True)
            self.assertTrue(env.action_space.contains(predicted[0]))
            for net in (policy.mlp_extractor.policy_net, policy.mlp_extractor.value_net):
                self.assertEqual([x.out_features for x in net if isinstance(x, nn.Linear)], [256, 256])
            self.assertEqual(policy.activation_fn, nn.Tanh)
            # All three keys actually reach the extractor, including both goals.
            for key in FEATURE_KEYS:
                changed = {k:v.copy() for k,v in obs.items()};changed[key][..., 0] += .002
                with torch.no_grad():other = policy.extract_features(policy.obs_to_tensor(changed)[0])
                self.assertFalse(torch.equal(actual, other))

    def test_modes_and_wrong_transforms_rejected_before_inference(self):
        with ReachEnv() as old, ReachEnv(observation_mode='fetch_like_v1') as new:
            a, _ = old.reset();b, _ = new.reset()
            for env, obs in ((old, b), (new, a)):
                with self.assertRaisesRegex(ValueError, 'modes cannot be mixed'):ReachTransform.from_env(env).encode(obs)
            with self.assertRaisesRegex(ValueError, 'does not match environment'):
                ScaledReachObservation(new, ReachTransform.from_env(old))
            transform = ReachTransform.from_env(new)
            config = dict(schema=2, environment=dict(observation_mode='fetch_like_v1'),
                          observation_transform=transform.config, **observation_metadata(new, transform))
            for key in ('observation_mode', 'features_dim', 'observation_space', 'feature_keys',
                        'normalization_version', 'normalization_config', 'observation_field_order'):
                broken = copy.deepcopy(config);broken.pop(key)
                with self.subTest(key=key), self.assertRaises(ValueError):checkpoint_transform(broken)
            broken = copy.deepcopy(config);broken['environment']['control_period'] = .1
            with self.assertRaisesRegex(ValueError, 'control_period'):checkpoint_transform(broken)
            with self.assertRaises(ValueError):checkpoint_transform(config, observation_mode='legacy')

    def test_legacy_only_consumers_refuse_new_mode_clearly(self):
        from realman_rl_simtoreal.rl.common.baselines import rule_action
        from realman_rl_simtoreal.rl.common.evaluation import evaluate
        from realman_rl_simtoreal.rl.sac import make_model
        from realman_rl_simtoreal.rl.sac_diagnostics import trace_policy
        from realman_rl_simtoreal.imitation.bc import collect_demonstrations, evaluate_policy, BCPolicy, BCNetwork
        config = dict(observation_mode='fetch_like_v1')
        with ReachEnv(**config) as env:
            obs, _ = env.reset()
            calls = [lambda:rule_action(obs, env.action_scale), lambda:make_model(env, 1),
                     lambda:evaluate('rule', [1], config=config), lambda:trace_policy('idle', [1], config=config),
                     lambda:collect_demonstrations([1], [2], environment=config),
                     lambda:evaluate_policy('rule', [1], environment=config),
                     lambda:BCPolicy(BCNetwork(), ReachTransform.from_env(env)),
                     lambda:script('scripts/train/train_sac_reach.py').train(Path('/tmp/unused-fetch-sac'), observation_mode='fetch_like_v1')]
            for call in calls:
                with self.assertRaisesRegex(ValueError, "legacy"):call()

    def test_g_historical_schema_load_and_deterministic_predictions(self):
        import torch
        from realman_rl_simtoreal.rl.common.env_factory import training_env
        from realman_rl_simtoreal.rl.ppo import make_ppo, load_ppo
        torch.set_num_threads(1)
        env, transform = training_env({});self.addCleanup(env.close)
        model, _ = make_ppo(env, 2026)
        obs = env.reset();before, _ = model.predict(obs, deterministic=True)
        config = dict(schema=1, algorithm='PPO', torch_num_threads=1, environment={}, observation_transform=transform.config)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            model.save(path/'model.zip');(path/'config.json').write_text(json.dumps(config))
            loaded, saved_transform, saved = load_ppo(path/'model.zip')
            self.assertEqual(saved, config);self.assertEqual(saved_transform.observation_mode, 'legacy')
            self.assertEqual(loaded.policy.features_extractor.features_dim, 33)
            np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], before)
            with self.assertRaisesRegex(ValueError, 'observation_mode mismatch'):
                load_ppo(path/'model.zip', observation_mode='fetch_like_v1')

    def test_g_actual_prechange_legacy_checkpoint_when_available(self):
        from realman_rl_simtoreal.rl.ppo import load_ppo
        baseline = json.loads((PROJECT_ROOT/'tests/fixtures/reach_legacy_policy_predictions.json').read_text())
        path = PROJECT_ROOT/baseline['path']
        if not path.exists():self.skipTest('Historical local artifact absent; portable schema=1 test still runs')
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), baseline['model_sha256'])
        self.assertEqual(hashlib.sha256((path.parent/'config.json').read_bytes()).hexdigest(), baseline['config_sha256'])
        model, transform, _ = load_ppo(path)
        for row in baseline['records']:
            raw = {k:np.array(v) for k,v in row['raw'].items()}
            encoded = transform.encode(raw)
            for key in encoded:np.testing.assert_array_equal(encoded[key], row['encoded'][key])
            np.testing.assert_array_equal(model.predict(encoded, deterministic=True)[0], row['prediction'])


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional SB3 absent')
class FetchPPOSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.folder = Path(os.environ.get('REALMAN_FETCH_PPO_TEST_ARTIFACTS', str(Path(cls.temp.name)/'run')))
        if not (cls.folder/'summary.json').exists():
            driver = script('scripts/train/train_ppo_reach.py')
            from realman_rl_simtoreal.rl.common import baselines, evaluation
            physical = mujoco.mj_step;original_step = ReachEnv.step;active = {};counts = dict(env_steps=0, physics_steps=0)
            def checked_physics(m, d):
                if id(d) in active:np.testing.assert_array_equal(d.qpos, active[id(d)])
                physical(m, d)
                if id(d) in active:active[id(d)] = d.qpos.copy();counts['physics_steps'] += 1
            def checked_step(env, action):
                active[id(env.data)] = env.data.qpos.copy();counts['env_steps'] += 1
                try:
                    result = original_step(env, action)
                    np.testing.assert_array_equal(env.data.qpos, active[id(env.data)])
                    return result
                finally:active.pop(id(env.data))
            with patch.object(mujoco, 'mj_step', side_effect=checked_physics), \
                 patch.object(ReachEnv, 'step', checked_step), \
                 patch.object(baselines, 'rule_action', side_effect=AssertionError('No expert')), \
                 patch.object(evaluation, 'rule_action', side_effect=AssertionError('No expert')):
                summary = driver.train(cls.folder, timesteps=512, task='axis_pair_reach', observation_mode='fetch_like_v1',
                                       seed=2026, eval_every=512, checkpoint_every=512)
            if summary['status'] != 'completed':raise AssertionError(summary['error'])
            (cls.folder/'test_verification.json').write_text(json.dumps(dict(**counts, no_experts=True, no_live_qpos_writes=True))+'\n')
        cls.summary = json.loads((cls.folder/'summary.json').read_text())
        cls.config = json.loads((cls.folder/'config.json').read_text())

    def test_f_one_complete_default_rollout_actor_critic_updates_and_finite_losses(self):
        from realman_rl_simtoreal.rl.ppo import PPO_CONFIG, load_ppo, optimizer_steps, TRAIN_METRICS
        self.assertEqual(self.config['ppo'], PPO_CONFIG)
        self.assertEqual(self.summary['status'], 'completed')
        self.assertEqual([self.summary[k] for k in ('actual_timesteps', 'completed_rollout_updates', 'epoch_updates', 'optimizer_steps')],
                         [512, 1, 10, 80])
        self.assertTrue(self.summary['actor_updated'] and self.summary['critic_updated'])
        self.assertEqual(self.config['init_mode'], 'scratch');self.assertTrue(self.config['pure_rl'])
        self.assertEqual(self.config['features_dim'], 12)
        with np.load(self.folder/'final/last_rollout.npz') as data:
            self.assertEqual(data['observation_observation'].shape[-1], 6)
            self.assertEqual(data['advantages'].size, 512)
            for key in data.files:self.assertTrue(np.isfinite(data[key]).all(), key)
        metrics = [json.loads(row) for row in (self.folder/'metrics.jsonl').read_text().splitlines()][-1]
        for key in TRAIN_METRICS:self.assertTrue(np.isfinite(metrics[key]), key)
        model, _, _ = load_ppo(self.folder/'final/model.zip')
        self.assertEqual(optimizer_steps(model), 80)
        evidence = json.loads((self.folder/'test_verification.json').read_text())
        self.assertTrue(evidence['no_experts'] and evidence['no_live_qpos_writes'])
        self.assertGreater(evidence['physics_steps'], 0)

    def test_f_all_checkpoints_metadata_reload_and_mismatch_refusal(self):
        from realman_rl_simtoreal.rl.ppo import load_ppo, save_ppo_checkpoint
        model, transform, config = load_ppo(self.folder/'final/model.zip')
        with ReachEnv(**config['environment']) as env:
            obs, _ = env.reset(seed=1, options={'goal_index':0})
            expected = model.predict(transform.encode(obs), deterministic=True)[0]
        other, t, c = load_ppo(self.folder/'checkpoints/step_000512/model.zip')
        np.testing.assert_array_equal(other.predict(t.encode(obs), deterministic=True)[0], expected)
        for name in ('initial', 'checkpoints/step_000000', 'checkpoints/step_000512', 'final'):
            saved = json.loads((self.folder/name/'config.json').read_text())
            checkpoint_transform(saved)
            self.assertEqual(saved['observation_mode'], 'fetch_like_v1');self.assertEqual(saved['features_dim'], 12)
            self.assertEqual(len(saved['observation_field_order']['observation']), 6)
        with self.assertRaisesRegex(ValueError, 'observation_mode mismatch'):
            load_ppo(self.folder/'final/model.zip', observation_mode='legacy')
        from realman_rl_simtoreal.rl.common.env_factory import training_env
        from realman_rl_simtoreal.rl.ppo import make_ppo
        legacy_env, _ = training_env({});self.addCleanup(legacy_env.close)
        old, _ = make_ppo(legacy_env, 1)
        with self.assertRaisesRegex(ValueError, 'observation_space'):
            save_ppo_checkpoint(old, Path(self.temp.name)/'bad', config, 0, completed_stage=False)

    def test_f_independent_cli_evaluation_and_goal_probe(self):
        target = self.folder/'independent_test_evaluation.json'
        if not target.exists():
            proc = subprocess.run([sys.executable, '-I', str(PROJECT_ROOT/'scripts/evaluate/evaluate_ppo_reach.py'),
                '--model', str(self.folder/'final/model.zip'), '--observation-mode', 'fetch_like_v1',
                '--episodes', '2', '--seed-start', '88000', '--json', str(target)], capture_output=True, text=True, timeout=120)
            self.assertEqual(proc.returncode, 0, proc.stdout+proc.stderr)
        result = json.loads(target.read_text())
        self.assertEqual(result['features_dim'], 12);self.assertTrue(result['parameters_unchanged'])
        self.assertEqual(set(result['per_goal']), {'+X', '-X'})
        probe = script('scripts/diagnostics/probe_ppo_goals.py').probe(self.folder/'final/model.zip')
        self.assertEqual(probe['features_dim'], 12)
        self.assertEqual(probe['changed_feature_indices'], [3])
        self.assertEqual(probe['expected_goal_features'], dict(desired_goal=[3, 4, 5]))
        self.assertTrue(probe['physical_state_identical'] and probe['parameters_unchanged'])


if __name__ == '__main__':unittest.main()
