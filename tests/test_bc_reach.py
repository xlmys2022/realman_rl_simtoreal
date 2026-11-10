"""BC data/mean-path/closed-loop contracts; no success-rate requirement."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch
    import mujoco
    from stable_baselines3.common.torch_layers import CombinedExtractor
    from stable_baselines3.sac.policies import Actor
    from realman_rl_simtoreal.imitation import bc as bc
    from realman_rl_simtoreal.rl.common.baselines import rule_action
    from realman_rl_simtoreal.tasks.reach import ENV_CONFIG
    from realman_rl_simtoreal.rl.common.env_factory import training_env
    from realman_rl_simtoreal.envs.reach_env import ReachEnv
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Optional SB3/PyTorch not installed')
class BehavioralCloningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        cls.data,cls.manifest=bc.collect_demonstrations([31000,31001],[31002])
        cls.dataset=cls.root/'data';cls.dataset.mkdir()
        np.savez_compressed(cls.dataset/'demonstrations.npz',**cls.data)
        manifest=dict(cls.manifest,archive_sha256=hashlib.sha256((cls.dataset/'demonstrations.npz').read_bytes()).hexdigest())
        (cls.dataset/'manifest.json').write_text(json.dumps(manifest))
        # Supervised training has no environment sampling at all.
        with patch.object(bc,'ReachEnv',side_effect=AssertionError('No rollout in BC fitting')):
            cls.summary=bc.train_bc(cls.dataset/'demonstrations.npz',cls.root/'fit',epochs=4,batch_size=16,seed=7)
        cls.policy,cls.config=bc.load_policy(cls.root/'fit/best_bc.pt')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_demonstrations_reproducible_including_physics(self):
        again,_=bc.collect_demonstrations([31000,31001],[31002])
        self.assertEqual(bc.dataset_digest(self.data),bc.dataset_digest(again))
        for key in self.data:
            np.testing.assert_array_equal(self.data[key],again[key])

    def test_labels_are_original_public_rule_actions_and_legal(self):
        for i,label in enumerate(self.data['action']):
            obs={k:self.data['raw_'+k][i] for k in bc.FEATURE_KEYS}
            np.testing.assert_array_equal(label,rule_action(obs,ENV_CONFIG['action_scale']))
            self.assertEqual(label.shape,(3,));self.assertEqual(label.dtype,np.float32)
            self.assertTrue(np.isfinite(label).all());self.assertTrue((np.abs(label) <= 1).all())

    def test_raw_positions_episode_success_and_split_are_explicit(self):
        np.testing.assert_array_equal(self.data['actual_position_m'],self.data['raw_achieved_goal'])
        np.testing.assert_array_equal(self.data['target_position_m'],self.data['raw_desired_goal'])
        np.testing.assert_array_equal(self.data['command_position_m'],self.data['raw_observation'][:,18:21])
        for e in self.manifest['episodes']:
            selected=self.data['episode_index'] == e['episode_index']
            np.testing.assert_array_equal(self.data['step_index'][selected],np.arange(e['length']))
            self.assertTrue(np.all(self.data['episode_success'][selected] == e['success']))

    def test_original_preprocessing_and_sb3_combined_extractor_order(self):
        env,transform=training_env(ENV_CONFIG)
        self.addCleanup(env.close)
        raw={k:self.data['raw_'+k] for k in bc.FEATURE_KEYS}
        encoded=transform.encode(raw)
        for k in bc.FEATURE_KEYS:
            np.testing.assert_array_equal(encoded[k],self.data['processed_'+k])
        extractor=CombinedExtractor(env.observation_space)
        self.assertEqual(tuple(extractor.extractors.keys()),bc.FEATURE_KEYS)
        tensors={k:torch.from_numpy(v).float() for k,v in encoded.items()}
        np.testing.assert_array_equal(extractor(tensors).numpy(),self.data['features'])
        self.assertEqual(transform.config,self.manifest['observation_transform'])

    def test_network_matches_sb3_deterministic_actor_mean_path(self):
        env,transform=training_env(ENV_CONFIG)
        self.addCleanup(env.close)
        actor=Actor(env.observation_space,env.action_space,[256,256],
                    features_extractor=CombinedExtractor(env.observation_space),features_dim=33)
        actor.latent_pi.load_state_dict(self.policy.network.latent_pi.state_dict())
        actor.mu.load_state_dict(self.policy.network.mu.state_dict())
        encoded={k:torch.from_numpy(self.data['processed_'+k]).float() for k in bc.FEATURE_KEYS}
        with torch.no_grad():
            expected=actor(encoded,deterministic=True).numpy()
        raw={k:self.data['raw_'+k] for k in bc.FEATURE_KEYS}
        np.testing.assert_array_equal(self.policy.predict(raw),expected)
        self.assertEqual(sum(p.numel() for p in self.policy.network.parameters()),75267)
        self.assertFalse(hasattr(self.policy.network,'log_std'))

    def test_whole_episode_split_excludes_final_targets(self):
        train=set(self.manifest['train_seeds']);validation=set(self.manifest['validation_seeds'])
        self.assertFalse(train & validation)
        self.assertFalse((train|validation) & set(bc.FINAL_SEEDS))
        for split,seeds in [(0,train),(1,validation)]:
            self.assertEqual(set(self.data['seed'][self.data['split'] == split]),seeds)
        for bad in [([1],[1]),([20000],[2]),([],[2]),([-1],[2])]:
            with self.assertRaises(ValueError):bc.collect_demonstrations(*bad)

    def test_training_updates_parameters_and_selects_validation_checkpoint(self):
        self.assertTrue(self.summary['parameters_updated'])
        self.assertTrue(self.summary['parameters_finite'])
        self.assertGreater(self.summary['optimizer_updates'],0)
        history=json.loads((self.root/'fit/history.json').read_text())
        self.assertEqual(self.summary['best_epoch'],min(history,key=lambda row:row['validation_mse'])['epoch'])
        self.assertNotEqual(self.summary['initial_weight_digest'],self.summary['trained_weight_digest'])

    def test_saved_loaded_predictions_exact_and_deterministic_seed(self):
        again,_=bc.load_policy(self.root/'fit/best_bc.pt')
        raw={k:self.data['raw_'+k][:5] for k in bc.FEATURE_KEYS}
        np.testing.assert_array_equal(again.predict(raw),self.policy.predict(raw))
        with tempfile.TemporaryDirectory() as temporary:
            bc.train_bc(self.dataset/'demonstrations.npz',Path(temporary)/'fit',epochs=4,batch_size=16,seed=7)
            repeated,_=bc.load_policy(Path(temporary)/'fit/best_bc.pt')
            np.testing.assert_array_equal(repeated.predict(raw),self.policy.predict(raw))

    def test_actions_finite_and_within_space_for_single_and_batched_input(self):
        raw={k:self.data['raw_'+k] for k in bc.FEATURE_KEYS}
        output=self.policy.predict(raw)
        self.assertEqual(output.shape,(len(self.data['action']),3))
        self.assertEqual(output.dtype,np.float32)
        self.assertTrue(np.isfinite(output).all());self.assertTrue((abs(output) <= 1).all())
        single={k:v[0].copy() for k,v in raw.items()}
        self.assertEqual(self.policy.predict(single).shape,(3,))
        single['achieved_goal'][0]=np.nan
        with self.assertRaises(ValueError):self.policy.predict(single)

    def test_metrics_distinguish_moving_small_stopping_and_near_goal(self):
        labels=np.array([[.2,0,0],[.05,0,0],[0,0,0]])
        raw=dict(achieved_goal=np.zeros((3,3)),desired_goal=np.array([[1.,0,0],[.001,0,0],[0,0,0]]),
                 observation=np.zeros((3,27)))
        result=bc.action_metrics(labels,labels,raw)
        self.assertEqual(result['mse'],0);self.assertEqual(result['mae'],0)
        self.assertEqual(result['moving']['count'],2);self.assertEqual(result['small_action']['count'],1)
        self.assertEqual(result['near_zero_stop']['count'],1);self.assertEqual(result['near_goal']['count'],2)
        self.assertEqual(result['direction_cosine']['mean'],1)
        empty=bc.action_metrics(labels[:1],labels[:1],{k:v[:1] for k,v in raw.items()})
        self.assertEqual(empty['near_zero_stop']['count'],0)
        self.assertIsNone(empty['near_zero_stop']['mse'])

    def test_dataset_hash_tampering_and_no_output_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            p=Path(temporary)
            (p/'manifest.json').write_text((self.dataset/'manifest.json').read_text())
            (p/'demonstrations.npz').write_bytes(b'not a dataset')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):bc.load_demonstrations(p/'demonstrations.npz')
        with self.assertRaises(ValueError):bc.empty_directory(self.dataset)

    def test_student_closed_loop_only_env_step_no_expert_fallback_no_network_update(self):
        before=bc.weight_digest(self.policy.network)
        original=ReachEnv.step;calls=[]
        def checked(env,action):
            calls.append(np.array(action));return original(env,action)
        with patch.object(ReachEnv,'step',checked),patch.object(bc,'rule_action',side_effect=AssertionError('No teacher during BC')):
            result=bc.evaluate_policy('bc',[40000],policy=self.policy,
                                      environment=dict(ENV_CONFIG,max_episode_steps=2))
        self.assertEqual(len(calls),2);self.assertEqual(len(result['records']),2)
        self.assertEqual(before,bc.weight_digest(self.policy.network))

    def test_no_qpos_teleport_between_mujoco_physics_steps(self):
        step=mujoco.mj_step;previous=None;calls=0
        def checked(model,data):
            nonlocal previous,calls
            if previous is not None:np.testing.assert_array_equal(data.qpos,previous)
            step(model,data);previous=data.qpos.copy();calls+=1
        with patch.object(mujoco,'mj_step',side_effect=checked):
            result=bc.evaluate_policy('bc',[40000],policy=self.policy,
                                      environment=dict(ENV_CONFIG,max_episode_steps=2))
        self.assertEqual(calls,50)
        self.assertEqual(result['episodes'][0]['physical_steps'],50)
        self.assertAlmostEqual(result['records'][-1]['simulation_time_s'],.1)

    def test_evaluation_repeatable_and_same_targets_for_three_policies(self):
        initial,_=bc.load_policy(self.root/'fit/untrained_bc.pt')
        cfg=dict(ENV_CONFIG,max_episode_steps=2)
        one=bc.evaluate_policy('bc',[40001],policy=self.policy,environment=cfg)
        two=bc.evaluate_policy('bc',[40001],policy=self.policy,environment=cfg)
        self.assertEqual(one['records'],two['records']);self.assertEqual(one['episodes'],two['episodes'])
        rule=bc.evaluate_policy('rule',[40001],environment=cfg)
        untrained=bc.evaluate_policy('untrained',[40001],policy=initial,environment=cfg)
        self.assertEqual(one['episodes'][0]['goal_m'],rule['episodes'][0]['goal_m'])
        self.assertEqual(one['episodes'][0]['goal_m'],untrained['episodes'][0]['goal_m'])

    def test_no_vendor_dependency(self):
        code="""
import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if 'Robotic_Arm' in fullname or '.vendor' in fullname:
            raise AssertionError('Vendor SDK must not load')
sys.meta_path.insert(0, Block())
from realman_rl_simtoreal.imitation.bc import collect_demonstrations
data, meta=collect_demonstrations([31000],[31001])
assert len(data['action']) > 0
"""
        result=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
