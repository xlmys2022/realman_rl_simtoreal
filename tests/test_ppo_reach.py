"""Official PPO contracts. All training assertions share ONE complete rollout."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from realman_rl_simtoreal.paths import PROJECT_ROOT


def read(path):
    return json.loads(Path(path).read_text())


def script(name):
    spec=importlib.util.spec_from_file_location(name,PROJECT_ROOT/'scripts'/('plots' if name.startswith('plot_') else 'train')/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


AVAILABLE=importlib.util.find_spec('stable_baselines3') is not None


@unittest.skipUnless(AVAILABLE,'Optional SB3/PyTorch not installed')
class PPOReachTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A fresh test run trains once. An explicit artifact folder permits
        # read-only reinspection without another rollout (e.g. training caps).
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.folder=Path(os.environ.get('REALMAN_PPO_TEST_ARTIFACTS',str(Path(cls.temp.name)/'run'))).resolve()
        if not (cls.folder/'test_verification.json').exists():
            code=r'''
import importlib.abc,importlib.util,json,sys
from pathlib import Path
import mujoco,numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if any(s in name for s in ('realman_rl_simtoreal.imitation','realman_rl_simtoreal.rl.sac','Robotic_Arm','realman_rl_simtoreal.vendor')):
   raise AssertionError('BC/SDK import forbidden: '+name)
sys.meta_path.insert(0,Block())
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.rl.common import evaluation, baselines
def forbidden(*a,**kw):raise AssertionError('SAC weights/expert actions forbidden')
evaluation.rule_action=baselines.rule_action=forbidden
from realman_rl_simtoreal.envs.reach_env import ReachEnv
old_env_step,old_physics=ReachEnv.step,mujoco.mj_step
active={};counts=dict(env_steps=0,physics_steps=0)
def physical(model,data):
 key=id(data)
 if key in active:np.testing.assert_array_equal(data.qpos,active[key])
 old_physics(model,data)
 if key in active:
  active[key]=data.qpos.copy();counts['physics_steps']+=1
def step(env,action):
 key=id(env.data);active[key]=env.data.qpos.copy();counts['env_steps']+=1
 try:
  result=old_env_step(env,action)
  np.testing.assert_array_equal(env.data.qpos,active[key])
  return result
 finally:active.pop(key)
ReachEnv.step=step;mujoco.mj_step=physical
spec=importlib.util.spec_from_file_location('ppo_driver',PROJECT_ROOT/'scripts/train/train_ppo_reach.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
folder=Path(sys.argv[1])
result=driver.train(folder,timesteps=128,n_steps=512,batch_size=64,n_epochs=10,
 seed=2026,eval_every=300,checkpoint_every=400,eval_episodes=1,
 tensorboard=importlib.util.find_spec('tensorboard') is not None)
assert result['status']=='completed',result['error']
assert result['completed_rollout_updates']==1 and result['actual_timesteps']==512
assert not any(n.startswith('realman_rl_simtoreal.imitation') for n in sys.modules)
counts.update(no_expert_or_sac_parameters=True,no_live_qpos_write=True)
(folder/'test_verification.json').write_text(json.dumps(counts,indent=2)+'\n')
print(json.dumps(result['timing']))
'''
            proc=subprocess.run([sys.executable,'-I','-c',code,str(cls.folder)],
                                capture_output=True,text=True,timeout=180)
            cls.folder.parent.mkdir(parents=True,exist_ok=True)
            (cls.folder.parent/'smoke_process.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode:raise AssertionError(proc.stdout+proc.stderr)
        from realman_rl_simtoreal.rl.ppo import load_ppo
        cls.model,cls.transform,cls.config=load_ppo(cls.folder/'final/model.zip')
        cls.initial,_,_=load_ppo(cls.folder/'initial/model.zip')
        cls.summary=read(cls.folder/'summary.json')
        cls.metrics=[json.loads(s) for s in (cls.folder/'metrics.jsonl').read_text().splitlines()]
        cls.actions=[json.loads(s) for s in (cls.folder/'actions.jsonl').read_text().splitlines()]
        cls.evaluations=read(cls.folder/'evaluations.json')

    def test_dict_features_and_native_policy_architecture(self):
        import torch
        from stable_baselines3.common.policies import MultiInputActorCriticPolicy
        from realman_rl_simtoreal.envs.reach_env import ReachEnv
        from realman_rl_simtoreal.rl.common.observation import ReachTransform
        self.assertIsInstance(self.model.policy,MultiInputActorCriticPolicy)
        self.assertEqual(self.model.policy.features_extractor.features_dim,33)
        self.assertEqual(list(self.model.observation_space.spaces),['achieved_goal','desired_goal','observation'])
        with ReachEnv(**self.config['environment']) as env:
            obs,_=env.reset(seed=2026)
            self.assertEqual(ReachTransform.from_env(env).config,self.transform.config)
            encoded=self.transform.encode(obs)
            tensors,_=self.model.policy.obs_to_tensor(encoded)
            with torch.no_grad():features=self.model.policy.extract_features(tensors).numpy()
            expected=np.concatenate([encoded[k].ravel() for k in self.model.observation_space.spaces]).astype(np.float32)
            np.testing.assert_array_equal(features[0],expected)
            action,_=self.model.predict(encoded,deterministic=True)
            self.assertTrue(env.action_space.contains(action))
        self.assertEqual(self.model.policy.activation_fn,torch.nn.Tanh)
        self.assertEqual([m.out_features for m in self.model.policy.mlp_extractor.policy_net if isinstance(m,torch.nn.Linear)],[256,256])
        self.assertEqual([m.out_features for m in self.model.policy.mlp_extractor.value_net if isinstance(m,torch.nn.Linear)],[256,256])
        self.assertFalse(self.model.policy.squash_output)
        np.testing.assert_array_equal(self.initial.policy.log_std.detach().numpy(),np.zeros(3))

    def test_single_complete_rollout_and_real_update_counts(self):
        from realman_rl_simtoreal.rl.ppo import optimizer_steps
        self.assertEqual(self.summary['status'],'completed')
        for key,value in dict(requested_timesteps=128,actual_timesteps=512,
            planned_complete_rollout_timesteps=512,completed_rollout_updates=1,
            collected_complete_rollouts=1,epoch_updates=10,optimizer_steps=80).items():
            self.assertEqual(self.summary[key],value,key)
        self.assertEqual(self.model.num_timesteps,512);self.assertEqual(self.model._n_updates,10)
        self.assertEqual(optimizer_steps(self.model),80)
        self.assertFalse(hasattr(self.model,'replay_buffer'))

    def test_actor_and_value_parameters_really_updated(self):
        from realman_rl_simtoreal.rl.ppo import parameter_hashes
        self.assertTrue(self.summary['actor_updated']);self.assertTrue(self.summary['critic_updated'])
        self.assertEqual(parameter_hashes(self.initial),self.summary['initial_parameter_hashes'])
        self.assertEqual(parameter_hashes(self.model),self.summary['final_parameter_hashes'])
        for group in ('actor','critic'):
            self.assertNotEqual(parameter_hashes(self.initial)[group],parameter_hashes(self.model)[group])

    def test_rollout_advantages_returns_losses_are_finite(self):
        import torch
        from realman_rl_simtoreal.rl.ppo import TRAIN_METRICS
        with np.load(self.folder/'final/last_rollout.npz') as data:
            for name in data.files:self.assertTrue(np.isfinite(data[name]).all(),name)
            self.assertEqual(data['advantages'].size,512)
            self.assertGreater(np.ptp(data['advantages']),0)
        for name in TRAIN_METRICS:self.assertTrue(np.isfinite(self.metrics[-1][name]),name)
        for p in self.model.policy.parameters():self.assertTrue(torch.isfinite(p).all())

    def test_box_clipping_separate_from_ppo_objective_clipping(self):
        raw=np.asarray([s['raw_action'] for s in self.actions]);actual=np.asarray([s['applied_action'] for s in self.actions])
        np.testing.assert_array_equal(actual,np.clip(raw,-1,1))
        self.assertTrue(np.isfinite(raw).all());self.assertGreater(np.max(abs(raw)),1)
        clipped=np.abs(raw)>1;stats=self.summary['training']['box_clipping']
        self.assertEqual(stats['samples'],512)
        self.assertEqual(stats['clipped_vectors'],int(clipped.any(axis=1).sum()))
        self.assertEqual(stats['clipped_components'],int(clipped.sum()))
        self.assertAlmostEqual(stats['vector_fraction'],clipped.any(axis=1).mean())
        self.assertAlmostEqual(stats['component_fraction'],clipped.mean())
        self.assertTrue(0<=self.metrics[-1]['clip_fraction']<=1)
        with np.load(self.folder/'final/last_rollout.npz') as data:
            np.testing.assert_array_equal(data['actions'].reshape(-1,3),raw.astype(np.float32))

    def test_only_completed_updates_trigger_snapshots(self):
        stages=read(self.folder/'stages.json')
        self.assertEqual([s['step'] for s in stages],[0,512])
        self.assertEqual([s['phase'] for s in stages],['initial','post_update'])
        self.assertEqual(stages[1]['evaluation_thresholds_crossed'],[300])
        self.assertEqual(stages[1]['checkpoint_thresholds_crossed'],[400])
        self.assertFalse((self.folder/'checkpoints/step_000300').exists())
        for label,step,complete in [('initial',0,False),('checkpoints/step_000000',0,False),
                                  ('checkpoints/step_000512',512,True),('final',512,True)]:
            state=read(self.folder/label/'state.json')
            self.assertEqual(state['actual_timesteps'],step)
            self.assertEqual(state['completed_update_stage'],complete)
            self.assertEqual(state['stage_kind'],'post_update' if complete else 'initial')

    def test_official_save_load_predictions_identical(self):
        from stable_baselines3 import PPO
        with np.load(self.folder/'final/last_rollout.npz') as data:
            obs={k:data['observation_'+k].reshape(-1,*space.shape)[:7]
                 for k,space in self.model.observation_space.spaces.items()}
        loaded=PPO.load(self.folder/'checkpoints/step_000512/model.zip',device='cpu')
        a,_=self.model.predict(obs,deterministic=True);b,_=loaded.predict(obs,deterministic=True)
        np.testing.assert_array_equal(a,b)

    def test_independent_evaluation_does_not_update_network_or_rng(self):
        import torch
        from realman_rl_simtoreal.rl.ppo import evaluate_ppo, parameter_hashes
        before=parameter_hashes(self.model);state=torch.random.get_rng_state().clone()
        config=dict(self.config['environment'],max_episode_steps=2)
        a=evaluate_ppo(self.model,self.transform,config,[88001])
        b=evaluate_ppo(self.model,self.transform,config,[88001])
        self.assertEqual(parameter_hashes(self.model),before)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertEqual(a['summary'],b['summary']);self.assertEqual(a['episodes'][0]['actions'],b['episodes'][0]['actions'])
        self.assertEqual(self.model.num_timesteps,512);self.assertTrue(a['parameters_unchanged'])

    def test_fixed_target_scope_is_not_random_generalization(self):
        from realman_rl_simtoreal.envs.reach_env import ReachEnv
        from realman_rl_simtoreal.rl.ppo import environment_config
        with ReachEnv(**environment_config()) as env:
            a,_=env.reset(seed=87000);b,_=env.reset(seed=88000)
            np.testing.assert_array_equal(a['desired_goal'],b['desired_goal'])
        for row in self.evaluations:self.assertIn('one fixed target',row['goal_scope'])
        self.assertEqual(environment_config('easy_random_reach')['target_radius'],.003)
        self.assertEqual(environment_config('random_reach')['target_radius'],.005)

    def test_no_experts_or_sac_parameters_and_no_live_qpos_writes(self):
        check=read(self.folder/'test_verification.json')
        self.assertTrue(check['no_expert_or_sac_parameters']);self.assertTrue(check['no_live_qpos_write'])
        self.assertGreaterEqual(check['env_steps'],512);self.assertGreater(check['physics_steps'],0)
        self.assertTrue(self.config['pure_rl']);self.assertEqual(self.config['init_mode'],'scratch')
        self.assertEqual(self.config['algorithm'],'PPO')
        self.assertFalse(any('bc' in k or 'replay' in k for k in self.config))
        for row in self.actions:
            self.assertIn(row['physics_steps'],(0,25))
            self.assertTrue(np.isfinite(row['actual_after_m']).all())

    def test_physical_configuration_and_source_models_unchanged(self):
        import ast
        import hashlib
        from realman_rl_simtoreal.rl.common.artifacts import asset_hashes
        env=self.config['environment']
        for key,value in dict(target_mode='fixed',reward_mode='reach_shaped_v1',action_scale=.00025,
            control_period=.05,max_episode_steps=200,success_threshold=.001,boundary_mode='terminate',
            gravity_compensation=True).items():self.assertEqual(env[key],value)
        # Discrete reset support intentionally changes ReachEnv source. Keep
        # physical hashes exact and freeze its legacy step/reward/safety AST.
        env_path='src/realman_rl_simtoreal/envs/reach_env.py'
        self.assertEqual({k:v for k,v in asset_hashes().items() if k!=env_path},
                         {k:v for k,v in self.config['assets'].items() if k!=env_path})
        contract=read(PROJECT_ROOT/'tests/fixtures/reach_env_legacy_contract.json')
        cls=next(n for n in ast.parse((PROJECT_ROOT/env_path).read_text()).body if isinstance(n,ast.ClassDef))
        methods={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
                 for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in contract['method_sha256']}
        self.assertEqual(methods,contract['method_sha256'])
        baseline=PROJECT_ROOT/'outputs/rl/ppo_reach_validation/baseline_manifest.json'
        if baseline.exists():
            # Optional developer integrity evidence; CI does not need old outputs.
            for name,digest in read(baseline).items():
                if name.startswith(('assets/','src/realman_rl_simtoreal/simulation/')):
                    self.assertEqual(hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_tensorboard_contains_measured_update_and_evaluation_metrics(self):
        if not self.config['tensorboard']:self.skipTest('Optional TensorBoard not installed')
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events=EventAccumulator(str(self.folder/'sb3'));events.Reload();tags=events.Tags()['scalars']
        for name in ('policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction','explained_variance'):
            self.assertIn('train/'+name,tags)
            points=events.Scalars('train/'+name);self.assertEqual([p.step for p in points],[512])
            self.assertAlmostEqual(points[-1].value,self.metrics[-1][name],delta=abs(self.metrics[-1][name])*1e-6+1e-7)
        for tag in ('evaluation/success_rate','evaluation/final_distance_mm','evaluation/episode_length_mean',
            'policy/action_std_mean','actions/box_clip_rollout/vector_fraction','training/episode_success_rate',
            'training/motion/toward_actual_fraction','training/motion/mean_actual_displacement_m'):
            self.assertIn(tag,tags)
        self.assertFalse(any(t.startswith('bc/') for t in tags))
        self.assertEqual([p.step for p in events.Scalars('evaluation/success_rate')],[0,512])
        self.assertEqual([p.value for p in events.Scalars('evaluation/success_rate')],
                         [r['summary']['success_rate'] for r in self.evaluations])

    def test_exported_plots_use_real_json_and_do_not_fabricate_missing_success(self):
        if importlib.util.find_spec('matplotlib') is None:self.skipTest('Optional Matplotlib not installed')
        plot=script('plot_ppo_training')
        out=Path(self.temp.name)/'plots';result=plot.export_plots(self.folder,out)
        self.assertEqual(len(result['generated']),11)
        self.assertEqual(result['curves']['evaluation_success_rate'],
            [[r['step'],r['summary']['success_rate']] for r in self.evaluations])
        self.assertEqual(result['curves']['value_loss'],[[512,self.metrics[-1]['value_loss']]])
        for name in result['generated']:self.assertEqual((out/name).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
        empty=Path(self.temp.name)/'no_eval';empty.mkdir()
        (empty/'metrics.jsonl').write_text(json.dumps(dict(step=0,policy_std_mean=1,rollout_box_clipping={}))+'\n')
        missing=plot.export_plots(empty,Path(self.temp.name)/'incomplete_plots')
        self.assertEqual(missing['curves']['evaluation_success_rate'],[])
        self.assertNotIn('evaluation_success.png',missing['generated'])

    def test_interval_validation_and_wrong_algorithm_refused(self):
        from realman_rl_simtoreal.rl.ppo import crossed_thresholds, load_ppo
        self.assertEqual(crossed_thresholds(512,1024,300),[600,900])
        with self.assertRaises(ValueError):crossed_thresholds(0,512,0)
        with self.assertRaises(ValueError):script('train_ppo_reach').train(Path(self.temp.name)/'invalid',n_steps=512,batch_size=63)
        with self.assertRaisesRegex(ValueError,'never overwrite'):script('train_ppo_reach').train(self.folder)
        wrong=Path(self.temp.name)/'wrong';wrong.mkdir();(wrong/'config.json').write_text('{"algorithm":"SAC"}')
        with self.assertRaisesRegex(ValueError,'PPO checkpoint'):load_ppo(wrong/'model.zip')


if __name__=='__main__':unittest.main()
