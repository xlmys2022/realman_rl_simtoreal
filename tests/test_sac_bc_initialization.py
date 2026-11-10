"""Actor-only transfer and monitoring contracts; never require short-run success."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

AVAILABLE=importlib.util.find_spec('stable_baselines3') is not None
if AVAILABLE:
    import torch
    from torch import nn
    from gymnasium import spaces
    from stable_baselines3 import SAC
    from realman_rl_simtoreal.paths import PROJECT_ROOT
    from realman_rl_simtoreal.experiments.bc_reach import BCNetwork,save_policy,weight_digest
    from realman_rl_simtoreal.experiments.sac_reach import ENV_CONFIG,training_env,make_model,ReachTransform
    from realman_rl_simtoreal.experiments.sac_initialization import initialize_actor_from_bc,make_probe


def entry(name):
    spec=importlib.util.spec_from_file_location(name,PROJECT_ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@unittest.skipUnless(AVAILABLE,'Optional SB3/PyTorch unavailable')
class ActorTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.probe,cls.transform=make_probe(ENV_CONFIG,[51000,51001])
        torch.manual_seed(41)
        # Test-only BC fixture; no production outputs or expert data dependency.
        cls.bc_network=BCNetwork()
        save_policy(cls.root/'bc.pt',cls.bc_network,dict(environment=ENV_CONFIG,
            observation_transform=cls.transform.config,train_seeds=[30000],validation_seeds=[30001]))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def model(self,**overrides):
        env,_=training_env(ENV_CONFIG);self.addCleanup(env.close)
        return make_model(env,41,overrides=overrides or None)

    def test_explicit_parameter_mapping_and_protected_heads(self):
        model=self.model()
        before={name:weight_digest(module) for name,module in
                [('critic',model.critic),('target',model.critic_target),('std',model.actor.log_std)]}
        reference,info=initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)
        for name in ('latent_pi','mu'):
            for key,value in getattr(reference.network,name).state_dict().items():
                self.assertTrue(torch.equal(value,getattr(model.actor,name).state_dict()[key]))
        for name,module in [('critic',model.critic),('target',model.critic_target),('std',model.actor.log_std)]:
            self.assertEqual(before[name],weight_digest(module))
        self.assertEqual(weight_digest(model.critic),weight_digest(model.critic_target))
        self.assertEqual(model.replay_buffer.size(),0)
        self.assertFalse(model.actor.optimizer.state);self.assertFalse(model.critic.optimizer.state)
        self.assertIn('unchanged',info['log_std_initialization'])

    def test_deterministic_actions_match_but_random_samples_need_not(self):
        model=self.model()
        reference,info=initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)
        expected=reference.predict(self.probe)
        actual,_=model.predict(self.transform.encode(self.probe),deterministic=True)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
        random,_=model.predict(self.transform.encode(self.probe),deterministic=False)
        self.assertFalse(np.array_equal(random,expected))
        self.assertLessEqual(info['consistency']['max_abs'],1e-7)

    def test_incompatible_preprocessing_missing_or_wrong_model_is_rejected(self):
        config=copy.deepcopy(self.transform.config);config['scale']['observation'][0]*=2
        with self.assertRaisesRegex(ValueError,'preprocessing'):
            initialize_actor_from_bc(self.model(),self.root/'bc.pt',ReachTransform(config),self.probe)
        with self.assertRaises(FileNotFoundError):
            initialize_actor_from_bc(self.model(),self.root/'missing.pt',self.transform,self.probe)
        torch.save(dict(kind='not_bc'),self.root/'bad.pt')
        with self.assertRaisesRegex(ValueError,'BC model'):
            initialize_actor_from_bc(self.model(),self.root/'bad.pt',self.transform,self.probe)

    def test_same_shape_wrong_activation_extractor_order_or_action_space_rejected(self):
        model=self.model();model.actor.latent_pi[1]=nn.Tanh()
        with self.assertRaises(ValueError):initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)
        model=self.model()
        model.actor.features_extractor.extractors=nn.ModuleDict(reversed(list(model.actor.features_extractor.extractors.items())))
        with self.assertRaises(ValueError):initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)
        model=self.model();model.action_space=spaces.Box(-1,2,(3,),dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'action space'):
            initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)

    def test_initialization_preserves_rng_and_frozen_reference(self):
        model=self.model();state=torch.get_rng_state().clone();numpy=np.random.get_state()
        reference,_=initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        np.testing.assert_array_equal(numpy[1],np.random.get_state()[1])
        self.assertTrue(all(not p.requires_grad for p in reference.network.parameters()))

    def test_both_critics_and_log_std_remain_trainable_via_stock_sac(self):
        model=self.model(learning_starts=4,batch_size=8,buffer_size=64)
        initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)
        before=[weight_digest(q) for q in model.critic.q_networks]
        model.learn(total_timesteps=12)
        self.assertGreater(model._n_updates,0)
        self.assertTrue(all(weight_digest(q)!=h for q,h in zip(model.critic.q_networks,before)))
        self.assertTrue(all(p.requires_grad for p in model.actor.log_std.parameters()))
        with self.assertRaisesRegex(ValueError,'fresh untrained'):
            initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)

    def test_initial_actor_save_load_actions_unchanged(self):
        model=self.model();initialize_actor_from_bc(model,self.root/'bc.pt',self.transform,self.probe)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'model.zip';model.save(path)
            loaded=SAC.load(path,device='cpu')
            a,_=model.predict(self.transform.encode(self.probe),deterministic=True)
            b,_=loaded.predict(self.transform.encode(self.probe),deterministic=True)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(loaded.num_timesteps,0);self.assertEqual(loaded._n_updates,0)


MONITORING=AVAILABLE and importlib.util.find_spec('tensorboard') is not None and importlib.util.find_spec('matplotlib') is not None


@unittest.skipUnless(MONITORING,'Optional TensorBoard/Matplotlib unavailable')
class MonitoringIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        probe,transform=make_probe(ENV_CONFIG,[52000])
        torch.manual_seed(53)
        save_policy(cls.root/'bc.pt',BCNetwork(),dict(environment=ENV_CONFIG,
            observation_transform=transform.config,train_seeds=[30000],validation_seeds=[30001]))
        cls.run_dir=cls.root/'run'
        cls.result=entry('train_sac_reach').train(cls.run_dir,timesteps=128,seed=53,eval_every=64,
            checkpoint_every=32,eval_episodes=1,eval_seed_start=52000,init_mode='bc_actor',
            bc_model=cls.root/'bc.pt',tensorboard=True,environment=dict(ENV_CONFIG,max_episode_steps=2))

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_separate_checkpoint_evaluation_schedule_and_unchanged_hyperparameters(self):
        self.assertEqual(self.result['status'],'completed',self.result['error'])
        self.assertEqual(self.result['total_timesteps'],128)
        self.assertEqual(self.result['buffer_size'],128);self.assertEqual(self.result['n_updates'],28)
        self.assertEqual([r['step'] for r in self.result['checkpoints']],[0,32,64,96,128])
        rows=json.loads((self.run_dir/'evaluations.json').read_text())
        self.assertEqual([r['step'] for r in rows],[0,64,128])
        self.assertTrue(self.result['actor_updated'] and self.result['critic_updated'])
        self.assertLessEqual(rows[0]['bc_action_deviation']['max_abs'],1e-7)

    def test_step_zero_and_trained_checkpoints_are_distinct(self):
        one=SAC.load(self.run_dir/'initial/model.zip',device='cpu')
        two=SAC.load(self.run_dir/'final/model.zip',device='cpu')
        self.assertEqual(one.num_timesteps,0);self.assertEqual(two.num_timesteps,128)
        self.assertNotEqual(weight_digest(one.actor),weight_digest(two.actor))
        self.assertEqual(json.loads((self.run_dir/'initial/state.json').read_text())['replay_size'],0)

    def test_tensorboard_real_evaluation_metrics_match_saved_json(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events=EventAccumulator(str(self.run_dir/'sb3'));events.Reload()
        actual={e.step:e.value for e in events.Scalars('evaluation/success_rate')}
        for row in json.loads((self.run_dir/'evaluations.json').read_text()):
            self.assertAlmostEqual(actual[row['step']],row['summary']['success_rate'])
        for tag in ['train/actor_loss','train/critic_loss','train/ent_coef','monitor/total_timesteps',
                    'evaluation/final_distance_mm','evaluation/timeouts','evaluation/safety_failures',
                    'training/action_rejection_fraction','bc/fixed_probe_mae','bc/evaluation_states_mae']:
            self.assertIn(tag,events.Tags()['scalars'])

    def test_png_data_matches_saved_evaluation_not_training_success(self):
        plot=entry('plot_sac_training');curves=plot.load_curves(self.run_dir)
        expected=json.loads((self.run_dir/'evaluations.json').read_text())
        self.assertEqual(curves['evaluation_success_rate'],[[r['step'],r['summary']['success_rate']] for r in expected])
        result=plot.export_plots(self.run_dir,self.root/'plots')
        self.assertEqual(len(result['generated']),4)
        for name in result['generated']:
            self.assertTrue((self.root/'plots'/name).read_bytes().startswith(b'\x89PNG'))

    def test_missing_evaluation_never_fabricates_success_curve(self):
        folder=self.root/'training_only';folder.mkdir()
        (folder/'diagnostics.jsonl').write_text(json.dumps(dict(step=1,**{'train/actor_loss':2.}))+'\n')
        result=entry('plot_sac_training').export_plots(folder,self.root/'training_only_plots')
        self.assertFalse((self.root/'training_only_plots/evaluation_success.png').exists())
        self.assertIn('evaluation_success.png',[r['file'] for r in result['skipped']])
        self.assertEqual(result['curves']['evaluation_success_rate'],[])


if __name__=='__main__':unittest.main()
