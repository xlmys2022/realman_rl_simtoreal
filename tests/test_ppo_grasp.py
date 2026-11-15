"""PPO-only grasp actions, discounted shaping, checkpoints and independent inference."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.envs.grasp_lift_env import GraspLiftEnv
from realman_rl_simtoreal.rl.ppo_grasp import (
    ENCODING,REWARD,PPOGraspEnv,encode,potential,shaped_reward,environment_config,
    aperture_score,make_training_env,load_checkpoint,evaluate_model,parameter_hashes,optimizer_steps)


class GraspPPOContracts(unittest.TestCase):
    def test_completion_bonus_dominates_indefinite_dense_reward(self):
        maximum=sum(REWARD[k] for k in ['dense_reach','dense_aperture','dense_grip','dense_height','dense_hold'])
        self.assertGreater(REWARD['success_bonus'],maximum/(1-REWARD['gamma']))

    def test_aperture_reward_encourages_room_to_approach_then_a_pinch(self):
        def score(distance,width):
            return aperture_score(dict(grasp_distance_m=distance,actual_width_m=width),REWARD)
        self.assertGreater(score(.02,.06),score(.02,.02))
        self.assertGreater(score(.001,.02),score(.001,.06))
        old={key:value for key,value in REWARD.items() if key not in
             ('pregrasp_width_m','close_distance_m','open_distance_m')}
        info=dict(grasp_distance_m=.01,actual_width_m=.04)
        expected=(1-np.tanh(.01/old['reach_scale_m']))*np.exp(-((.04-.02)/old['width_scale_m'])**2)
        self.assertAlmostEqual(aperture_score(info,old),expected)

    def test_equal_lift_and_grip_are_rewarded_less_away_from_the_goal(self):
        cfg=environment_config('pinch')
        with GraspLiftEnv(**cfg) as env:
            _,info=env.reset(seed=12)
            info.update(contact_forces_n=[.1,.1,0],actual_width_m=.02,support_clearance_m=.005,
                        object_position_m=info['object_goal_m'],grasp_position_m=info['object_goal_m'],grasp_distance_m=0.)
            away=copy.deepcopy(info)
            delta=env.support_rotation[:,0]*.01
            away['object_position_m']=(np.array(info['object_position_m'])+delta).tolist()
            away['grasp_position_m']=(np.array(info['grasp_position_m'])+delta).tolist()
            self.assertGreater(potential(info,cfg),potential(away,cfg))
            a,_=shaped_reward(0,info,False,cfg);b,_=shaped_reward(0,away,False,cfg)
            self.assertGreater(a,b)

    def test_pinch_curriculum_starts_open_and_supported_without_a_preset_grasp(self):
        with GraspLiftEnv(**environment_config('pinch')) as env:
            _,info=env.reset(seed=12)
            self.assertAlmostEqual(info['actual_width_m'],.022,delta=1e-6)
            self.assertFalse(info['bilateral_contact']);self.assertFalse(info['is_success'])
            self.assertEqual(info['hold_time_s'],0.)
            self.assertLess(abs(info['support_clearance_m']),1e-8)
            self.assertEqual(env.data.time,0.)

    def test_random_height_curriculum_is_seeded_and_changes_only_the_goal(self):
        cfg=environment_config('full')
        with PPOGraspEnv(cfg,randomize_height=True,randomize_approach=True) as first,PPOGraspEnv(cfg,randomize_height=True,randomize_approach=True) as second:
            goals=[];distances=[]
            for seed in (12,None,None):
                a,ai=first.reset(seed=seed);b,bi=second.reset(seed=seed)
                np.testing.assert_array_equal(a,b)
                height=ai['training_goal_height_m'];goals.append(height)
                distance=ai['training_approach_distance_m'];distances.append(distance)
                self.assertEqual(distance,bi['training_approach_distance_m'])
                self.assertGreaterEqual(distance,.005);self.assertLessEqual(distance,.02)
                self.assertAlmostEqual(np.linalg.norm(np.array(ai['grasp_reference_m'])-ai['pregrasp_reference_m']),distance)
                self.assertEqual(height,bi['training_goal_height_m'])
                self.assertGreaterEqual(height,.005);self.assertLessEqual(height,.03)
                self.assertAlmostEqual(first.episode_config['height_tolerance']/height,1/15)
                self.assertEqual(first.task_config,cfg)
                self.assertFalse(ai['bilateral_contact']);self.assertFalse(ai['is_success'])
                self.assertEqual(ai['hold_time_s'],0.)
                self.assertLess(abs(ai['support_clearance_m']),1e-8)
                self.assertEqual(first.unwrapped.data.time,0.)
                self.assertAlmostEqual(ai['actual_width_m'],cfg['initial_width'],delta=1e-6)
            self.assertGreater(len(set(goals)),1)
            self.assertGreater(len(set(distances)),1)

    def test_fixed_encoding_scales_millimetres_and_preserves_input(self):
        with GraspLiftEnv() as env:
            obs,_=env.reset(seed=2027)
            before=obs.copy();value=encode(obs)
            np.testing.assert_array_equal(obs,before)
            self.assertEqual(value.dtype,np.float32);self.assertEqual(value.shape,(41,))
            np.testing.assert_allclose(value[12:21],np.arcsinh(before[12:21]*50),atol=1e-7)
            np.testing.assert_array_equal(value[:12],before[:12])
            batch=encode(np.stack([obs,obs]))
            np.testing.assert_array_equal(batch[0],value)

    def test_training_wrappers_only_change_observations_and_rewards(self):
        cfg=environment_config('easy')
        with GraspLiftEnv(**cfg) as raw,PPOGraspEnv(cfg) as wrapped:
            obs,a=raw.reset(seed=12);encoded,b=wrapped.reset(seed=12)
            np.testing.assert_array_equal(encoded,encode(obs))
            initial=potential(a,cfg)
            shaping=[]
            for step in range(10):
                action=np.array([.05,0,.05,-.2],np.float32)
                obs,base,t,tr,a=raw.step(action)
                encoded,reward,u,ur,b=wrapped.step(action)
                self.assertEqual((t,tr),(u,ur))
                np.testing.assert_array_equal(raw.data.qpos,wrapped.unwrapped.data.qpos)
                np.testing.assert_array_equal(raw.data.qvel,wrapped.unwrapped.data.qvel)
                np.testing.assert_array_equal(raw.data.ctrl,wrapped.unwrapped.data.ctrl)
                np.testing.assert_array_equal(encoded,encode(obs))
                self.assertEqual(base,b['environment_reward'])
                self.assertEqual(sum(b['reward_components'].values()),reward)
                shaping.append(b['reward_components']['potential_change'])
                if t or tr:break
            gamma=REWARD['gamma'];n=len(shaping)
            expected=(0 if t else gamma**n*potential(a,cfg))-initial
            self.assertAlmostEqual(sum(gamma**i*s for i,s in enumerate(shaping)),expected,places=10)

    def test_timeout_retains_potential_and_real_failure_zeroes_it(self):
        cfg=environment_config('easy');cfg['max_episode_steps']=1
        with PPOGraspEnv(cfg) as env:
            _,reset=env.reset(seed=12);start=potential(reset,cfg)
            _,reward,t,tr,info=env.step([0,0,0,0])
            self.assertFalse(t);self.assertTrue(tr)
            self.assertAlmostEqual(info['reward_components']['potential_change'],REWARD['gamma']*potential(info,cfg)-start)
            env.reset(seed=12)
            env.unwrapped.data.qfrc_applied[0]=.1
            _,reward,t,tr,info=env.step([0,0,0,0])
            self.assertTrue(t);self.assertFalse(tr)
            self.assertEqual(info['reward_components']['potential_change'],-start)
            self.assertEqual(info['reward_components']['safety_penalty'],-25.)

    def test_private_kinematics_optimization_matches_full_forward_geometry(self):
        with GraspLiftEnv() as env:
            env.reset(seed=12)
            full=mujoco.MjData(env.model);fast=mujoco.MjData(env.model)
            for shift in (0,.001,-.001):
                full.qpos[:]=env.data.qpos
                full.qpos[env.kinematics.qadr[0]]+=shift
                fast.qpos[:]=full.qpos
                mujoco.mj_forward(env.model,full);mujoco.mj_kinematics(env.model,fast)
                np.testing.assert_array_equal(full.xpos,fast.xpos)
                np.testing.assert_array_equal(full.xmat,fast.xmat)
                np.testing.assert_array_equal(full.site_xpos,fast.site_xpos)

    def test_two_subprocesses_return_four_action_transitions(self):
        env=make_training_env(dict(environment_config('easy'),max_episode_steps=2),2,2027)
        try:
            obs=env.reset();self.assertEqual(obs.shape,(2,41))
            obs,reward,done,info=env.step(np.zeros((2,4),np.float32))
            self.assertTrue(np.isfinite(obs).all());self.assertTrue(np.isfinite(reward).all())
            self.assertTrue(all('ppo_reward' in i for i in info))
            self.assertTrue(all(i['physics_steps']==50 for i in info))
        finally:env.close()


class GraspPPORun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory()
        cls.folder=Path(cls.directory.name)/'ppo'
        spec=importlib.util.spec_from_file_location('train_grasp',PROJECT_ROOT/'scripts/train/train_ppo_grasp_lift.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.trainer=module
        args=SimpleNamespace(seed=2027,timesteps=64,n_envs=1,n_steps=32,batch_size=16,epochs=1,
            eval_every=32,eval_episodes=1,run_dir=cls.folder,stage='easy',episode_steps=8,
            init_model=None,eval_seed_start=240000,tensorboard=False)
        with patch('realman_rl_simtoreal.evaluation.grasp_lift.GraspLiftController',side_effect=AssertionError('No staged actions')):
            cls.summary=module.train(args)
        cls.model,cls.config=load_checkpoint(cls.folder/'final/model.zip')

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_complete_rollouts_update_actor_and_critic_without_demonstrations(self):
        s=self.summary
        self.assertEqual(s['status'],'completed',s['error'])
        self.assertTrue(s['actor_updated']);self.assertTrue(s['critic_updated'])
        self.assertEqual(s['actual_timesteps'],64);self.assertEqual(s['epoch_updates'],2)
        self.assertEqual(s['optimizer_steps'],4)
        self.assertEqual(s['training']['collected_rollouts'],2)
        self.assertTrue(self.config['pure_ppo']);self.assertFalse(self.config['demonstrations'])
        self.assertFalse(self.config['staged_actions']);self.assertEqual(self.config['initialization'],'scratch')

    def test_saved_parameters_and_four_actions_round_trip(self):
        self.assertEqual(parameter_hashes(self.model),self.summary['final_parameter_hashes'])
        with GraspLiftEnv(**self.config['environment']) as env:
            obs,_=env.reset(seed=2027)
            action,_=self.model.predict(encode(obs),deterministic=True)
            self.assertTrue(env.action_space.contains(action));self.assertEqual(action.shape,(4,))
        state=json.loads((self.folder/'final/state.json').read_text())
        self.assertTrue(state['completed_update_stage']);self.assertEqual(state['optimizer_steps'],4)

    def test_evaluation_is_read_only_and_never_constructs_the_staged_controller(self):
        before=parameter_hashes(self.model);counters=(self.model.num_timesteps,optimizer_steps(self.model))
        with patch('realman_rl_simtoreal.evaluation.grasp_lift.GraspLiftController',side_effect=AssertionError('No fallback')):
            result=evaluate_model(self.model,self.config,[250000])
        self.assertEqual(parameter_hashes(self.model),before)
        self.assertEqual((self.model.num_timesteps,optimizer_steps(self.model)),counters)
        self.assertTrue(result['parameters_unchanged'])
        self.assertTrue(all(r['controller_phase']=='ppo' and r['controller_centering_offset_m'] is None
                            for r in result['episodes'][0]['records']))
        for record in result['episodes'][0]['records']:
            np.testing.assert_array_equal(np.clip(record['raw_action'],-1,1),record['action'])

    def test_changed_encoding_and_reward_contracts_are_rejected(self):
        from realman_rl_simtoreal.rl.ppo_grasp import validate_model
        for key in ('encoding','reward'):
            config=copy.deepcopy(self.config)
            if key=='encoding':config[key]['name']='different'
            else:config[key]['gamma']=.99
            with self.subTest(key=key),self.assertRaises(ValueError):validate_model(self.model,config)

    def test_explicit_transfer_bias_reset_preserves_the_other_three_actions(self):
        args=SimpleNamespace(seed=2027,timesteps=32,n_envs=1,n_steps=32,batch_size=16,epochs=1,
            eval_every=32,eval_episodes=1,run_dir=Path(self.directory.name)/'transfer',stage='easy',episode_steps=8,
            init_model=self.folder/'final/model.zip',eval_seed_start=240000,tensorboard=False,
            reset_gripper_bias=True,gripper_mean=.5,reset_exploration=True,xyz_std=.02,gripper_std=.6,
            learning_rate=.00003,approach_distance=.012)
        with patch('realman_rl_simtoreal.evaluation.grasp_lift.GraspLiftController',side_effect=AssertionError('No expert actions')):
            summary=self.trainer.train(args)
        initial,config=load_checkpoint(args.run_dir/'initial/model.zip')
        for name,tensor in initial.policy.state_dict().items():
            if name=='log_std':continue
            if name=='action_net.bias':
                torch.testing.assert_close(tensor[:3],self.model.policy.state_dict()[name][:3],rtol=0,atol=0)
                self.assertEqual(float(tensor[3]),.5)
            else:torch.testing.assert_close(tensor,self.model.policy.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(summary['status'],'completed')
        self.assertEqual(config['environment']['approach_distance'],.012)
        self.assertEqual(config['ppo']['learning_rate'],.00003)
        self.assertEqual(config['transfer']['transformed_parameters'],['log_std','action_net.bias'])


if __name__=='__main__':unittest.main()
