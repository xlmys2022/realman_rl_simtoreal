"""Horizontal tabletop geometry, exact zero reset, real PPO and read-only inference."""
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
from realman_rl_simtoreal.envs.grasp.tabletop_pick_env import TabletopPickEnv
from realman_rl_simtoreal.rl.ppo_tabletop import (
    PPOTabletopEnv,ENCODING,REWARD,encode,environment_config,potential,
    load_checkpoint,evaluate_model,make_training_env)
from realman_rl_simtoreal.rl.ppo_grasp import parameter_hashes,optimizer_steps


class TabletopPhysics(unittest.TestCase):
    def test_flat_desk_and_cube_share_the_base_plane_without_object_constraints(self):
        with TabletopPickEnv() as env:
            _,info=env.reset(seed=2027);m=env.model
            self.assertAlmostEqual(m.body_pos[m.body('base_link').id,2],env.table_height)
            np.testing.assert_array_equal(env.support_normal,[0,0,1])
            np.testing.assert_array_equal(env.data.xmat[env.support_body].reshape(3,3),np.eye(3))
            self.assertAlmostEqual(info['object_position_m'][2],env.table_height+.01)
            self.assertLess(abs(info['support_clearance_m']),1e-12)
            self.assertEqual(m.jnt_type[env.object_joint],mujoco.mjtJoint.mjJNT_FREE)
            self.assertEqual(m.body_gravcomp[env.object_body],0)
            self.assertTrue(np.all(m.eq_type==mujoco.mjtEq.mjEQ_JOINT))

    def test_full_reset_is_exact_zero_at_time_zero_and_desk_never_follows_the_cube(self):
        with TabletopPickEnv(object_radius=.02) as env:
            table=env.model.body_pos[env.support_body].copy();positions=[]
            for seed in (2027,2028):
                obs,info=env.reset(seed=seed)
                np.testing.assert_array_equal(env.data.qpos[env.kinematics.qadr],np.zeros(6))
                np.testing.assert_array_equal(env.data.ctrl[env.actuator_ids],np.zeros(6))
                np.testing.assert_array_equal(env.model.body_pos[env.support_body],table)
                self.assertEqual(env.data.time,0.);self.assertEqual(info['hold_time_s'],0.)
                self.assertFalse(info['bilateral_contact']);self.assertGreater(info['grasp_distance_m'],1.)
                self.assertEqual(obs.shape,(56,));self.assertEqual(env.action_space.shape,(7,))
                positions.append(info['object_position_m'])
            self.assertNotEqual(positions[0],positions[1])

    def test_seven_actions_advance_one_clock_and_record_real_support_thickness(self):
        with TabletopPickEnv(record_physics=True) as env:
            env.reset(seed=2027)
            _,_,t,tr,info=env.step([0,0,0,0,0,0,1])
            self.assertFalse(t);self.assertFalse(tr);self.assertEqual(info['physics_steps'],50)
            self.assertAlmostEqual(env.data.time,.05)
            self.assertEqual(len(env.recorder.rows),50)
            self.assertAlmostEqual(env.recorder.support_half_height,.025)
            self.assertLess(abs(env.recorder.last['support_clearance_m']),2e-6)
            self.assertGreater(info['contact_forces_n'][2],.05)
            self.assertFalse(np.any(env.data.qfrc_applied));self.assertFalse(np.any(env.data.xfrc_applied))

    def test_collision_shield_only_holds_joint_reference_and_keeps_neural_jaw_action(self):
        with TabletopPickEnv() as env:
            env.reset(seed=2027);before=env.data.ctrl[env.actuator_ids].copy()
            with patch.object(env,'_safe_reference',return_value=(False,'robot_support')):
                _,_,_,_,info=env.step(np.ones(7,np.float32))
            np.testing.assert_array_equal(env.data.ctrl[env.actuator_ids],before)
            self.assertTrue(info['action_rejected']);self.assertEqual(info['rejection_detail'],'robot_support')
            self.assertGreater(info['requested_width_m'],.06)
            self.assertAlmostEqual(env.data.time,.05)

    def test_shield_checks_deep_pad_cube_contact_as_well_as_unexpected_contacts(self):
        with TabletopPickEnv(**environment_config('pinch')) as env:
            env.reset(seed=2027)
            direction=env.data.site_xpos[env.gripper.left]-env.data.site_xpos[env.gripper.right]
            direction/=np.linalg.norm(direction)
            solved=env._solve_goal(env._initial_object+.003*direction,env.grasp_q)
            self.assertTrue(solved.success)
            actual=env.data.qpos.copy();command=env.data.ctrl.copy()
            safe,category=env._safe_reference(solved.q)
            self.assertFalse(safe);self.assertTrue(category.endswith('_pad_target'))
            np.testing.assert_array_equal(env.data.qpos,actual)
            np.testing.assert_array_equal(env.data.ctrl,command)

    def test_jaw_commands_keep_a_recorded_margin_from_mechanical_end_stops(self):
        with TabletopPickEnv() as env:
            env.reset(seed=2027)
            _,_,_,_,info=env.step([0,0,0,0,0,0,-1])
            self.assertEqual(info['requested_width_m'],0.)
            self.assertEqual(info['commanded_width_m'],.002)
            self.assertTrue(info['jaw_command_limited'])
            _,_,_,_,info=env.step([0,0,0,0,0,0,1])
            self.assertAlmostEqual(info['requested_width_m']-info['commanded_width_m'],.002)

    def test_invalid_pose_and_actions_are_rejected_before_integration(self):
        for q in ([0]*5,[0,0,np.nan,0,0,0],[200,0,0,0,0,0]):
            with self.subTest(q=q),self.assertRaises(ValueError):TabletopPickEnv(initial_q_deg=q)
        with TabletopPickEnv() as env:
            env.reset(seed=2027)
            for action in (np.zeros(4),[0]*6+[np.nan],[2]*7):
                with self.assertRaises(ValueError):env.step(action)
                self.assertEqual(env.data.time,0.)

    def test_actual_contact_updates_only_a_grasp_relative_goal_feature(self):
        with TabletopPickEnv(**environment_config('pinch')) as env:
            env.reset(seed=2027);reference=env.data.ctrl[env.actuator_ids].copy()
            for _ in range(20):
                _,_,t,tr,info=env.step(np.r_[np.rad2deg(reference),-.75].astype(np.float32))
                if info['grasp_relative_lift_goal']:break
                self.assertFalse(t or tr)
            self.assertTrue(info['bilateral_contact']);self.assertTrue(info['grasp_relative_lift_goal'])
            np.testing.assert_allclose(env.data.ctrl[env.actuator_ids],reference,rtol=0,atol=2e-7)
            expected=env.object_goal+np.asarray(info['grasp_position_m'])-info['object_position_m']+[0,0,.002]
            np.testing.assert_allclose(env.lift_reference,expected,atol=1e-10,rtol=0)
            p,r=env.kinematics.fk(env.lift_q)
            np.testing.assert_allclose(p+r@env._tcp_offset,expected,atol=1e-6,rtol=0)

    def test_held_object_motion_is_checked_by_physics_instead_of_a_frozen_cube_probe(self):
        with TabletopPickEnv(**environment_config('pinch')) as env:
            env.reset(seed=2027)
            reference=env.data.ctrl[env.actuator_ids].copy()
            for _ in range(20):
                _,_,t,tr,info=env.step(np.r_[np.rad2deg(reference),-.75].astype(np.float32))
                if info['bilateral_contact']:break
                self.assertFalse(t or tr)
            self.assertTrue(info['bilateral_contact'])
            direction=env.data.site_xpos[env.gripper.left]-env.data.site_xpos[env.gripper.right]
            direction/=np.linalg.norm(direction)
            tcp=env.data.site_xpos[env.tcp_site].copy()
            solved=env._solve_goal(tcp+.001*direction,reference)
            self.assertTrue(solved.success)
            safe,category=env._safe_reference(solved.q)
            self.assertTrue(safe,category)
            # The real policy checker remains active on the same candidate.
            with self.assertRaisesRegex(ValueError,'pad_target'):
                env.contact_policy.check(env._candidate)

    def test_fixed_encoding_and_wrapper_preserve_the_raw_dynamics(self):
        cfg=environment_config('pinch')
        with TabletopPickEnv(**cfg) as raw,PPOTabletopEnv(cfg) as wrapped:
            obs,a=raw.reset(seed=2027);coded,b=wrapped.reset(seed=2027);original=obs.copy()
            np.testing.assert_array_equal(coded,encode(obs));np.testing.assert_array_equal(obs,original)
            phi=potential(a,cfg);changes=[]
            for i in range(5):
                action=np.r_[np.rad2deg(raw.data.ctrl[raw.actuator_ids]),-.75].astype(np.float32)
                obs,base,t,tr,a=raw.step(action);coded,reward,u,ur,b=wrapped.step(action)
                np.testing.assert_array_equal(raw.data.qpos,wrapped.unwrapped.data.qpos)
                np.testing.assert_array_equal(raw.data.qvel,wrapped.unwrapped.data.qvel)
                self.assertEqual((t,tr),(u,ur));self.assertEqual(base,b['environment_reward'])
                self.assertEqual(sum(b['reward_components'].values()),reward)
                changes.append(b['reward_components']['potential_change']/REWARD['scale'])
            gamma=REWARD['gamma']
            self.assertAlmostEqual(sum(gamma**i*s for i,s in enumerate(changes)),gamma**5*potential(a,cfg)-phi,places=10)

    def test_absolute_joint_reference_does_not_accumulate_and_is_rate_limited(self):
        with TabletopPickEnv() as env:
            env.reset(seed=2027);action=np.array([np.rad2deg(.002),0,0,0,0,0,1],np.float32)
            for _ in range(5):env.step(action)
            self.assertAlmostEqual(env.data.ctrl[env.actuator_ids[0]],float(np.deg2rad(float(action[0]))),places=10)
            action[1]=60.
            before=env.data.ctrl[env.actuator_ids].copy()
            _,_,_,_,info=env.step(action)
            delta=env.data.ctrl[env.actuator_ids]-before
            self.assertLessEqual(max(abs(delta)),np.deg2rad(.5)+1e-12)
            self.assertAlmostEqual(delta[1],np.deg2rad(.5))
            self.assertAlmostEqual(info['requested_joint_position_rad'][1],np.pi/3)

    def test_waypoint_changes_goal_features_while_zero_reset_and_controls_stay_exact(self):
        with TabletopPickEnv(approach_waypoint_height=.06) as env:
            obs,info=env.reset(seed=2027)
            self.assertFalse(info['approach_complete'])
            self.assertGreater(np.linalg.norm(env.reach_q-env.grasp_q),.1)
            np.testing.assert_allclose(obs[44:50],env.reach_q/.1,atol=2e-6,rtol=0)
            np.testing.assert_array_equal(env.data.qpos[env.kinematics.qadr],np.zeros(6))
            np.testing.assert_array_equal(env.data.ctrl[env.actuator_ids],np.zeros(6))
            self.assertEqual(env.data.time,0.)

    def test_approach_corridor_allows_height_error_but_requires_horizontal_alignment(self):
        with TabletopPickEnv() as probe:
            probe.reset(seed=2027)
            configurations=[]
            for offset,ready in (([.006,0,0],False),([0,0,-.01],True)):
                solved=probe._solve_goal(probe._initial_object+[0,0,.06]+np.asarray(offset),probe.pregrasp_q)
                self.assertTrue(solved.success);configurations.append((np.rad2deg(solved.q).tolist(),ready))
        for q,ready in configurations:
            with TabletopPickEnv(initial_q_deg=q,approach_waypoint_height=.06,approach_waypoint_tolerance=.015) as env:
                _,info=env.reset(seed=2027)
                self.assertEqual(info['approach_complete'],ready)
                self.assertEqual(env.data.time,0.)
                self.assertGreater(info['grasp_position_m'][2]-info['initial_object_position_m'][2],.045)

    def test_curriculum_threshold_can_change_without_changing_policy_goal_features(self):
        with TabletopPickEnv(**environment_config('approach',goal_height=.03)) as easy,TabletopPickEnv(**environment_config('lift',goal_height=.03,hold_seconds=.2)) as full:
            a,ia=easy.reset(seed=2027);b,ib=full.reset(seed=2027)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(easy.lift_q,full.lift_q)
            self.assertEqual(ia['goal_height_m'],.03);self.assertEqual(ib['goal_height_m'],.03)
            self.assertEqual(ia['required_lift_height_m'],.005)
            self.assertEqual(ib['required_lift_height_m'],.03)

    def test_waypoint_completion_requires_actual_motion_and_does_not_command_the_next_goal(self):
        cfg=environment_config('approach',pregrasp_height=.08,approach_waypoint_height=.06)
        with TabletopPickEnv(**cfg) as env:
            _,info=env.reset(seed=2027);self.assertFalse(info['approach_complete'])
            target=env.reach_q.copy()
            with patch.object(env.kinematics,'solve_ik',side_effect=AssertionError('No runtime IK action')):
                for _ in range(25):
                    _,_,t,tr,info=env.step(np.r_[np.rad2deg(target),1].astype(np.float32))
                    self.assertFalse(t or tr)
                    if info['approach_complete']:break
            self.assertTrue(info['approach_complete'])
            self.assertGreater(env.data.time,0.)
            np.testing.assert_array_equal(env.reach_q,env.grasp_q)
            self.assertGreater(np.linalg.norm(env.data.ctrl[env.actuator_ids]-env.grasp_q),.1)

    def test_action_regularizer_uses_the_observed_goal_even_when_physics_completes_a_waypoint(self):
        cfg=environment_config('approach',pregrasp_height=.08,approach_waypoint_height=.06)
        with PPOTabletopEnv(cfg) as env:
            _,info=env.reset(seed=2027);self.assertFalse(info['approach_complete'])
            action=np.r_[np.rad2deg(info['reach_goal_joint_rad']),1].astype(np.float32)
            for _ in range(25):
                _,_,t,tr,info=env.step(action)
                self.assertFalse(t or tr)
                self.assertGreater(info['reward_components']['action_target'],-1e-10)
                if info['approach_complete']:break
            self.assertTrue(info['approach_complete'])
            # Repeating the old goal after the observation changes is penalized.
            _,_,_,_,info=env.step(action)
            self.assertLess(info['reward_components']['action_target'],-.01)

    def test_joint_reference_uses_one_fraction_to_preserve_the_commanded_path(self):
        with TabletopPickEnv() as env:
            env.reset(seed=2027)
            requested=np.array([0,20,90,0,70,0,1],np.float32)
            before=env.data.ctrl[env.actuator_ids].copy()
            _,_,_,_,info=env.step(requested)
            delta=env.data.ctrl[env.actuator_ids]-before
            np.testing.assert_allclose(delta,np.deg2rad(requested[:6].astype(float))/180,rtol=0,atol=1e-12)
            self.assertEqual(info['physics_steps'],50)

    def test_subprocesses_expose_all_seven_policy_actions(self):
        env=make_training_env(environment_config('pinch',max_episode_steps=2),2,2027)
        try:
            self.assertEqual(env.reset().shape,(2,56))
            obs,reward,done,infos=env.step(np.zeros((2,7),np.float32))
            self.assertTrue(np.isfinite(obs).all());self.assertTrue(np.isfinite(reward).all())
            self.assertTrue(all(i['physics_steps']==50 for i in infos))
        finally:env.close()


class TabletopPPO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory();cls.folder=Path(cls.directory.name)/'run'
        spec=importlib.util.spec_from_file_location('train_tabletop',PROJECT_ROOT/'scripts/train/train_ppo_tabletop.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        args=SimpleNamespace(seed=2027,timesteps=64,n_envs=1,n_steps=32,batch_size=16,epochs=1,
            learning_rate=.0003,joint_std=.1,gripper_std=.3,gripper_mean=0.,init_model=None,
            reset_exploration=False,run_dir=cls.folder,stage='full',max_episode_steps=8,
            eval_seed_start=340000,eval_every=32,eval_episodes=1,tensorboard=False,
            sde_sample_freq=1,reference_smoothing=.05)
        cls.train=staticmethod(module.train);cls.base_args=copy.deepcopy(args)
        with patch('realman_rl_simtoreal.evaluation.grasp_lift.GraspLiftController',side_effect=AssertionError('No teacher/fallback')):
            cls.summary=module.train(args)
        cls.model,cls.config=load_checkpoint(cls.folder/'final/model.zip')

    @classmethod
    def tearDownClass(cls):cls.directory.cleanup()

    def test_real_ppo_updates_actor_and_critic_and_saves_completed_state(self):
        s=self.summary
        self.assertEqual(s['status'],'completed',s['error']);self.assertTrue(s['actor_updated']);self.assertTrue(s['critic_updated'])
        self.assertEqual(s['actual_timesteps'],64)
        self.assertGreater(s['optimizer_steps'],0);self.assertLessEqual(s['optimizer_steps'],4)
        self.assertEqual(s['optimizer_steps'],optimizer_steps(self.model))
        self.assertEqual(s['training']['collected_rollouts'],2)
        self.assertEqual(parameter_hashes(self.model),s['final_parameter_hashes'])
        state=json.loads((self.folder/'final/state.json').read_text())
        self.assertTrue(state['completed_update_stage'])
        initial=json.loads((self.folder/'initial/state.json').read_text());self.assertFalse(initial['completed_update_stage'])
        self.assertFalse(self.config['demonstrations']);self.assertFalse(self.config['scripted_actions']);self.assertFalse(self.config['ik_actions'])

    def test_inference_is_immutable_from_exact_zero_with_seven_neural_outputs(self):
        before=parameter_hashes(self.model);counters=(self.model.num_timesteps,optimizer_steps(self.model))
        with patch('realman_rl_simtoreal.evaluation.grasp_lift.GraspLiftController',side_effect=AssertionError('No scripted actions')):
            result=evaluate_model(self.model,self.config,[350000])
        self.assertEqual(parameter_hashes(self.model),before);self.assertEqual((self.model.num_timesteps,optimizer_steps(self.model)),counters)
        self.assertTrue(result['parameters_unchanged']);self.assertFalse(result['staged_controller_used'])
        for row in result['episodes'][0]['records']:
            np.testing.assert_array_equal(row['initial_q_deg'],np.zeros(6))
            self.assertEqual(row['controller_phase'],'ppo');self.assertEqual(len(row['action']),7)
            np.testing.assert_array_equal(np.clip(row['raw_action'],self.model.action_space.low,self.model.action_space.high),row['action'])

    def test_incompatible_contracts_are_rejected(self):
        from realman_rl_simtoreal.rl.ppo_tabletop import validate_model
        cfg=copy.deepcopy(self.config);cfg['encoding']=dict(ENCODING,name='other')
        with self.assertRaises(ValueError):validate_model(self.model,cfg)

    def test_transfer_inherits_noise_refresh_and_actuator_filter_without_cli_overrides(self):
        args=copy.deepcopy(self.base_args)
        args.init_model=self.folder/'final/model.zip';args.run_dir=Path(self.directory.name)/'transfer'
        args.timesteps=32;args.sde_sample_freq=None;args.reference_smoothing=None
        result=self.train(args)
        self.assertEqual(result['status'],'completed',result['error'])
        model,config=load_checkpoint(args.run_dir/'final/model.zip')
        self.assertEqual(model.sde_sample_freq,1)
        self.assertEqual(config['environment']['reference_smoothing'],.05)
        self.assertEqual(config['transfer']['source_parameter_hashes'],parameter_hashes(self.model))
        self.assertTrue(config['transfer']['optimizer_reset']);self.assertEqual(model.num_timesteps,32)

    def test_persistent_exploration_does_not_echo_contact_feature_jitter(self):
        self.assertTrue(self.model.use_sde);self.assertTrue(self.config['smooth_exploration'])
        self.model.policy.reset_noise(1)
        first=np.zeros(56,np.float32);second=first.copy();second[30:33]=2.
        def residual(obs):
            with torch.no_grad():
                tensor,_=self.model.policy.obs_to_tensor(obs)
                distribution=self.model.policy.get_distribution(tensor)
                return (distribution.get_actions(deterministic=False)-distribution.get_actions(deterministic=True)).cpu().numpy()
        a=residual(first);b=residual(second)
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
        self.model.policy.reset_noise(1)
        self.assertFalse(np.allclose(a,residual(first)))

    def test_correlated_gaussian_scores_each_action_with_its_full_covariance(self):
        policy=self.model.policy
        observations=torch.zeros((3,56))
        distribution=policy.get_distribution(observations)
        gaussian=distribution.distribution
        transform=torch.as_tensor(self.config['ppo']['policy_kwargs']['exploration_transform'])
        variance=policy.log_std.exp().square().mean(dim=0)
        expected=(transform*variance)@transform.T
        torch.testing.assert_close(gaussian.covariance_matrix,expected.expand(3,7,7))
        self.assertGreater(float(expected[:6,:6].triu(1).abs().max().detach()),1e-6)
        actions=gaussian.mean+torch.arange(7)*.01
        self.assertEqual(distribution.log_prob(actions).shape,(3,))
        self.assertEqual(distribution.entropy().shape,(3,))
        reference=torch.distributions.MultivariateNormal(gaussian.mean,covariance_matrix=expected)
        torch.testing.assert_close(distribution.log_prob(actions),reference.log_prob(actions),rtol=1e-5,atol=1e-4)
        torch.testing.assert_close(distribution.entropy(),reference.entropy())
        policy.zero_grad()
        (-distribution.log_prob(actions.detach()).mean()).backward()
        self.assertTrue(torch.isfinite(policy.action_net.weight.grad).all())
        self.assertTrue(torch.isfinite(policy.log_std.grad).all())


if __name__=='__main__':unittest.main()
