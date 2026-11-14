"""Physical pick/lift, negative controls, state continuity and the RL interface."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np

from realman_rl_simtoreal.envs.grasp_lift_env import GraspLiftEnv
from realman_rl_simtoreal.rl.grasp_lift import rollout,summarize,experiment_metadata
from realman_rl_simtoreal.simulation.kinematics import IKResult
from realman_rl_simtoreal.tasks.grasp_lift import GRASP_LIFT_CONFIG


class GraspInterfaceTests(unittest.TestCase):
    def test_seeded_reset_reproduces_object_support_arm_and_open_gripper(self):
        with GraspLiftEnv() as env:
            obs,info=env.reset(seed=2027)
            q=env.data.qpos.copy();support=env.model.body_pos[env.support_body].copy()
            self.assertEqual(obs.shape,(41,));self.assertEqual(obs.dtype,np.float32)
            self.assertEqual(env.action_space.shape,(4,))
            self.assertEqual(env.physics_steps,50)
            self.assertAlmostEqual(info['grasp_distance_m'],.02,delta=1e-6)
            self.assertAlmostEqual(info['actual_width_m'],.06,delta=1e-6)
            self.assertLess(abs(info['support_clearance_m']),1e-8)
            self.assertAlmostEqual(env.model.body_mass[env.object_body],.01)
            self.assertEqual(env.model.body_gravcomp[env.object_body],0)
            np.testing.assert_array_equal(env.data.qvel,0)
            env.step([.1,0,0,1])
            again,reset_info=env.reset(seed=2027)
            np.testing.assert_array_equal(obs,again)
            np.testing.assert_array_equal(env.data.qpos,q)
            np.testing.assert_array_equal(env.model.body_pos[env.support_body],support)
            self.assertEqual(env.data.time,0)
            self.assertEqual(info['initial_object_position_m'],reset_info['initial_object_position_m'])

    def test_random_object_locations_stay_inside_the_declared_local_disk(self):
        with GraspLiftEnv() as env:
            starts=[env.reset(seed=s)[1]['initial_object_position_m'] for s in range(6)]
            self.assertEqual(len(set(map(tuple,starts))),6)
            offsets=np.array(starts)-env._object_pose[:3]
            self.assertTrue((np.linalg.norm(offsets,axis=1)<=GRASP_LIFT_CONFIG['object_radius']+1e-12).all())
            np.testing.assert_allclose(offsets@env.support_normal,0,atol=1e-12,rtol=0)

    def test_invalid_actions_do_not_advance_or_write_state(self):
        with GraspLiftEnv() as env:
            with self.assertRaises(gym.error.ResetNeeded):env.step([0,0,0,1])
            env.reset(seed=2027)
            before=env.data.qpos.copy();ctrl=env.data.ctrl.copy()
            for action in ([0,0,0],[0,0,0,1.01],[0,0,np.nan,1],[0,0,0,np.inf]):
                with self.subTest(action=action),self.assertRaises(ValueError):env.step(action)
                np.testing.assert_array_equal(env.data.qpos,before)
                np.testing.assert_array_equal(env.data.ctrl,ctrl)
                self.assertEqual(env.data.time,0)
                self.assertEqual(env._elapsed_steps,0)

    def test_ik_failure_stops_without_moving_or_resetting_the_object(self):
        with GraspLiftEnv() as env:
            env.reset(seed=2027)
            before=env.data.qpos.copy()
            failed=IKResult(False,None,600,.1,.1,'Injected IK failure')
            with patch.object(env,'_ik',return_value=failed):
                obs,reward,terminated,truncated,info=env.step([0,0,0,1])
            self.assertTrue(terminated);self.assertFalse(truncated)
            self.assertTrue(info['is_safety_failure']);self.assertFalse(info['is_success'])
            self.assertEqual(info['termination_reason'],'ik_failure')
            self.assertEqual(info['physics_steps'],0)
            np.testing.assert_array_equal(env.data.qpos,before)
            self.assertTrue(np.isfinite(obs).all());self.assertTrue(np.isfinite(reward))
            with self.assertRaises(gym.error.ResetNeeded):env.step([0,0,0,1])

    def test_external_force_is_rejected_before_any_integration(self):
        with GraspLiftEnv() as env:
            env.reset(seed=2027)
            before=env.data.qpos.copy()
            env.data.qfrc_applied[env.kinematics.vadr[0]]=.01
            _,_,terminated,_,info=env.step([0,0,0,1])
            self.assertTrue(terminated);self.assertTrue(info['is_safety_failure'])
            self.assertIn('External force',info['detail'])
            self.assertEqual(info['physics_steps'],0)
            np.testing.assert_array_equal(env.data.qpos,before)

    def test_numerical_failure_returns_a_marked_finite_snapshot_without_repairing_state(self):
        with GraspLiftEnv() as env:
            previous,_=env.reset(seed=2027)
            env.data.qvel[env.object_vadr]=np.nan
            obs,reward,terminated,truncated,info=env.step([0,0,0,1])
            self.assertTrue(terminated);self.assertFalse(truncated)
            self.assertEqual(info['termination_reason'],'numerical_failure')
            self.assertFalse(info['observation_valid']);self.assertTrue(info['is_safety_failure'])
            self.assertTrue(np.isnan(env.data.qvel[env.object_vadr]))
            np.testing.assert_array_equal(obs,previous)
            self.assertTrue(np.isfinite(reward));self.assertEqual(info['physics_steps'],0)
            json.dumps(info,allow_nan=False)

    def test_timeout_and_close_contracts(self):
        with GraspLiftEnv(max_episode_steps=1,render_mode='ansi') as env:
            env.reset(seed=2027)
            _,_,terminated,truncated,info=env.step([0,0,0,1])
            self.assertFalse(terminated);self.assertTrue(truncated)
            self.assertEqual(info['termination_reason'],'time_limit')
            self.assertFalse(info['is_success']);self.assertEqual(info['physics_steps'],50)
            self.assertIn('clearance=',env.render())
            with self.assertRaises(gym.error.ResetNeeded):env.step([0,0,0,1])
        with self.assertRaises(RuntimeError):env.reset()

    def test_configuration_rejects_invalid_task_bounds(self):
        for config in [dict(lift_height=0),dict(object_radius=.1),dict(hold_seconds=np.inf),
                       dict(hold_seconds='wrong'),dict(action_scale=[.001]),dict(max_episode_steps=True),
                       dict(record_physics=1),dict(mass=0),dict(friction=-1),dict(control_period=.1),
                       dict(physics_timestep=0),dict(physics_timestep=.0015),dict(unknown=1)]:
            with self.subTest(config=config),self.assertRaises(ValueError):GraspLiftEnv(**config)

    def test_gymnasium_checker(self):
        with GraspLiftEnv() as env:check_env(env,skip_render_check=True)

    def test_ppo_rollout_update_and_checkpoint_use_the_four_action_interface(self):
        import torch
        from stable_baselines3 import PPO
        threads=torch.get_num_threads();torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads,threads)
        with GraspLiftEnv(max_episode_steps=8) as env:
            model=PPO('MlpPolicy',env,n_steps=32,batch_size=16,n_epochs=1,seed=2027,
                      policy_kwargs=dict(net_arch=[32,32]),device='cpu')
            before={k:v.clone() for k,v in model.policy.state_dict().items()}
            model.learn(64)
            self.assertEqual(model.num_timesteps,64)
            self.assertEqual(model._n_updates,2)
            self.assertTrue(any(not torch.equal(v,before[k]) for k,v in model.policy.state_dict().items()))
            obs,_=env.reset(seed=2027)
            action,_=model.predict(obs,deterministic=True)
            self.assertTrue(env.action_space.contains(action))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'model.zip';model.save(path)
                loaded=PPO.load(path,device='cpu')
                np.testing.assert_array_equal(loaded.predict(obs,deterministic=True)[0],action)
                result=rollout(env,2027,model=loaded)
                self.assertEqual(result['length'],8)
                self.assertTrue(result['observation_valid'])


class PhysicalGraspLiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with GraspLiftEnv(record_physics=True) as env:
            original=mujoco.mj_step
            previous=None
            cls.physics_calls=0
            def audited(model,data):
                nonlocal previous
                if previous is not None:np.testing.assert_array_equal(data.qpos,previous)
                original(model,data)
                previous=data.qpos.copy()
                cls.physics_calls+=1
            with patch.object(mujoco,'mj_step',side_effect=audited):
                cls.normal=rollout(env,2027)
        with GraspLiftEnv() as env:
            cls.no_close=rollout(env,2027,close=False)
            cls.no_lift=rollout(env,2027,lift=False)

    def test_real_pick_and_lift_meet_the_full_hold_criterion(self):
        r=self.normal
        self.assertTrue(r['success'],r['detail'])
        self.assertGreaterEqual(r['hold_time_s'],2.)
        self.assertGreater(r['final_support_clearance_m'],.028)
        self.assertGreater(r['world_z_lift_m'],.024)
        self.assertAlmostEqual(r['actual_width_m'],.020,delta=.0002)
        self.assertEqual(r['gripper_statistics']['target_width_m'],0.)
        tail=r['physics_records'][-2000:]
        self.assertTrue(all(min(x['left_normal_n'],x['right_normal_n'])>.005 for x in tail))
        self.assertTrue(all(x['support_normal_n']<.001 and x['support_clearance_m']>=.028 for x in tail))
        self.assertTrue(all(np.linalg.norm(x['object_qvel'][:3])<.002 for x in tail))
        relative=np.array([x['object_relative_to_pads_m'] for x in tail])
        self.assertLess(np.linalg.norm(relative-relative[0],axis=1).max(),.002)

    def test_arm_gripper_and_object_share_one_physical_clock_without_teleporting(self):
        r=self.normal
        self.assertEqual(self.physics_calls,50*r['length'])
        self.assertEqual(len(r['physics_records']),self.physics_calls)
        self.assertAlmostEqual(r['physics_records'][-1]['time_s'],.05*r['length'],places=8)
        self.assertEqual(sum(x['physics_steps'] for x in r['records']),self.physics_calls)
        self.assertIn('approach',{x['phase'] for x in r['physics_records']})
        self.assertIn('lift',{x['phase'] for x in r['physics_records']})

    def test_contact_world_force_supports_the_free_objects_weight(self):
        tail=[r for r in self.normal['physics_records'][-2000:] if r['contacts_sampled']]
        forces=[c['force_on_target_world_n'] for c in tail[-1]['contacts'] if c['force_on_target_world_n'] is not None]
        np.testing.assert_allclose(np.sum(forces,axis=0),[0,0,.01*9.81],atol=.003,rtol=0)

    def test_no_close_does_not_count_arm_lift_as_object_lift(self):
        r=self.no_close
        self.assertFalse(r['success'])
        self.assertLess(r['final_support_clearance_m'],.028)
        self.assertEqual(r['contact_forces_n'][:2],[0.,0.])
        self.assertFalse(any(x['bilateral_contact'] for x in r['records']))
        self.assertEqual(r['hold_time_s'],0.)

    def test_no_lift_does_not_count_closure_as_pick_success(self):
        r=self.no_lift
        self.assertFalse(r['success']);self.assertFalse(r['is_safety_failure'],r['detail'])
        self.assertTrue(any(x['bilateral_contact'] for x in r['records']))
        self.assertLess(r['final_support_clearance_m'],.003)
        self.assertEqual(r['hold_time_s'],0.)

    def test_metrics_and_resource_records_include_all_failed_controls(self):
        summary=summarize([self.normal,self.no_close,self.no_lift])
        self.assertEqual(summary['episodes'],3);self.assertEqual(summary['success_rate'],1/3)
        metadata=experiment_metadata()
        self.assertEqual(metadata['action_dim'],4)
        self.assertIn('assets/scenes/grasp_contact.xml',metadata['assets'])
        for r in (self.normal,self.no_close,self.no_lift):
            for step in r['records']:
                self.assertAlmostEqual(sum(step['reward_components'].values()),step['reward'],places=12)


if __name__=='__main__':unittest.main()
