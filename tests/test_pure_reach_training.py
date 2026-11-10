"""Pure-RL task and discount-aware reward contracts, independent of learning quality."""
import importlib.util
import json
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
from realman_rl_simtoreal.tasks.reach import TASKS, task_environment
from realman_rl_simtoreal.rl.common.metrics import MotionStats
from realman_rl_simtoreal.paths import PROJECT_ROOT


class PureTaskRewardTests(unittest.TestCase):
    def env(self,**kw):
        env=ReachEnv(**dict(dict(reward_mode='reach_shaped_v1'),**kw));self.addCleanup(env.close)
        return env

    def info(self,previous,failed=False,rejected=False):
        return dict(previous_achieved_goal=np.asarray(previous),is_safety_failure=failed,is_action_rejected=rejected)

    def test_formula_success_failure_and_nonterminal(self):
        env=self.env();goal=np.zeros(3);previous=np.array([.005,0,0]);next_=np.array([.004,0,0])
        phi0=np.expm1(-1);phi1=np.expm1(-.8)
        self.assertAlmostEqual(env.compute_reward(next_,goal,self.info(previous)),.99*phi1-phi0)
        self.assertAlmostEqual(env.compute_reward([.0005,0,0],goal,self.info(previous)),env.shaped_terminal_value-phi0)
        self.assertAlmostEqual(env.compute_reward(next_,goal,self.info(previous,failed=True)),-env.shaped_terminal_value-phi0)
        self.assertAlmostEqual(env.compute_reward(next_,goal,self.info(previous,rejected=True)),.99*phi1-phi0-.01)

    def test_scalar_batch_and_goal_relabel_do_not_reuse_old_success(self):
        env=self.env();previous=np.array([[.005,0,0],[.003,0,0]])
        achieved=np.array([[.004,0,0],[.0005,0,0]]);goal=np.zeros((2,3))
        infos=[self.info(previous[0]),self.info(previous[1],failed=True)]
        batch=dict(previous_achieved_goal=previous,is_safety_failure=np.array([False,True]),is_action_rejected=np.array([False,False]))
        a=env.compute_reward(achieved,goal,infos);b=env.compute_reward(achieved,goal,batch)
        np.testing.assert_array_equal(a,b)
        for i in range(2):self.assertEqual(a[i],env.compute_reward(achieved[i],goal[i],infos[i]))
        bad=dict(infos[0],is_success=True,distance_m=0,termination_reason='success')
        self.assertEqual(env.compute_reward(achieved[0],goal[0],bad),a[0])
        changed=env.compute_reward(achieved[0],achieved[0],infos[0])
        self.assertGreater(changed,env.shaped_terminal_value)

    def test_missing_flags_illegal_types_and_nonfinite_previous_rejected(self):
        env=self.env()
        for info in (None,{},self.info([np.nan,0,0]),self.info([0,0,0],failed=1),self.info([0,0,0],rejected=1)):
            with self.assertRaises(ValueError):env.compute_reward([.005,0,0],[0,0,0],info)
        with self.assertRaises(ValueError):self.env(reward_gamma=1)
        with self.assertRaises(ValueError):self.env(reward_sigma_m=0)

    def test_discounted_worst_case_success_timeout_failure_order_including_rejections(self):
        env=self.env();H=200;gamma=.99;goal=np.zeros(3)
        def total(kind):
            previous=np.array([.005,0,0]);value=0.
            for t in range(H):
                last=t==H-1
                next_=np.array([.0005 if kind=='success' and last else 100.,0,0])
                failed=kind=='failure' and last
                # Give success/timeout every possible rejection; failure none.
                reward=env.compute_reward(next_,goal,self.info(previous,failed=failed,rejected=kind!='failure'))
                value+=gamma**t*reward;previous=next_
            return value
        initial_cost=-np.expm1(-1)
        self.assertGreater(total('success'),initial_cost)
        self.assertGreater(total('timeout'),total('failure'))
        self.assertAlmostEqual(total('timeout'),initial_cost-1,places=11)
        self.assertAlmostEqual(total('failure'),initial_cost-1.01,places=11)

    def test_naive_negative_exponential_small_failure_penalty_has_early_exit_loophole(self):
        cost=1-np.exp(-1)
        idle=-cost*sum(.99**i for i in range(200))
        immediate_failure=-cost-1
        self.assertGreater(immediate_failure,idle)

    def test_real_timeout_and_step_compute_reward_match(self):
        env=self.env(max_episode_steps=1);obs,_=env.reset(seed=1);before=obs['achieved_goal'].copy()
        obs,r,t,tr,info=env.step([0,0,0])
        self.assertFalse(t);self.assertTrue(tr);self.assertEqual(info['termination_reason'],'time_limit')
        np.testing.assert_array_equal(info['previous_achieved_goal'],before)
        self.assertEqual(r,env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info))
        self.assertAlmostEqual(r,.01*(1-np.exp(-1)))

    def test_actual_numerical_failure_still_terminates_without_state_repair(self):
        env=self.env();env.reset(seed=1);env.data.qvel[0]=np.nan
        obs,r,t,tr,info=env.step([0,0,0])
        self.assertTrue(t);self.assertFalse(tr);self.assertEqual(info['termination_reason'],'numerical_failure')
        self.assertTrue(np.isnan(env.data.qvel[0]));self.assertEqual(info['physics_steps'],0)
        self.assertLess(r,-6)
        self.assertEqual(r,env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info))

    def test_rejected_command_has_bounded_cost_and_normal_physics(self):
        env=self.env(boundary_mode='reject_and_hold',action_scale=.01);obs,_=env.reset(seed=1)
        obs,r,t,tr,info=env.step([1,1,1])
        self.assertFalse(t or tr);self.assertTrue(info['is_action_rejected']);self.assertFalse(info['is_safety_failure'])
        self.assertEqual(info['physics_steps'],25);self.assertAlmostEqual(env.data.time,.05)
        self.assertAlmostEqual(r,.01*(1-np.exp(-1))-.01)

    def test_reward_change_preserves_real_physics_and_original_safe_dense(self):
        a=self.env();b=ReachEnv(reward_mode='safe_dense');self.addCleanup(b.close)
        a.reset(seed=2);b.reset(seed=2)
        for i in range(10):
            action=[1,0,0] if i<5 else [0,0,0]
            ra,rb=a.step(action),b.step(action)
            for key in ra[0]:np.testing.assert_array_equal(ra[0][key],rb[0][key])
            self.assertEqual(ra[2:4],rb[2:4])
            if ra[2] or ra[3]:break
        self.assertEqual(ra[4]['termination_reason'],'success')
        self.assertNotEqual(ra[1],rb[1])

    def test_task_goal_definitions_reachable_and_reproducible(self):
        for task in TASKS:
            config=task_environment(task,dict(reward_mode='safe_dense',action_scale=.001))
            env=ReachEnv(**config);self.addCleanup(env.close)
            for seed in range(6):
                obs,_=env.reset(seed=seed);again,_=env.reset(seed=seed)
                np.testing.assert_array_equal(obs['desired_goal'],again['desired_goal'])
                radius=np.linalg.norm(obs['desired_goal']-env.kinematics.home_position)
                if task=='fixed_reach':self.assertAlmostEqual(radius,.005)
                elif task=='easy_random_reach':self.assertTrue(.002<=radius<=.003)
                else:self.assertTrue(.003<=radius<=.005)
                result=env.kinematics.solve_ik(obs['desired_goal'],env.kinematics.home_q)
                self.assertTrue(result.success)
                self.assertLess(np.max(abs(result.q-env.kinematics.home_q)),np.deg2rad(5))
                # Original solver defaults: 10um translation, 1e-4 rad rotation.
                self.assertLessEqual(result.position_error,1e-5)
                self.assertLessEqual(result.orientation_error,1e-4)
                self.assertEqual(env.data.time,0)

    def test_random_task_preserves_old_seeded_target_and_action_space(self):
        config=task_environment('random_reach',dict(reward_mode='safe_dense',action_scale=.001))
        a=ReachEnv(**config);b=ReachEnv(target_mode='random',reward_mode='safe_dense')
        self.addCleanup(a.close);self.addCleanup(b.close)
        one,two=a.reset(seed=3),b.reset(seed=3)
        for key in one[0]:np.testing.assert_array_equal(one[0][key],two[0][key])
        self.assertEqual(a.action_space,b.action_space)
        self.assertEqual(a.action_space.shape,(3,))

    def test_smaller_scale_and_fixed_cycle_never_teleport_qpos(self):
        env=self.env(action_scale=.00025);obs,_=env.reset(seed=1)
        previous=env.data.qpos.copy();step=mujoco.mj_step;calls=0
        def checked(m,d):
            nonlocal previous,calls
            np.testing.assert_array_equal(d.qpos,previous);step(m,d);previous=d.qpos.copy();calls+=1
        with patch.object(mujoco,'mj_step',side_effect=checked):
            obs,r,t,tr,info=env.step([1,0,0])
        self.assertEqual(calls,25);self.assertEqual(info['physics_steps'],25)
        self.assertAlmostEqual(info['attempted_position_m'][0]-env.kinematics.home_position[0],.00025)
        self.assertGreater(np.linalg.norm(env.data.qpos-env.kinematics.home_q),1e-8)

    def test_official_gym_checker_all_task_modes(self):
        for task in TASKS:
            env=ReachEnv(**task_environment(task,dict(reward_mode='reach_shaped_v1',action_scale=.001)))
            self.addCleanup(env.close);check_env(env,skip_render_check=True)

    def test_motion_diagnostics_are_episode_aware(self):
        m=MotionStats()
        m.observe([0,0,0],[.001,0,0],[.005,0,0],[0,0,0],[.001,0,0],[1,0,0],True)
        m.observe([0,0,0],[-.001,0,0],[.005,0,0],[0,0,0],[-.001,0,0],[-1,0,0],False)
        s=m.summary();self.assertEqual(s['toward_actual_fraction'],.5)
        self.assertEqual(s['toward_command_fraction'],.5);self.assertIsNone(s['reversal_fraction'])
        self.assertAlmostEqual(s['mean_actual_displacement_m'],.001)


AVAILABLE=importlib.util.find_spec('stable_baselines3') is not None


@unittest.skipUnless(AVAILABLE,'Optional SB3/PyTorch unavailable')
class PureTrainingTests(unittest.TestCase):
    def driver(self):
        spec=importlib.util.spec_from_file_location('pure_sac_entry',PROJECT_ROOT/'scripts/train/train_sac_reach.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

    def test_pure_task_rejects_all_bc_initialization_or_reference(self):
        for init,path in [('bc_actor',Path('unused.pt')),('scratch',Path('unused.pt'))]:
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(ValueError,'Pure RL'):
                    self.driver().train(Path(temp)/'run',timesteps=1,task='fixed_reach',init_mode=init,bc_model=path)

    def test_pure_training_import_isolated_from_bc_sdk_and_expert_calls(self):
        code='''
import importlib.abc,importlib.util,sys,tempfile,json
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if any(s in name for s in ['realman_rl_simtoreal.imitation','Robotic_Arm','realman_rl_simtoreal.vendor']):
   raise AssertionError('BC/SDK import forbidden: '+name)
sys.meta_path.insert(0,Block())
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.rl.common import evaluation, baselines
def forbidden(*a,**kw):raise AssertionError('Expert action forbidden')
evaluation.rule_action=baselines.rule_action=forbidden
spec=importlib.util.spec_from_file_location('pure_driver',PROJECT_ROOT/'scripts/train/train_sac_reach.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
with tempfile.TemporaryDirectory() as temp:
 folder=Path(temp)/'run'
 monitoring=importlib.util.find_spec('tensorboard') is not None
 result=driver.train(folder,timesteps=4,task='fixed_reach',reward_mode='reach_shaped_v1',
  eval_every=4,eval_episodes=1,environment={'max_episode_steps':2},seed=91,tensorboard=monitoring)
 assert result['status']=='completed',result['error']
 assert result['buffer_size']==4 and result['n_updates']==0
 config=json.loads((folder/'config.json').read_text())
 assert config['pure_rl'] and config['init_mode']=='scratch' and config['bc_model'] is None
 from stable_baselines3 import SAC
 model=SAC.load(folder/'final/model.zip',device='cpu')
 assert model.num_timesteps==4
 assert (folder/'final/replay_buffer.pkl.gz').exists()
 if monitoring:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  events=EventAccumulator(str(folder/'sb3'));events.Reload()
  tags=events.Tags()['scalars']
  assert 'training/motion/mean_actual_displacement_m' in tags
  assert 'evaluation/motion/toward_actual_fraction' in tags
  assert not any(tag.startswith('bc/') for tag in tags)
assert not any(n.startswith('realman_rl_simtoreal.imitation') for n in sys.modules)
'''
        result=subprocess.run([sys.executable,'-I','-c',code],capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_sb3_processed_goal_reward_contract(self):
        from stable_baselines3.common.env_checker import check_env as sb3_check
        from realman_rl_simtoreal.rl.common.observation import ScaledReachObservation
        with ScaledReachObservation(ReachEnv(reward_mode='reach_shaped_v1')) as env:
            sb3_check(env,warn=True)

    def test_new_reward_terminal_td_labels_come_from_flags_not_large_negative_threshold(self):
        import torch
        from realman_rl_simtoreal.rl.common.env_factory import training_env
        from realman_rl_simtoreal.rl.sac import make_model
        from realman_rl_simtoreal.tasks.reach import ENV_CONFIG
        from realman_rl_simtoreal.rl.sac_diagnostics import td_diagnostic
        torch.set_num_threads(1);env,_=training_env(ENV_CONFIG);self.addCleanup(env.close)
        model=make_model(env,11,overrides=dict(buffer_size=8,policy_kwargs=dict(net_arch=[16,16])))
        obs=env.reset()
        for reward in [-2.,3.]:
            model.replay_buffer.add(obs,obs,np.zeros((1,3),np.float32),np.array([reward]),np.array([True]),[{}])
        summary,arrays=td_diagnostic(model,terminal_labels={0:'safety_terminal',1:'success_terminal'})
        self.assertEqual(summary['categories']['safety_terminal']['count'],1)
        self.assertEqual(summary['categories']['success_terminal']['count'],1)
        np.testing.assert_allclose(arrays['td_target'],[-2,3])


if __name__=='__main__':unittest.main()
