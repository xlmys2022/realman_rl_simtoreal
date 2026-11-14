"""Progress reward contracts and exactly one shared default 512-step PPO smoke."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.tasks.reach import ENV_CONFIG, task_environment
from realman_rl_simtoreal.tasks.rewards import REACH_PROGRESS_V1_DEFAULTS, REWARD_COMPONENT_KEYS, progress_parameters


def script(path):
    spec=importlib.util.spec_from_file_location(Path(path).stem,PROJECT_ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class ProgressRewardTests(unittest.TestCase):
    def env(self,**kw):
        e=ReachEnv(**dict(dict(reward_mode='reach_progress_v1',action_scale=.00025,observation_mode='fetch_like_v1'),**kw))
        self.addCleanup(e.close);return e

    def info(self,previous,**kw):
        return dict(previous_achieved_goal=np.asarray(previous),is_safety_failure=False,is_action_rejected=False,**kw)

    def reward(self,e,previous,current,**kw):
        info=self.info([previous,0,0]);info.update(kw)
        return e.compute_reward([current,0,0],[0,0,0],info)

    def test_default_parameters_and_nominal_numerical_scale(self):
        e=self.env();self.assertEqual(e.reward_progress_parameters,dict(REACH_PROGRESS_V1_DEFAULTS))
        self.assertAlmostEqual(self.reward(e,.005,.00475),.25-.0475-.01)
        self.assertAlmostEqual(self.reward(e,.005,.00525),-.25-.0525-.01)
        self.assertAlmostEqual(self.reward(e,.005,.005),-.06)

    def test_a_direction_order_in_analytic_cases(self):
        e=self.env()
        for d in (.0015,.003,.005,.01):
            self.assertGreater(self.reward(e,d,d-.00025),self.reward(e,d,d))
            self.assertGreater(self.reward(e,d,d),self.reward(e,d,d+.00025))

    def test_b_stationary_has_zero_progress_and_negative_total(self):
        e=self.env();terms=e.reward_components([.005,0,0],[0,0,0],self.info([.005,0,0]))
        self.assertEqual(terms['progress_reward'],0);self.assertEqual(terms['distance_change_m'],0)
        self.assertLess(terms['total_reward'],0)

    def test_c_success_is_inclusive_and_recomputed_for_current_goal(self):
        e=self.env();info=self.info([.00125,0,0],is_success=False)
        t=e.reward_components([.001,0,0],[0,0,0],info)
        self.assertEqual(t['success_bonus'],10.)
        bad=dict(info,is_success=True,termination_reason='success',distance_m=0)
        t=e.reward_components([.001,0,0],[.01,0,0],bad)
        self.assertEqual(t['success_bonus'],0.)
        self.assertEqual(info['is_success'],False)

    def test_d_timeout_is_not_failure_or_success(self):
        e=self.env(max_episode_steps=1);e.reset(seed=1)
        obs,r,terminated,truncated,info=e.step([0,0,0])
        self.assertFalse(terminated);self.assertTrue(truncated)
        self.assertEqual(info['termination_reason'],'time_limit')
        self.assertFalse(info['is_safety_failure']);self.assertFalse(info['is_success'])
        self.assertEqual(info['reward_components']['success_bonus'],0.)
        self.assertEqual(info['reward_components']['safety_penalty'],0.)
        self.assertAlmostEqual(r,-.06)
        self.assertEqual(r,e.compute_reward(obs['achieved_goal'],obs['desired_goal'],info))

    def test_e_failure_suppresses_success_and_charges_step(self):
        e=self.env();info=self.info([.005,0,0]);info['is_safety_failure']=True
        t=e.reward_components([.0005,0,0],[0,0,0],info)
        self.assertEqual(t['success_bonus'],0.)
        self.assertEqual(t['safety_penalty'],-20.)
        self.assertEqual(t['step_penalty'],-.01)

    def test_e_terminate_safety_does_not_invent_rejection(self):
        e=self.env(action_scale=.01);e.reset(seed=1)
        _,r,t,tr,info=e.step([1,1,1])
        self.assertTrue(t);self.assertFalse(tr);self.assertTrue(info['is_safety_failure'])
        self.assertEqual(info['physics_steps'],0)
        self.assertFalse(info['is_action_rejected'])
        self.assertEqual(info['reward_components']['rejection_penalty'],0.)
        self.assertAlmostEqual(r,-20.06)

    def test_reject_and_hold_keeps_real_previous_goal_and_penalizes(self):
        e=self.env(action_scale=.01,boundary_mode='reject_and_hold');o,_=e.reset(seed=1)
        previous=o['achieved_goal'].copy();o,r,t,tr,info=e.step([1,1,1])
        self.assertFalse(t or tr);self.assertEqual(info['physics_steps'],25)
        np.testing.assert_array_equal(info['previous_achieved_goal'],previous)
        self.assertTrue(info['is_action_rejected']);self.assertFalse(info['is_safety_failure'])
        self.assertEqual(info['reward_components']['rejection_penalty'],-.1)
        self.assertAlmostEqual(r,-.16)

    def test_f_default_round_trip_not_profitable(self):
        e=self.env();a=self.reward(e,.005,.00475);b=self.reward(e,.00475,.005)
        self.assertLess(a+b,0);self.assertLess(a+.99*b,0)
        distances=(.005,.00475,.00525,.005)
        rewards=[self.reward(e,x,y) for x,y in zip(distances,distances[1:])]
        self.assertLess(sum(rewards),0);self.assertLess(np.dot(rewards,.99**np.arange(3)),0)

    def test_f_discounted_pure_progress_does_not_cancel(self):
        e=self.env(reward_progress_parameters=dict(w_distance=0,step_cost=0))
        a=self.reward(e,.005,.00475);b=self.reward(e,.00475,.005)
        self.assertAlmostEqual(a+b,0)
        self.assertAlmostEqual(a+.99*b,.0025)
        self.assertGreater(a+.99*b,0)

    def test_no_unverified_global_undiscounted_failure_preference(self):
        # Abstract 15mm state, not a sampled reset or a physical trajectory.
        # Fixed penalty cannot dominate every possible undiscounted timeout.
        e=self.env();idle=200*self.reward(e,.015,.015)
        failure=self.reward(e,.015,.015,is_safety_failure=True)
        self.assertGreater(failure,idle)
        self.assertLess(failure,self.reward(e,.015,.015)*sum(.99**i for i in range(200)))

    def test_g_success_ends_episode_and_cannot_be_collected_twice(self):
        e=self.env();e.reset(seed=1)
        for _ in range(30):
            o,r,t,tr,i=e.step([1,0,0])
            if t or tr:break
        self.assertTrue(i['is_success']);self.assertTrue(t);self.assertFalse(tr)
        self.assertEqual(i['reward_components']['success_bonus'],10.)
        with self.assertRaises(gym.error.ResetNeeded):e.step([0,0,0])

    def test_h_old_rewards_match_frozen_prechange_physical_samples(self):
        fixture=json.loads((PROJECT_ROOT/'tests/fixtures/reach_reward_legacy_values.json').read_text())
        for row in fixture['rows']:
            with ReachEnv(reward_mode=row['mode']) as e:
                info=dict(previous_achieved_goal=row['previous_achieved_goal'],is_safety_failure=row['is_safety_failure'],
                          is_action_rejected=row['is_action_rejected'])
                self.assertEqual(e.compute_reward(row['achieved_goal'],row['desired_goal'],info),row['reward'])

    def test_scalar_batch_list_dict_and_multidimensional_inputs(self):
        e=self.env();achieved=np.array([[.00475,0,0],[.00075,0,0],[.00525,0,0]])
        desired=np.zeros((3,3));previous=np.array([[.005,0,0],[.00125,0,0],[.005,0,0]])
        rows=[self.info(x) for x in previous];rows[2]['is_safety_failure']=True
        batch=dict(previous_achieved_goal=previous,is_safety_failure=np.array([False,False,True]),
                   is_action_rejected=np.array([False,False,False]))
        a=e.compute_reward(achieved,desired,rows);b=e.compute_reward(achieved,desired,batch)
        np.testing.assert_array_equal(a,b)
        for j in range(3):self.assertEqual(a[j],e.compute_reward(achieved[j],desired[j],rows[j]))
        shaped=np.stack([achieved,achieved]);infos=np.array([rows,rows],dtype=object)
        r=e.compute_reward(shaped,np.zeros_like(shaped),infos)
        np.testing.assert_array_equal(r,np.stack([a,a]))
        relabeled=e.compute_reward(achieved,achieved,batch)
        self.assertGreater(relabeled[0],a[0]);self.assertLess(relabeled[2],0)

    def test_components_reproduce_reward_and_info_is_not_mutated(self):
        e=self.env();o,_=e.reset(seed=1)
        for action in ([1,0,0],[0,0,0],[-1,0,0]):
            previous=o['achieved_goal'].copy();o,r,t,tr,info=e.step(action)
            frozen=copy.deepcopy(info);terms=info['reward_components']
            self.assertEqual(sum(terms[k] for k in REWARD_COMPONENT_KEYS),r)
            self.assertEqual(terms['total_reward'],r)
            np.testing.assert_array_equal(info['previous_achieved_goal'],previous)
            self.assertEqual(e.compute_reward(o['achieved_goal'],o['desired_goal'],info),r)
            self.assertEqual(info['reward_components'],frozen['reward_components'])
            self.assertEqual(set(o),{'observation','achieved_goal','desired_goal'})
            self.assertEqual(o['observation'].shape,(6,))

    def test_invalid_info_and_parameter_values_fail_clearly(self):
        e=self.env()
        for info in (None,{},self.info([np.nan,0,0]),dict(self.info([0,0,0]),is_safety_failure=1),
                     dict(self.info([0,0,0]),is_action_rejected=0)):
            with self.assertRaises(ValueError):e.compute_reward([.005,0,0],[0,0,0],info)
        for p in ({'unknown':1},{'progress_scale':0},{'distance_scale':-1},{'step_cost':np.nan},{'success_bonus':True}):
            with self.assertRaises(ValueError):self.env(reward_progress_parameters=p)
        with self.assertRaises(ValueError):ReachEnv(reward_mode='distance',reward_progress_parameters={})
        original=progress_parameters();original['step_cost']=99
        self.assertEqual(progress_parameters()['step_cost'],.01)

    def test_real_direction_order_and_actual_motion_for_both_targets(self):
        c=task_environment('axis_pair_reach',ENV_CONFIG,reward_mode='reach_progress_v1',action_scale=.00025,
                           observation_mode='fetch_like_v1')
        with ReachEnv(**c) as e:
            for index,sign in ((0,1),(1,-1)):
                values={}
                for direction in (-1,0,1):
                    o,_=e.reset(seed=2026,options={'goal_index':index});before=o['achieved_goal'].copy()
                    o,r,_,_,i=e.step([direction,0,0]);values[direction]=r
                    expected=np.linalg.norm(before-o['desired_goal'])-np.linalg.norm(o['achieved_goal']-o['desired_goal'])
                    self.assertAlmostEqual(i['reward_components']['progress_reward'],expected/.001)
                    if direction:self.assertGreater(direction*(o['achieved_goal'][0]-before[0]),0)
                self.assertGreater(values[sign],values[0]);self.assertGreater(values[0],values[-sign])

    def test_new_reward_does_not_change_physics_actions_or_termination(self):
        for mode in ('distance','safe_dense','reach_shaped_v1'):
            with ReachEnv(reward_mode=mode,action_scale=.00025,observation_mode='fetch_like_v1') as old, self.env() as new:
                old.reset(seed=1);new.reset(seed=1)
                for _ in range(25):
                    a,b=old.step([1,0,0]),new.step([1,0,0])
                    for field in ('qpos','qvel','ctrl'):np.testing.assert_array_equal(getattr(old.data,field),getattr(new.data,field))
                    for key in a[0]:np.testing.assert_array_equal(a[0][key],b[0][key])
                    self.assertEqual(a[2:4],b[2:4]);self.assertEqual(a[4]['termination_reason'],b[4]['termination_reason'])
                    self.assertEqual(a[4]['physics_steps'],b[4]['physics_steps'])
                    if a[2] or a[3]:break

    def test_typical_real_success_timeout_failure_returns(self):
        values={}
        for name,direction,pulse in (('success',1,20),('timeout',0,0),('failure',-1,200)):
            with self.env() as e:
                e.reset(seed=1);rewards=[]
                for step in range(200):
                    _,r,t,tr,i=e.step([direction if step<pulse else 0,0,0]);rewards.append(r)
                    if t or tr:break
                values[name]=(sum(rewards),np.dot(rewards,.99**np.arange(len(rewards))))
        for index in (0,1):
            self.assertGreater(values['success'][index],values['timeout'][index])
            self.assertGreater(values['timeout'][index],values['failure'][index])

    def test_gym_and_sb3_checker_both_observation_modes(self):
        from gymnasium.utils.env_checker import check_env
        from stable_baselines3.common.env_checker import check_env as sb3_check
        for mode in ('legacy','fetch_like_v1'):
            e=self.env(observation_mode=mode);check_env(e,skip_render_check=True);sb3_check(e)
            o,_=e.reset(seed=1);self.assertTrue(e.observation_space.contains(o))

    def test_task_config_preserves_full_nondefault_parameters_and_distributions(self):
        for task in ('fixed_reach','axis_pair_reach','easy_random_reach','random_reach'):
            old=task_environment(task,ENV_CONFIG)
            new=task_environment(task,ENV_CONFIG,reward_mode='reach_progress_v1',
                                 reward_progress_parameters=dict(step_cost=.02))
            for key in ('target_mode','target_radius','target_offsets'):
                self.assertEqual(old.get(key),new.get(key))
            self.assertEqual(new['reward_progress_parameters']['step_cost'],.02)
            self.assertEqual(set(new['reward_progress_parameters']),set(REACH_PROGRESS_V1_DEFAULTS))

    def test_diagnostic_cli_is_independent_and_does_not_train(self):
        p=subprocess.run([sys.executable,'-I',str(PROJECT_ROOT/'scripts/diagnostics/diagnose_reach_rewards.py'),'--help'],
                         cwd='/tmp',capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stderr);self.assertIn('--compare-from',p.stdout)


class ProgressPPOSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.folder=Path(os.environ.get('REALMAN_PROGRESS_PPO_TEST_ARTIFACTS',str(Path(cls.temp.name)/'run'))).resolve()
        if not (cls.folder/'summary.json').exists():
            code=r'''
import importlib.abc,importlib.util,json,sys
from pathlib import Path
import mujoco,numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.startswith(('realman_rl_simtoreal.imitation','realman_rl_simtoreal.rl.sac','realman_rl_simtoreal.vendor','Robotic_Arm')):
   raise AssertionError('No BC/SAC/SDK in PPO reward smoke')
sys.meta_path.insert(0,Block())
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.rl.common import baselines,evaluation
def forbidden(*a,**kw):raise AssertionError('No expert actions')
baselines.rule_action=evaluation.rule_action=forbidden
physical=mujoco.mj_step;original=ReachEnv.step;active={};counts=dict(env_steps=0,physics_steps=0)
def checked(m,d):
 if id(d) in active:np.testing.assert_array_equal(d.qpos,active[id(d)])
 physical(m,d)
 if id(d) in active:active[id(d)]=d.qpos.copy();counts['physics_steps']+=1
def step(e,a):
 active[id(e.data)]=e.data.qpos.copy();counts['env_steps']+=1
 try:
  r=original(e,a);np.testing.assert_array_equal(e.data.qpos,active[id(e.data)]);return r
 finally:active.pop(id(e.data))
mujoco.mj_step=checked;ReachEnv.step=step
spec=importlib.util.spec_from_file_location('driver',PROJECT_ROOT/'scripts/train/train_ppo_reach.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
p=Path(sys.argv[1]);s=m.train(p,timesteps=512,task='axis_pair_reach',observation_mode='fetch_like_v1',
 reward_mode='reach_progress_v1',seed=2026,eval_every=512,checkpoint_every=512)
assert s['status']=='completed',s['error']
(p/'test_verification.json').write_text(json.dumps(dict(**counts,no_experts=True,no_live_qpos_writes=True))+'\n')
'''
            p=subprocess.run([sys.executable,'-I','-c',code,str(cls.folder)],capture_output=True,text=True,timeout=180)
            cls.folder.parent.mkdir(parents=True,exist_ok=True)
            (cls.folder.parent/'progress_smoke_process.log').write_text(p.stdout+p.stderr)
            if p.returncode:raise AssertionError(p.stdout+p.stderr)
        cls.config=json.loads((cls.folder/'config.json').read_text());cls.summary=json.loads((cls.folder/'summary.json').read_text())

    def test_complete_rollout_real_updates_finite_losses_and_12_inputs(self):
        from realman_rl_simtoreal.rl.ppo import PPO_CONFIG,load_ppo
        self.assertEqual(self.config['ppo'],PPO_CONFIG)
        self.assertEqual([self.summary[k] for k in ('actual_timesteps','completed_rollout_updates','epoch_updates','optimizer_steps')],[512,1,10,80])
        self.assertTrue(self.summary['actor_updated'] and self.summary['critic_updated'])
        self.assertEqual(self.config['reward_parameters'],dict(REACH_PROGRESS_V1_DEFAULTS))
        self.assertEqual(self.config['environment']['reward_progress_parameters'],self.config['reward_parameters'])
        model,_,_=load_ppo(self.folder/'final/model.zip')
        self.assertEqual(model.policy.features_extractor.features_dim,12)
        with np.load(self.folder/'final/last_rollout.npz') as f:
            for key in f.files:self.assertTrue(np.isfinite(f[key]).all())
        evidence=json.loads((self.folder/'test_verification.json').read_text())
        self.assertTrue(evidence['no_experts'] and evidence['no_live_qpos_writes'])

    def test_component_logs_sum_and_checkpoint_reload(self):
        from realman_rl_simtoreal.rl.ppo import load_ppo
        a,t,c=load_ppo(self.folder/'final/model.zip');b,_,_=load_ppo(self.folder/'checkpoints/step_000512/model.zip')
        with ReachEnv(**c['environment']) as e:
            o,_=e.reset(seed=1);np.testing.assert_array_equal(a.predict(t.encode(o),deterministic=True)[0],b.predict(t.encode(o),deterministic=True)[0])
        rows=[json.loads(x) for x in (self.folder/'actions.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),512)
        for row in rows:
            terms=row['reward_components'];self.assertEqual(sum(terms[k] for k in REWARD_COMPONENT_KEYS),row['reward'])
        self.assertEqual(self.summary['training']['reward_component_steps'],512)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);broken=copy.deepcopy(c);broken['reward_parameters']['step_cost']=.2
            (p/'config.json').write_text(json.dumps(broken))
            with self.assertRaisesRegex(ValueError,'reward parameters'):load_ppo(p/'model.zip')
            broken=copy.deepcopy(c);broken['environment']['reward_mode']='reach_shaped_v1'
            broken['reward_parameters']=None
            (p/'config.json').write_text(json.dumps(broken));shutil.copyfile(self.folder/'final/model.zip',p/'model.zip')
            with self.assertRaisesRegex(ValueError,'reward configuration'):load_ppo(p/'model.zip')

    def test_independent_process_evaluation_uses_saved_reward(self):
        target=self.folder/'independent_evaluation.json'
        if not target.exists():
            p=subprocess.run([sys.executable,'-I',str(PROJECT_ROOT/'scripts/evaluate/evaluate_ppo_reach.py'),
                '--model',str(self.folder/'final/model.zip'),'--observation-mode','fetch_like_v1',
                '--episodes','2','--seed-start','96000','--json',str(target)],capture_output=True,text=True,timeout=120)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        d=json.loads(target.read_text());self.assertEqual(d['environment']['reward_mode'],'reach_progress_v1')
        self.assertTrue(d['parameters_unchanged']);self.assertEqual(d['features_dim'],12)
        for e in d['episodes']:
            self.assertEqual(len(e['reward_components']),e['length'])
            self.assertAlmostEqual(e['reward_component_totals']['total_reward'],e['return_'],places=10)


if __name__=='__main__':unittest.main()
