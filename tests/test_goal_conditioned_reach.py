"""Discrete-goal contracts and one shared PPO rollout, not performance assertions."""
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
from realman_rl_simtoreal.tasks.reach import task_environment,ENV_CONFIG
from realman_rl_simtoreal.paths import PROJECT_ROOT


def config(task='axis_pair_reach'):
    return task_environment(task,dict(ENV_CONFIG,boundary_mode='terminate'),reward_mode='reach_shaped_v1',action_scale=.00025)


def module(path):
    spec=importlib.util.spec_from_file_location(Path(path).stem,PROJECT_ROOT/path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class AxisEnvironmentTests(unittest.TestCase):
    def test_pair_exact_targets_both_validated_and_readonly(self):
        with ReachEnv(**config()) as env:
            home=env.kinematics.home_position
            np.testing.assert_allclose(env.goal_candidates-home,[[.005,0,0],[-.005,0,0]],rtol=0,atol=1e-15)
            self.assertEqual(len(env.candidate_validation),2)
            self.assertTrue(all(r['valid'] for r in env.candidate_validation))
            for r in env.candidate_validation:
                self.assertLessEqual(r['position_error_m'],1e-5);self.assertLessEqual(r['orientation_error_rad'],1e-4)
            a=env.goal_candidates;a[0]=0;self.assertFalse(np.array_equal(a,env.goal_candidates))

    def test_seeded_direction_sequence_and_forced_goal_reset(self):
        with ReachEnv(**config()) as env:
            def sequence():return [env.reset(seed=2026 if i==0 else None)[1]['goal_index'] for i in range(30)]
            a=sequence();self.assertEqual(a,sequence());self.assertEqual(set(a),{0,1})
            for i in (0,1):
                obs,info=env.reset(seed=1,options={'goal_index':i})
                self.assertEqual(info['goal_index'],i);np.testing.assert_array_equal(obs['desired_goal'],env.goal_candidates[i])
                np.testing.assert_array_equal(env.data.qpos,env.kinematics.home_q)
                np.testing.assert_array_equal(env.data.qvel,np.zeros(env.model.nv));self.assertEqual(env.data.time,0)

    def test_real_step_goal_fixed_and_no_direct_qpos_motion(self):
        with ReachEnv(**config()) as env:
            obs,_=env.reset(seed=3,options={'goal_index':1});goal=obs['desired_goal'].copy();last=env.data.qpos.copy()
            physical=mujoco.mj_step;calls=0
            def step(m,d):
                nonlocal last,calls
                np.testing.assert_array_equal(d.qpos,last);physical(m,d);last=d.qpos.copy();calls+=1
            with patch.object(mujoco,'mj_step',side_effect=step):
                for _ in range(3):
                    obs,_,t,tr,_=env.step(np.array([-.1,0,0],np.float32))
                    np.testing.assert_array_equal(obs['desired_goal'],goal);self.assertFalse(t or tr)
            np.testing.assert_array_equal(env.data.qpos,last);self.assertEqual(calls,75)
            self.assertAlmostEqual(env.data.time,.15)

    def test_plus_x_physics_and_reward_identical_to_original_fixed(self):
        legacy=config('fixed_reach')
        with ReachEnv(**legacy) as a,ReachEnv(**config()) as b:
            a.reset(seed=5);b.reset(seed=5,options={'goal_index':0})
            for action in ([.5,0,0],[0,0,0],[.2,0,0]):
                x,y=a.step(action),b.step(action)
                for key in x[0]:np.testing.assert_array_equal(x[0][key],y[0][key])
                self.assertEqual(x[1:4],y[1:4]);self.assertEqual(x[1],a.compute_reward(x[0]['achieved_goal'],x[0]['desired_goal'],x[4]))

    def test_legacy_golden_trajectories_exact(self):
        fixture=json.loads((PROJECT_ROOT/'tests/fixtures/reach_env_legacy_contract.json').read_text())
        for row in fixture['rollouts']:
            with ReachEnv(**row['config']) as env:
                env.reset(seed=row['seed'])
                for expected in row['steps']:
                    obs,r,t,tr,info=env.step(expected['action'])
                    for key in obs:np.testing.assert_array_equal(obs[key],expected['observation'][key])
                    for key in ('qpos','qvel','ctrl'):np.testing.assert_array_equal(getattr(env.data,key),expected[key])
                    self.assertEqual((r,t,tr,info['termination_reason'],env.data.time),
                                     (expected['reward'],expected['terminated'],expected['truncated'],expected['reason'],expected['time']))

    def test_six_5mm_reports_all_invalid_candidates_without_dropping_them(self):
        with self.assertRaisesRegex(ValueError,'candidate 2.*candidate 3') as error:ReachEnv(**config('axis_six_reach'))
        self.assertIn('5 degrees',str(error.exception))
        # 3mm is explicit diagnostic-only configuration, not a new stage default.
        c=config('axis_six_reach');c['target_offsets']=(np.array(c['target_offsets'])*.6).tolist();c['target_radius']=.003
        with ReachEnv(**c) as env:self.assertEqual(len(env.goal_candidates),6)
        self.assertEqual(config('axis_six_reach')['target_radius'],.005)

    def test_invalid_candidates_or_options_fail_instead_of_clipping(self):
        for offsets in (None,[[.005,0,0],[.005,0,0]],[[np.nan,0,0]],[[.02,0,0]],[[0,0,0]]):
            with self.assertRaises(ValueError):ReachEnv(**dict(config(),target_offsets=offsets))
        with ReachEnv(**config()) as env:
            for option in ({'goal_index':True},{'goal_index':2},{'goal_index':-.1},{'unknown':0}):
                with self.assertRaises(ValueError):env.reset(options=option)

    def test_official_gym_checker_pair_and_33d_preprocessing(self):
        from realman_rl_simtoreal.rl.common.observation import ReachTransform,features
        with ReachEnv(**config()) as env:
            check_env(env,skip_render_check=True)
            transform=ReachTransform.from_env(env)
            for i,sign in ((0,1),(1,-1)):
                obs,_=env.reset(options={'goal_index':i});encoded=transform.encode(obs)
                self.assertEqual(features(encoded).shape,(33,))
                np.testing.assert_array_equal(encoded['observation'][15:18],[sign*.5,0,0])


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'),'Optional SB3 absent')
class AxisPPOSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.folder=Path(os.environ.get('REALMAN_AXIS_PPO_SMOKE_ARTIFACTS',str(Path(cls.temp.name)/'axis_smoke')))
        if not (cls.folder/'test_verification.json').exists():
            code=r'''
import importlib.abc,importlib.util,json,sys
from pathlib import Path
import mujoco,numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*a):
  if name.startswith(('realman_rl_simtoreal.imitation','realman_rl_simtoreal.rl.sac','realman_rl_simtoreal.vendor','Robotic_Arm')):raise AssertionError('No BC/SAC/vendor')
sys.meta_path.insert(0,Block())
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.rl.common import evaluation,baselines
def forbidden(*a,**k):raise AssertionError('No expert actions')
evaluation.rule_action=baselines.rule_action=forbidden
from realman_rl_simtoreal.envs.reach_env import ReachEnv
original=ReachEnv.step;physical=mujoco.mj_step;active={};count=0
def mj_step(m,d):
 if id(d) in active:np.testing.assert_array_equal(d.qpos,active[id(d)])
 physical(m,d)
 if id(d) in active:active[id(d)]=d.qpos.copy()
def step(env,a):
 global count
 count+=1;active[id(env.data)]=env.data.qpos.copy()
 try:
  result=original(env,a);np.testing.assert_array_equal(env.data.qpos,active[id(env.data)]);return result
 finally:active.pop(id(env.data))
mujoco.mj_step=mj_step;ReachEnv.step=step
spec=importlib.util.spec_from_file_location('axis_driver',PROJECT_ROOT/'scripts/train/train_ppo_reach.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
p=Path(sys.argv[1]);s=driver.train(p,timesteps=512,task='axis_pair_reach',seed=2026,
 eval_every=512,checkpoint_every=512,tensorboard=importlib.util.find_spec('tensorboard') is not None)
assert s['status']=='completed',s['error']
assert s['completed_rollout_updates']==1 and s['actual_timesteps']==512
(p/'test_verification.json').write_text(json.dumps(dict(env_steps=count,no_experts=True,no_live_qpos_writes=True))+'\n')
'''
            p=subprocess.run([sys.executable,'-I','-c',code,str(cls.folder)],capture_output=True,text=True,timeout=180)
            cls.folder.parent.mkdir(parents=True,exist_ok=True);(cls.folder.parent/'axis_smoke_process.log').write_text(p.stdout+p.stderr)
            if p.returncode:raise AssertionError(p.stdout+p.stderr)
        cls.config=json.loads((cls.folder/'config.json').read_text());cls.summary=json.loads((cls.folder/'summary.json').read_text())

    def test_exact_one_rollout_actor_value_update_finite_and_scratch(self):
        from realman_rl_simtoreal.rl.ppo import PPO_CONFIG
        s=self.summary
        self.assertEqual((s['actual_timesteps'],s['completed_rollout_updates'],s['epoch_updates'],s['optimizer_steps']),(512,1,10,80))
        self.assertTrue(s['actor_updated'] and s['critic_updated']);self.assertEqual(self.config['ppo'],PPO_CONFIG)
        self.assertEqual(self.config['init_mode'],'scratch');self.assertEqual(self.config['environment']['target_mode'],'discrete')
        with np.load(self.folder/'final/last_rollout.npz') as b:
            for key in b.files:self.assertTrue(np.isfinite(b[key]).all())

    def test_enumerated_targets_not_random_seed_repeats_and_per_goal_motion(self):
        from realman_rl_simtoreal.rl.ppo import load_ppo,evaluate_ppo,parameter_hashes
        model,t,_=load_ppo(self.folder/'final/model.zip');before=parameter_hashes(model)
        result=evaluate_ppo(model,t,dict(config(),max_episode_steps=2),[123,123])
        self.assertEqual([e['goal_index'] for e in result['episodes']],[0,1])
        self.assertEqual(set(result['per_goal']),{'+X','-X'});self.assertIn('not random-position generalization',result['goal_scope'])
        self.assertEqual(parameter_hashes(model),before)
        # Same reset seed is fine for distinct explicitly enumerated positions.
        self.assertEqual(len(module('scripts/evaluate/analyze_reach_results.py').analyze(result)['discrete_goals']),2)
        for e in result['episodes']:
            p=np.r_[np.asarray(e['actual_positions_before_m']),np.asarray(e['final_position_m'])[None,:]]
            self.assertAlmostEqual(np.linalg.norm(np.diff(p,axis=0),axis=1).sum(),e['motion']['actual_path_m'])
        with self.assertRaisesRegex(ValueError,'exactly 2'):evaluate_ppo(model,t,config(),[1]*100)

    def test_cli_auto_two_targets_and_rejects_fake_100_goal_evaluation(self):
        driver=module('scripts/evaluate/evaluate_ppo_reach.py')
        with self.assertRaisesRegex(ValueError,'exactly 2'):driver.run(self.folder/'final/model.zip',episodes=100)
        train=module('scripts/train/train_ppo_reach.py')
        with self.assertRaisesRegex(ValueError,'eval_episodes=2'):train.train(Path(self.temp.name)/'bad',task='axis_pair_reach',eval_episodes=20)

    def test_tensorboard_goal_scalars_match_real_evaluation(self):
        if not self.config['tensorboard']:self.skipTest('TensorBoard absent')
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        a=EventAccumulator(str(self.folder/'sb3'));a.Reload()
        evaluation=json.loads((self.folder/'checkpoints/step_000512/evaluation.json').read_text())
        for label,tag in (('+X','plus_x'),('-X','minus_x')):
            for key in ('success','first_action_x','first_action_y','first_action_z','distance_mm','action_direction_fraction','actual_path_mm'):
                self.assertIn(f'evaluation/goals/{tag}/{key}',a.Tags()['scalars'])
            self.assertEqual(a.Scalars(f'evaluation/goals/{tag}/success')[-1].value,float(evaluation['per_goal'][label]['success']))

    def test_goal_png_curves_come_from_real_metrics(self):
        if importlib.util.find_spec('matplotlib') is None:self.skipTest('Matplotlib absent')
        plot=module('scripts/plots/plot_ppo_training.py');out=Path(self.temp.name)/'plots'
        result=plot.export_plots(self.folder,out)
        self.assertIn('goal_success.png',result['generated']);self.assertIn('goal_first_action_x.png',result['generated'])
        self.assertIn('training_episode_success.png',result['generated'])
        evaluations=json.loads((self.folder/'evaluations.json').read_text())
        self.assertEqual(result['curves']['goal/-X/success'],[[e['step'],float(e['per_goal']['-X']['success'])] for e in evaluations])

    def test_frozen_sensitivity_probe_uses_only_consistent_reset_observations(self):
        probe=module('scripts/diagnostics/probe_ppo_goals.py')
        r=probe.probe(self.folder/'final/model.zip',task='axis_six_reach',distance_mm=3)
        self.assertEqual(r['changed_feature_indices'],[3,4,5,21,22,23]);self.assertTrue(r['physical_state_identical'])
        self.assertTrue(r['parameters_unchanged']);self.assertEqual(len(r['rows']),6)
        with self.assertRaisesRegex(ValueError,'candidate 2'):probe.probe(self.folder/'final/model.zip',task='axis_six_reach',distance_mm=5)

    def test_discrete_readonly_analysis_preserves_individual_goal_results(self):
        analysis=module('scripts/evaluate/analyze_reach_results.py')
        data=json.loads((self.folder/'checkpoints/step_000512/evaluation.json').read_text())
        result=analysis.analyze(data)
        self.assertEqual(set(result['discrete_goals']),{'+X','-X'})
        for label in ('+X','-X'):
            self.assertEqual(result['discrete_goals'][label]['samples'],1)
            self.assertEqual(result['discrete_goals'][label]['successes'],int(data['per_goal'][label]['success']))


if __name__=='__main__':unittest.main()
