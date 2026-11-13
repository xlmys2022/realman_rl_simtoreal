"""Random-goal contracts and read-only audits; one shared 512-step PPO smoke."""
from collections import Counter
import copy
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

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.tasks.reach import ENV_CONFIG,task_environment
from realman_rl_simtoreal.simulation.control import validate_target
from realman_rl_simtoreal.simulation.ee_control import check_motion


def module(path):
    spec=importlib.util.spec_from_file_location(Path(path).stem,PROJECT_ROOT/path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


CONFIG=task_environment('random_reach',dict(ENV_CONFIG,boundary_mode='terminate'),
                        reward_mode='reach_shaped_v1',action_scale=.00025)
ANALYSIS='scripts/evaluate/analyze_reach_results.py'


class RandomTargetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records=[];cls.encoded=[]
        from realman_rl_simtoreal.rl.common.observation import ReachTransform
        with ReachEnv(**CONFIG) as env:
            transform=ReachTransform.from_env(env)
            for seed in range(83000,83200):
                obs,info=env.reset(seed=seed);before=env.data.qpos.copy()
                result=env.kinematics.solve_ik(obs['desired_goal'],env.kinematics.home_q,env.kinematics.home_rotation)
                if not result.success:raise AssertionError(result.reason)
                validate_target(env.model,env.joint_ids,env.kinematics.home_q,result.q)
                check_motion(env.model,env.data,env.joint_ids,env.actuator_ids,env.kinematics,result,obs['desired_goal'])
                np.testing.assert_array_equal(env.data.qpos,before)
                np.testing.assert_array_equal(env.data.qpos[env.kinematics.qadr],env.kinematics.home_q)
                np.testing.assert_array_equal(env.data.ctrl[env.actuator_ids],env.kinematics.home_q)
                np.testing.assert_array_equal(env.data.qvel,np.zeros(env.model.nv))
                assert env.data.time==0
                delta=obs['desired_goal']-obs['achieved_goal']
                cls.records.append(dict(seed=seed,initial_m=obs['achieved_goal'].tolist(),goal_m=obs['desired_goal'].tolist(),
                    offset_m=delta.tolist(),distance_m=float(np.linalg.norm(delta)),
                    octant=''.join('+' if x>0 else '-' for x in delta),sampling=info['goal_sampling'],
                    joint_target_rad=result.q.tolist(),position_error_m=result.position_error,
                    orientation_error_rad=result.orientation_error))
                cls.encoded.append(transform.encode(obs))
            cls.home_q=env.kinematics.home_q.copy();cls.limits=env.kinematics.limits.copy()
        cls.summary=dict(samples=len(cls.records),seeds=[83000,83199],config=CONFIG,
            octants=dict(Counter(r['octant'] for r in cls.records)),
            offset_min_mm=(np.min([r['offset_m'] for r in cls.records],axis=0)*1000).tolist(),
            offset_max_mm=(np.max([r['offset_m'] for r in cls.records],axis=0)*1000).tolist(),
            radius_min_mm=min(r['distance_m'] for r in cls.records)*1000,
            radius_max_mm=max(r['distance_m'] for r in cls.records)*1000,
            radius_mean_mm=float(np.mean([r['distance_m'] for r in cls.records])*1000),
            attempts=sum(r['sampling']['attempts'] for r in cls.records),
            rejections=dict(sum((Counter(r['sampling']['rejections']) for r in cls.records),Counter())),
            records=cls.records)
        if os.environ.get('REALMAN_RANDOM_REACH_TEST_ARTIFACTS'):
            path=Path(os.environ['REALMAN_RANDOM_REACH_TEST_ARTIFACTS'])/'target_sampling.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():assert json.loads(path.read_text())==cls.summary
            else:path.write_text(json.dumps(cls.summary,indent=2,allow_nan=False)+'\n')

    def test_200_validated_goals_cover_xyz_and_octants(self):
        d=np.array([r['offset_m'] for r in self.records])
        self.assertTrue((d>0).any(axis=0).all());self.assertTrue((d<0).any(axis=0).all())
        self.assertEqual(len(self.summary['octants']),8)
        self.assertGreaterEqual(self.summary['radius_min_mm'],3-1e-9)
        self.assertLessEqual(self.summary['radius_max_mm'],5+1e-9)
        self.assertEqual(len(set(tuple(r['goal_m']) for r in self.records)),200)

    def test_all_goals_hold_home_orientation_and_joint_limits(self):
        q=np.array([r['joint_target_rad'] for r in self.records])
        self.assertTrue((q>self.limits[:,0]).all());self.assertTrue((q<self.limits[:,1]).all())
        self.assertLessEqual(np.max(abs(q-self.home_q)),np.deg2rad(5))
        self.assertLessEqual(max(r['position_error_m'] for r in self.records),1e-5)
        self.assertLessEqual(max(r['orientation_error_rad'] for r in self.records),1e-4)

    def test_seeded_sequence_reproducible_and_unseeded_reset_resamples(self):
        with ReachEnv(**CONFIG) as env:
            first=[env.reset(seed=84000)[0]['desired_goal']]+[env.reset()[0]['desired_goal'] for _ in range(4)]
            again=[env.reset(seed=84000)[0]['desired_goal']]+[env.reset()[0]['desired_goal'] for _ in range(4)]
            np.testing.assert_array_equal(first,again)
            self.assertEqual(len(set(map(tuple,first))),5)
            for row in self.records[:4]:np.testing.assert_array_equal(env.reset(seed=row['seed'])[0]['desired_goal'],row['goal_m'])

    def test_preprocessing_retains_all_three_direction_components(self):
        from realman_rl_simtoreal.rl.common.observation import features
        for row,obs in zip(self.records,self.encoded):
            np.testing.assert_allclose(obs['observation'][15:18],np.array(row['offset_m'])/.01,rtol=1e-6,atol=1e-8)
            np.testing.assert_array_equal(np.sign(obs['observation'][15:18]),np.sign(row['offset_m']))
            np.testing.assert_array_equal(obs['desired_goal'],row['goal_m'])
            self.assertEqual(features(obs).shape,(33,));self.assertTrue(np.isfinite(features(obs)).all())

    def test_goal_fixed_during_real_steps_with_no_qpos_shortcut(self):
        with ReachEnv(**CONFIG) as env:
            obs,_=env.reset(seed=84001);goal=obs['desired_goal'].copy();previous=env.data.qpos.copy()
            physical=mujoco.mj_step;calls=0
            def checked(m,d):
                nonlocal previous,calls
                np.testing.assert_array_equal(d.qpos,previous);physical(m,d);previous=d.qpos.copy();calls+=1
            with patch.object(mujoco,'mj_step',side_effect=checked):
                for _ in range(3):
                    obs,_,t,tr,info=env.step(np.zeros(3,np.float32))
                    np.testing.assert_array_equal(obs['desired_goal'],goal);self.assertFalse(t or tr)
            self.assertEqual(calls,75);self.assertAlmostEqual(env.data.time,.15)
            np.testing.assert_array_equal(env.data.qpos,previous)

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'),'Optional SB3 absent')
    def test_vecenv_automatically_resets_home_and_changes_episode_goal(self):
        from realman_rl_simtoreal.rl.common.env_factory import training_env
        env,_=training_env(dict(CONFIG,max_episode_steps=2));self.addCleanup(env.close)
        env.seed(84003);obs=env.reset();goal=obs['desired_goal'].copy()
        next_,_,done,_=env.step(np.zeros((1,3),np.float32));self.assertFalse(done[0])
        np.testing.assert_array_equal(next_['desired_goal'],goal)
        next_,_,done,info=env.step(np.zeros((1,3),np.float32));self.assertTrue(done[0])
        np.testing.assert_array_equal(info[0]['terminal_observation']['desired_goal'],goal[0])
        self.assertFalse(np.array_equal(next_['desired_goal'],goal))
        raw=env.venv.envs[0].unwrapped
        np.testing.assert_array_equal(raw.data.qpos[raw.kinematics.qadr],self.home_q);self.assertEqual(raw.data.time,0)


class DirectionAnalysisTests(unittest.TestCase):
    def examples(self):
        from itertools import product
        rows=[]
        for i,signs in enumerate(product((1,-1),repeat=3)):
            delta=np.array(signs)*(.0035 if i<4 else .0045)/np.sqrt(3);success=i%2==0
            action=np.array(signs)/np.sqrt(3)
            rows.append(dict(seed=1000+i,initial_position_m=[0,0,0],target_position_m=delta.tolist(),
                final_position_m=delta.tolist() if success else [0,0,0],final_distance_m=0. if success else float(np.linalg.norm(delta)),
                success=success,reason='success' if success else 'time_limit',length=1,actions=[action.tolist()],
                raw_actions=[action.tolist()],actual_positions_before_m=[[0,0,0]]))
        return dict(policy='ppo',deterministic=True,episodes=rows)

    def test_overlapping_directions_and_disjoint_octants_radii(self):
        result=module(ANALYSIS).analyze(self.examples())
        self.assertEqual(result['overall']['success_rate'],.5)
        self.assertEqual(sum(r['samples'] for r in result['directions'].values()),24)
        self.assertEqual(sum(r['samples'] for r in result['octants'].values()),8)
        for key,row in result['octants'].items():self.assertEqual(row['samples'],0 if key=='on_coordinate_plane' else 1)
        self.assertEqual(result['radius_bins']['3-4mm']['samples'],4)
        self.assertEqual(result['radius_bins']['4-5mm']['samples'],4)
        self.assertTrue(result['directions']['+X']['small_sample'])

    def test_clipping_direction_distortion_and_goal_projection(self):
        doc=self.examples();e=doc['episodes'][0];doc['episodes']=[e]
        e.update(target_position_m=[.001,-.003,0],final_position_m=[0,0,0],final_distance_m=np.sqrt(.00001),
                 success=False,reason='time_limit',actions=[[1,1,0]],raw_actions=[[100,1,0]])
        m=module(ANALYSIS).analyze(doc)['overall']['actions']
        self.assertEqual(m['box_clip_vector_fraction'],1);self.assertGreater(m['clipping_direction_mean_deg'],40)
        self.assertEqual(m['raw_action_toward_actual_goal_fraction'],1)
        self.assertEqual(m['action_toward_actual_goal_fraction'],0)
        self.assertEqual(m['toward_to_away_after_clipping_fraction'],1)

    def test_historical_json_does_not_invent_missing_raw_or_trajectory(self):
        doc=self.examples()
        for e in doc['episodes']:e.pop('raw_actions');e.pop('actual_positions_before_m')
        result=module(ANALYSIS).analyze(doc)['overall']['actions']
        self.assertFalse(result['raw_actions_complete']);self.assertFalse(result['actual_trajectory_complete'])
        self.assertNotIn('box_clip_vector_fraction',result)
        self.assertNotIn('action_toward_actual_goal_fraction',result)

    def test_invalid_actions_repeated_seeds_and_inconsistent_distance_fail(self):
        for mutate in (lambda d:d['episodes'][0].update(raw_actions=[[100,1,0]]),
                       lambda d:d['episodes'][1].update(seed=1000),
                       lambda d:d['episodes'][0].update(final_distance_m=10)):
            doc=self.examples();mutate(doc)
            with self.assertRaises(ValueError):module(ANALYSIS).analyze(doc)
        with self.assertRaises(ValueError):module(ANALYSIS).analyze(dict(episodes=[]))

    def test_axis_plane_empty_groups_and_four_mm_boundary(self):
        doc=self.examples();e=doc['episodes'][0];doc['episodes']=[e]
        e.update(target_position_m=[.004,0,0],final_position_m=[.004,0,0],actions=[[0,0,0]],raw_actions=[[0,0,0]])
        r=module(ANALYSIS).analyze(doc)
        self.assertEqual(r['octants']['on_coordinate_plane']['samples'],1)
        self.assertEqual(r['radius_bins']['4-5mm']['samples'],1)
        self.assertIsNone(r['directions']['-X']['success_rate'])
        self.assertIsNone(r['overall']['actions']['clipping_direction_mean_deg'])

    def test_readonly_cli_from_tmp_and_no_overwrite(self):
        help_=subprocess.run([sys.executable,'-I',str(PROJECT_ROOT/ANALYSIS),'--help'],
                             cwd='/tmp',capture_output=True,text=True,timeout=30)
        self.assertEqual(help_.returncode,0,help_.stderr);self.assertIn('--evaluation',help_.stdout)
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'evaluation.json';out=Path(temp)/'analysis.json'
            source.write_text(json.dumps(self.examples()));before=source.read_bytes()
            command=[sys.executable,'-I',str(PROJECT_ROOT/ANALYSIS),'--evaluation',str(source),'--json',str(out)]
            p=subprocess.run(command,cwd='/tmp',capture_output=True,text=True,timeout=30)
            self.assertEqual(p.returncode,0,p.stdout+p.stderr);self.assertEqual(source.read_bytes(),before)
            p=subprocess.run(command,cwd='/tmp',capture_output=True,text=True,timeout=30)
            self.assertNotEqual(p.returncode,0);self.assertIn('exists',p.stderr)


@unittest.skipUnless(importlib.util.find_spec('stable_baselines3'),'Optional SB3 absent')
class RandomPPOSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.temp.cleanup)
        cls.folder=Path(os.environ.get('REALMAN_RANDOM_PPO_SMOKE_ARTIFACTS',str(Path(cls.temp.name)/'random_smoke')))
        if not (cls.folder/'test_verification.json').exists():
            code=r'''
import importlib.abc,importlib.util,json,sys
from pathlib import Path
import mujoco,numpy as np
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.startswith(('realman_rl_simtoreal.imitation','realman_rl_simtoreal.rl.sac','realman_rl_simtoreal.vendor','Robotic_Arm')):
   raise AssertionError('No BC/SAC/vendor in random PPO: '+name)
sys.meta_path.insert(0,Block())
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.rl.common import evaluation,baselines
def forbidden(*a,**k):raise AssertionError('No expert actions')
evaluation.rule_action=baselines.rule_action=forbidden
from realman_rl_simtoreal.envs.reach_env import ReachEnv
physical=mujoco.mj_step;original=ReachEnv.step;active={};steps=0
def mj_step(m,d):
 if id(d) in active:np.testing.assert_array_equal(d.qpos,active[id(d)])
 physical(m,d)
 if id(d) in active:active[id(d)]=d.qpos.copy()
def step(env,action):
 global steps
 steps+=1;active[id(env.data)]=env.data.qpos.copy()
 try:
  result=original(env,action);np.testing.assert_array_equal(env.data.qpos,active[id(env.data)]);return result
 finally:active.pop(id(env.data))
mujoco.mj_step=mj_step;ReachEnv.step=step
spec=importlib.util.spec_from_file_location('random_ppo_driver',PROJECT_ROOT/'scripts/train/train_ppo_reach.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
folder=Path(sys.argv[1]);s=driver.train(folder,timesteps=512,task='random_reach',seed=2026,
 eval_every=512,eval_episodes=2,eval_seed_start=86000,checkpoint_every=512,
 tensorboard=importlib.util.find_spec('tensorboard') is not None)
assert s['status']=='completed',s['error']
assert s['actual_timesteps']==512 and s['completed_rollout_updates']==1
(folder/'test_verification.json').write_text(json.dumps(dict(env_steps=steps,no_live_qpos_writes=True,no_experts=True))+'\n')
'''
            proc=subprocess.run([sys.executable,'-I','-c',code,str(cls.folder)],capture_output=True,text=True,timeout=180)
            cls.folder.parent.mkdir(parents=True,exist_ok=True)
            (cls.folder.parent/'random_smoke_process.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode:raise AssertionError(proc.stdout+proc.stderr)
        cls.summary=json.loads((cls.folder/'summary.json').read_text());cls.config=json.loads((cls.folder/'config.json').read_text())

    def test_single_random_rollout_real_actor_value_update_and_physics(self):
        s=self.summary
        self.assertEqual(s['actual_timesteps'],512);self.assertEqual(s['completed_rollout_updates'],1)
        self.assertEqual(s['epoch_updates'],10);self.assertEqual(s['optimizer_steps'],80)
        self.assertTrue(s['actor_updated'] and s['critic_updated'])
        self.assertEqual(self.config['environment'],CONFIG);self.assertEqual(self.config['init_mode'],'scratch')
        self.assertTrue(json.loads((self.folder/'test_verification.json').read_text())['no_live_qpos_writes'])
        rows=[json.loads(x) for x in (self.folder/'actions.jsonl').read_text().splitlines()]
        self.assertGreater(len(set(tuple(x['goal_m']) for x in rows)),1)
        for a,b in zip(rows,rows[1:]):
            if a['terminated'] or a['truncated']:self.assertNotEqual(a['goal_m'],b['goal_m'])
            else:self.assertEqual(a['goal_m'],b['goal_m'])
        with np.load(self.folder/'final/last_rollout.npz') as buf:
            for name in buf.files:self.assertTrue(np.isfinite(buf[name]).all())

    def test_evaluation_records_raw_actions_and_correct_episode_slices(self):
        measured=json.loads((self.folder/'checkpoints/step_000512/evaluation.json').read_text())
        self.assertEqual(measured['summary']['episodes'],2)
        for e in measured['episodes']:
            self.assertEqual(len(e['actions']),len(e['raw_actions']));self.assertEqual(len(e['actions']),len(e['actual_positions_before_m']))
            np.testing.assert_array_equal(e['actions'],np.clip(e['raw_actions'],-1,1))
            np.testing.assert_array_equal(e['actual_positions_before_m'][0],e['initial_position_m'])
        result=module(ANALYSIS).analyze(measured)
        self.assertTrue(result['overall']['actions']['raw_actions_complete'])
        self.assertTrue(result['overall']['actions']['actual_trajectory_complete'])

    def test_twenty_target_api_and_frozen_loaded_policy(self):
        from realman_rl_simtoreal.rl.ppo import load_ppo,evaluate_ppo,parameter_hashes
        model,transform,_=load_ppo(self.folder/'final/model.zip');before=parameter_hashes(model)
        # Interface test only; H=1 is explicitly NOT the production evaluation.
        result=evaluate_ppo(model,transform,dict(CONFIG,max_episode_steps=1),range(86020,86040))
        self.assertEqual(result['summary']['episodes'],20)
        self.assertEqual(len(set(tuple(e['target_position_m']) for e in result['episodes'])),20)
        self.assertEqual(parameter_hashes(model),before);self.assertEqual(model.num_timesteps,512)
        self.assertTrue(result['parameters_unchanged'])

    def test_tensorboard_and_saved_policy_metadata(self):
        if not self.config['tensorboard']:self.skipTest('TensorBoard absent')
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        a=EventAccumulator(str(self.folder/'sb3'));a.Reload()
        for tag in ('evaluation/success_rate','evaluation/motion/toward_actual_fraction',
                    'evaluation/motion/distance_improving_fraction','actions/box_clip_rollout/vector_fraction',
                    'policy/action_std_mean','train/value_loss','train/explained_variance'):
            self.assertIn(tag,a.Tags()['scalars'])
        self.assertFalse(any(t.startswith('bc/') for t in a.Tags()['scalars']))
        self.assertEqual(self.config['validation_seeds'],[86000,86001])


if __name__=='__main__':unittest.main()
