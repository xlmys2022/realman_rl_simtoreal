"""Random resets, reward handovers and physically continuous goal sequences."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np

from realman_rl_simtoreal.envs.factory import make_reach_env
from realman_rl_simtoreal.envs.sequence_reach_env import SequenceReachEnv
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.rl.ppo import environment_config,make_ppo,load_ppo,evaluate_ppo,save_ppo_checkpoint
from realman_rl_simtoreal.rl.common.env_factory import training_env
from realman_rl_simtoreal.rl.common.evaluation import evaluate,summarize_episodes
from realman_rl_simtoreal.rl.common.observation import ReachTransform,observation_metadata
from realman_rl_simtoreal.simulation.kinematics import IKResult


def small_config(**overrides):
    config=environment_config('random_start_multi_reach_20cm')
    config.update(target_radius=.005,target_min_radius=.003,min_goal_distance=.002,
                  goals_per_episode=3,action_scale=.00025)
    config.update(overrides)
    return config


class SequenceReachTests(unittest.TestCase):
    def test_seeded_random_start_and_complete_plan_reproduce_after_motion(self):
        with make_reach_env(environment_config('random_start_multi_reach_20cm')) as env:
            obs,info=env.reset(seed=180000)
            start=env.data.qpos.copy()
            goals=env.goal_plan
            self.assertGreater(np.linalg.norm(obs['achieved_goal']-env.kinematics.home_position),.005)
            self.assertEqual(goals.shape,(5,3))
            self.assertTrue((np.linalg.norm(goals-env.kinematics.home_position,axis=1)<=.2+1e-12).all())
            np.testing.assert_array_equal(env.data.ctrl[env.actuator_ids],env.data.qpos[env.kinematics.qadr])
            np.testing.assert_array_equal(env.data.qvel,np.zeros(env.model.nv))
            self.assertEqual(env.data.time,0)
            self.assertEqual(info['goals_completed'],0)
            self.assertEqual(env.episode_step_limit,5000)
            env.step([.1,-.1,0])
            again,again_info=env.reset(seed=180000)
            np.testing.assert_array_equal(env.data.qpos,start)
            np.testing.assert_array_equal(env.goal_plan,goals)
            np.testing.assert_array_equal(again['desired_goal'],obs['desired_goal'])
            self.assertEqual(again_info['goal_plan_m'],info['goal_plan_m'])
            exposed=env.goal_plan
            exposed[:]=0
            np.testing.assert_array_equal(env.goal_plan,goals)

    def test_distinct_seeds_generate_distinct_start_poses(self):
        with make_reach_env(environment_config('random_start_multi_reach_20cm')) as env:
            starts=[tuple(env.reset(seed=s)[0]['achieved_goal']) for s in range(180000,180006)]
            self.assertEqual(len(set(starts)),6)

    def test_goal_handover_preserves_state_and_rewards_the_completed_goal(self):
        with make_reach_env(small_config()) as env:
            obs,_=env.reset(seed=23)
            physical=mujoco.mj_step
            previous=env.data.qpos.copy()
            physics_steps=0
            results=[]
            def checked(model,data):
                nonlocal previous,physics_steps
                np.testing.assert_array_equal(data.qpos,previous)
                physical(model,data)
                previous=data.qpos.copy()
                physics_steps+=1
            with patch.object(mujoco,'mj_step',side_effect=checked):
                for step in range(env.episode_step_limit):
                    desired_before=obs['desired_goal'].copy()
                    action=obs['observation'][:3]/env.action_scale
                    action*=min(1.,.5/max(np.linalg.norm(action),1e-30))
                    obs,reward,terminated,truncated,info=env.step(action)
                    self.assertEqual(reward,env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info))
                    if info['goal_result'] is not None:
                        result=info['goal_result']
                        results.append(result)
                        np.testing.assert_array_equal(info['transition_desired_goal'],desired_before)
                        self.assertEqual(env._physical_state(),result['end_state'])
                        if info['goal_changed']:
                            self.assertFalse(terminated or truncated)
                            self.assertFalse(info['is_success'])
                            self.assertTrue(info['is_goal_success'])
                            self.assertEqual(info['reward_components']['success_bonus'],20.)
                            self.assertFalse(np.array_equal(obs['desired_goal'],desired_before))
                            batch=env.compute_reward(np.tile(obs['achieved_goal'],(2,1)),
                                np.tile(obs['desired_goal'],(2,1)),np.array([info,info],dtype=object))
                            np.testing.assert_allclose(batch,[reward,reward],atol=1e-12,rtol=0)
                    if terminated or truncated:break
            self.assertTrue(info['is_success'],info)
            self.assertEqual(info['goals_completed'],3)
            self.assertEqual(len(results),3)
            self.assertEqual(physics_steps,25*(step+1))
            self.assertEqual(sum(r['length'] for r in results),step+1)
            for before,after in zip(results,results[1:]):
                self.assertEqual(before['end_state'],after['start_state'])
            self.assertGreater(np.linalg.norm(results[0]['end_state']['qvel']),1e-8)
            with self.assertRaises(gym.error.ResetNeeded):env.step([0,0,0])

    def test_timeout_ends_sequence_and_counts_unattempted_goals(self):
        with make_reach_env(small_config(max_episode_steps=1)) as env:
            env.reset(seed=23)
            _,_,terminated,truncated,info=env.step([0,0,0])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(info['termination_reason'],'time_limit')
            self.assertEqual(info['goals_completed'],0)
            self.assertEqual(info['goals_planned'],3)
            self.assertEqual(info['goal_result']['length'],1)
            with self.assertRaises(gym.error.ResetNeeded):env.step([0,0,0])

    def test_failed_planning_leaves_reset_state_inactive(self):
        with make_reach_env(small_config(max_target_attempts=2)) as env:
            failed=IKResult(False,None,1,.1,.1,'Injected IK failure')
            with patch.object(env.kinematics,'solve_ik',return_value=failed):
                with self.assertRaisesRegex(RuntimeError,'No validated sequence pose'):env.reset(seed=23)
            self.assertIsNone(env.goal_plan)
            self.assertEqual(env.data.time,0)
            np.testing.assert_array_equal(env.data.qpos,env.kinematics.home_q)
            with self.assertRaises(gym.error.ResetNeeded):env.step([0,0,0])

    def test_invalid_sequence_parameters(self):
        for values in [dict(goals_per_episode=0),dict(goals_per_episode=True),dict(goals_per_episode=1.5),
                       dict(start_mode='unknown'),dict(min_goal_distance=np.nan),dict(min_goal_distance=.001),
                       dict(min_goal_distance=.5),dict(track_issued_command=True,cartesian_rate_limit=False),
                       dict(control_ik_damping=0),dict(control_ik_seed_weight=-1)]:
            config=environment_config('random_start_multi_reach_20cm')
            config.update(values)
            with self.subTest(values=values),self.assertRaises(ValueError):make_reach_env(config)

    def test_command_tracks_the_issued_pose_without_accumulating_unexecuted_motion(self):
        with make_reach_env(environment_config('random_start_multi_reach_20cm')) as env:
            obs,_=env.reset(seed=180000)
            origin=env._command_position.copy()
            previous=env._joint_reference.copy()
            _,_,_,_,info=env.step(np.ones(3))
            self.assertIsNone(info['termination_reason'])
            np.testing.assert_array_equal(env._command_position,info['ik']['control_target_m'])
            np.testing.assert_allclose(env.kinematics.fk(env._joint_reference)[0],env._command_position,atol=1e-5,rtol=0)
            self.assertLessEqual(np.max(np.abs(env._joint_reference-previous)),np.deg2rad(.5)+1e-12)
            self.assertLess(np.linalg.norm(env._command_position-origin),np.sqrt(3)*env.action_scale)

    def test_failed_full_command_can_use_a_valid_shorter_cartesian_step(self):
        with make_reach_env(environment_config('random_start_multi_reach_20cm')) as env:
            env.reset(seed=180000)
            origin=env.kinematics.fk(env._joint_reference)[0]
            target=origin+np.array([.001,0,0])
            failed=IKResult(False,None,600,.1,.1,'Injected full-command failure')
            shorter=env.kinematics.solve_ik(origin+.5*(target-origin),env._joint_reference)
            self.assertTrue(shorter.success)
            with patch.object(env,'_solve_rate_limited_ik',side_effect=[(failed,target),(shorter,origin+.5*(target-origin))]) as solve:
                result,waypoint=env._solve_control_ik(target)
            self.assertTrue(result.success)
            self.assertEqual(solve.call_count,2)
            np.testing.assert_array_equal(waypoint,origin+.5*(target-origin))
            np.testing.assert_array_equal(env._command_position,origin)

    def test_invalid_reset_arguments_preserve_the_active_plan_and_state(self):
        with make_reach_env(small_config()) as env:
            with self.assertRaises(gym.error.ResetNeeded):env.step([0,0,0])
            env.reset(seed=23)
            plan=env.goal_plan
            state=env._physical_state()
            with self.assertRaises(gym.error.Error):env.reset(seed=-1)
            with self.assertRaises(ValueError):env.reset(options={'goal_index':1})
            np.testing.assert_array_equal(env.goal_plan,plan)
            self.assertEqual(env._physical_state(),state)
            env.step([0,0,0])

    def test_goal_and_sequence_success_denominators_include_early_failures(self):
        rows=[]
        for success,completed,attempted in [(False,2,3),(True,5,5)]:
            rows.append(dict(reason='success' if success else 'time_limit',success=success,safety_failure=False,
                             return_=0.,discounted_return=0.,length=10,final_distance_m=.001,
                             goals_planned=5,goals_completed=completed,goals_attempted=attempted))
        summary=summarize_episodes(rows)
        self.assertEqual(summary['sequence_success_rate'],.5)
        self.assertEqual(summary['goal_success_rate'],7/8)
        self.assertEqual(summary['goal_completion_rate'],.7)

    def test_evaluation_and_analysis_keep_per_goal_physics(self):
        config=small_config()
        with make_reach_env(config) as env:transform=ReachTransform.from_env(env)
        class Feedback:
            def predict(self,obs,deterministic):
                action=np.sinh(obs['observation'][:3].astype(float))*.001/config['action_scale']
                action*=min(1.,.5/max(np.linalg.norm(action),1e-30))
                return action.astype(np.float32),None
        measured=evaluate('ppo',[23],model=Feedback(),transform=transform,config=config,collect_motion=True)
        self.assertEqual(measured['summary']['goals_completed'],3)
        self.assertEqual(measured['summary']['sequence_success_rate'],1.)
        episode=measured['episodes'][0]
        self.assertEqual(sum(len(r['actions']) for r in episode['goal_results']),episode['length'])
        spec=importlib.util.spec_from_file_location('sequence_analysis',PROJECT_ROOT/'scripts/evaluate/analyze_reach_results.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        analyzed=module.analyze(measured)
        self.assertEqual(analyzed['overall']['samples'],3)
        self.assertTrue(analyzed['physical_handover_verified'])
        broken=copy.deepcopy(measured)
        broken['episodes'][0]['goal_results'][1]['start_state']['time_s']+=1
        with self.assertRaisesRegex(ValueError,'discontinuous'):module.analyze(broken)

    def test_checkpoint_reloads_sequence_factory_and_evaluates_full_episode(self):
        config=environment_config('random_start_multi_reach_20cm')
        config.update(max_episode_steps=2,goals_per_episode=3)
        env,transform=training_env(config)
        self.addCleanup(env.close)
        model,_=make_ppo(env,2027)
        metadata=dict(schema=2,algorithm='PPO',seed=2027,torch_num_threads=1,environment=config,
                      observation_transform=transform.config,reward_parameters=config['reward_progress_parameters'],
                      **observation_metadata(env.venv.envs[0].unwrapped,transform))
        model.reach_reward_configuration=dict(mode='reach_progress_v1',parameters=metadata['reward_parameters'])
        with tempfile.TemporaryDirectory() as folder:
            save_ppo_checkpoint(model,folder,metadata,0,completed_stage=False)
            loaded,saved,saved_config=load_ppo(Path(folder)/'model.zip')
            measured=evaluate_ppo(loaded,saved,saved_config['environment'],[180000])
            self.assertEqual(measured['summary']['goals_planned'],3)
            self.assertEqual(measured['episodes'][0]['length'],2)
            self.assertEqual(len(measured['episodes'][0]['goal_results']),1)
            self.assertTrue(measured['parameters_unchanged'])

    def test_gymnasium_goal_checker(self):
        with make_reach_env(small_config()) as env:check_env(env,skip_render_check=True)


if __name__=='__main__':unittest.main()
