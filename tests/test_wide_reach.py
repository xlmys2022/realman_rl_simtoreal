"""20 cm sampling, physical control and goal-relative PPO transfer contracts."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.rl.ppo import environment_config, make_ppo, parameter_hashes, optimizer_steps, save_ppo_checkpoint
from realman_rl_simtoreal.rl.common.env_factory import training_env
from realman_rl_simtoreal.rl.common.observation import ReachTransform, observation_metadata
from realman_rl_simtoreal.simulation.control import advance_control_cycle
from realman_rl_simtoreal.simulation.kinematics import rotation_vector


class WideReachTests(unittest.TestCase):
    def test_samples_reach_twenty_centimetres_and_reset_reproduces_targets(self):
        with ReachEnv(**environment_config('random_reach_20cm')) as env:
            goals=[]
            for seed in range(130000,130020):
                obs,info=env.reset(seed=seed)
                radius=np.linalg.norm(obs['desired_goal']-obs['achieved_goal'])
                self.assertGreaterEqual(radius,.005-1e-12)
                self.assertLessEqual(radius,.2+1e-12)
                self.assertEqual(env.data.time,0)
                np.testing.assert_array_equal(env.data.qpos,env.kinematics.home_q)
                goals.append(obs['desired_goal'])
                self.assertGreaterEqual(info['goal_sampling']['attempts'],1)
            self.assertGreater(max(np.linalg.norm(g-env.kinematics.home_position) for g in goals),.19)
            np.testing.assert_array_equal(env.reset(seed=130000)[0]['desired_goal'],goals[0])

    def test_wide_goal_executes_real_physics_beyond_old_envelope(self):
        config=environment_config('random_reach_20cm')
        config.update(target_mode='discrete',target_offsets=[[-.10,.04,-.06]])
        with ReachEnv(**config) as env:
            obs,_=env.reset(seed=1)
            physical=mujoco.mj_step
            previous=env.data.qpos.copy()
            count=0
            def checked(model,data):
                nonlocal previous,count
                np.testing.assert_array_equal(data.qpos,previous)
                physical(model,data)
                previous=data.qpos.copy()
                count+=1
            with patch.object(mujoco,'mj_step',side_effect=checked):
                for step in range(env.max_episode_steps):
                    delta=obs['desired_goal']-env._command_position
                    action=delta/env.action_scale
                    action*=min(1.,.5/max(np.linalg.norm(action),1e-30))
                    obs,_,terminated,truncated,info=env.step(action)
                    if terminated or truncated:break
            self.assertTrue(info['is_success'],info)
            self.assertLessEqual(info['distance_m'],.001)
            self.assertGreater(np.linalg.norm(obs['achieved_goal']-env.kinematics.home_position),.1)
            self.assertEqual(count,25*(step+1))
            np.testing.assert_array_equal(env.data.qpos,previous)

    def test_model_joint_limits_apply_to_wide_targets(self):
        with ReachEnv(**environment_config('random_reach_20cm')) as env:
            outside=env.kinematics.home_q.copy()
            outside[0]=env.kinematics.limits[0,1]+.01
            with self.assertRaisesRegex(ValueError,'joint limits'):
                env._validate_joint_target(outside)

    def test_synchronized_rate_preserves_joint_delta_ratios(self):
        with ReachEnv(**environment_config('random_reach_20cm')) as env:
            env.reset(seed=1)
            start=env.data.ctrl.copy()
            delta=np.array([.02,-.01,.005,.001,-.01,.002])
            advance_control_cycle(env.model,env.data,env.joint_ids,env.actuator_ids,start+delta,
                                  max_joint_delta=.005,synchronized=True)
            np.testing.assert_allclose(env.data.ctrl-start,delta*.25,atol=1e-15,rtol=0)

    def test_wide_transform_preserves_si_goals(self):
        with ReachEnv(**environment_config('random_reach_20cm')) as env:
            raw,_=env.reset(seed=130004)
            transform=ReachTransform.from_env(env)
            encoded=transform.encode(raw)
            self.assertEqual(transform.config['schema'],6)
            np.testing.assert_array_equal(encoded['observation'],np.arcsinh(raw['observation']/.001).astype(np.float32))
            for key in ('achieved_goal','desired_goal'):
                np.testing.assert_array_equal(encoded[key],raw[key])

    def test_transfer_copies_weights_with_fresh_optimizer_and_wide_metadata(self):
        small={'observation_mode':'goal_relative_v1'}
        source_env,source_transform=training_env(small)
        wide_env,wide_transform=training_env(environment_config('random_reach_20cm'))
        self.addCleanup(source_env.close)
        self.addCleanup(wide_env.close)
        source,_=make_ppo(source_env,2027)
        config=dict(schema=2,algorithm='PPO',seed=2027,torch_num_threads=1,environment=small,
                    observation_transform=source_transform.config,
                    **observation_metadata(source_env.venv.envs[0].unwrapped,source_transform))
        with tempfile.TemporaryDirectory() as folder:
            save_ppo_checkpoint(source,folder,config,0,completed_stage=False)
            model,_=make_ppo(wide_env,2028,initial_model=Path(folder)/'model.zip')
            self.assertEqual(parameter_hashes(source),parameter_hashes(model))
            self.assertEqual(model.reach_observation_mode,'goal_relative_v3')
            self.assertEqual(optimizer_steps(model),0)
            self.assertEqual(model.num_timesteps,0)
            self.assertTrue(model.reach_transfer_configuration['optimizer_reset'])
            self.assertEqual(wide_transform.features_dim,12)

    def test_limited_cartesian_ik_preserves_each_issued_pose_orientation(self):
        with ReachEnv(**environment_config('random_reach_20cm')) as env:
            env.reset(seed=130004)
            for _ in range(30):
                previous=env.data.ctrl.copy()
                _,_,terminated,truncated,info=env.step([1,1,-1])
                self.assertFalse(terminated or truncated,info)
                self.assertLessEqual(np.max(np.abs(env.data.ctrl-previous)),np.deg2rad(.5)+1e-12)
                _,rotation=env.kinematics.fk(env.data.ctrl[env.actuator_ids])
                self.assertLessEqual(np.linalg.norm(rotation_vector(env.kinematics.home_rotation,rotation)),1e-4)
                self.assertIn('control_target_m',info['ik'])

    def test_wide_analysis_uses_centimetre_radius_groups(self):
        spec=importlib.util.spec_from_file_location('analysis',PROJECT_ROOT/'scripts/evaluate/analyze_reach_results.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        episodes=[]
        for i,radius in enumerate([.02,.07,.12,.19]):
            episodes.append(dict(seed=i,target_position_m=[radius,0,0],initial_position_m=[0,0,0],
                final_position_m=[radius,0,0],final_distance_m=0.,success=True,length=1,reason='success',
                actions=[[1,0,0]]))
        result=module.analyze(dict(environment={'target_radius':.2},episodes=episodes))
        self.assertEqual([result['radius_bins'][k]['samples'] for k in ['0-5cm','5-10cm','10-15cm','15-20cm']],[1]*4)


if __name__=='__main__':unittest.main()
