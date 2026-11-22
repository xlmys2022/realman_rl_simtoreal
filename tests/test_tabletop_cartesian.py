"""Exercise the policy/controller boundary and real MuJoCo dynamics."""
import ast
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig as Config
from realman_rl_simtoreal.envs.grasp.tabletop_cartesian_env import TabletopCartesianEnv as TabletopEnv, ACTION_CONTRACT, OBSERVATION_FIELDS
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian_reward import calculate, potential, REWARD
from realman_rl_simtoreal.paths import PROJECT_ROOT as ROOT
from realman_rl_simtoreal.simulation.motion.kinematics import IKResult, rotation_vector
from realman_rl_simtoreal.simulation.runtime.observation_presets import OBSERVATION_POSES, startup_contract


class CartesianPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # This is an actual zero -> observation rollout, reused only at resets.
        cls.config = replace(Config(), cache_startup=True)
        cls.env = TabletopEnv(cls.config)
        cls.first_observation, cls.first_info = cls.env.reset(seed=2027)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.obs, self.info = self.env.reset(seed=2027)

    def test_zero_to_observation_uses_physics_before_policy(self):
        report = self.first_info['startup']
        self.assertEqual(report['mode'], 'simulated_traditional_control')
        np.testing.assert_array_equal(report['initial_q_deg'], np.zeros(6))
        self.assertEqual(report['policy_actions'], 0)
        self.assertGreater(report['simulated_seconds'], 20)
        self.assertAlmostEqual(report['cycles'] * .05, report['simulated_seconds'], places=7)
        np.testing.assert_allclose(report['actual_tcp_m'], (.1, 0, 1.1), atol=.001)
        camera = report['camera']
        np.testing.assert_allclose(camera['forward'], (0, 0, -1), atol=.001)
        np.testing.assert_allclose(camera['up'], (1, 0, 0), atol=.001)
        self.assertFalse(camera['optical_frame_calibrated'])
        self.assertEqual(self.first_info['policy_steps'], 0)
        self.assertAlmostEqual(self.first_info['policy_time_s'], 0.)

    def test_cached_training_reset_is_explicit_and_preserves_new_cube(self):
        self.assertEqual(self.info['startup']['mode'], 'cached_verified_observation_reset')
        self.assertEqual(self.info['startup']['simulated_seconds'], 0)
        self.assertAlmostEqual(self.env.robot.data.time, 0)
        env = TabletopEnv(replace(self.config, object_radius=.01))
        try:
            env.robot._cache = copy.deepcopy(self.env.robot._cache)
            _, first = env.reset(seed=11)
            _, second = env.reset(seed=12)
            self.assertGreater(np.linalg.norm(np.array(first['object_position_m']) - second['object_position_m']), .001)
            np.testing.assert_allclose(second['tcp_position_m'], self.config.observation_tcp, atol=.001)
        finally:
            env.close()

    def test_joint_targets_are_debug_metadata_only(self):
        self.assertEqual(self.env.action_space.shape, (4,))
        self.assertEqual(self.obs.shape, (29,))
        self.assertEqual(sum(size for _, size in OBSERVATION_FIELDS), 29)
        self.assertTrue(np.isfinite(self.obs).all())
        altered = copy.deepcopy(self.info)
        altered.update(joint_position_rad=[99] * 6, joint_reference_rad=[-99] * 6,
                       grasp_joint_target=[123] * 6, lift_joint_target=[-123] * 6)
        np.testing.assert_array_equal(self.env._observation(altered), self.obs)
        self.assertEqual(potential(altered, self.config), potential(self.info, self.config))
        self.assertEqual(calculate(0, altered, False, np.zeros(4), self.config),
                         calculate(0, self.info, False, np.zeros(4), self.config))

    def test_fk_and_ik_are_pure_and_control_the_actual_tcp(self):
        robot = self.env.robot
        kin = robot.kinematics
        before = {name: getattr(robot.data, name).copy() for name in ('qpos', 'qvel', 'ctrl', 'qacc')}
        time_before = robot.data.time
        q = robot.data.qpos[kin.qadr].copy()
        position, rotation = kin.fk(q)
        np.testing.assert_allclose(position, robot.tcp_pose()[0], atol=1e-12)
        target = position + np.array([.001, -.001, 0])
        result = kin.solve_ik(target, q, rotation, position_tolerance=2e-6)
        self.assertTrue(result.success, result.reason)
        solved_position, solved_rotation = kin.fk(result.q)
        np.testing.assert_allclose(solved_position, target, atol=2e-6)
        self.assertLess(np.linalg.norm(rotation_vector(rotation, solved_rotation)), 1e-4)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(robot.data, name), value)
        self.assertEqual(robot.data.time, time_before)
        unreachable = kin.solve_ik((100., 0, 0), q, rotation)
        self.assertFalse(unreachable.success)
        self.assertIsNone(unreachable.q)

    def test_jacobian_matches_tcp_finite_difference(self):
        kin = self.env.robot.kinematics
        q = self.env.robot.data.qpos[kin.qadr].copy()
        jacobian, _ = kin.jacobian(q)
        for axis in range(6):
            offset = np.eye(6)[axis] * 1e-6
            finite = (kin.fk(q + offset)[0] - kin.fk(q - offset)[0]) / 2e-6
            np.testing.assert_allclose(jacobian[:, axis], finite, atol=1e-7)

    def test_actions_advance_shared_physics_with_bounded_joint_references(self):
        robot = self.env.robot
        old_reference = robot.data.ctrl[robot.actuator_ids].copy()
        old_tcp = robot.tcp_pose()[0]
        old_time = robot.data.time
        for _ in range(8):
            _, _, terminated, truncated, info = self.env.step(np.array([.4, 0, 0, .8], dtype=np.float32))
            self.assertFalse(terminated, info['detail'])
            self.assertFalse(truncated)
            new_reference = robot.data.ctrl[robot.actuator_ids].copy()
            self.assertLessEqual(max(abs(new_reference - old_reference)), self.config.joint_step + 1e-9)
            self.assertEqual(info['physics_steps'], 50)
            old_reference = new_reference
        self.assertAlmostEqual(robot.data.time - old_time, .4, places=9)
        self.assertGreater(robot.tcp_pose()[0][0] - old_tcp[0], .0005)
        self.assertFalse(np.any(robot.data.qfrc_applied))
        self.assertFalse(np.any(robot.data.xfrc_applied))

    def test_zero_motion_keeps_cartesian_reference_fixed(self):
        reference = self.env.controller.reference.copy()
        for _ in range(10):
            self.env.step(np.array([0, 0, 0, .8], dtype=np.float32))
        np.testing.assert_allclose(self.env.controller.reference, reference, atol=1e-12)

    def test_incremental_jaw_holds_zero_and_integrates_bounded_width_requests(self):
        env=TabletopEnv(replace(self.config,jaw_control='delta'))
        try:
            env.robot._cache=copy.deepcopy(self.env.robot._cache)
            env.reset(seed=2027)
            for _ in range(10):env.step(np.zeros(4,dtype=np.float32))
            self.assertAlmostEqual(env._jaw_reference,.06)
            for _ in range(10):
                _,_,terminated,_,info=env.step(np.array([0,0,0,-.25],dtype=np.float32))
                self.assertFalse(terminated,info['detail'])
            self.assertAlmostEqual(env._jaw_reference,.055)
            self.assertLess(env.robot.gripper.get_width(),.057)
            for _ in range(10):env.step(np.zeros(4,dtype=np.float32))
            self.assertAlmostEqual(env._jaw_reference,.055)
            for _ in range(40):
                _,_,terminated,_,info=env.step(np.array([0,0,0,-1],dtype=np.float32))
                self.assertFalse(terminated,info['detail'])
            self.assertAlmostEqual(env._jaw_reference,env.config.jaw_margin)
            self.assertEqual(info['jaw_control'],'delta')
        finally:env.close()

    def test_workspace_limits_do_not_cancel_a_valid_tangential_request(self):
        env=TabletopEnv(replace(self.config,workspace_low=(-.15,-.3,1.0995)))
        try:
            env.robot._cache=copy.deepcopy(self.env.robot._cache)
            env.reset(seed=2027)
            start=env.robot.tcp_pose()[0]
            for _ in range(12):
                _,_,terminated,_,info=env.step(np.array([.5,0,-1,.8],dtype=np.float32))
                self.assertFalse(terminated,info['detail'])
                self.assertTrue(info['workspace_limited'])
            self.assertGreater(env.robot.tcp_pose()[0][0]-start[0],.002)
            self.assertGreaterEqual(env.controller.reference[2],1.0995-1e-9)
        finally:env.close()

    def test_failed_ik_holds_arm_but_executes_neural_jaw(self):
        robot = self.env.robot
        before = robot.data.ctrl.copy()
        failure = IKResult(False, None, 0, None, None, 'test rejection')
        with patch.object(robot.kinematics, 'solve_ik', return_value=failure) as solve:
            _, _, terminated, _, info = self.env.step(np.array([.2, 0, 0, -.8], dtype=np.float32))
        self.assertGreater(solve.call_count, 0)
        self.assertFalse(terminated)
        self.assertEqual(info['ik_fraction'], 0)
        np.testing.assert_array_equal(robot.data.ctrl[robot.actuator_ids], before[robot.actuator_ids])
        self.assertLess(robot.data.ctrl[robot.gripper.actuator_id], before[robot.gripper.actuator_id])
        self.assertEqual(info['physics_steps'], 50)
        self.assertAlmostEqual(robot.data.time, .05, places=9)

    def test_invalid_actions_do_not_advance_time(self):
        for action in (np.zeros(7), [0, 0, np.nan, 0], [0, 0, 0, 1.01]):
            with self.assertRaises(ValueError):
                self.env.step(action)
        self.assertEqual(self.env.robot.data.time, 0)

    def test_cartesian_execution_can_physically_pick_and_hold(self):
        # A controller diagnostic, not a PPO success claim or training data.
        from realman_rl_simtoreal.simulation.devices.mujoco_gripper import MAX_WIDTH
        env = self.env
        final = None
        initial_potential=potential(self.info,env.config)
        idle=calculate(initial_potential,self.info,False,np.zeros(4),env.config)[0]
        discounted=0.;discount=1.
        stages = [('approach', (.4, 0, .79), .06, 1., 550),
                  ('descend', (.4, 0, .763), .06, .4, 150),
                  ('close', (.4, 0, .763), .002, .4, 30),
                  ('lift', (.4, 0, .795), .002, .3, 200),
                  ('hold', (.4, 0, .795), .002, .3, 90)]
        for phase, target, width, limit, budget in stages:
            for _ in range(budget):
                increment = (np.asarray(target) - env.controller.reference) / (
                    env.config.action_scale * env.config.cartesian_smoothing)
                action = np.r_[np.clip(increment, -limit, limit), width / MAX_WIDTH * 2 - 1].astype(np.float32)
                _, reward, terminated, truncated, final = env.step(action)
                discounted+=discount*reward;discount*=REWARD['gamma']
                self.assertFalse(final['is_safety_failure'], final['detail'])
                self.assertFalse(truncated)
                if terminated:
                    break
                if phase in ('approach', 'descend', 'lift') and np.linalg.norm(
                        np.asarray(target) - final['tcp_position_m']) < .0002:
                    break
            if terminated:
                break
        self.assertTrue(final['is_success'], final['termination_reason'])
        self.assertGreaterEqual(final['clearance_m'], .028)
        self.assertGreaterEqual(final['hold_seconds'], 2.)
        self.assertGreater(min(final['contact_forces_n'][:2]), .005)
        self.assertLess(final['contact_forces_n'][2], .001)
        idle_return=idle*(1-REWARD['gamma']**env.config.max_steps)/(1-REWARD['gamma'])
        self.assertGreater(discounted,idle_return)


class PackageTests(unittest.TestCase):
    def test_invalid_workspace_is_rejected_before_robot_creation(self):
        for values in (dict(workspace_low=(0,0)), dict(workspace_high=(0,0,0)),
                       dict(observation_pose='object_relative'),dict(control_mode='joint'),
                       dict(cache_startup='false'), dict(axis_scale=(0,1,1))):
            with self.assertRaises(ValueError):
                replace(Config(), **values)
        with self.assertRaises(TypeError):
            Config(observation_tcp=(.4,0,.763))

    def test_feature_uses_shared_physics_and_has_no_joint_goal_coupling(self):
        from realman_rl_simtoreal.simulation.runtime import tabletop as tabletop_runtime
        from realman_rl_simtoreal.simulation.physics import joint_control as control
        from realman_rl_simtoreal.simulation.devices import mujoco_gripper as gripper
        from realman_rl_simtoreal.simulation.motion.kinematics import Kinematics
        from realman_rl_simtoreal.simulation.motion.tool_kinematics import ToolKinematics
        self.assertIs(ToolKinematics.solve_ik, Kinematics.solve_ik)
        self.assertIs(ToolKinematics.fk, Kinematics.fk)
        self.assertIs(ToolKinematics.jacobian, Kinematics.jacobian)
        self.assertIs(tabletop_runtime.advance_synchronized_cycle, control.advance_synchronized_cycle)
        self.assertIs(tabletop_runtime.Gripper, gripper.Gripper)
        environment=(ROOT/'src/realman_rl_simtoreal/envs/grasp/tabletop_cartesian_env.py').read_text()
        self.assertNotIn('solve_ik',environment)
        self.assertNotIn('grasp_q',environment)
        self.assertNotIn('tabletop_pick_env',environment)
        for path in (ROOT/'src/realman_rl_simtoreal').rglob('*.py'):
            self.assertNotIn('from realman_ee',path.read_text())

    def test_camera_additions_do_not_change_dynamics(self):
        source = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/scenes/tabletop_pick.xml'))
        derived = mujoco.MjModel.from_xml_path(str(ROOT / 'assets/scenes/tabletop_cartesian.xml'))
        for name in ('body_mass', 'body_inertia', 'body_pos', 'body_quat', 'jnt_range', 'jnt_axis',
                     'geom_size', 'geom_friction', 'geom_contype', 'geom_conaffinity',
                     'actuator_gainprm', 'actuator_biasprm', 'actuator_forcerange',
                     'pair_friction', 'pair_solref', 'pair_solimp', 'eq_data'):
            np.testing.assert_array_equal(getattr(derived, name), getattr(source, name), err_msg=name)



class RandomEvaluationTests(unittest.TestCase):
    def setUp(self):
        import torch
        from gymnasium import spaces
        from realman_rl_simtoreal.simulation.physics.grasp_metrics import CONTACT_MEASUREMENT
        policy=torch.nn.Module();policy.optimizer=SimpleNamespace(state={},state_dict=lambda:dict(state={},param_groups=[]))
        self.model=SimpleNamespace(policy=policy,action_space=spaces.Box(-1.,1.,(4,),dtype=np.float32),
                                   num_timesteps=123,_n_updates=4)
        # Full evaluation must restore completion criteria, while an explicit
        # random-position request must survive that restoration.
        self.config=dict(schema=2,environment=replace(Config(),object_radius=.01,
            success_height=.005,height_tolerance=.0005,hold_seconds=.2,max_steps=8,cache_startup=True).to_dict(),
            startup_contract=startup_contract('table_down'),contact_measurement=CONTACT_MEASUREMENT)

    def test_full_evaluation_preserves_explicit_randomization_and_strict_criteria(self):
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import evaluate
        def environment(cfg,**kwargs):return SimpleNamespace(config=cfg,close=lambda:None)
        with patch('realman_rl_simtoreal.evaluation.tabletop_cartesian.TabletopEnv',side_effect=environment), \
             patch('realman_rl_simtoreal.evaluation.tabletop_cartesian.rollout',return_value=dict(success=False,reason='time_limit')):
            for radius in (None,0.,.02,.05):
                result=evaluate(self.model,self.config,[8],full=True,object_radius=radius)
                cfg=result['environment']
                self.assertEqual(cfg['object_radius'],0. if radius is None else radius)
                self.assertTrue(result['full_task']);self.assertTrue(result['parameters_unchanged'])
                self.assertIsNone(cfg['success_height']);self.assertEqual(cfg['lift_height'],.03)
                self.assertEqual(cfg['height_tolerance'],.002);self.assertEqual(cfg['hold_seconds'],2.)
                self.assertEqual(cfg['max_steps'],1600);self.assertFalse(cfg['cache_startup'])
                self.assertEqual(result['startup_contract'],startup_contract('table_down'))

    def test_invalid_randomization_is_rejected_before_robot_allocation(self):
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import evaluate
        with patch('realman_rl_simtoreal.evaluation.tabletop_cartesian.TabletopEnv') as allocate:
            for radius in (-.001,.051,np.nan,np.inf):
                with self.assertRaises(ValueError):evaluate(self.model,self.config,[8],full=True,object_radius=radius)
            allocate.assert_not_called()

    def test_custom_environment_cannot_hide_a_randomization_override(self):
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import evaluate
        with self.assertRaisesRegex(ValueError,'must construct its own'):
            evaluate(self.model,self.config,[8],environment=object(),object_radius=.02)


class PPOIntegrationTests(unittest.TestCase):
    def test_source_capture_does_not_include_training_outputs(self):
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import capture_sources
        with tempfile.TemporaryDirectory(dir=ROOT, prefix='source-copy-test-') as temp:
            capture_sources(temp)
            import zipfile
            with zipfile.ZipFile(Path(temp)/'source_snapshot.zip') as archive:
                names=archive.namelist()
            self.assertIn('assets/scenes/tabletop_cartesian.xml',names)
            self.assertIn('src/realman_rl_simtoreal/envs/grasp/tabletop_cartesian_env.py',names)
            self.assertFalse(any(name.startswith('outputs/') for name in names))
            self.assertFalse(any(name.startswith(Path(temp).name+'/') for name in names))

    def test_real_ppo_updates_both_networks_and_saved_inference_is_immutable(self):
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import train, load, evaluate, hashes
        with tempfile.TemporaryDirectory(prefix='realman-ee-test-') as temp:
            args = SimpleNamespace(stage='pinch', timesteps=64, n_envs=1, n_steps=32,
                                   eval_every=64, learning_rate=.0003, std=.2,jaw_std=.6,
                                   run_dir=Path(temp) / 'run', init_model=None,
                                   reset_exploration=False, max_steps=32, seed=901,cache_startup=True)
            result = train(args)
            self.assertEqual(result['status'], 'completed', result['error'])
            self.assertTrue(result['actor_updated'])
            self.assertTrue(result['critic_updated'])
            self.assertGreater(result['optimizer_steps'], 0)
            checkpoint = args.run_dir / 'final/model.zip'
            model, config = load(checkpoint)
            self.assertEqual(model.policy.exploration_profile,'grasp_lift')
            self.assertEqual(config['exploration']['exploration_profile'],'grasp_lift')
            np.testing.assert_allclose(config['initial_action_std'],[.2,.2,.2,.6],atol=1e-6)
            self.assertEqual(config['action_contract'], ACTION_CONTRACT)
            self.assertFalse(config['kinematic_joint_supervision'])
            self.assertEqual(config['startup_contract'],startup_contract('table_down'))
            self.assertNotIn('observation_tcp',config['environment'])
            state = json.loads(checkpoint.with_name('state.json').read_text())
            self.assertTrue(state['completed_update'])
            self.assertEqual(hashes(model), result['final_parameter_hashes'])
            with self.assertRaisesRegex(ValueError,'must construct its own'):
                evaluate(model,config,[902],full=True,environment=object())
            evaluated = evaluate(model, config, [902])
            self.assertTrue(evaluated['parameters_unchanged'])
            first = evaluated['episodes'][0]
            self.assertEqual(first['initial_startup']['mode'], 'simulated_traditional_control')
            self.assertEqual(first['initial_startup']['policy_actions'], 0)
            np.testing.assert_allclose(first['initial_startup']['actual_tcp_m'],(.1,0,1.1),atol=.001)
            for row in first['records']:
                np.testing.assert_array_equal(np.clip(row['raw_action'], -1, 1), row['action'])
                self.assertEqual(row['physics_steps'], 50)
            self.assertEqual(hashes(model), result['final_parameter_hashes'])
            # Prevent silently loading the old project's joint policy.
            config_path = checkpoint.with_name('config.json')
            config['action_contract']['shape'] = 7
            config_path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'Cartesian checkpoint'):
                load(checkpoint)

    def test_pose_mode_ppo_updates_and_restores_the_fixed_forward_start(self):
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import train,load,evaluate
        from realman_rl_simtoreal.envs.grasp.tabletop_cartesian_env import POSE_ACTION_CONTRACT
        with tempfile.TemporaryDirectory(prefix='realman-pose-test-') as temp:
            args=SimpleNamespace(stage='easy',timesteps=64,n_envs=1,n_steps=32,eval_every=64,
                learning_rate=.0003,std=.15,run_dir=Path(temp)/'run',init_model=None,
                reset_exploration=False,max_steps=32,seed=904,cache_startup=True,observation_pose='forward')
            result=train(args)
            self.assertEqual(result['status'],'completed',result['error'])
            self.assertTrue(result['actor_updated']);self.assertTrue(result['critic_updated'])
            model,config=load(args.run_dir/'final/model.zip')
            self.assertEqual(model.policy.exploration_profile,'grasp_lift')
            self.assertEqual(config['action_contract'],POSE_ACTION_CONTRACT)
            self.assertEqual(model.observation_space.shape,(38,))
            evaluation=evaluate(model,config,[905])
            self.assertTrue(evaluation['parameters_unchanged'])
            initial=evaluation['episodes'][0]['initial_startup']
            self.assertEqual(initial['startup_contract'],startup_contract('forward'))
            self.assertEqual(initial['mode'],'simulated_traditional_control')
            np.testing.assert_allclose(initial['actual_tcp_m'],(.15,0,1.15),atol=.001)

    def test_legacy_near_object_policy_is_not_silently_evaluated(self):
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import load
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.zip'
            path.with_name('config.json').write_text(json.dumps(dict(schema=1,environment=dict(observation_tcp=[.4,0,.763]))))
            with self.assertRaisesRegex(ValueError,'Legacy near-object startup'):
                load(path)


class FixedObservationTests(unittest.TestCase):
    def test_all_curricula_use_the_task_preset_and_full_translation_range(self):
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import STAGES,environment_config
        for name,profile in OBSERVATION_POSES.items():
            for stage in STAGES:
                cfg=environment_config(stage,observation_pose=name)
                self.assertEqual(cfg.observation_tcp,profile.tcp_m)
                self.assertEqual(cfg.initial_width,.06)
                self.assertEqual(cfg.axis_scale,(1.,1.,1.))
                self.assertFalse(cfg.cache_startup)
                self.assertNotIn('observation_tcp',STAGES[stage])

    def test_actual_startup_target_and_trajectory_do_not_follow_objects(self):
        from realman_rl_simtoreal.simulation.runtime.tabletop import TabletopRuntime
        for name,profile in OBSERVATION_POSES.items():
            robot=TabletopRuntime(Config(observation_pose=name))
            traces=[]
            for xy in ((.37,-.025),(.43,.025)):
                trace=[]
                def tick(runtime,phase):
                    trace.append(runtime.data.ctrl[runtime.actuator_ids].copy())
                with patch.object(robot.kinematics,'solve_ik',side_effect=AssertionError('Startup must not target an object')):
                    report,_=robot.prepare(xy,tick=tick)
                traces.append(np.array(trace))
                np.testing.assert_array_equal(report['target_q_deg'],profile.joints_deg)
                np.testing.assert_allclose(report['actual_tcp_m'],profile.tcp_m,atol=.001)
                np.testing.assert_allclose(report['tool_forward'],np.array(profile.tool_rotation)[:,2],atol=.001)
                np.testing.assert_allclose(report['camera']['forward'],np.array(profile.tool_rotation)[:,2],atol=.001)
                self.assertEqual(report['policy_actions'],0)
                self.assertTrue(report['startup_contract']['object_independent'])
            np.testing.assert_array_equal(traces[0],traces[1])

    def test_blocked_observation_path_aborts_without_choosing_another_pose(self):
        from realman_rl_simtoreal.simulation.runtime.tabletop import TabletopRuntime
        from realman_rl_simtoreal.simulation.runtime.observation_startup import observation_solution
        robot=TabletopRuntime(Config());robot.reset_zero((.4,0))
        with patch.object(robot,'path_is_safe',return_value=False),patch.object(robot.kinematics,'solve_ik') as solve:
            with self.assertRaisesRegex(ValueError,'observation path is obstructed'):
                observation_solution(robot)
        solve.assert_not_called()
        self.assertEqual(robot.data.time,0)


class PoseControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env=TabletopEnv(Config(observation_pose='forward',cache_startup=True))
        cls.initial_obs,cls.initial_info=cls.env.reset(seed=906)

    @classmethod
    def tearDownClass(cls):cls.env.close()

    def setUp(self):self.obs,self.info=self.env.reset(seed=906)

    def test_handoff_has_no_hidden_approach_or_automatic_turn(self):
        from realman_rl_simtoreal.simulation.devices.mujoco_gripper import MAX_WIDTH
        self.assertEqual(self.obs.shape,(38,));self.assertEqual(self.env.action_space.shape,(7,))
        self.assertEqual(self.info['policy_steps'],0)
        self.assertEqual(self.info['policy_time_s'],0)
        self.assertGreater(np.linalg.norm(np.array(self.info['object_position_m'])-self.info['tcp_position_m']),.4)
        position,rotation=self.env.robot.tcp_pose()
        reference=self.env.controller.reference.copy();orientation=self.env.controller.orientation.copy()
        action=np.zeros(7,dtype=np.float32);action[-1]=.06/MAX_WIDTH*2-1
        for _ in range(8):
            _,_,terminated,_,info=self.env.step(action)
            self.assertFalse(terminated,info['detail'])
        np.testing.assert_array_equal(self.env.controller.reference,reference)
        np.testing.assert_allclose(self.env.controller.orientation,orientation,atol=1e-12)
        np.testing.assert_allclose(self.env.robot.tcp_pose()[0],position,atol=.001)
        np.testing.assert_allclose(self.env.robot.tcp_pose()[1],rotation,atol=.001)
        self.assertAlmostEqual(info['policy_time_s'],.4)

    def test_policy_rotation_actions_turn_the_actual_tool_through_ik(self):
        robot=self.env.robot
        before=robot.data.ctrl[robot.actuator_ids].copy()
        initial_error=self.info['grasp_orientation_error_rad']
        for _ in range(12):
            _,_,terminated,_,info=self.env.step(np.array([0,0,0,0,.4,0,.8],dtype=np.float32))
            self.assertFalse(terminated,info['detail'])
            now=robot.data.ctrl[robot.actuator_ids].copy()
            self.assertLessEqual(max(abs(now-before)),self.env.config.joint_step+1e-9)
            self.assertEqual(info['physics_steps'],50)
            before=now
        self.assertLess(info['grasp_orientation_error_rad'],initial_error-.01)
        np.testing.assert_allclose(info['tcp_position_m'],(.15,0,1.15),atol=.002)

    def test_pose_observations_and_rewards_do_not_include_joint_targets(self):
        altered=copy.deepcopy(self.info)
        altered.update(joint_position_rad=[100]*6,joint_reference_rad=[-100]*6)
        np.testing.assert_array_equal(self.env._observation(altered),self.obs)
        self.assertEqual(potential(altered,self.env.config),potential(self.info,self.env.config))


if __name__ == '__main__':
    unittest.main()
