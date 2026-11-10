"""Default golden regression, explicit boundary experiment and read-only soft TD."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import mujoco
import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.simulation.kinematics import IKResult
from realman_rl_simtoreal.simulation import control


def native(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,dict):return {k:native(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [native(v) for v in x]
    return x


class ReachBoundaryTests(unittest.TestCase):
    def env(self,**kwargs):
        e=ReachEnv(**dict(dict(boundary_mode='reject_and_hold',reward_mode='safe_dense'),**kwargs))
        self.addCleanup(e.close)
        e.reset(seed=0)
        return e

    def test_original_default_matches_frozen_prechange_trajectories_exactly(self):
        cases=json.loads((PROJECT_ROOT/'tests/fixtures/reach_terminate_04a.json').read_text())
        for case in cases:
            for explicit in ({},{'boundary_mode':'terminate'}):
                with ReachEnv(**case['kwargs'],**explicit) as e:
                    self.assertEqual(native(e.reset(seed=case['seed'])),case['reset'])
                    for step in case['steps']:
                        self.assertEqual(native(e.step(np.array(step['action'],dtype=np.float32))),step['result'])

    def test_default_workspace_still_terminates_without_advancing(self):
        with ReachEnv(action_scale=.01,reward_mode='safe_dense') as e:
            e.reset(seed=0)
            _,reward,t,tr,info=e.step(np.ones(3,dtype=np.float32))
            self.assertTrue(t and not tr)
            self.assertEqual(info['termination_reason'],'workspace_limit')
            self.assertEqual(e.data.time,0)
            self.assertLess(reward,-201)

    def test_rejected_workspace_keeps_reference_and_advances_one_cycle(self):
        e=self.env(action_scale=.01)
        command,joint=e._command_position.copy(),e._joint_reference.copy()
        with patch.object(e.kinematics,'solve_ik',side_effect=AssertionError('Outside candidate must not run IK')):
            obs,r,t,tr,info=e.step(np.ones(3,dtype=np.float32))
        self.assertFalse(t or tr)
        self.assertTrue(info['is_action_rejected'])
        self.assertFalse(info['is_safety_failure'])
        self.assertEqual(info['rejection_reason'],'workspace_limit')
        self.assertEqual(info['physics_steps'],25)
        self.assertAlmostEqual(e.data.time,.05)
        np.testing.assert_array_equal(e._command_position,command)
        np.testing.assert_array_equal(e._joint_reference,joint)
        np.testing.assert_array_equal(e.data.ctrl,joint)
        self.assertAlmostEqual(r,-1.25)
        self.assertEqual(r,e.compute_reward(obs['achieved_goal'],obs['desired_goal'],info))

    def test_soft_joint_candidate_is_not_executed_or_clipped(self):
        e=self.env()
        candidate=e.kinematics.home_q.copy();candidate[0]+=np.deg2rad(6)
        answer=IKResult(True,candidate,1,0,0)
        with patch.object(e.kinematics,'solve_ik',return_value=answer),patch(
                'realman_rl_simtoreal.envs.reach_env.check_motion',side_effect=AssertionError('Do not preflight rejected candidate')):
            _,_,t,tr,info=e.step([1,0,0])
        self.assertFalse(t or tr)
        self.assertEqual(info['rejection_reason'],'joint_command_limit')
        np.testing.assert_array_equal(e.data.ctrl,e.kinematics.home_q)
        np.testing.assert_array_equal(candidate,answer.q)
        self.assertEqual(info['physics_steps'],25)

    def test_reject_keeps_last_issued_joint_reference_after_real_motion(self):
        e=self.env()
        e.step([1,0,0]);command,joint=e._command_position.copy(),e._joint_reference.copy()
        candidate=e.kinematics.home_q.copy();candidate[0]+=np.deg2rad(6)
        with patch.object(e.kinematics,'solve_ik',return_value=IKResult(True,candidate,1,0,0)):
            _,_,_,_,info=e.step([0,1,0])
        self.assertTrue(info['is_action_rejected'])
        np.testing.assert_array_equal(e._command_position,command)
        np.testing.assert_array_equal(e.data.ctrl,joint)

    def test_illegal_ik_shape_nan_or_model_limit_still_terminates(self):
        for q in (np.full(6,np.nan),np.full(6,np.inf),np.zeros(5),np.full(6,100.)):
            e=self.env()
            with patch.object(e.kinematics,'solve_ik',return_value=IKResult(True,q,1,0,0)):
                _,_,t,tr,info=e.step([1,0,0])
            self.assertTrue(t and not tr)
            self.assertEqual(info['termination_reason'],'joint_limit')
            self.assertFalse(info['is_action_rejected'])
            self.assertEqual(e.data.time,0)

    def test_ik_solver_failure_is_not_recoverable_boundary(self):
        e=self.env()
        with patch.object(e.kinematics,'solve_ik',return_value=IKResult(False,None,150,.1,.1,'not converged')):
            _,_,t,tr,info=e.step([1,0,0])
        self.assertTrue(t and not tr)
        self.assertEqual(info['termination_reason'],'ik_failure')
        self.assertFalse(info['is_action_rejected'])
        self.assertEqual(e.data.time,0)

    def test_actual_joint_limit_and_nonfinite_state_override_candidate_boundary(self):
        for kind in ('joint','nan'):
            e=self.env(action_scale=.01)
            if kind=='joint':
                e.data.qpos[0]=e.kinematics.limits[0,1]+.001
                mujoco.mj_forward(e.model,e.data)
            else:e.data.qvel[0]=np.nan
            _,_,t,tr,info=e.step([1,1,1])
            self.assertTrue(t and not tr)
            self.assertEqual(info['termination_reason'],'joint_limit' if kind=='joint' else 'numerical_failure')
            self.assertFalse(info['is_action_rejected'])
            self.assertEqual(e.data.time,0)

    def test_actual_speed_collision_and_drive_fault_not_masked_by_hold(self):
        e=self.env(action_scale=.01)
        e.data.qvel[0]=np.deg2rad(31)
        _,_,t,_,info=e.step([1,1,1])
        self.assertTrue(t);self.assertEqual(info['termination_reason'],'velocity_limit')
        e=self.env(action_scale=.01)
        e.model.geom('floor').pos[2]=.5
        mujoco.mj_setConst(e.model,mujoco.MjData(e.model));mujoco.mj_forward(e.model,e.data)
        _,_,t,_,info=e.step([1,1,1])
        self.assertTrue(t);self.assertEqual(info['termination_reason'],'collision')
        e=self.env(action_scale=.01)
        with patch.object(control.ForceMonitor,'observe',side_effect=ValueError('Injected audit fault')):
            _,_,t,_,info=e.step([1,1,1])
        self.assertTrue(t);self.assertEqual(info['termination_reason'],'drive_failure')
        self.assertEqual(info['physics_steps'],1)

    def test_real_orientation_guard_still_terminates_during_rejected_hold(self):
        e=self.env()
        for _ in range(5):result=e.step([1,1,1])
        _,reward,t,tr,info=result
        self.assertTrue(t and not tr)
        self.assertEqual(info['rejection_reason'],'joint_command_limit')
        self.assertEqual(info['termination_reason'],'orientation_limit')
        self.assertGreater(info['orientation_error_rad'],np.deg2rad(1))
        self.assertLess(reward,-202)

    def test_reward_scalar_batch_relabel_and_missing_flags(self):
        e=self.env()
        achieved=np.array([[0.,0,0],[.004,0,0]])
        desired=np.zeros((2,3))
        infos=[dict(is_safety_failure=False,is_action_rejected=True),dict(is_safety_failure=True,is_action_rejected=False)]
        np.testing.assert_allclose(e.compute_reward(achieved,desired,infos),[-1,-201.2])
        batch=dict(is_safety_failure=np.array([False,True]),is_action_rejected=np.array([True,False]))
        np.testing.assert_array_equal(e.compute_reward(achieved,desired,infos),e.compute_reward(achieved,desired,batch))
        self.assertEqual(e.compute_reward(achieved[0],desired[0],infos[0]),-1.)
        with self.assertRaises(ValueError):e.compute_reward(achieved[0],desired[0],dict(is_safety_failure=False))
        with self.assertRaises(ValueError):e.compute_reward(achieved[0],desired[0],dict(is_safety_failure=False,is_action_rejected=1))

    def test_timeout_and_success_distinct_from_rejection(self):
        e=self.env(action_scale=.01,max_episode_steps=1)
        _,_,t,tr,info=e.step([1,1,1])
        self.assertFalse(t);self.assertTrue(tr)
        self.assertEqual(info['termination_reason'],'time_limit')
        self.assertTrue(info['is_action_rejected']);self.assertFalse(info['is_safety_failure'])
        e=self.env(action_scale=.01)
        e._goal=e.kinematics.home_position.copy()  # test terminal precedence only
        _,_,t,tr,info=e.step([1,1,1])
        self.assertTrue(t);self.assertFalse(tr);self.assertTrue(info['is_success'])
        self.assertTrue(info['is_action_rejected'])

    def test_repeating_invalid_commands_is_costly_bounded_and_physically_integrated(self):
        e=self.env(action_scale=.01,max_episode_steps=5)
        total=0
        previous=e.data.qpos.copy();step=mujoco.mj_step;calls=0
        def checked(m,d):
            nonlocal previous,calls
            np.testing.assert_array_equal(d.qpos,previous)
            step(m,d);previous=d.qpos.copy();calls+=1
        with patch.object(mujoco,'mj_step',side_effect=checked):
            for _ in range(5):
                _,r,t,tr,info=e.step([1,1,1]);total+=r
                self.assertTrue(info['is_action_rejected'])
                self.assertEqual(info['physics_steps'],25)
        self.assertEqual(calls,125);self.assertAlmostEqual(e.data.time,.25)
        self.assertLess(total,-5)
        self.assertFalse(t);self.assertTrue(tr)
        with self.assertRaises(gym.error.ResetNeeded):e.step([0,0,0])

    def test_fixed_distance_units_and_official_checker_both_modes(self):
        for target in ('fixed','random'):
            for reward in ('distance','safe_dense'):
                e=self.env(target_mode=target,reward_mode=reward)
                check_env(e,skip_render_check=True)
        e=self.env(action_scale=.01,reward_mode='distance')
        obs,r,_,_,info=e.step([1,1,1])
        self.assertAlmostEqual(r,-.025)
        self.assertEqual(r,e.compute_reward(obs['achieved_goal'],obs['desired_goal'],info))

    def test_parameter_validation(self):
        for options in (dict(boundary_mode='clip'),dict(rejection_penalty=0),dict(rejection_penalty=np.nan)):
            with self.assertRaises(ValueError):ReachEnv(**options)


try:
    import torch
    from realman_rl_simtoreal.rl.common.env_factory import training_env
    from realman_rl_simtoreal.rl.sac import make_model
    from realman_rl_simtoreal.rl.common.evaluation import evaluate
    from realman_rl_simtoreal.tasks.reach import ENV_CONFIG
    from realman_rl_simtoreal.rl.sac_diagnostics import td_diagnostic, fingerprint
    AVAILABLE=True
except ImportError:AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Optional SB3/PyTorch not installed')
class SoftTDDiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.env,self.transform=training_env(dict(ENV_CONFIG,max_episode_steps=4,boundary_mode='reject_and_hold'))
        self.addCleanup(self.env.close)
        self.model=make_model(self.env,7,overrides=dict(buffer_size=32,learning_starts=4,batch_size=8,
                                                      policy_kwargs=dict(net_arch=[16,16])))
        self.model.learn(total_timesteps=12)

    def test_td_weights_counters_rng_unchanged_and_reproducible(self):
        before=fingerprint(self.model);tstate=torch.get_rng_state().clone();nstate=np.random.get_state()
        one,arr=td_diagnostic(self.model,seed=9,batch_size=12)
        two,arr2=td_diagnostic(self.model,seed=9,batch_size=12)
        self.assertEqual(one,two)
        np.testing.assert_array_equal(arr['td_target'],arr2['td_target'])
        self.assertEqual(before,fingerprint(self.model))
        self.assertTrue(torch.equal(tstate,torch.get_rng_state()))
        np.testing.assert_array_equal(nstate[1],np.random.get_state()[1])
        self.assertEqual(nstate[2:],np.random.get_state()[2:])

    def test_terminal_mask_and_entropy_target_hand_calculation(self):
        rb=self.model.replay_buffer
        rb.reset()
        obs=self.env.reset()
        for reward,done,timeout in [(-.25,False,False),(-.04,True,False),(-201.25,True,False),(-.25,True,True)]:
            rb.add(obs,obs,np.zeros((1,3),np.float32),np.array([reward],np.float32),np.array([done]),[{'TimeLimit.truncated':timeout}])
        alpha=float(self.model.log_ent_coef.detach().exp())
        def next_action(observations):
            n=len(observations['observation'])
            return torch.zeros(n,3),torch.full((n,),.5)
        def target(observations,actions):
            n=len(actions);return torch.full((n,1),2.),torch.full((n,1),3.)
        with patch.object(self.model.actor,'action_log_prob',side_effect=next_action),patch.object(self.model.critic_target,'forward',side_effect=target):
            summary,a=td_diagnostic(self.model,batch_size=4)
        np.testing.assert_array_equal(a['effective_done'],[0,1,1,0])
        np.testing.assert_allclose(a['td_target'],[-.25+.99*(2-.5*alpha),-.04,-201.25,-.25+.99*(2-.5*alpha)],atol=1e-6)
        self.assertEqual(summary['timeouts'],1)
        self.assertEqual(summary['effective_terminations'],2)
        self.assertAlmostEqual(a['td_target_wrong_done'][-1],-.25)

    def test_evaluation_no_training_and_same_goals_across_boundary_modes(self):
        before=fingerprint(self.model)
        a=evaluate('sac',[10000,10001],model=self.model,transform=self.transform,config=dict(ENV_CONFIG,max_episode_steps=2))
        b=evaluate('sac',[10000,10001],model=self.model,transform=self.transform,config=dict(ENV_CONFIG,max_episode_steps=2,boundary_mode='reject_and_hold'))
        self.assertEqual([e['target_position_m'] for e in a['episodes']],[e['target_position_m'] for e in b['episodes']])
        self.assertEqual(before,fingerprint(self.model))


if __name__=='__main__':unittest.main()
