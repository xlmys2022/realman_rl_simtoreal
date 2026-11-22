"""Regress the failed full-task reward ranking and PPO contact precision."""
import copy
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian_reward import potential,calculate,quantities,REWARD
from realman_rl_simtoreal.simulation.physics.grasp_metrics import inward_normal,grasp_forces
from realman_rl_simtoreal.simulation.devices.mujoco_gripper import width_to_angle,angle_to_width
from realman_rl_simtoreal.rl.cartesian_policy import CartesianPPOPolicy,reset_critic,reset_jaw_head
from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import hashes
from realman_rl_simtoreal.rl.cartesian_reach_policy import (ReachGraspPPOPolicy,initialize_reach_arm,branch_hashes)


def info(tcp=(.1,0,1.1),width=.06,grip=0.,height=0.):
    return dict(tcp_position_m=list(tcp),object_position_m=[.4,0,.76+height],object_goal_m=[.4,0,.79],
                contact_forces_n=[grip,grip,.0981 if height==0 else 0.],clearance_m=height,
                gripper_width_m=width,hold_seconds=0.,ik_fraction=1.,object_velocity_m_s=[0,0,0],
                is_success=False,is_safety_failure=False,termination_reason=None)


class TaskRewardTests(unittest.TestCase):
    def setUp(self):self.config=CartesianTabletopConfig()

    def idle(self,state):
        return calculate(potential(state,self.config),state,False,np.zeros(4),self.config)[0]

    def test_far_idle_is_negative_and_worse_than_correct_approach(self):
        far=info();near=info(tcp=(.4,0,.79),width=.06)
        self.assertLess(self.idle(far),0)
        self.assertGreater(self.idle(near),self.idle(far))

    def test_real_grip_and_lift_are_better_than_stationary_empty_jaw(self):
        empty=info(tcp=(.4,0,.763),width=.02)
        grip=info(tcp=(.4,0,.763),width=.02,grip=.13)
        lifted=info(tcp=(.4,0,.793),width=.02,grip=.13,height=.03)
        self.assertGreater(self.idle(grip),self.idle(empty))
        self.assertGreater(self.idle(lifted),self.idle(grip))
        lifted.update(is_success=True,termination_reason='success',hold_seconds=2.)
        value,_=calculate(potential(lifted,self.config),lifted,True,np.zeros(4),self.config)
        self.assertGreater(value,8.)

    def test_opening_a_prematurely_closed_empty_jaw_still_improves_reward(self):
        narrow=info(tcp=(.35,0,.78),width=.004)
        opening=info(tcp=(.35,0,.78),width=.012)
        self.assertGreater(self.idle(opening),self.idle(narrow))

    def test_ungrasped_approach_has_no_standoff_reward_peak(self):
        states=[(.06,.06),(.04,.06),(.03,.06),(.02,.06),(.01,.06),(.006,.06),(.003,.06)]
        rewards=[self.idle(info(tcp=(.4,0,.76+distance),width=width)) for distance,width in states]
        for farther,nearer in zip(rewards,rewards[1:]):
            self.assertGreater(nearer,farther)

    def test_alignment_is_rewarded_before_descending_into_the_cube(self):
        above=info(tcp=(.37,0,.812),width=.06)
        premature=info(tcp=(.37,0,.755),width=.06)
        self.assertGreater(self.idle(above),self.idle(premature))
        self.assertGreater(potential(above,self.config),potential(premature,self.config))

    def test_jaw_stays_open_until_the_cube_is_centered_between_the_pads(self):
        offset=info(tcp=(.4,.01,.763),width=.06)
        aligned=info(tcp=(.4,0,.763),width=.02)
        self.assertGreater(quantities(offset,self.config)['desired_width_m'],.059)
        self.assertAlmostEqual(quantities(aligned,self.config)['desired_width_m'],.02,places=8)

    def test_aligned_closure_has_progress_feedback_before_force_contact(self):
        previous=info(tcp=(.4,0,.763),width=.06)
        for width in [.05,.04,.03,.02]:
            closed=info(tcp=(.4,0,.763),width=width)
            self.assertGreater(potential(closed,self.config),potential(previous,self.config))
            self.assertGreater(self.idle(closed),self.idle(previous))
            previous=closed

    def test_failure_cannot_skip_the_negative_return_of_continuing(self):
        state=info()
        continuation=self.idle(state)*sum(REWARD['gamma']**step for step in range(self.config.max_steps))
        for reason in ('safety_failure','dropped'):
            failed=dict(state,termination_reason=reason,is_safety_failure=reason=='safety_failure')
            reward,_=calculate(potential(state,self.config),failed,True,np.zeros(4),self.config)
            self.assertLess(reward,continuation)

    def test_safety_failure_is_worse_than_reopening_and_continuing(self):
        state=info(width=.002)
        angles=np.linspace(width_to_angle(.002),width_to_angle(.06),41)
        self.assertLess(np.diff(angles).max()/self.config.control_period,.5)
        previous=potential(state,self.config);continuation=0.
        # A feasible recovery is the comparison, rather than intentionally
        # keeping an empty jaw closed for the full episode.
        for step in range(self.config.max_steps):
            reopened=info(width=angle_to_width(angles[min(step,40)]))
            reward,_=calculate(previous,reopened,False,np.zeros(4),self.config)
            previous=potential(reopened,self.config)
            continuation+=REWARD['gamma']**step*reward
        failed=dict(state,termination_reason='safety_failure',is_safety_failure=True)
        reward,_=calculate(potential(state,self.config),failed,True,np.zeros(4),self.config)
        self.assertLess(reward,continuation)

    def test_completing_full_hold_is_better_than_collecting_dense_rewards_forever(self):
        held=info(tcp=(.4,0,.793),width=.02,grip=.13,height=.03)
        held['hold_seconds']=2.
        infinite_hold=self.idle(held)/(1-REWARD['gamma'])
        success=dict(held,is_success=True,termination_reason='success')
        reward,_=calculate(potential(held,self.config),success,True,np.zeros(4),self.config)
        self.assertGreater(reward,infinite_hold)

    def test_opening_an_empty_jaw_takes_priority_over_rushing_closer(self):
        initial=info(tcp=(.35,0,.80),width=.004)
        opening=info(tcp=(.35,0,.80),width=.012)
        rushing=info(tcp=(.352,0,.80),width=.004)
        previous=potential(initial,self.config)
        opened=calculate(previous,opening,False,np.zeros(4),self.config)[0]
        rushed=calculate(previous,rushing,False,np.zeros(4),self.config)[0]
        self.assertGreater(opened,rushed)

    def test_potential_does_not_pay_for_failing_or_stopping_below_height(self):
        far=info();far.update(is_safety_failure=True,termination_reason='safety_failure')
        _,terms=calculate(potential(far,self.config),far,True,np.zeros(4),self.config)
        self.assertLessEqual(terms['potential'],0)
        initial=info(tcp=(.4,0,.790),width=.02,grip=.13,height=.027)
        raised=info(tcp=(.4,0,.791),width=.02,grip=.13,height=.028)
        raised['object_velocity_m_s']=[0,0,.005]
        value,terms=calculate(potential(initial,self.config),raised,False,np.zeros(4),self.config)
        self.assertEqual(terms['speed'],0)
        self.assertGreater(value,0)


class GraspMeasurementTests(unittest.TestCase):
    def test_top_pressure_is_not_an_inward_grip_normal(self):
        vertical=SimpleNamespace(geom=[4,7],frame=np.array([[0,0,1],[1,0,0],[0,1,0]]).ravel())
        self.assertEqual(inward_normal(vertical,.7,7,np.array([0,1,0])),0)
        side=SimpleNamespace(geom=[4,7],frame=np.array([[0,1,0],[1,0,0],[0,0,1]]).ravel())
        self.assertAlmostEqual(inward_normal(side,.13,7,np.array([0,1,0])),.13)
        reversed_contact=SimpleNamespace(geom=[7,4],frame=np.array([[0,-1,0],[1,0,0],[0,0,1]]).ravel())
        self.assertAlmostEqual(inward_normal(reversed_contact,.13,7,np.array([0,1,0])),.13)

    def test_empty_closed_jaw_cannot_count_as_straddling_a_twenty_mm_cube(self):
        recorder=SimpleNamespace(last=dict(actual_width_m=.002,support_normal_n=.0981))
        np.testing.assert_array_equal(grasp_forces(None,None,None,None,recorder),[0,0,.0981])


class ExplorationTests(unittest.TestCase):
    @staticmethod
    def policy(actions=4,observations=29):
        torch.set_num_threads(1)
        return CartesianPPOPolicy(spaces.Box(-np.inf,np.inf,(observations,),np.float32),
            spaces.Box(-1,1,(actions,),np.float32),lambda _:3e-4,
            net_arch=dict(pi=[16,16],vf=[16,16]),log_std_init=float(np.log(.3)))

    def test_sampling_and_update_log_probabilities_use_the_same_std(self):
        for dimensions in ((4,29),(7,38)):
            policy=self.policy(*dimensions)
            obs=torch.zeros((2,dimensions[1]));obs[0,6]=np.arcsinh(.3/.01);obs[1,6]=np.arcsinh(.003/.01)
            with torch.no_grad():
                actions,values,log_prob=policy(obs)
                checked_values,checked_prob,_=policy.evaluate_actions(obs,actions)
                std=policy.get_distribution(obs).distribution.stddev.clone()
                mean=policy.get_distribution(obs).distribution.mean.clone()
                policy.adaptive_exploration=False
                plain_mean=policy.get_distribution(obs).distribution.mean.clone()
            torch.testing.assert_close(values,checked_values)
            torch.testing.assert_close(log_prob,checked_prob)
            torch.testing.assert_close(mean,plain_mean)
            self.assertAlmostEqual(std[0,0].item(),.3,places=5)
            self.assertLess(std[1,0].item(),.025)
            self.assertGreater(std[1,-1].item(),std[1,0].item())

    def test_value_reset_does_not_modify_actor_mean_or_exploration(self):
        policy=self.policy();model=SimpleNamespace(policy=policy)
        before=hashes(model);std=policy.log_std.detach().clone()
        reset_critic(policy);after=hashes(model)
        self.assertEqual(before['actor_mean'],after['actor_mean'])
        self.assertNotEqual(before['critic'],after['critic'])
        torch.testing.assert_close(std,policy.log_std)

    def test_jaw_keeps_exploring_until_the_gripper_can_straddle_the_cube(self):
        policy=self.policy();policy.jaw_exploration_distance=.01
        obs=torch.zeros((1,29));obs[0,6]=np.arcsinh(.015/.01)
        with torch.no_grad():std=policy.get_distribution(obs).distribution.stddev.clone()
        self.assertAlmostEqual(std[0,-1].item(),.3,places=5)
        self.assertLess(std[0,0].item(),.15)

    def test_jaw_head_reset_preserves_arm_outputs_and_value_and_is_trainable(self):
        for dimensions in ((4,29),(7,38)):
            policy=self.policy(*dimensions);obs=torch.randn(3,dimensions[1])
            before=copy.deepcopy(policy.state_dict())
            with torch.no_grad():old=policy.get_distribution(obs).distribution.mean.clone()
            reset_jaw_head(policy,.812)
            with torch.no_grad():new=policy.get_distribution(obs).distribution.mean.clone()
            torch.testing.assert_close(new[:,:-1],old[:,:-1])
            torch.testing.assert_close(new[:,-1],torch.full((3,),.812))
            for name,value in policy.state_dict().items():
                if not name.startswith('action_net.'):torch.testing.assert_close(value,before[name])
            loss=policy.get_distribution(obs).distribution.mean[:,-1].sum()
            loss.backward()
            self.assertGreater(policy.action_net.weight.grad[-1].abs().sum().item(),0)

    def test_grasp_profile_explores_closure_only_when_aligned_and_quiets_a_real_grip(self):
        policy=self.policy();policy.exploration_profile='grasp_lift'
        obs=torch.zeros(3,29);obs[:,8]=np.arcsinh(-.003/.01)
        obs[0,6]=np.arcsinh(.05/.01);obs[2,21:23]=.5
        with torch.no_grad():std=policy.get_distribution(obs).distribution.stddev.clone()
        self.assertAlmostEqual(std[0,-1].item(),.03,places=5)
        self.assertAlmostEqual(std[1,-1].item(),.3,places=5)
        self.assertAlmostEqual(std[2,-1].item(),.03,places=5)

    def test_grasp_profile_explores_lift_until_the_goal_then_becomes_precise(self):
        policy=self.policy();policy.exploration_profile='grasp_lift'
        obs=torch.zeros(3,29);obs[:,8]=np.arcsinh(-.003/.01)
        obs[:2,21:23]=.5;obs[0,11]=obs[2,11]=np.arcsinh(.03/.01)
        with torch.no_grad():std=policy.get_distribution(obs).distribution.stddev.clone()
        self.assertAlmostEqual(std[0,2].item(),.45,places=5)
        self.assertLess(std[0,0].item(),.025)
        self.assertLess(std[1,2].item(),.025)
        self.assertLess(std[2,2].item(),.025)

    def test_grasp_profile_requires_orientation_and_preserves_ppo_likelihoods_and_means(self):
        for dimensions in ((4,29),(7,38)):
            policy=self.policy(*dimensions);policy.exploration_profile='grasp_lift'
            obs=torch.zeros(2,dimensions[1]);obs[:,8]=np.arcsinh(-.003/.01)
            if dimensions[1]==38:obs[1,29]=.5
            with torch.no_grad():
                actions,values,log_prob=policy(obs)
                checked_values,checked_prob,_=policy.evaluate_actions(obs,actions)
                distribution=policy.get_distribution(obs).distribution
                mean=distribution.mean.clone();std=distribution.stddev.clone()
                policy.adaptive_exploration=False
                plain_mean=policy.get_distribution(obs).distribution.mean.clone()
            torch.testing.assert_close(values,checked_values)
            torch.testing.assert_close(log_prob,checked_prob)
            torch.testing.assert_close(mean,plain_mean)
            if dimensions[1]==38:self.assertLess(std[1,-1].item(),std[0,-1].item()/3)


class ReachTransferTests(unittest.TestCase):
    @staticmethod
    def policies():
        torch.set_num_threads(1)
        source=ActorCriticPolicy(spaces.Box(-np.inf,np.inf,(12,),np.float32),
            spaces.Box(-1,1,(3,),np.float32),lambda _:3e-4,
            net_arch=dict(pi=[256,256],vf=[256,256]))
        target=ReachGraspPPOPolicy(spaces.Box(-np.inf,np.inf,(29,),np.float32),
            spaces.Box(-1,1,(4,),np.float32),lambda _:3e-4,
            features_extractor_kwargs=dict(source_home=[.22,0,.5],observation_anchor=[.1,0,.35]),
            exploration_profile='grasp_lift')
        initialize_reach_arm(target,source,.25)
        reset_jaw_head(target,0.)
        return source,target

    @staticmethod
    def observations():
        obs=torch.zeros(2,29)
        obs[:,:3]=torch.tensor([.8,0,.026])
        obs[:,8]=np.arcsinh(-.003/.01);obs[:,11]=np.arcsinh(.03/.01)
        obs[1,21:23]=.5
        return obs

    def test_transferred_neural_means_match_the_original_reach_network(self):
        source,target=self.policies();obs=self.observations()
        before=copy.deepcopy(source.state_dict())
        with torch.no_grad():
            features=target.extract_features(obs)
            expected=source.action_net(source.mlp_extractor.policy_net(features[:,:12]))*.25
            actual=target.get_distribution(obs).distribution.mean
        torch.testing.assert_close(actual[:,:3],expected)
        torch.testing.assert_close(actual[:,-1],torch.zeros(2))
        for name,value in before.items():torch.testing.assert_close(value,source.state_dict()[name])

    def test_lift_goal_gate_is_trainable_while_the_neural_arm_is_frozen(self):
        _,target=self.policies();obs=self.observations()
        target.set_arm_trainable(False);before=branch_hashes(target)
        mean=target.get_distribution(obs).distribution.mean
        mean.sum().backward()
        self.assertTrue(all(p.grad is None for p in target.mlp_extractor.policy_net['arm'].parameters()))
        self.assertGreater(target.features_extractor.grip_gate.bias.grad.abs().sum().item(),0)
        self.assertGreater(target.jaw_action_head.weight.grad.abs().sum().item(),0)
        target.optimizer.step();after=branch_hashes(target)
        self.assertEqual(before['arm'],after['arm'])
        self.assertNotEqual(before['jaw'],after['jaw'])
        self.assertNotEqual(before['goal_gate'],after['goal_gate'])

    def test_identity_projection_retains_ppo_likelihood_and_parameter_coverage(self):
        _,target=self.policies();obs=self.observations()
        with torch.no_grad():
            actions,values,probability=target(obs)
            checked_values,checked_probability,_=target.evaluate_actions(obs,actions)
        torch.testing.assert_close(values,checked_values)
        torch.testing.assert_close(probability,checked_probability)
        actual={id(p) for group in target.optimizer.param_groups for p in group['params']}
        self.assertEqual(actual,{id(p) for p in target.parameters()})
        self.assertTrue(isinstance(target.action_net,torch.nn.Identity))


if __name__=='__main__':unittest.main()
