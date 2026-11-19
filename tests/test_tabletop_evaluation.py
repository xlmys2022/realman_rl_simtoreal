"""Algorithm-free tabletop episodes, ownership and read-only state guards."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from realman_rl_simtoreal.evaluation import tabletop_cartesian as evaluation
from realman_rl_simtoreal.tasks.tabletop_cartesian import CartesianTabletopConfig as Config
from realman_rl_simtoreal.simulation.observation_profiles import startup_contract
from realman_rl_simtoreal.simulation.grasp_metrics import CONTACT_MEASUREMENT


class FakeEnv:
    def __init__(self,config=None,**kwargs):
        self.config=config or replace(Config(),max_steps=3)
        self.action_space=SimpleNamespace(shape=(4,));self.closed=False;self.actions=[];self.seeds=[]
        self.robot=SimpleNamespace(gripper=SimpleNamespace(statistics=lambda:{'ticks':150}))
    def reset(self,*,seed):
        self.seeds.append(seed);self.actions=[]
        return np.zeros(29,dtype=np.float32),{'object_position_m':[.4,0.,.8],'startup':{'mode':'physical'}}
    def step(self,action):
        self.actions.append(action.copy());done=len(self.actions)==2
        info=dict(is_success=done,termination_reason='success' if done else None,detail=None,
                  clearance_m=.03,hold_seconds=2.,step=len(self.actions))
        return np.full(29,len(self.actions),dtype=np.float32),1.25,done,False,info
    def close(self):self.closed=True


def saved(config):
    return dict(schema=2,environment=config.to_dict(),startup_contract=startup_contract(config.observation_pose),
                contact_measurement=CONTACT_MEASUREMENT)


class CartesianEvaluationTests(unittest.TestCase):
    def test_plain_callable_uses_original_seed_clipping_rewards_and_record_keys(self):
        env=FakeEnv();observed=[];ticks=[]
        def predict(obs):observed.append(obs.copy());return np.array([2.,-2.,.25,1.],dtype=np.float32)
        result=evaluation.rollout(env,predict,77,tick=lambda robot,phase:ticks.append(phase))
        self.assertEqual(env.seeds,[77]);self.assertEqual(result['length'],2)
        self.assertEqual(result['return_'],2.5);self.assertTrue(result['success']);self.assertEqual(ticks,['ppo']*3)
        np.testing.assert_array_equal(env.actions[0],[1.,-1.,.25,1.]);self.assertEqual(env.actions[0].dtype,np.float32)
        self.assertEqual(result['records'][0]['raw_action'],[2.,-2.,.25,1.])
        self.assertNotIn('policy_observation',result['records'][0]);np.testing.assert_array_equal(observed[1],np.ones(29))

    def test_visual_records_keep_pre_action_observation(self):
        env=FakeEnv(replace(Config(),observation_source='rgbd_hsv',max_steps=3))
        result=evaluation.rollout(env,lambda obs:np.zeros(4),88)
        self.assertEqual(result['records'][0]['policy_observation'],[0.]*29)
        self.assertEqual(result['records'][1]['policy_observation'],[1.]*29)

    def test_external_environment_stays_caller_owned_and_json_fields_remain(self):
        env=FakeEnv();state={'weights':'fixed','optimizer':'fixed'}
        result=evaluation.evaluate(lambda obs:np.zeros(4),saved(env.config),[9,10],action_shape=(4,),environment=env,
            state_fingerprint=lambda:dict(state),sources=lambda:{'file':'hash'},algorithm='rules')
        self.assertFalse(env.closed);self.assertEqual(env.seeds,[9,10]);self.assertTrue(result['parameters_unchanged'])
        self.assertEqual(result['summary'],dict(episodes=2,successes=2,success_rate=1.,safety_failures=0))
        self.assertEqual(result['sources'],{'file':'hash'});self.assertEqual(result['algorithm'],'rules')
        self.assertEqual(set(result),{'task','algorithm','full_task','environment','observation_source','vision_contract',
            'startup_contract','parameters_unchanged','scripted_task_actions','ik_joint_targets_in_policy','summary','episodes','sources'})

    def test_owned_environment_closes_on_prediction_failure(self):
        env=FakeEnv()
        with patch.object(evaluation,'TabletopEnv',return_value=env):
            with self.assertRaisesRegex(RuntimeError,'predict failed'):
                evaluation.evaluate(lambda obs:(_ for _ in ()).throw(RuntimeError('predict failed')),
                    saved(env.config),[3],action_shape=(4,))
        self.assertTrue(env.closed)

    def test_optimizer_only_mutation_is_rejected(self):
        env=FakeEnv();state={'adam_moment':1.}
        def predict(obs):state['adam_moment']+=1;return np.zeros(4)
        with self.assertRaisesRegex(AssertionError,'policy state'):
            evaluation.evaluate(predict,saved(env.config),[3],action_shape=(4,),environment=env,
                state_fingerprint=lambda:dict(state))
        self.assertFalse(env.closed)

    def test_missing_guard_does_not_claim_parameter_verification(self):
        env=FakeEnv()
        result=evaluation.evaluate(lambda obs:np.zeros(4),saved(env.config),[9],action_shape=(4,),environment=env)
        self.assertIsNone(result['parameters_unchanged'])

    def test_model_guard_includes_moments_and_groups_without_step_changes(self):
        import torch
        from realman_rl_simtoreal.rl.common.artifacts import optimizer_state_hash,optimizer_steps
        parameter=torch.nn.Parameter(torch.tensor([1.]))
        optimizer=torch.optim.Adam([parameter]);parameter.grad=torch.ones(1);optimizer.step()
        model=SimpleNamespace(policy=SimpleNamespace(optimizer=optimizer))
        before=optimizer_state_hash(model);steps=optimizer_steps(model)
        optimizer.state[parameter]['exp_avg'].add_(.01)
        self.assertNotEqual(before,optimizer_state_hash(model));self.assertEqual(steps,optimizer_steps(model))
        before=optimizer_state_hash(model);optimizer.param_groups[0]['lr']*=.5
        self.assertNotEqual(before,optimizer_state_hash(model))


if __name__=='__main__':unittest.main()
