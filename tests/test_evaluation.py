"""Independent evaluation contracts; no policy training or production-output writes."""
import copy
import importlib.util
import subprocess
import sys
import unittest

import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.evaluation import baselines, grasp_lift, metrics, reach
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.rl.common.observation import ReachTransform
from realman_rl_simtoreal.tasks.grasp_lift import GRASP_LIFT_CONFIG
from realman_rl_simtoreal.utils.metrics import MotionStats


def episode():
    return dict(reason='success', success=True, return_=2., discounted_return=1.8,
                length=2, final_distance_m=.0005, safety_failure=False,
                rejected_actions=1)


def without_times(document):
    result=copy.deepcopy(document)
    result.pop('timing',None)
    for row in result.get('episodes',[]):
        row.pop('episode_wall_s',None)
    return result


class EvaluationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.rows=[episode(),dict(episode(),reason='time_limit',success=False,
            return_=-4.,discounted_return=-3.,length=6,final_distance_m=.003,
            safety_failure=True,rejected_actions=2)]

    def test_reach_statistics_keep_episode_and_success_denominators(self):
        result=metrics.summarize_episodes(self.rows)
        self.assertEqual(result['success_rate'],.5)
        self.assertEqual(result['mean_return'],-1.)
        self.assertAlmostEqual(result['mean_discounted_return'],-.6)
        self.assertEqual(result['mean_episode_length'],4.)
        self.assertEqual(result['mean_success_steps'],2.)
        self.assertEqual(result['mean_final_distance_m'],.00175)
        self.assertEqual(result['rejected_actions'],3)
        self.assertEqual(result['safety_failures'],1)
        self.assertEqual(result['termination_counts']['time_limit'],1)
        self.assertEqual(result['termination_counts']['collision'],0)

    def test_empty_reach_metrics_remain_null_with_original_reason_keys(self):
        result=metrics.summarize_episodes([])
        self.assertIsNone(result['success_rate'])
        self.assertIsNone(result['mean_final_distance_m'])
        self.assertEqual(result['episodes'],0)
        self.assertEqual(result['termination_counts'],dict.fromkeys(metrics.REASONS,0))

    def test_sequence_attempted_and_planned_goal_denominators_stay_distinct(self):
        rows=[dict(self.rows[0],goals_planned=5,goals_completed=5,goals_attempted=5),
              dict(self.rows[1],goals_planned=5,goals_completed=2,goals_attempted=3)]
        result=metrics.summarize_episodes(rows)
        self.assertEqual(result['sequence_success_rate'],.5)
        self.assertEqual(result['goal_success_rate'],7/8)
        self.assertEqual(result['goal_completion_rate'],.7)
        self.assertEqual(result['goals_attempted'],8)

    def test_grasp_success_metrics_exclude_failed_clearances(self):
        rows=[dict(success=True,reason='success',is_safety_failure=False,
                   final_support_clearance_m=.03,world_z_lift_m=.026,length=10),
              dict(success=False,reason='time_limit',is_safety_failure=False,
                   final_support_clearance_m=.001,world_z_lift_m=.0005,length=20)]
        result=metrics.summarize_grasp_episodes(rows)
        self.assertEqual(result['successes'],1)
        self.assertEqual(result['mean_success_clearance_m'],.03)
        self.assertEqual(result['mean_success_world_z_lift_m'],.026)
        self.assertEqual(result['mean_success_steps'],10.)
        self.assertIsNone(metrics.summarize_grasp_episodes([])['mean_success_steps'])

    def test_historical_baseline_schema_does_not_gain_rl_only_fields(self):
        result=metrics.summarize_baseline_episodes(self.rows)
        self.assertEqual(result['timeouts'],1)
        self.assertEqual(result['mean_success_steps'],2.)
        self.assertNotIn('mean_discounted_return',result)
        self.assertNotIn('safety_failures',result)
        self.assertEqual(result['collisions'],0)

    def test_historical_bc_motion_metrics_keep_aggregate_denominators(self):
        rows=[dict(r,actual_toward=1,actual_directions=2,command_toward=2,
                   command_directions=2,reversed_steps=1,reversal_pairs=3,
                   command_path_m=.004,actual_path_m=.002) for r in self.rows]
        result=metrics.summarize_bc_episodes(rows)
        self.assertEqual(result['actual_toward_fraction'],.5)
        self.assertEqual(result['command_toward_fraction'],1.)
        self.assertEqual(result['reversed_fraction'],1/3)
        self.assertEqual(result['actual_to_command_path_ratio'],.5)
        self.assertNotIn('collision',result['termination_counts'])

    def test_neutral_motion_statistics_reset_direction_history_between_episodes(self):
        stats=MotionStats()
        stats.observe([0,0,0],[.001,0,0],[.01,0,0],[0,0,0],[.002,0,0],[1,0,0],True)
        stats.observe([0,0,0],[-.001,0,0],[.01,0,0],[0,0,0],[-.002,0,0],[-1,0,0],False)
        result=stats.summary()
        self.assertIsNone(result['reversal_fraction'])
        self.assertEqual(result['toward_actual_fraction'],.5)
        self.assertEqual(result['actual_to_command_path_ratio'],.5)


class EvaluationInterfaceTests(unittest.TestCase):
    config=dict(target_mode='fixed',max_episode_steps=2)

    def transform(self):
        with ReachEnv(**self.config) as env:
            return ReachTransform.from_env(env)

    def test_generic_callable_matches_the_existing_deterministic_protocol(self):
        class ZeroPolicy:
            def predict(self,observation,deterministic):
                if not deterministic:raise AssertionError('Stochastic evaluation')
                return np.zeros(3,dtype=np.float32),None
        transform=self.transform()
        original=reach.evaluate('ppo',[21,22],model=ZeroPolicy(),transform=transform,
                                config=self.config,collect_motion=True)
        generic=reach.evaluate_policy(lambda obs:np.zeros(3,dtype=np.float32),[21,22],
            name='ppo',transform=transform,config=self.config,collect_motion=True)
        self.assertEqual(without_times(original),without_times(generic))
        self.assertTrue(generic['deterministic'])

    def test_rule_random_and_bc_outputs_keep_their_separate_json_protocols(self):
        rule=reach.evaluate('rule',[21],config=self.config)
        random=reach.evaluate('random',[21],config=self.config)
        detailed=reach.evaluate_bc('rule',[21],environment=self.config)
        self.assertEqual(rule['episodes'][0]['target_position_m'],random['episodes'][0]['target_position_m'])
        self.assertIn('records',detailed)
        self.assertNotIn('records',rule)
        self.assertEqual(detailed['episodes'][0]['goal_m'],rule['episodes'][0]['target_position_m'])
        self.assertEqual(detailed['episodes'][0]['return_'],rule['episodes'][0]['return_'])

    def test_invalid_generic_action_is_rejected_by_the_original_action_contract(self):
        with self.assertRaisesRegex(ValueError,'finite action space'):
            reach.evaluate_policy(lambda obs:np.zeros(4,dtype=np.float32),[21],config=self.config)
        with self.assertRaises(TypeError):
            reach.evaluate_policy(object(),[21],config=self.config)
        with self.assertRaises(ValueError):
            reach.evaluate('unknown',[21],config=self.config)

    def test_evaluation_uses_a_separate_environment_and_private_target_rng(self):
        with ReachEnv(**self.config) as training:
            training.reset(seed=8)
            before=[training.data.qpos.copy(),training.data.qvel.copy(),training.data.ctrl.copy(),training.data.time]
            rng=np.random.get_state()
            reach.evaluate('random',[21],config=self.config)
            for saved,current in zip(before,[training.data.qpos,training.data.qvel,training.data.ctrl,training.data.time]):
                np.testing.assert_array_equal(saved,current)
            after=np.random.get_state()
            self.assertEqual(rng[0],after[0]);np.testing.assert_array_equal(rng[1],after[1])
            self.assertEqual(rng[2:],after[2:])

    def test_caller_owned_bc_fingerprint_detects_parameter_mutation(self):
        transform=self.transform()
        class MutatingPolicy:
            def __init__(self):self.transform=transform;self.version=0
            def predict(self,obs):
                self.version+=1
                return np.zeros(3,dtype=np.float32)
        policy=MutatingPolicy()
        with self.assertRaisesRegex(ValueError,'changed network parameters'):
            reach.evaluate_bc('bc',[21],policy=policy,environment=self.config,
                              state_digest=lambda:policy.version)

    def test_stateless_bc_predictor_can_use_the_common_callable_interface(self):
        class RawPolicy:
            def predict(self,obs):return np.zeros(3,dtype=np.float32)
        result=reach.evaluate_policy(RawPolicy().predict,[21],name='bc',config=self.config)
        self.assertEqual(result['policy'],'bc')
        self.assertTrue(result['deterministic'])
        self.assertEqual(result['summary']['termination_counts']['time_limit'],1)

    def test_generic_caller_fingerprint_protects_unknown_algorithm_state(self):
        state={'updates':0}
        def predict(obs):
            state['updates']+=1
            return np.zeros(3,dtype=np.float32)
        with self.assertRaisesRegex(ValueError,'parameters or counters'):
            reach.evaluate_policy(predict,[21],config=self.config,
                                  state_digest=lambda:dict(state))

    def test_grasp_accepts_a_predictor_without_any_training_methods(self):
        class RawPolicy:
            def predict(self,obs,deterministic):
                if not deterministic:raise AssertionError('Stochastic evaluation')
                return np.zeros(4,dtype=np.float32),None
        result=grasp_lift.evaluate(dict(GRASP_LIFT_CONFIG,max_episode_steps=2),[21],model=RawPolicy())
        self.assertEqual(result['summary']['episodes'],1)
        self.assertEqual(result['episodes'][0]['length'],2)
        self.assertEqual(result['episodes'][0]['records'][0]['controller_phase'],'ppo')

    def test_evaluation_and_baselines_run_with_learning_libraries_blocked(self):
        code='''
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('torch','stable_baselines3') or name.startswith((
    'realman_rl_simtoreal.rl.ppo','realman_rl_simtoreal.rl.sac','realman_rl_simtoreal.imitation')):
   raise AssertionError('Learning dependency: '+name)
sys.meta_path.insert(0,Block())
from realman_rl_simtoreal.evaluation import reach,baselines,grasp_lift
from realman_rl_simtoreal.tasks.grasp_lift import GRASP_LIFT_CONFIG
assert reach.evaluate('rule',[21],config=dict(target_mode='fixed',max_episode_steps=2))['summary']['episodes']==1
assert baselines.evaluate_policy('rule',[21],max_episode_steps=2)['summary']['episodes']==1
assert grasp_lift.evaluate(dict(GRASP_LIFT_CONFIG,max_episode_steps=2),[21])['summary']['episodes']==1
assert not any(n in sys.modules for n in ('torch','stable_baselines3'))
'''
        process=subprocess.run([sys.executable,'-I','-c',code],cwd='/tmp',capture_output=True,text=True,timeout=30)
        self.assertEqual(process.returncode,0,process.stdout+process.stderr)

    def test_evaluate_script_delegates_its_original_helpers_to_the_package(self):
        path=PROJECT_ROOT/'scripts/evaluate/evaluate_reach_baselines.py'
        spec=importlib.util.spec_from_file_location('baseline_entry_contract',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertIs(module.evaluate_policy,baselines.evaluate_policy)
        self.assertIs(module.latency_summary,metrics.latency_summary)


if __name__=='__main__':unittest.main(verbosity=2)
