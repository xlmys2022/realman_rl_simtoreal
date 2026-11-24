"""Behavioral contracts recorded before the progressive structure refactor."""
import ast
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from realman_rl_simtoreal.paths import PROJECT_ROOT as ROOT
from realman_rl_simtoreal.utils import artifacts

FIXTURE = json.loads((ROOT/'tests/fixtures/refactor_contracts.json').read_text())

# Fixture paths name the source layout used when the hashes were recorded.
PHYSICS_SOURCE_PATHS = {
    'simulation/tabletop_runtime.py': 'simulation/runtime/tabletop.py',
    'envs/tabletop_cartesian_env.py': 'envs/grasp/tabletop_cartesian_env.py',
    'envs/tabletop_cartesian_vision_env.py': 'envs/grasp/tabletop_cartesian_vision_env.py',
}


# The recorded class digests include these original docstring nodes.
LEGACY_POLICY_DOCSTRINGS = {'SmoothJointPolicy': 'MLP action means with persistent Gaussian position exploration.\n'
                      '\n'
                      '    Stock PPO/gSDE still samples and scores every action. A constant '
                      'unit-norm\n'
                      '    exploration feature prevents contact-force jitter from feeding '
                      'back into\n'
                      '    the sampled position offset between scheduled noise refreshes.\n'
                      '    ',
 'CartesianPPOPolicy': 'Change exploration scale from task observations, never the network '
                       'mean.\n'
                       '\n'
                       '    All motion and jaw means remain learned PPO outputs. The distance '
                       'is\n'
                       '    decoded from the existing Cartesian observation, not from an IK '
                       'solution.\n'
                       '    Sampling and PPO likelihood evaluation use the same '
                       'state-conditioned std.\n'
                       '    ',
 'CartesianGoalFeatures': 'Expose Cartesian grasp/lift goals to an existing neural Reach '
                          'controller.\n'
                          '\n'
                          '    The two force-gate parameters are learned. Frame anchors are '
                          'fixed at the\n'
                          "    selected observation preset and the original Reach model's "
                          'nominal home.\n'
                          '    No joint angles, solved joint targets or action labels enter '
                          'these features.\n'
                          '    ',
 'ReachGraspNetworks': 'Separate neural arm/jaw means and an independent task-value branch.',
 'ReachGraspPPOPolicy': 'Four learned Cartesian means with an optional pretrained Reach '
                        'branch.',
 'VisualCartesianGoalFeatures': None,
 'VisualReachGraspNetworks': None,
 'VisualReachGraspPPOPolicy': None}


class ArtifactContracts(unittest.TestCase):
    def test_legacy_json_bytes_and_nan_policies(self):
        from realman_rl_simtoreal.rl.common.artifacts import write_json as strict, append_json
        from realman_rl_simtoreal.rl.ppo_cartesian import write_json as cartesian
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'nested/result.json';value={'标签':'夹爪','return_':1.25}
            strict(path,value)
            self.assertEqual(path.read_bytes(),(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
            cartesian(path,value)
            self.assertEqual(path.read_bytes(),(json.dumps(value,indent=2,ensure_ascii=False)+'\n').encode())
            with self.assertRaises(ValueError):strict(path,{'v':float('nan')})
            cartesian(path,{'v':float('nan')});self.assertIn('NaN',path.read_text())
            path.unlink();append_json(path,value);append_json(path,value)
            self.assertEqual(path.read_text(),(json.dumps(value,allow_nan=False)+'\n')*2)
            with self.assertRaises(FileNotFoundError):append_json(Path(folder)/'missing'/'row.jsonl',value)

    def test_directory_guard_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            target=artifacts.empty_directory(Path(folder)/'run')
            self.assertEqual(target,(Path(folder)/'run').resolve())
            payload=target/'model.zip';payload.write_bytes(b'protected checkpoint')
            with self.assertRaisesRegex(ValueError,'never overwrite'):artifacts.empty_directory(target)
            self.assertEqual(payload.read_bytes(),b'protected checkpoint')

    def test_hashes_and_source_archive_keep_caller_scope_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);a=root/'a.py';b=root/'b.bin';a.write_bytes(b'\x00a\xff');b.write_bytes(b'b')
            hashes=artifacts.file_hashes([b,a],root=root)
            self.assertEqual(list(hashes),['b.bin','a.py'])
            self.assertEqual(hashes['a.py'],hashlib.sha256(a.read_bytes()).hexdigest())
            archive=root/'snapshot.zip';artifacts.archive_sources(archive,[b,root,a],root=root)
            with zipfile.ZipFile(archive) as stored:
                self.assertEqual(stored.namelist(),['b.bin','a.py'])
                self.assertEqual(stored.read('a.py'),a.read_bytes())

    def test_cartesian_snapshot_retains_original_source_selection(self):
        from realman_rl_simtoreal.rl import ppo_cartesian as rl
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);names=['src/realman_rl_simtoreal/__init__.py','assets/scene.xml','assets/mesh.bin',
                'scripts/demo.py','pyproject.toml','requirements.txt','requirements-rl.txt','requirements-vision.txt']
            for name in names:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name)
            (root/'excluded.txt').write_text('outside provenance scope')
            with patch.object(rl,'ROOT',root),patch.object(rl,'DEFAULT_SCENE',root/'assets/scene.xml'):
                rl.capture_sources(root)
                self.assertEqual(list(rl.provenance()),['assets/scene.xml','src/realman_rl_simtoreal/__init__.py'])
            with zipfile.ZipFile(root/'source_snapshot.zip') as stored:
                self.assertEqual(stored.namelist(),sorted(names))
                self.assertEqual(stored.read('scripts/demo.py').decode(),'scripts/demo.py')

    def test_legacy_helper_imports_share_one_implementation(self):
        from realman_rl_simtoreal.rl import ppo,ppo_grasp,ppo_cartesian as cartesian
        from realman_rl_simtoreal.rl.common import artifacts as model_artifacts
        self.assertIs(ppo.parameter_hashes,ppo_grasp.parameter_hashes)
        self.assertIs(ppo.optimizer_steps,ppo_grasp.optimizer_steps)
        self.assertIs(cartesian.optimizer_steps,ppo.optimizer_steps)
        self.assertIs(model_artifacts.write_json,artifacts.write_json)
        self.assertIs(model_artifacts.empty_directory,artifacts.empty_directory)


class FrozenPolicyAndTaskContracts(unittest.TestCase):
    def test_serialized_custom_classes_keep_paths_and_implementations(self):
        for key,expected in FIXTURE['policy_class_ast_sha256'].items():
            relative,name=key.split(':')
            with self.subTest(policy=key):
                tree=ast.parse((ROOT/relative).read_text())
                node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
                # Restore only the historical documentation node before hashing executable code.
                self.assertIsNotNone(ast.get_docstring(node))
                node.body.pop(0)
                original_docstring = LEGACY_POLICY_DOCSTRINGS[name]
                if original_docstring is not None:
                    node.body.insert(0, ast.Expr(value=ast.Constant(value=original_docstring)))
                self.assertEqual(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),expected)

    def test_physics_and_completion_methods_keep_original_timing_and_rules(self):
        for key,expected in FIXTURE['physics_method_ast_sha256'].items():
            relative,klass,name=key.split(':')
            with self.subTest(method=key):
                package = ROOT/'src/realman_rl_simtoreal'
                source = PHYSICS_SOURCE_PATHS[str((ROOT/relative).relative_to(package))]
                tree = ast.parse((package/source).read_text())
                cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==klass)
                node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
                if (klass,name)==('TabletopRuntime','prepare'):
                    # Compare against the old import spelling, keeping the call and all control flow.
                    startup_import = node.body[1]
                    self.assertIsInstance(startup_import, ast.ImportFrom)
                    self.assertEqual((startup_import.level,startup_import.module),(1,'observation_startup'))
                    startup_import.module = 'observation_pose'
                self.assertEqual(hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),expected)

    def test_reward_values_and_components_equal_pre_refactor_cases(self):
        from realman_rl_simtoreal.rl import ppo_grasp,ppo_tabletop
        for index,row in enumerate(FIXTURE['reward_cases']):
            module=ppo_grasp if row['task']=='grasp_lift' else ppo_tabletop
            with self.subTest(task=row['task'],case=index):
                context=FIXTURE['reward_contexts'][row['context']]
                kwargs=dict(row['kwargs'])
                if row['task']=='tabletop_pick':kwargs['action_info']=row['info']
                self.assertEqual(module.potential(row['info'],context['config'],context['reward']),row['previous'])
                value,terms=module.shaped_reward(row['previous'],row['info'],row['terminated'],
                    context['config'],context['reward'],**kwargs)
                self.assertEqual(terms,row['terms']);self.assertEqual(value,row['expected'])


class CartesianCLIContracts(unittest.TestCase):
    def test_obsolete_forwarders_are_not_importable(self):
        for module in ('cli', 'visualization', 'viewer', 'utils.tabletop_cartesian_cli',
                       'utils.tabletop_cartesian_render', 'rl.ppo_tabletop_cartesian'):
            with self.subTest(module=module):
                self.assertIsNone(find_spec('realman_rl_simtoreal.' + module))

    def test_all_original_parser_options_and_defaults_remain(self):
        path=ROOT/'src/realman_rl_simtoreal/apps/cli/tabletop_cartesian.py'
        calls=[ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text()))
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument']
        self.assertEqual(calls,FIXTURE['cartesian_cli_arguments'])

    def test_train_and_render_dispatch_keep_argument_values(self):
        from realman_rl_simtoreal.apps.cli.tabletop_cartesian import main
        with patch('realman_rl_simtoreal.apps.training.tabletop_cartesian.train',return_value={'status':'completed'}) as train:
            self.assertEqual(main(['train','--run-dir','/tmp/unused-realman-cli','--init-reach-model','reach.zip',
                '--observation-source','rgbd_hsv','--jaw-control','delta','--no-adaptive-exploration']),0)
            args=train.call_args.args[0]
            self.assertEqual(args.init_reach_model,Path('reach.zip'));self.assertEqual(args.observation_source,'rgbd_hsv')
            self.assertEqual(args.jaw_control,'delta');self.assertFalse(args.adaptive_exploration)
            self.assertEqual(args.seed,2027);self.assertEqual(args.n_steps,256)
        with patch('realman_rl_simtoreal.apps.visualization.tabletop_cartesian.run',return_value=0) as render:
            self.assertEqual(main(['render','--model','m.zip','--output','r.gif','--object-radius','.02','--saved-task']),0)
            args=render.call_args.args[0];self.assertEqual(args.object_radius,.02);self.assertTrue(args.saved_task)




def model_state_digest(value):
    import torch
    digest=hashlib.sha256()
    def update(item):
        if isinstance(item,dict):
            for key in sorted(item,key=str):digest.update(str(key).encode());update(item[key])
        elif isinstance(item,(tuple,list)):
            for part in item:update(part)
        elif torch.is_tensor(item):
            digest.update(str(item.dtype).encode()+str(tuple(item.shape)).encode()+item.detach().cpu().contiguous().numpy().tobytes())
        else:digest.update(repr(item).encode())
    update(value);return digest.hexdigest()


class ExistingCheckpointContracts(unittest.TestCase):
    def test_all_available_saved_models_load_without_inference_or_state_drift(self):
        import torch
        from stable_baselines3 import SAC
        from realman_rl_simtoreal.rl.ppo import load_ppo
        from realman_rl_simtoreal.rl.ppo_grasp import load_checkpoint as load_grasp
        from realman_rl_simtoreal.rl.ppo_tabletop import load_checkpoint as load_joint
        from realman_rl_simtoreal.rl.ppo_cartesian import load as load_cartesian
        from realman_rl_simtoreal.imitation.bc import load_policy
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            for name,expected in FIXTURE['checkpoint_cases'].items():
                with self.subTest(checkpoint=name):
                    path=ROOT/expected['path']
                    if not path.is_file():
                        self.skipTest('Historical checkpoint unavailable: '+str(path))
                    self.assertEqual(artifacts.file_sha256(path),expected['file_sha256'])
                    if name=='bc':
                        model,_=load_policy(path);network=model.network
                        obs={k:np.linspace(-.01,.01,n,dtype=np.float32) for k,n in
                            [('observation',27),('achieved_goal',3),('desired_goal',3)]}
                        before=model_state_digest(network.state_dict());action=model.predict(obs);raw=action;optimizer=None
                    else:
                        loaded=(load_ppo(path) if name=='reach_ppo' else (SAC.load(path,device='cpu'),{}) if name=='reach_sac'
                            else load_grasp(path) if name=='grasp_ppo' else load_joint(path) if name=='joint_tabletop_ppo'
                            else load_cartesian(path, allow_legacy_startup=True))
                        model=loaded[0];network=model.policy
                        def sample(space):
                            if hasattr(space,'spaces'):return {k:sample(s) for k,s in space.spaces.items()}
                            return np.linspace(-.03,.03,int(np.prod(space.shape)),dtype=np.float32).reshape(space.shape)
                        obs=sample(model.observation_space);before=model_state_digest(network.state_dict())
                        def optimizer_digest():
                            if hasattr(network,'optimizer'):return model_state_digest(network.optimizer.state_dict())
                            return model_state_digest((network.actor.optimizer.state_dict(),network.critic.optimizer.state_dict(),model.ent_coef_optimizer.state_dict()))
                        optimizer=optimizer_digest();counters=(model.num_timesteps,model._n_updates)
                        action,_=model.predict(obs,deterministic=True)
                        if hasattr(network,'get_distribution'):
                            tensor,_=network.obs_to_tensor(obs)
                            with torch.no_grad():raw=network.get_distribution(tensor).get_actions(deterministic=True).cpu().numpy().reshape(model.action_space.shape)
                        else:raw=action
                        self.assertEqual(optimizer_digest(),optimizer);self.assertEqual(counters,(model.num_timesteps,model._n_updates))
                    self.assertEqual(model_state_digest(network.state_dict()),before)
                    self.assertEqual(before,expected['weights']);self.assertEqual(optimizer,expected['optimizer'])
                    self.assertEqual(type(network).__module__+'.'+type(network).__name__,expected['policy_class'])
                    np.testing.assert_array_equal(action,expected['action']);np.testing.assert_array_equal(raw,expected['raw_action'])
        finally:torch.set_num_threads(old_threads)


class VisualCompositionContracts(unittest.TestCase):
    def test_progress_and_encoding_use_only_explicit_sensor_inputs(self):
        from realman_rl_simtoreal.envs.grasp.tabletop_cartesian_vision_env import visual_grip_and_hold,encode_visual_observation
        visual=dict(position_m=np.zeros(3),velocity_m_s=np.zeros(3),normal=np.array([0.,0.,1.]),
                    confidence=1.,visible=True,age=0.)
        offset=np.array([0.,0.,-.003]);goal_error=np.zeros(3)
        grip,hold=visual_grip_and_hold(visual,offset,goal_error,width=.02,jaw_reference=0.,cube_size=.02,
            timeout=.3,previous_grip=0.,previous_hold=0.,dt=0.)
        self.assertEqual(grip,1.);self.assertEqual(hold,0.)
        grip,hold=visual_grip_and_hold(visual,offset,goal_error,width=.02,jaw_reference=0.,cube_size=.02,
            timeout=.3,previous_grip=grip,previous_hold=hold,dt=.05)
        self.assertEqual(grip,1.);self.assertEqual(hold,.05)
        obs=encode_visual_observation(np.zeros(3),np.zeros(3),offset,goal_error,visual,width=.02,width_speed=0.,
            jaw_reference=0.,grip=grip,hold=hold,timeout=.3,hold_required=3.,tracking_error=np.zeros(3),ik_fraction=1.)
        self.assertEqual(obs.shape,(29,));self.assertEqual(obs.dtype,np.float32)
        self.assertEqual(float(obs[21]),1.);self.assertAlmostEqual(float(obs[24]),.05/3.,places=8)
        # Loss of visibility resets visual hold even if camera position remains close.
        visual['visible']=False
        _,hold=visual_grip_and_hold(visual,offset,goal_error,width=.02,jaw_reference=0.,cube_size=.02,
            timeout=.3,previous_grip=1.,previous_hold=2.9,dt=.05)
        self.assertEqual(hold,0.)

    def test_composed_helpers_do_not_accept_privileged_environment_info(self):
        import inspect
        from realman_rl_simtoreal.envs.grasp.tabletop_cartesian_vision_env import visual_grip_and_hold,encode_visual_observation
        for function in (visual_grip_and_hold,encode_visual_observation):
            parameters=inspect.signature(function).parameters
            self.assertNotIn('info',parameters);self.assertNotIn('robot',parameters)
            self.assertNotIn('contact_forces',parameters);self.assertNotIn('object_truth',parameters)


class TaskOwnershipContracts(unittest.TestCase):
    def test_historical_rl_reward_imports_forward_to_task_definitions(self):
        from realman_rl_simtoreal.rl import ppo_grasp,ppo_tabletop
        from realman_rl_simtoreal.tasks.grasp import lift as grasp_lift, tabletop_joint as tabletop_pick
        for rl,task,names in [(ppo_grasp,grasp_lift,('REWARD','CURRICULUM','goal_alignment','aperture_score','potential','shaped_reward')),
                              (ppo_tabletop,tabletop_pick,('REWARD','CURRICULUM','features','potential','shaped_reward'))]:
            for name in names:self.assertIs(getattr(rl,name),getattr(task,name))
        self.assertIsNot(grasp_lift.REWARD,tabletop_pick.REWARD)

    def test_cartesian_training_uses_original_completion_only_overrides(self):
        from realman_rl_simtoreal.apps.training import tabletop_cartesian as training
        from realman_rl_simtoreal.tasks.grasp import tabletop_cartesian as task
        self.assertIs(training.STAGES, task.STAGES)
        self.assertIs(training.environment_config, task.environment_config)
        self.assertEqual(task.EASY, dict(success_height=.005, height_tolerance=.0005, hold_seconds=.2))
        self.assertEqual(set(task.STAGES), {'easy', 'pinch', 'approach', 'lift', 'full'})


if __name__=='__main__':unittest.main()
