"""Standard four-action PPO. Kinematics is solely an environment execution service."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecCheckNan
from stable_baselines3.common.logger import configure
from ..utils.artifacts import write_json as _write_json, append_json, file_hashes, archive_sources, file_sha256
from .common.artifacts import optimizer_steps
from ..tasks.tabletop_cartesian import CartesianTabletopConfig as Config
from ..paths import PROJECT_ROOT as ROOT, TABLETOP_CARTESIAN_SCENE as DEFAULT_SCENE
from ..envs.factory import make_tabletop_cartesian_env as TabletopEnv
from ..envs.tabletop_cartesian_env import (ACTION_CONTRACT,OBSERVATION_FIELDS,
                                         action_contract,observation_fields)
from ..tasks.tabletop_cartesian_reward import REWARD
from ..simulation.observation_profiles import startup_contract
from ..simulation.grasp_metrics import CONTACT_MEASUREMENT
from .cartesian_policy import CartesianPPOPolicy,reset_critic,reset_jaw_head
from .cartesian_reach_policy import (ReachGraspPPOPolicy,CartesianGoalFeatures,initialize_reach_arm,branch_hashes,validate_reach_policy)
from .vision_policy import VisualReachGraspPPOPolicy,VisualCartesianGoalFeatures,transfer_visual_observation_inputs
from ..envs.tabletop_cartesian_vision_env import VISION_CONTRACT
from ..simulation.gripper import MAX_WIDTH

EASY=dict(success_height=.005,height_tolerance=.0005,hold_seconds=.2)
# Curriculum changes completion criteria only. Every stage starts at the same
# selected observation preset, with the same open jaw and full XYZ authority.
STAGES=dict(easy=dict(EASY),pinch=dict(EASY),approach=dict(EASY),lift=dict(),full=dict())


def environment_config(stage,*,observation_pose='table_down',control_mode='auto',cache_startup=False,
                       jaw_control='absolute',jaw_delta_scale=.002,observation_source='state'):
    cfg=replace(Config(observation_pose=observation_pose,control_mode=control_mode,
                       jaw_control=jaw_control,jaw_delta_scale=jaw_delta_scale,observation_source=observation_source),
                **STAGES[stage],cache_startup=cache_startup)
    if cfg.observation_pose=='forward' and not cfg.uses_orientation_actions:
        raise ValueError('A forward tabletop grasp start requires pose control so PPO can turn the gripper')
    return cfg


def write_json(path,value):
    return _write_json(path,value,ensure_ascii=False,allow_nan=True)


def hashes(model):
    result={key:hashlib.sha256() for key in ('actor_mean','critic','all')}
    for name,tensor in model.policy.state_dict().items():
        blob=name.encode()+tensor.detach().cpu().contiguous().numpy().tobytes()
        result['all'].update(blob)
        if name.startswith(('mlp_extractor.policy_net.','action_net.')):result['actor_mean'].update(blob)
        if name.startswith(('mlp_extractor.value_net.','value_net.')):result['critic'].update(blob)
    return {k:v.hexdigest() for k,v in result.items()}


def provenance():
    files=list((ROOT/'src/realman_rl_simtoreal').rglob('*.py'))+[DEFAULT_SCENE]
    return file_hashes(sorted(files),root=ROOT)


def capture_sources(folder):
    """Archive this feature's sources and shared assets as training provenance."""
    files={ROOT/name for name in provenance()}
    files.update((ROOT/'assets').rglob('*'))
    files.update((ROOT/'scripts').rglob('*.py'))
    files.update(ROOT/name for name in ('pyproject.toml','requirements.txt','requirements-rl.txt','requirements-vision.txt'))
    archive_sources(Path(folder)/'source_snapshot.zip',sorted(files),root=ROOT)


def save(model,folder,config,complete=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);model.save(folder/'model.zip')
    write_json(folder/'config.json',config)
    write_json(folder/'state.json',dict(timesteps=int(model.num_timesteps),optimizer_steps=optimizer_steps(model),
        parameter_hashes=hashes(model),completed_update=bool(complete and model.num_timesteps>0)))


def load(path,*,allow_legacy_startup=False,allow_legacy_observation=False):
    path=Path(path);config=json.loads((path.parent/'config.json').read_text());torch.set_num_threads(1)
    legacy=config.get('schema',1)<2 or 'startup_contract' not in config
    if legacy:
        if not allow_legacy_startup:
            raise ValueError('Legacy near-object startup checkpoint: evaluate only a model trained from a fixed observation pose; '
                             'old weights may be supplied to --init-model for retraining')
        expected_action,expected_fields=ACTION_CONTRACT,OBSERVATION_FIELDS
    else:
        cfg=Config(**config['environment'])
        if config['startup_contract']!=startup_contract(cfg.observation_pose):
            raise ValueError('Checkpoint fixed observation pose contract does not match the configured preset')
        expected_action,expected_fields=action_contract(cfg),observation_fields(cfg)
    if config.get('action_contract')!=expected_action or config.get('observation_fields')!=[list(v) for v in expected_fields]:
        raise ValueError('Expected this project\'s Cartesian checkpoint; joint-space legacy models are incompatible')
    if config.get('contact_measurement')!=CONTACT_MEASUREMENT and not allow_legacy_observation:
        raise ValueError('Checkpoint uses the old contact-force observation; use its archived source for historical evaluation '
                         'or --init-model to retrain with inward grip forces')
    model=PPO.load(path,device='cpu')
    vision=not legacy and cfg.observation_source=='rgbd_hsv'
    if vision!=isinstance(model.policy,VisualReachGraspPPOPolicy):
        raise ValueError('Checkpoint policy and camera observation contract do not match; retrain an explicit transfer')
    if vision and config.get('vision_contract')!=VISION_CONTRACT:
        if not (allow_legacy_observation and config.get('vision_contract',{}).get('name') in
                ('wrist_rgbd_hsv_cube_v1','wrist_rgbd_hsv_cube_v2','wrist_rgbd_hsv_cube_v3')):
            raise ValueError('Unsupported visual sensor contract; transfer old visual weights through --init-model')
    if (model.observation_space.shape!=(sum(n for _,n in expected_fields),)
            or model.action_space.shape!=(expected_action['shape'],)):raise ValueError('Invalid policy dimensions')
    return model,config


def rollout(env,model,seed,tick=None):
    obs,initial=env.reset(seed=seed);records=[];total=0.
    if tick:tick(env.robot,'ppo')
    for _ in range(env.config.max_steps):
        policy_observation=obs.copy()
        tensor,_=model.policy.obs_to_tensor(obs)
        with torch.no_grad():raw=model.policy.get_distribution(tensor).get_actions(deterministic=True).cpu().numpy().reshape(env.action_space.shape)
        action=np.clip(raw,-1,1).astype(np.float32)
        obs,reward,t,tr,info=env.step(action);total+=reward
        records.append(dict(info,raw_action=raw.tolist(),action=action.tolist(),reward=reward))
        if env.config.observation_source=='rgbd_hsv':records[-1]['policy_observation']=policy_observation.tolist()
        if tick:tick(env.robot,'ppo')
        if t or tr:break
    return dict(seed=seed,initial_object_position_m=initial['object_position_m'],
                success=info['is_success'],reason=info['termination_reason'],detail=info['detail'],
                length=len(records),return_=total,clearance_m=info['clearance_m'],hold_seconds=info['hold_seconds'],
                initial_startup=initial['startup'],gripper_statistics=env.robot.gripper.statistics(),records=records)


def evaluate(model,config,seeds,*,full=False,tick=None,environment=None,observation_pose=None,object_radius=None):
    before=hashes(model);counters=(model.num_timesteps,model._n_updates,optimizer_steps(model))
    if config.get('schema',1)<2 or 'startup_contract' not in config:
        raise ValueError('Evaluation requires the fixed observation pose startup contract')
    if config.get('contact_measurement')!=CONTACT_MEASUREMENT:
        raise ValueError('Evaluation requires the current inward grip-force observation contract')
    if config['startup_contract']!=startup_contract(Config(**config['environment']).observation_pose):
        raise ValueError('Checkpoint fixed observation pose contract does not match the configured preset')
    if environment is not None and (full or observation_pose is not None or object_radius is not None):
        raise ValueError('Full or overridden-pose evaluation must construct its own physical-startup environment')
    values=dict(config['environment']);values['cache_startup']=False
    if observation_pose is not None:values['observation_pose']=observation_pose
    if full:
        default=Config()
        for key in ('initial_width','success_height','lift_height','height_tolerance','hold_seconds','max_steps','object_radius','axis_scale'):
            values[key]=getattr(default,key)
    # Explicit evaluation randomization follows the full-task defaults. Otherwise
    # full=True silently restores radius zero and repeats the training position.
    if object_radius is not None:values['object_radius']=object_radius
    if model.action_space.shape!=(action_contract(Config(**values))['shape'],):
        raise ValueError('The checkpoint cannot control this observation pose; use pose control to turn from a forward start')
    own=environment is None
    env=TabletopEnv(Config(**values),startup_tick=tick) if own else environment
    try:episodes=[rollout(env,model,int(seed),tick=tick) for seed in seeds]
    finally:
        if own:env.close()
    assert hashes(model)==before and counters==(model.num_timesteps,model._n_updates,optimizer_steps(model))
    return dict(task='tabletop_cartesian',algorithm='PPO',full_task=full,environment=env.config.to_dict(),
                observation_source=env.config.observation_source,
                vision_contract=VISION_CONTRACT if env.config.observation_source=='rgbd_hsv' else None,
                startup_contract=startup_contract(env.config.observation_pose),
                parameters_unchanged=True,scripted_task_actions=False,ik_joint_targets_in_policy=False,
                summary=dict(episodes=len(episodes),successes=sum(e['success'] for e in episodes),
                success_rate=float(np.mean([e['success'] for e in episodes])),
                safety_failures=sum(e['reason'] in ('safety_failure','numerical_failure') for e in episodes)),
                episodes=episodes,sources=provenance())


class Audit(BaseCallback):
    def __init__(self,folder):super().__init__();self.folder=Path(folder);self.episodes=[]
    def _on_step(self):
        raw=np.asarray(self.locals['actions']);applied=np.asarray(self.locals['clipped_actions'])
        if raw.shape[-1]!=self.model.action_space.shape[0] or not np.isfinite(raw).all() or not np.array_equal(np.clip(raw,-1,1),applied):
            raise ValueError('Invalid PPO Cartesian actions')
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            if not info['observation_valid']:raise ValueError('Invalid observation')
            if done:
                row=dict(step=int(self.model.num_timesteps),success=info['is_success'],reason=info['termination_reason'],
                         detail=info['detail'],length=info['episode']['l'],return_=info['episode']['r'],
                         clearance_m=info['clearance_m'],max_hold_seconds=info['max_hold_seconds'],
                         max_clearance_m=info['max_clearance_m'],max_bilateral_force_n=info['max_bilateral_force_n'],
                         min_tcp_object_distance_m=info['min_tcp_object_distance_m'])
                self.episodes.append(row)
                append_json(self.folder/'episodes.jsonl',row,allow_nan=True)
        return True


def train(args):
    torch.set_num_threads(1)
    if args.stage not in STAGES:raise ValueError('Unknown curriculum')
    if min(args.timesteps,args.n_envs,args.n_steps,args.eval_every)<1 or args.n_steps<2 or not 1<=args.n_envs<=16:
        raise ValueError('Invalid training budget')
    if not (0<args.learning_rate<=.001 and 0<args.std<=1):raise ValueError('Invalid PPO distribution')
    jaw_std=getattr(args,'jaw_std',None)
    if jaw_std is not None and not 0<jaw_std<=1:raise ValueError('Invalid PPO jaw distribution')
    init_reach=getattr(args,'init_reach_model',None)
    freeze_arm_steps=getattr(args,'freeze_arm_steps',0)
    arm_speed_factor=getattr(args,'arm_speed_factor',.25)
    if (args.init_model and init_reach or not isinstance(freeze_arm_steps,int) or freeze_arm_steps<0
            or not 0<arm_speed_factor<=1):raise ValueError('Invalid Reach initialization')
    source=saved=None
    if args.init_model:source,saved=load(args.init_model,allow_legacy_startup=True,allow_legacy_observation=True)
    reach_source=reach_transform=reach_saved=None
    if init_reach:
        from .ppo import load_ppo as load_reach_ppo
        reach_source,reach_transform,reach_saved=load_reach_ppo(init_reach)
        if (reach_transform.observation_mode!='goal_relative_v3' or reach_source.num_timesteps<=0
                or optimizer_steps(reach_source)<=0):
            raise ValueError('Reach initialization requires a completed goal_relative_v3 PPO update')
        validate_reach_policy(reach_source.policy)
    saved_environment=saved.get('environment',{}) if saved and saved.get('schema',1)>=2 else {}
    selected_source=getattr(args,'observation_source',None) or saved_environment.get('observation_source','state')
    selected_pose=getattr(args,'observation_pose',None) or saved_environment.get('observation_pose','table_down')
    selected_control=getattr(args,'control_mode',None) or saved_environment.get('control_mode','auto')
    selected_jaw=getattr(args,'jaw_control',None) or saved_environment.get('jaw_control','delta' if init_reach else 'absolute')
    jaw_scale=getattr(args,'jaw_delta_scale',None)
    if jaw_scale is None:jaw_scale=saved_environment.get('jaw_delta_scale',.002)
    cfg=environment_config(args.stage,observation_pose=selected_pose,control_mode=selected_control,
                           cache_startup=getattr(args,'cache_startup',False),jaw_control=selected_jaw,jaw_delta_scale=jaw_scale,
                           observation_source=selected_source)
    structured=bool(init_reach or isinstance(getattr(source,'policy',None),ReachGraspPPOPolicy))
    vision=cfg.observation_source=='rgbd_hsv'
    if vision and not structured:raise ValueError('Initialize RGB-D PPO from --init-reach-model or a Cartesian Reach/grasp checkpoint')
    if source is not None and isinstance(source.policy,VisualReachGraspPPOPolicy) and not vision:
        raise ValueError('A visual policy cannot be silently transferred back to truth-state observations')
    visual_transfer=bool(vision and source is not None and saved_environment.get('observation_source','state')!='rgbd_hsv')
    vision_contract_changed=bool(vision and source is not None and saved.get('vision_contract')!=VISION_CONTRACT)
    if structured and (cfg.uses_orientation_actions or cfg.observation_pose!='table_down'):
        raise ValueError('Reach warm start supports table_down position control; use ordinary Cartesian PPO for pose control')
    if freeze_arm_steps and not structured:raise ValueError('Arm freezing requires the Reach-initialized policy')
    if getattr(args,'reset_goal_gate',False) and not structured:raise ValueError('Goal gate reset requires the Reach-initialized policy')
    if source is not None and (source.action_space.shape!=(action_contract(cfg)['shape'],)
            or source.observation_space.shape!=(sum(n for _,n in observation_fields(cfg)),)):
        raise ValueError('Source policy has a different Cartesian control mode; select a compatible checkpoint or train from scratch')
    folder=Path(args.run_dir)
    if folder.exists() and any(folder.iterdir()):raise ValueError('Choose a new run directory')
    folder.mkdir(parents=True,exist_ok=True)
    if getattr(args,'max_steps',None) is not None:cfg=replace(cfg,max_steps=args.max_steps)
    span=args.n_envs*args.n_steps
    batch=getattr(args,'batch_size',None) or min(256,span)
    if not 2<=batch<=span or span%batch:raise ValueError('Rollout must contain complete minibatches')
    capture_sources(folder)
    dev=TabletopEnv(cfg)
    goal_features=None
    if init_reach:
        import mujoco
        from ..paths import DEFAULT_SCENE as REACH_SCENE
        from ..simulation.kinematics import Kinematics
        if reach_saved['environment']['control_period']!=cfg.control_period:
            dev.close();raise ValueError('Reach transfer requires the same physical control period')
        home=Kinematics(mujoco.MjModel.from_xml_path(str(REACH_SCENE))).home_position
        goal_features=dict(source_home=home.tolist(),
            observation_anchor=(np.asarray(cfg.observation_tcp)-dev.robot.base_position).tolist(),
            grasp_offset_z=.003,approach_clearance=.05,approach_xy_scale=.015,
            initial_gate_weight=80.,initial_gate_bias=-40.)
    elif structured:
        goal_features=dict(saved['cartesian_goal_features'])
        if getattr(args,'reset_goal_gate',False):goal_features.update(initial_gate_weight=80.,initial_gate_bias=-40.)
    if vision:goal_features.update(visual_xy_deadband=.001,visual_z_deadband=.0005)
    def factory():return Monitor(TabletopEnv(cfg))
    vector=DummyVecEnv([factory]) if args.n_envs==1 else SubprocVecEnv([factory for _ in range(args.n_envs)],start_method='fork')
    vector=VecCheckNan(vector,raise_exception=True);vector.seed(args.seed)
    adaptive=getattr(args,'adaptive_exploration',True)
    exploration=dict(adaptive_exploration=adaptive,
        exploration_profile='distance' if vision else getattr(args,'exploration_profile','grasp_lift'),
        exploration_distance=.04,motion_std_floor=.03,jaw_std_floor=.2,jaw_exploration_distance=.01,
        cage_xy_scale=.004,cage_z_scale=.005,grasp_offset_z=.003,jaw_far_std_floor=.1,
        jaw_held_std_floor=.1,grip_force_scale=.1,lift_exploration_gain=2.)
    policy_kwargs=dict(log_std_init=float(np.log(args.std)),**exploration)
    if structured:policy_kwargs.update(features_extractor_class=VisualCartesianGoalFeatures if vision else CartesianGoalFeatures,
                                      features_extractor_kwargs=goal_features)
    else:policy_kwargs['net_arch']=dict(pi=[128,128],vf=[128,128])
    policy_class=VisualReachGraspPPOPolicy if vision else ReachGraspPPOPolicy if structured else CartesianPPOPolicy
    model=PPO(policy_class,vector,
              learning_rate=args.learning_rate,n_steps=args.n_steps,batch_size=batch,
              n_epochs=10,gamma=REWARD['gamma'],gae_lambda=.95,clip_range=.2,ent_coef=.001,vf_coef=.5,
              max_grad_norm=.5,target_kl=.04,policy_kwargs=policy_kwargs,
              seed=args.seed,device='cpu',verbose=0)
    transfer=None
    if init_reach:
        gain=arm_speed_factor*reach_saved['environment']['action_scale']/(cfg.action_scale*cfg.cartesian_smoothing)
        initialize_reach_arm(model.policy,reach_source.policy,gain)
        transfer=dict(kind='ppo_reach_arm',model=str(Path(init_reach).resolve()),
            sha256=file_sha256(init_reach),source_steps=reach_source.num_timesteps,
            source_parameter_hashes=hashes(reach_source),optimizer_reset=True,source_observation_transform=reach_transform.config,
            action_gain=gain,arm_speed_factor=arm_speed_factor,source_environment=reach_saved['environment'])
    elif args.init_model:
        state=json.loads((Path(args.init_model).parent/'state.json').read_text())
        if not state['completed_update']:raise ValueError('Transfer requires a completed PPO update')
        model.policy.load_state_dict(source.policy.state_dict(),strict=True)
        transfer=dict(model=str(Path(args.init_model).resolve()),sha256=file_sha256(args.init_model),
                      source_steps=source.num_timesteps,source_parameter_hashes=hashes(source),optimizer_reset=True,
                      source_startup_contract=saved.get('startup_contract'),
                      startup_changed=saved.get('startup_contract')!=startup_contract(cfg.observation_pose),
                      source_reward=saved.get('reward'),reward_changed=saved.get('reward')!=REWARD,
                      source_contact_measurement=saved.get('contact_measurement','pad_normal_sum_v0'),
                      contact_observation_changed=saved.get('contact_measurement')!=CONTACT_MEASUREMENT,
                      source_environment=saved['environment'],source_exploration=saved.get('exploration'))
    else:
        with torch.no_grad():
            model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    jaw_changed=bool(source and saved_environment.get('jaw_control','absolute')!=cfg.jaw_control)
    jaw_reset=not args.init_model or getattr(args,'reset_jaw',False) or jaw_changed
    if jaw_reset:reset_jaw_head(model.policy,0. if cfg.jaw_control=='delta' else 2*cfg.initial_width/MAX_WIDTH-1)
    goal_gate_reset=bool(structured and (init_reach or getattr(args,'reset_goal_gate',False)))
    if goal_gate_reset:model.policy.features_extractor.initialize_lift_gate()
    if visual_transfer:transfer_visual_observation_inputs(model.policy)
    if not args.init_model or args.reset_exploration:
        with torch.no_grad():model.policy.log_std.fill_(float(np.log(args.std)))
    if jaw_std is not None:
        with torch.no_grad():model.policy.log_std[-1]=float(np.log(jaw_std))
    critic_reset=getattr(args,'reset_critic',False) or visual_transfer or vision_contract_changed
    if critic_reset:
        actor_before=hashes(model)['actor_mean'];reset_critic(model.policy)
        assert hashes(model)['actor_mean']==actor_before
    if transfer is not None:
        transfer['critic_reset']=critic_reset;transfer['jaw_head_reset']=jaw_reset
        transfer['jaw_control_changed']=jaw_changed
        transfer['jaw_delta_scale_changed']=bool(source and saved_environment.get('jaw_delta_scale',.002)!=cfg.jaw_delta_scale)
        transfer['goal_gate_reset']=goal_gate_reset
        transfer['visual_observation_transfer']=visual_transfer
        transfer['vision_contract_changed']=vision_contract_changed
    if structured:model.policy.set_arm_trainable(freeze_arm_steps==0)
    config=dict(schema=2,algorithm='PPO',task='tabletop_cartesian',stage=args.stage,seed=args.seed,
                environment=cfg.to_dict(),action_contract=action_contract(cfg),observation_fields=observation_fields(cfg),
                startup_contract=startup_contract(cfg.observation_pose),
                contact_measurement=CONTACT_MEASUREMENT,adaptive_exploration=adaptive,
                exploration=exploration,critic_reset=critic_reset,
                jaw_head_reset=jaw_reset,jaw_initial_opening_m=cfg.initial_width if jaw_reset else None,
                policy_type='visual_reach_grasp' if vision else 'reach_grasp' if structured else 'cartesian',cartesian_goal_features=goal_features,
                observation_source=cfg.observation_source,object_state_in_policy='rgbd_hsv' if vision else 'simulator_state',
                vision_contract=VISION_CONTRACT if vision else None,
                privileged_reward_and_success_audit=vision,
                freeze_arm_steps=freeze_arm_steps,
                goal_gate_reset=goal_gate_reset,
                initial_branch_hashes=branch_hashes(model.policy) if structured else None,
                kinematic_joint_supervision=False,demonstrations=False,scripted_task_actions=False,
                transfer=transfer,n_envs=args.n_envs,n_steps=args.n_steps,std=args.std,jaw_std=jaw_std,
                initial_action_std=model.policy.log_std.detach().exp().cpu().tolist(),
                learning_rate=args.learning_rate,batch_size=batch,n_epochs=10,gamma=REWARD['gamma'],
                gae_lambda=.95,clip_range=.2,entropy_coefficient=.001,target_kl=.04,reward=REWARD,
                initial_parameter_hashes=hashes(model),sources=provenance())
    write_json(folder/'config.json',config);save(model,folder/'initial',config,complete=False)
    model.set_logger(configure(str(folder/'sb3'),['csv','tensorboard']))
    audit=Audit(folder);audit.init_callback(model);stages=[];started=perf_counter();status='running';error=None
    def snapshot(measure):
        step=int(model.num_timesteps);checkpoint=folder/'checkpoints'/f'step_{step:08d}';save(model,checkpoint,config)
        if step:
            buffer=model.rollout_buffer
            np.savez_compressed(checkpoint/'rollout.npz',observations=buffer.observations,raw_actions=buffer.actions,
                                actions=np.clip(buffer.actions,-1,1),rewards=buffer.rewards.T.reshape(-1),
                                episode_starts=buffer.episode_starts.T.reshape(-1),returns=buffer.returns,
                                advantages=buffer.advantages,values=buffer.values,log_probs=buffer.log_probs)
            model.logger.dump(step=step)
        row=dict(step=step,optimizer_steps=optimizer_steps(model),episodes=len(audit.episodes),
                 recent_success=float(np.mean([e['success'] for e in audit.episodes[-100:]])) if audit.episodes else None)
        if measure:
            result=evaluate(model,config,[340000],environment=dev)
            write_json(checkpoint/'evaluation.json',result);row['evaluation']=result['summary']
        stages.append(row);write_json(folder/'stages.json',stages)
        print('step',step,'stage',args.stage,'train',row['recent_success'],'eval',row.get('evaluation',{}).get('success_rate'),
              'wall_s',round(perf_counter()-started,1),flush=True)
    try:
        snapshot(True)
        while model.num_timesteps<args.timesteps:
            if (folder/'STOP_TRAINING').exists():status='stopped';break
            if structured:model.policy.set_arm_trainable(model.num_timesteps>=freeze_arm_steps)
            old=int(model.num_timesteps)
            model.learn(total_timesteps=span,reset_num_timesteps=old==0,callback=audit,log_interval=None)
            snapshot(model.num_timesteps>=args.timesteps or model.num_timesteps//args.eval_every>old//args.eval_every)
        if status=='running':status='completed'
    except KeyboardInterrupt:status='interrupted';error='KeyboardInterrupt'
    except Exception as exc:status='error';error=f'{type(exc).__name__}: {exc}'
    finally:
        save(model,folder/'final',config,complete=status=='completed')
        final=hashes(model)
        summary=dict(status=status,error=error,timesteps=int(model.num_timesteps),optimizer_steps=optimizer_steps(model),
                     actor_updated=final['actor_mean']!=config['initial_parameter_hashes']['actor_mean'],
                     critic_updated=final['critic']!=config['initial_parameter_hashes']['critic'],
                     final_parameter_hashes=final,wall_seconds=perf_counter()-started)
        if structured:
            final_branches=branch_hashes(model.policy)
            summary['final_branch_hashes']=final_branches
            for name,value in final_branches.items():summary[name+'_updated']=value!=config['initial_branch_hashes'][name]
        write_json(folder/'summary.json',summary);dev.close();vector.close();model.logger.close()
    return summary
