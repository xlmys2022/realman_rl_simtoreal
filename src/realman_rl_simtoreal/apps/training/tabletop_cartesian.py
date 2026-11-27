"""运行笛卡尔桌面 PPO、Reach 权重热启动与 RGB-D 迁移训练。"""
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

from ...envs.factory import make_tabletop_cartesian_env
from ...envs.grasp.tabletop_cartesian_env import action_contract, observation_fields
from ...tasks.grasp.tabletop_policy_io import vision_contract
from ...rl.cartesian_policy import reset_critic, reset_jaw_head
from ...rl.cartesian_reach_policy import (
    ReachGraspPPOPolicy, branch_hashes, initialize_reach_arm, validate_reach_policy,
)
from ...rl.common.artifacts import optimizer_steps
from ...rl.ppo_cartesian import (
    Audit, capture_sources, evaluate, hashes, load, make_model, provenance, save, write_json,
)
from ...rl.vision_policy import VisualReachGraspPPOPolicy, transfer_visual_observation_inputs
from ...simulation.devices.mujoco_gripper import MAX_WIDTH
from ...simulation.physics.grasp_metrics import CONTACT_MEASUREMENT
from ...simulation.runtime.observation_presets import REALMAN_TRANSFER_PROFILE, startup_contract
from ...tasks.grasp.tabletop_cartesian import STAGES, environment_config
from ...tasks.grasp.tabletop_cartesian_reward import REWARD
from ...utils.artifacts import file_sha256


def train(args):
    """按 args 的观测模式、课程及迁移来源训练笛卡尔 PPO，返回实验摘要。"""
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
        from ...rl.ppo import load_ppo as load_reach_ppo
        reach_source,reach_transform,reach_saved=load_reach_ppo(init_reach)
        if (reach_transform.observation_mode!='goal_relative_v3' or reach_source.num_timesteps<=0
                or optimizer_steps(reach_source)<=0):
            raise ValueError('Reach initialization requires a completed goal_relative_v3 PPO update')
        validate_reach_policy(reach_source.policy)
    saved_environment=saved.get('environment',{}) if saved and saved.get('schema',1)>=2 else {}
    scene_profile = getattr(args, 'scene_profile', None) or saved_environment.get('scene_profile', 'legacy')
    default_source = 'rgbd_hsv' if scene_profile == REALMAN_TRANSFER_PROFILE else saved_environment.get('observation_source','state')
    selected_source=getattr(args,'observation_source',None) or default_source
    selected_pose=getattr(args,'observation_pose',None) or saved_environment.get('observation_pose','table_down')
    selected_control=getattr(args,'control_mode',None) or saved_environment.get('control_mode','auto')
    selected_jaw=getattr(args,'jaw_control',None) or saved_environment.get('jaw_control','delta' if init_reach else 'absolute')
    hand_eye_file = getattr(args, 'hand_eye_file', None)
    if hand_eye_file is None and scene_profile == saved_environment.get('scene_profile'):
        hand_eye_file = saved_environment.get('hand_eye_file')
    jaw_scale=getattr(args,'jaw_delta_scale',None)
    if jaw_scale is None:jaw_scale=saved_environment.get('jaw_delta_scale',.002)
    cfg=environment_config(args.stage,observation_pose=selected_pose,control_mode=selected_control,
                           cache_startup=getattr(args,'cache_startup',False),jaw_control=selected_jaw,jaw_delta_scale=jaw_scale,
                           observation_source=selected_source, scene_profile=scene_profile,
                           hand_eye_file=str(Path(hand_eye_file).resolve()) if hand_eye_file else None)
    if getattr(args, 'object_radius', None) is not None:
        cfg = replace(cfg, object_radius=args.object_radius)
    vision=cfg.observation_source=='rgbd_hsv'
    visual_scratch = vision and source is None and init_reach is None
    structured=bool(init_reach or isinstance(getattr(source,'policy',None),ReachGraspPPOPolicy) or visual_scratch)
    if vision and not structured:raise ValueError('RGB-D transfer requires an initialized Cartesian Reach/grasp checkpoint')
    if source is not None and isinstance(source.policy,VisualReachGraspPPOPolicy) and not vision:
        raise ValueError('A visual policy cannot be silently transferred back to truth-state observations')
    visual_transfer=bool(vision and source is not None and saved_environment.get('observation_source','state')!='rgbd_hsv')
    sensor_contract=vision_contract(cfg) if vision else None
    vision_contract_changed=bool(vision and source is not None and saved.get('vision_contract')!=sensor_contract)
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
    dev=make_tabletop_cartesian_env(cfg)
    goal_features=None
    if init_reach or visual_scratch:
        import mujoco
        from ...paths import DEFAULT_SCENE as REACH_SCENE
        from ...simulation.motion.kinematics import Kinematics
        if init_reach and reach_saved['environment']['control_period']!=cfg.control_period:
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
    def factory():return Monitor(make_tabletop_cartesian_env(cfg))
    vector=DummyVecEnv([factory]) if args.n_envs==1 else SubprocVecEnv([factory for _ in range(args.n_envs)],start_method='fork')
    vector=VecCheckNan(vector,raise_exception=True);vector.seed(args.seed)
    adaptive=getattr(args,'adaptive_exploration',True)
    model,exploration=make_model(vector,seed=args.seed,learning_rate=args.learning_rate,
        n_steps=args.n_steps,batch_size=batch,std=args.std,adaptive=adaptive,
        exploration_profile=getattr(args,'exploration_profile','grasp_lift'),
        structured=structured,vision=vision,goal_features=goal_features)
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
                      startup_changed=saved.get('startup_contract')!=startup_contract(cfg.observation_pose, cfg.scene_profile),
                      source_reward=saved.get('reward'),reward_changed=saved.get('reward')!=REWARD,
                      source_contact_measurement=saved.get('contact_measurement','pad_normal_sum_v0'),
                      contact_observation_changed=saved.get('contact_measurement')!=CONTACT_MEASUREMENT,
                      source_environment=saved['environment'],source_exploration=saved.get('exploration'))
    elif not structured:
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
        transfer['learning_rate_schedule_reset'] = True
        transfer['normalization'] = 'fixed_si_scaling_no_running_statistics'
        transfer['critic_reset']=critic_reset;transfer['jaw_head_reset']=jaw_reset
        transfer['jaw_control_changed']=jaw_changed
        transfer['jaw_delta_scale_changed']=bool(source and saved_environment.get('jaw_delta_scale',.002)!=cfg.jaw_delta_scale)
        transfer['goal_gate_reset']=goal_gate_reset
        transfer['visual_observation_transfer']=visual_transfer
        transfer['vision_contract_changed']=vision_contract_changed
        if source is not None:
            transfer['action_contract_changed']=saved.get('action_contract')!=action_contract(cfg)
    if structured:model.policy.set_arm_trainable(freeze_arm_steps==0)
    config=dict(schema=2,algorithm='PPO',task='tabletop_cartesian',stage=args.stage,seed=args.seed,
                environment=cfg.to_dict(),action_contract=action_contract(cfg),observation_fields=observation_fields(cfg),
                startup_contract=startup_contract(cfg.observation_pose, cfg.scene_profile),
                contact_measurement=CONTACT_MEASUREMENT,adaptive_exploration=adaptive,
                exploration=exploration,critic_reset=critic_reset,
                jaw_head_reset=jaw_reset,jaw_initial_opening_m=cfg.initial_width if jaw_reset else None,
                policy_type='visual_reach_grasp' if vision else 'reach_grasp' if structured else 'cartesian',cartesian_goal_features=goal_features,
                observation_source=cfg.observation_source,object_state_in_policy='rgbd_hsv' if vision else 'simulator_state',
                vision_contract=sensor_contract,
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
    if cfg.scene_profile != 'legacy':
        config['scene_sha256'] = file_sha256(dev.robot.scene_path)
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
