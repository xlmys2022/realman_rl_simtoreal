"""训练七动作关节桌面 PPO，管理课程、探索参数继承和独立评估。"""
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from ...rl.ppo_tabletop import (
    ACTION_CONTRACT, ENCODING, PPO_PARAMETERS, TabletopAudit, evaluate_model, load_checkpoint,
    make_model, make_training_env, metadata, reset_exploration_std, save_checkpoint,
)
from ...rl.common.artifacts import (
    empty_directory, optimizer_steps, parameter_hashes, versions, write_json,
)
from ...tasks.grasp.tabletop_joint import REWARD, environment_config
from ...utils.artifacts import file_sha256


def train(args):
    """构建关节桌面环境并执行 PPO 更新，在新 run 目录保存评估和模型。"""
    if args.seed<0 or min(args.timesteps,args.n_envs,args.n_steps,args.batch_size,args.epochs,args.eval_every,args.eval_episodes)<1:
        raise ValueError('Nonnegative seed and positive budgets required')
    span=args.n_envs*args.n_steps
    if args.n_steps<2 or args.batch_size<2 or span%args.batch_size:raise ValueError('Complete PPO minibatches required')
    if not np.isfinite(args.learning_rate) or not 0<args.learning_rate<=.001:raise ValueError('Learning rate must be in (0,.001]')
    if not 0<args.joint_std<=5 or not 0<args.gripper_std<=1 or not -1<=args.gripper_mean<=1:
        raise ValueError('Invalid initial action distribution')
    torch.set_num_threads(1)
    source=saved=None
    if args.init_model:
        source,saved=load_checkpoint(args.init_model)
        state=json.loads((args.init_model.parent/'state.json').read_text())
        if not state['completed_update_stage']:raise ValueError('Source must be a completed PPO update')
        if saved['reward']!=REWARD and not args.allow_reward_change:
            raise ValueError('Source reward differs; explicitly use --allow-reward-change')
    use_sde=getattr(args,'use_sde',None)
    use_sde=(source.use_sde if source is not None else True) if use_sde is None else use_sde
    if source is not None and source.use_sde!=use_sde:raise ValueError('Source exploration parameterization differs')
    smooth=getattr(args,'smooth_exploration',None)
    smooth=(saved.get('smooth_exploration',False) if saved else True) if smooth is None else smooth
    smooth=bool(smooth and use_sde)
    if source is not None and smooth!=saved.get('smooth_exploration',False) and not args.reset_exploration:
        raise ValueError('Exploration-feature changes require --reset-exploration')
    overrides={}
    for key in ('start_fraction','object_radius','pregrasp_height','initial_width','lift_height','hold_seconds','max_episode_steps',
                'reference_smoothing','gripper_reference_speed','approach_waypoint_height','approach_waypoint_tolerance','goal_height'):
        value=getattr(args,key,None)
        if value is not None:overrides[key]=value
    if saved and getattr(args,'reference_smoothing',None) is None:
        overrides['reference_smoothing']=saved['environment'].get('reference_smoothing',1.)
    if saved and getattr(args,'gripper_reference_speed',None) is None:
        overrides['gripper_reference_speed']=saved['environment']['gripper_reference_speed']
    if saved and getattr(args,'approach_waypoint_height',None) is None:
        overrides['approach_waypoint_height']=saved['environment'].get('approach_waypoint_height',0.)
    if saved and getattr(args,'approach_waypoint_tolerance',None) is None:
        overrides['approach_waypoint_tolerance']=saved['environment'].get('approach_waypoint_tolerance',.005)
    if saved and getattr(args,'goal_height',None) is None:
        overrides['goal_height']=saved['environment'].get('goal_height')
    if getattr(args,'initial_q_deg',None) is not None:
        overrides.update(initial_q_deg=args.initial_q_deg,initial_pose_name='explicit_joint_configuration',initial_pose_source='user-specified CLI degrees')
    elif saved:
        overrides.update({key:saved['environment'][key] for key in ('initial_q_deg','initial_pose_name','initial_pose_source')})
    environment=environment_config(args.stage,**overrides)
    folder=empty_directory(args.run_dir)
    parameters=copy.deepcopy(PPO_PARAMETERS)
    frequency=getattr(args,'sde_sample_freq',None)
    if frequency is None:frequency=saved['ppo'].get('sde_sample_freq',64) if saved else 64
    if not isinstance(frequency,int) or (frequency!=-1 and frequency<1):raise ValueError('Exploration frequency must be positive or -1')
    parameters.update(learning_rate=args.learning_rate,n_steps=args.n_steps,batch_size=args.batch_size,n_epochs=args.epochs,
                      use_sde=use_sde,sde_sample_freq=frequency)
    env=make_training_env(environment,args.n_envs,args.seed,REWARD)
    initial_obs=env.reset()
    initial_reference=np.asarray(env.get_attr('_reset_q')[0])
    old_transform=saved['ppo']['policy_kwargs'].get('exploration_transform') if saved else None
    transform=old_transform
    if smooth and transform is None:
        direction=np.asarray(env.get_attr('lift_q')[0])-np.asarray(env.get_attr('grasp_q')[0])
        direction/=np.linalg.norm(direction)
        matrix=np.eye(7)
        matrix[:6,:6]=.05*np.eye(6)+.95*np.outer(direction,direction)
        transform=matrix.tolist()
    if transform is not None:parameters['policy_kwargs']['exploration_transform']=transform
    anchor=saved['ppo']['policy_kwargs'].get('action_anchor') if saved else np.r_[np.rad2deg(initial_reference),0.].tolist()
    if smooth and anchor is not None:parameters['policy_kwargs']['action_anchor']=anchor
    transform_changed=source is not None and transform!=old_transform
    if transform_changed and not args.reset_exploration:raise ValueError('Exploration-basis changes require --reset-exploration')
    model=make_model(env,args.seed,parameters,smooth=smooth,initial_observation=initial_obs[0],
        reference_rad=initial_reference,joint_std=args.joint_std,gripper_std=args.gripper_std,
        gripper_mean=args.gripper_mean)
    transfer=None
    if source is not None:
        model.policy.load_state_dict(source.policy.state_dict(),strict=True)
        if args.reset_exploration:
            reset_exploration_std(model,initial_obs[0],joint_std=args.joint_std,
                                  gripper_std=args.gripper_std,smooth=smooth)
        transfer=dict(model=str(args.init_model.resolve()),sha256=file_sha256(args.init_model),
            source_timesteps=int(source.num_timesteps),source_parameter_hashes=parameter_hashes(source),
            optimizer_reset=True,counters_reset=True,transformed_parameters=['log_std'] if args.reset_exploration else [],
            reward_changed=saved['reward']!=REWARD,source_reward=saved['reward'],
            exploration_features_changed=smooth!=saved.get('smooth_exploration',False),
            exploration_transform_changed=transform_changed)
        model.set_random_seed(args.seed)
    model.set_random_seed(args.seed)
    model.tabletop_contract=dict(encoding=copy.deepcopy(ENCODING),reward=copy.deepcopy(REWARD),action_contract=copy.deepcopy(ACTION_CONTRACT))
    model.set_logger(configure(str(folder/'sb3'),['csv','tensorboard'] if args.tensorboard else ['csv']))
    config=dict(schema=1,algorithm='PPO',task='tabletop_pick',pure_ppo=True,demonstrations=False,scripted_actions=False,
        initialization='scratch' if source is None else 'ppo_weights',transfer=transfer,
        ik_goal_features=True,ik_actions=False,curriculum_stage=args.stage,environment=environment,
        kinematic_action_regularizer=bool(REWARD.get('action_target_weight',0)),
        encoding=ENCODING,reward=REWARD,ppo=parameters,n_envs=args.n_envs,seed=args.seed,
        action_contract=ACTION_CONTRACT,action_space_low=env.action_space.low.tolist(),action_space_high=env.action_space.high.tolist(),
        scratch_initial_joint_reference_rad=initial_reference.tolist(),scratch_initial_joint_head='constant reset pose; no target/teacher actions',
        initial_action_bias=model.policy.action_net.bias.detach().cpu().tolist(),
        initial_policy_std=model.policy.log_std.detach().exp().cpu().tolist(),
        smooth_exploration=smooth,
        exploration_basis='nominal lift direction with 5% full-rank orthogonal exploration' if transform is not None else 'independent actions',
        requested_timesteps=args.timesteps,planned_timesteps=((args.timesteps+span-1)//span)*span,
        validation_seeds=list(range(args.eval_seed_start,args.eval_seed_start+args.eval_episodes)),
        stop_file=str(folder/'STOP_TRAINING'),**metadata())
    with torch.no_grad():
        tensor,_=model.policy.obs_to_tensor(initial_obs[0])
        config['initial_action_std']=model.policy.get_distribution(tensor).distribution.stddev[0].cpu().tolist()
        config['initial_deterministic_action']=model.policy.get_distribution(tensor).get_actions(deterministic=True)[0].cpu().tolist()
    config['algorithm_sources']={str(Path(__file__).resolve()):file_sha256(Path(__file__))}
    write_json(folder/'config.json',config);write_json(folder/'dependencies.json',versions())
    initial=parameter_hashes(model);save_checkpoint(model,folder/'initial',config)
    audit=TabletopAudit(folder,args.n_envs);audit.init_callback(model)
    evaluations=[];stages=[];status='running';error=None;started=perf_counter()
    def snapshot(evaluate_now):
        current=int(model.num_timesteps);checkpoint=folder/'checkpoints'/f'step_{current:08d}'
        save_checkpoint(model,checkpoint,config);stats=audit.summary()
        row=dict(step=current,epochs=int(model._n_updates),optimizer_steps=optimizer_steps(model),training=stats)
        for key in ('policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction','explained_variance','std'):
            value=model.logger.name_to_value.get('train/'+key)
            row[key]=float(value) if value is not None and np.isfinite(value) else None
        if evaluate_now:
            measured=evaluate_model(model,config,config['validation_seeds'])
            write_json(checkpoint/'evaluation.json',measured);row['evaluation']=measured['summary']
            evaluations.append(dict(step=current,checkpoint=str(checkpoint/'model.zip'),summary=measured['summary']))
            write_json(folder/'evaluations.json',evaluations)
            model.logger.record('evaluation/success_rate',measured['summary']['success_rate'])
        for key,value in stats.items():
            if isinstance(value,(float,int)) and value is not None:model.logger.record('training/'+key,value)
        model.logger.record('monitor/actual_timesteps',current);model.logger.dump(step=current)
        stages.append(row);write_json(folder/'stages.json',stages)
        print(f"step={current}/{config['planned_timesteps']} stage={args.stage} "
            f"success={stats['recent_success_rate']} contact={stats['contact_episodes']} lifted={stats['lifted_episodes']} "
            f"eval={row.get('evaluation',{}).get('success_rate')} wall_s={perf_counter()-started:.1f}",flush=True)
    try:
        snapshot(True)
        while model.num_timesteps<args.timesteps:
            if (folder/'STOP_TRAINING').exists():status='stopped';error='Explicit stop file';break
            before=int(model.num_timesteps)
            model.learn(total_timesteps=span,reset_num_timesteps=before==0,callback=audit,log_interval=None)
            if audit.abort_reason:status='safety_aborted';error=audit.abort_reason;break
            if model.num_timesteps!=before+span:raise ValueError('Incomplete PPO update')
            snapshot(model.num_timesteps>=args.timesteps or model.num_timesteps//args.eval_every>before//args.eval_every)
        if status=='running':status='completed'
    except KeyboardInterrupt:status='interrupted';error='KeyboardInterrupt'
    except Exception as exc:status='error';error=f'{type(exc).__name__}: {exc}'
    finally:
        if status=='running':status='interrupted';error='Incomplete run'
        save_checkpoint(model,folder/'final',config,complete=status=='completed')
        final=parameter_hashes(model)
        summary=dict(status=status,error=error,actual_timesteps=int(model.num_timesteps),epoch_updates=int(model._n_updates),
            optimizer_steps=optimizer_steps(model),initial_parameter_hashes=initial,final_parameter_hashes=final,
            actor_updated=initial['actor']!=final['actor'],critic_updated=initial['critic']!=final['critic'],
            training=audit.summary(),evaluations=evaluations,wall_s=perf_counter()-started)
        write_json(folder/'summary.json',summary)
        try:env.close()
        except (EOFError,OSError):pass
        model.logger.close()
    return summary
