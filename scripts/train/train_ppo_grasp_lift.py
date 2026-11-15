#!/usr/bin/env python3
"""Train four-action PPO from scratch or a recorded PPO curriculum checkpoint."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

from realman_rl_simtoreal.viewer import configure_backend
configure_backend()

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from realman_rl_simtoreal.rl.ppo_grasp import (
    CURRICULUM,ENCODING,REWARD,PPO_PARAMETERS,GraspAudit,environment_config,
    make_training_env,parameter_hashes,optimizer_steps,save_checkpoint,load_checkpoint,evaluate_model)
from realman_rl_simtoreal.utils.artifacts import experiment_metadata
from realman_rl_simtoreal.rl.common.artifacts import empty_directory,write_json,versions


def train(args):
    if args.seed<0 or min(args.timesteps,args.n_envs,args.n_steps,args.batch_size,args.epochs,args.eval_every,args.eval_episodes)<1:
        raise ValueError('Expected nonnegative seed and positive budgets/sizes')
    span=args.n_envs*args.n_steps
    if args.n_steps<2 or args.batch_size<2 or span%args.batch_size:
        raise ValueError('Rollouts must contain full minibatches of at least two samples')
    folder=empty_directory(args.run_dir)
    torch.set_num_threads(1)
    environment=environment_config(args.stage)
    if getattr(args,'approach_distance',None) is not None:
        environment['approach_distance']=args.approach_distance
    if args.episode_steps is not None:environment['max_episode_steps']=args.episode_steps
    parameters=copy.deepcopy(PPO_PARAMETERS)
    parameters.update(n_steps=args.n_steps,batch_size=args.batch_size,n_epochs=args.epochs)
    if getattr(args,'learning_rate',None) is not None:
        if not np.isfinite(args.learning_rate) or not 0<args.learning_rate<=.001:
            raise ValueError('Learning rate must be in (0,.001]')
        parameters['learning_rate']=args.learning_rate
    reward=copy.deepcopy(REWARD);reward['scale']=getattr(args,'reward_scale',.01)
    if not 0<reward['scale']<=1:raise ValueError('Reward scale must be in (0,1]')
    xyz_std=getattr(args,'xyz_std',.1);gripper_std=getattr(args,'gripper_std',.3)
    if not (0<xyz_std<=1 and 0<gripper_std<=1):raise ValueError('Exploration standard deviations must be in (0,1]')
    mean=getattr(args,'gripper_mean',0.)
    if not np.isfinite(mean) or not -1<=mean<=1:raise ValueError('Initial gripper mean must be in [-1,1]')
    randomize_height=getattr(args,'randomize_height',False)
    randomize_approach=getattr(args,'randomize_approach',False)
    env=make_training_env(environment,args.n_envs,args.seed,reward,randomize_height,randomize_approach)
    model=PPO('MlpPolicy',env,seed=args.seed,verbose=0,**parameters)
    with torch.no_grad():
        model.policy.log_std[:3].fill_(float(np.log(xyz_std)))
        model.policy.log_std[3].fill_(float(np.log(gripper_std)))
        model.policy.action_net.bias[3]=mean
    transfer=None
    if args.init_model is not None:
        source,saved=load_checkpoint(args.init_model)
        state=json.loads((args.init_model.parent/'state.json').read_text())
        if not state['completed_update_stage']:raise ValueError('Curriculum sources require a completed PPO update')
        if saved['encoding']!=ENCODING:raise ValueError('Incompatible PPO curriculum encoding')
        changed=saved['reward']!=reward
        if changed and not getattr(args,'allow_reward_change',False):
            raise ValueError('Reward differs from source; use --allow-reward-change for an explicit recorded change')
        model.policy.load_state_dict(source.policy.state_dict(),strict=True)
        reset_exploration=getattr(args,'reset_exploration',False)
        if reset_exploration:
            with torch.no_grad():
                model.policy.log_std[:3].fill_(float(np.log(xyz_std)))
                model.policy.log_std[3].fill_(float(np.log(gripper_std)))
        reset_gripper_bias=getattr(args,'reset_gripper_bias',False)
        if reset_gripper_bias:
            with torch.no_grad():model.policy.action_net.bias[3]=mean
        transformed=(['log_std'] if reset_exploration else [])+(['action_net.bias'] if reset_gripper_bias else [])
        transfer=dict(model=str(args.init_model.resolve()),sha256=hashlib.sha256(args.init_model.read_bytes()).hexdigest(),
            source_timesteps=int(source.num_timesteps),source_parameter_hashes=parameter_hashes(source),
            optimizer_reset=True,counters_reset=True,reward_changed=changed,source_reward=saved['reward'],
            exploration_reset=reset_exploration,gripper_bias_reset=reset_gripper_bias,
            transformed_parameters=transformed)
        model.set_random_seed(args.seed)
    model.grasp_ppo_contract=dict(encoding=copy.deepcopy(ENCODING),reward=copy.deepcopy(reward))
    model.set_logger(configure(str(folder/'sb3'),['tensorboard'] if args.tensorboard else ['csv']))
    metadata=experiment_metadata()
    metadata['algorithm_sources']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),Path(__import__('inspect').getfile(GraspAudit)))}
    config=dict(schema=1,algorithm='PPO',task='grasp_lift',pure_ppo=True,curriculum_stage=args.stage,
        initialization='scratch' if transfer is None else 'ppo_weights',transfer=transfer,
        demonstrations=False,staged_actions=False,environment=environment,encoding=ENCODING,reward=reward,
        ppo=parameters,n_envs=args.n_envs,seed=args.seed,requested_timesteps=args.timesteps,
        initial_policy_std=model.policy.log_std.detach().exp().cpu().tolist(),
        initial_action_bias=model.policy.action_net.bias.detach().cpu().tolist(),
        training_reset_distribution=dict(randomize_height=randomize_height,
            height_range_m=[.005,environment['lift_height']] if randomize_height else [environment['lift_height']]*2,
            randomize_approach=randomize_approach,
            approach_range_m=[.005,environment['approach_distance']] if randomize_approach else [environment['approach_distance']]*2),
        planned_timesteps=((args.timesteps+span-1)//span)*span,
        validation_seeds=list(range(args.eval_seed_start,args.eval_seed_start+args.eval_episodes)),**metadata)
    write_json(folder/'config.json',config);write_json(folder/'dependencies.json',versions())
    audit=GraspAudit(folder,args.n_envs);audit.init_callback(model)
    initial=parameter_hashes(model);save_checkpoint(model,folder/'initial',config)
    evaluations=[];stages=[];status='running';error=None;started=perf_counter()
    def snapshot(evaluate_now):
        current=int(model.num_timesteps)
        checkpoint=folder/'checkpoints'/f'step_{current:08d}'
        save_checkpoint(model,checkpoint,config)
        stats=audit.summary()
        values=dict(step=current,epochs=int(model._n_updates),optimizer_steps=optimizer_steps(model),training=stats)
        for key in ('policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction','explained_variance','std'):
            value=model.logger.name_to_value.get('train/'+key)
            values[key]=float(value) if value is not None and np.isfinite(value) else None
        if evaluate_now:
            measured=evaluate_model(model,config,config['validation_seeds'])
            write_json(checkpoint/'evaluation.json',measured)
            values['evaluation']=measured['summary']
            evaluations.append(dict(step=current,summary=measured['summary'],checkpoint=str(checkpoint/'model.zip')))
            write_json(folder/'evaluations.json',evaluations)
            model.logger.record('evaluation/success_rate',measured['summary']['success_rate'])
        for key,value in stats.items():
            if isinstance(value,(int,float)) and value is not None:model.logger.record('training/'+key,value)
        model.logger.record('monitor/actual_timesteps',current)
        model.logger.record('monitor/optimizer_steps',optimizer_steps(model))
        model.logger.dump(step=current)
        stages.append(values);write_json(folder/'stages.json',stages)
        rate=values.get('evaluation',{}).get('success_rate')
        print(f"step={current}/{config['planned_timesteps']} stage={args.stage} training={stats['recent_success_rate']} "
              f"contact={stats['contact_episodes']} lifted={stats['lifted_episodes']} eval={rate} "
              f"wall_s={perf_counter()-started:.1f}",flush=True)
    try:
        snapshot(True)
        while model.num_timesteps<args.timesteps:
            before=int(model.num_timesteps)
            model.learn(total_timesteps=span,reset_num_timesteps=before==0,callback=audit,log_interval=None)
            if audit.abort_reason:
                status='safety_aborted';error=audit.abort_reason;break
            if model.num_timesteps!=before+span:raise ValueError('Incomplete PPO rollout/update')
            snapshot(model.num_timesteps>=args.timesteps or model.num_timesteps//args.eval_every>before//args.eval_every)
        if status=='running':status='completed'
    except KeyboardInterrupt:
        status='interrupted';error='KeyboardInterrupt'
    except Exception as exc:
        status='error';error=f'{type(exc).__name__}: {exc}'
    finally:
        if status=='running':status='interrupted';error='Execution ended before a completed training run'
        save_checkpoint(model,folder/'final',config,complete=status=='completed')
        final=parameter_hashes(model)
        summary=dict(status=status,error=error,actual_timesteps=int(model.num_timesteps),
            epoch_updates=int(model._n_updates),optimizer_steps=optimizer_steps(model),training=audit.summary(),
            initial_parameter_hashes=initial,final_parameter_hashes=final,
            actor_updated=initial['actor']!=final['actor'],critic_updated=initial['critic']!=final['critic'],
            wall_s=perf_counter()-started,evaluations=evaluations)
        write_json(folder/'summary.json',summary)
        try:env.close()
        except (OSError,EOFError):pass
        model.logger.close()
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--stage',choices=list(CURRICULUM),default='full')
    p.add_argument('--init-model',type=Path)
    p.add_argument('--allow-reward-change',action='store_true',help='Explicitly record a reward change when transferring PPO weights')
    p.add_argument('--randomize-height',action='store_true',help='Train on seeded lift heights from 5mm to the stage maximum')
    p.add_argument('--randomize-approach',action='store_true',help='Train on seeded start distances from 5mm to the stage maximum')
    p.add_argument('--reset-exploration',action='store_true',help='Reset only log-std after transferring a PPO policy; record the transformation')
    p.add_argument('--reset-gripper-bias',action='store_true',help='Explicitly reinitialize only the fourth action bias after PPO transfer')
    p.add_argument('--timesteps',type=int,default=65536)
    p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--n-envs',type=int,default=8);p.add_argument('--n-steps',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--learning-rate',type=float,help='Explicit PPO learning rate for curriculum fine tuning')
    p.add_argument('--xyz-std',type=float,default=.1,help='Initial Cartesian exploration std for scratch runs')
    p.add_argument('--gripper-std',type=float,default=.3,help='Initial aperture exploration std for scratch runs')
    p.add_argument('--gripper-mean',type=float,default=0.,help='Initial aperture-action bias for scratch or --reset-gripper-bias')
    p.add_argument('--reward-scale',type=float,default=.01,help='Fixed reward multiplier for PPO optimization')
    p.add_argument('--eval-every',type=int,default=16384);p.add_argument('--eval-episodes',type=int,default=4)
    p.add_argument('--eval-seed-start',type=int,default=240000)
    p.add_argument('--episode-steps',type=int,help='Explicit episode time-budget override, recorded in the configuration')
    p.add_argument('--approach-distance',type=float,help='Explicit curriculum start distance in metres, recorded in the configuration')
    p.add_argument('--tensorboard',action='store_true')
    args=p.parse_args()
    try:
        summary=train(args)
        print('status=',summary['status'],'actual_timesteps=',summary['actual_timesteps'],'error=',summary['error'])
        return int(summary['status']!='completed')
    except (ValueError,RuntimeError,OSError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
