#!/usr/bin/env python3
"""Pure official PPO Reach; evaluate/save only after complete rollout updates."""
import argparse
import hashlib
import inspect
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from realman_rl_simtoreal.rl.ppo import (
    PPOAudit,
    environment_config,
    make_ppo,
    parameter_hashes,
    optimizer_steps,
    crossed_thresholds,
    post_update_metrics,
    evaluate_ppo,
    save_ppo_checkpoint,
)
from realman_rl_simtoreal.tasks.reach import TASKS, OBSERVATION_MODES
from realman_rl_simtoreal.rl.common.env_factory import training_env
from realman_rl_simtoreal.rl.common.artifacts import asset_hashes, versions, write_json, append_json
from realman_rl_simtoreal.rl.common.evaluation import record_evaluation
from realman_rl_simtoreal.tasks.rewards import REWARD_MODES
from realman_rl_simtoreal.rl.common.observation import observation_metadata, validate_model_observation


def train(folder,*,timesteps=16384,seed=2026,task='fixed_reach',reward_mode=None,
          action_scale=None,eval_every=2048,eval_episodes=None,checkpoint_every=2048,
          eval_seed_start=87000,tensorboard=False,n_steps=512,batch_size=64,n_epochs=10,
          observation_mode=None, reward_progress_parameters=None, initial_model=None):
    folder=Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):raise ValueError('Run directory must be empty; never overwrite')
    environment=environment_config(task,reward_mode,action_scale,observation_mode,reward_progress_parameters)
    reward_mode = environment['reward_mode']
    action_scale = environment['action_scale']
    candidate_count=len(environment['target_offsets']) if environment['target_mode']=='discrete' else None
    if eval_episodes is None:eval_episodes=candidate_count or 1
    if candidate_count is not None and eval_episodes!=candidate_count:
        raise ValueError(f'Discrete task requires eval_episodes={candidate_count}, one per distinct target')
    if min(timesteps,eval_every,eval_episodes,checkpoint_every)<1 or min(seed,eval_seed_start)<0:
        raise ValueError('Positive step/interval/count and nonnegative seeds required')
    if min(n_steps,batch_size)<2 or n_steps%batch_size or n_epochs<1:
        raise ValueError('Require n_steps/batch>=2, full minibatches and positive epochs')
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    folder.mkdir(parents=True,exist_ok=True)
    env,transform=training_env(environment,folder/'train.monitor.csv')
    model,parameters=make_ppo(env,seed,n_steps=n_steps,batch_size=batch_size,n_epochs=n_epochs,
                             initial_model=initial_model)
    if environment.get('reward_gamma',model.gamma)!=model.gamma:
        env.close();raise ValueError('PPO and reward shaping gamma must match')
    model.set_logger(configure(str(folder/'sb3'),['csv','json']+(['tensorboard'] if tensorboard else [])))
    rounded_budget=math.ceil(timesteps/n_steps)*n_steps
    raw=env.venv.envs[0].unwrapped
    validate_model_observation(model, transform)
    config=dict(schema=2,algorithm='PPO',policy='MultiInputPolicy',pure_rl=True,
        init_mode='scratch' if initial_model is None else 'ppo_transfer',
        seed=seed,training_task=task,environment=environment,ppo=parameters,
        requested_timesteps=timesteps,planned_complete_rollout_timesteps=rounded_budget,n_envs=1,
        torch_num_threads=1,deterministic_algorithms=True,tensorboard=tensorboard,
        observation_transform=transform.config,reward_transform='none',
        reward_parameters=dict(sigma_m=raw.reward_sigma_m,gamma=raw.reward_gamma,
            terminal_value=raw.shaped_terminal_value,rejection_cost=1-raw.reward_gamma) if reward_mode=='reach_shaped_v1'
            else raw.reward_progress_parameters.copy() if reward_mode=='reach_progress_v1' else None,
        validation_seeds=list(range(eval_seed_start,eval_seed_start+eval_episodes)),
        evaluation_every_steps=eval_every,checkpoint_every_steps=checkpoint_every,
        scheduling='Intervals are thresholds; run once at the next completed PPO update, plus initial/final. '
                   'No partial-rollout training or snapshots presented as complete updates.',
        policy_architecture=dict(features=model.policy.features_extractor.features_dim,actor=[256,256,3],critic=[256,256,1],
            activation=model.policy.activation_fn.__name__,distribution=type(model.policy.action_dist).__name__,
            squash_output=model.policy.squash_output,orthogonal_init=model.policy.ortho_init,
            log_std_initial=model.policy.log_std.detach().cpu().tolist(),
            action_handling='Stock Gaussian raw actions; SB3 clips to the original Box before env.step'),
        assets=asset_hashes(),experiment_sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__).resolve(),Path(inspect.getfile(PPOAudit)).resolve(),Path(inspect.getfile(training_env)).resolve())},
        checkpoint_note='Independent random run. No BC/SAC weights, expert data, replay prefill or implicit resume. '
                        'One complete rollout per learn call, constant learning rate/clip range; policy and env stay alive.')
    config.update(observation_metadata(raw, transform))
    if initial_model is not None:
        config['transfer'] = model.reach_transfer_configuration
        config['checkpoint_note'] = ('PPO policy/value/log-std weights transferred from the recorded model. '
            'New optimizer, rollout buffer, counters and environment; requested timesteps are additional training steps. '
            'Observation and action scales follow this run configuration.')
    if reward_mode=='reach_progress_v1':
        # Stock SB3 serializes this plain metadata; no policy/optimizer change.
        model.reach_reward_configuration=dict(mode=reward_mode,parameters=config['reward_parameters'].copy())
    write_json(folder/'config.json',config);write_json(folder/'observation_transform.json',transform.config)
    write_json(folder/'dependencies.json',versions())
    initial=parameter_hashes(model);model.save(folder/'initial_model.zip')
    save_ppo_checkpoint(model,folder/'initial',config,0,completed_stage=False)
    audit=PPOAudit(folder,action_scale);audit.init_callback(model)
    stages=[];evaluations=[];trained_rollouts=0;training_wall=0.;evaluation_wall=0.
    status,error='completed',None;started_run=perf_counter()

    def snapshot(previous,*,initial_stage=False,final_stage=False):
        nonlocal evaluation_wall
        current=int(model.num_timesteps)
        eval_thresholds=[] if initial_stage else crossed_thresholds(previous,current,eval_every)
        checkpoint_thresholds=[] if initial_stage else crossed_thresholds(previous,current,checkpoint_every)
        do_eval=initial_stage or final_stage or bool(eval_thresholds)
        do_save=initial_stage or final_stage or bool(checkpoint_thresholds)
        metrics=post_update_metrics(model,audit);audit.record_scalars()
        metrics['trained_rollouts']=trained_rollouts
        metrics['phase']='initial' if initial_stage else 'post_update'
        for key,value in [('actual_timesteps',current),('completed_rollout_updates',trained_rollouts),
                          ('epoch_updates',model._n_updates),('optimizer_steps',optimizer_steps(model))]:
            model.logger.record('monitor/'+key,value)
        evaluation=None
        checkpoint=folder/'checkpoints'/f'step_{current:06d}'
        if do_save:save_ppo_checkpoint(model,checkpoint,config,trained_rollouts,completed_stage=not initial_stage)
        if do_eval:
            tick=perf_counter()
            measured=evaluate_ppo(model,transform,environment,config['validation_seeds'])
            evaluation_wall+=perf_counter()-tick
            destination=checkpoint if do_save else folder/'evaluations'/f'step_{current:06d}'
            write_json(destination/'evaluation.json',measured)
            record_evaluation(model.logger,measured)
            for key,value in measured['box_clipping'].items():
                if value is not None:model.logger.record('evaluation/box_clipping/'+key,value)
            evaluation=measured['summary']
            evaluations.append(dict(step=current,summary=evaluation,motion=measured['motion'],
                                    box_clipping=measured['box_clipping'],goal_scope=measured['goal_scope']))
            if 'per_goal' in measured:evaluations[-1]['per_goal']=measured['per_goal']
            write_json(folder/'evaluations.json',evaluations)
        append_json(folder/'metrics.jsonl',metrics)
        row=dict(step=current,phase=metrics['phase'],metrics=metrics,training=audit.summary(),evaluation=evaluation,
            checkpoint_saved=do_save,evaluation_thresholds_crossed=eval_thresholds,
            checkpoint_thresholds_crossed=checkpoint_thresholds)
        stages.append(row);write_json(folder/'stages.json',stages)
        model.logger.dump(step=current)
        print(f"PPO step={current} requested={timesteps} rollouts={trained_rollouts} "
              f"epochs={model._n_updates} optimizer_steps={optimizer_steps(model)} "
              f"eval_success={evaluation['success_rate'] if evaluation else None} "
              f"box_clip={metrics['rollout_box_clipping']['vector_fraction']}",flush=True)

    try:
        snapshot(0,initial_stage=True)
        while model.num_timesteps<timesteps:
            previous=int(model.num_timesteps);epochs_before=model._n_updates;tick=perf_counter()
            try:
                # Stock PPO collects n_steps then trains. Never shorten its buffer.
                model.learn(total_timesteps=n_steps,reset_num_timesteps=previous==0,
                            callback=audit,log_interval=None)
            finally:training_wall+=perf_counter()-tick
            if audit.abort_reason is not None:
                status,error='safety_aborted',audit.abort_reason;break
            if model.num_timesteps!=previous+n_steps or model._n_updates<=epochs_before:
                raise ValueError('PPO returned without exactly one complete rollout update')
            trained_rollouts+=1
            snapshot(previous,final_stage=model.num_timesteps>=timesteps)
    except KeyboardInterrupt:
        status,error='interrupted','KeyboardInterrupt; final may contain an incomplete rollout/update'
    except (ValueError,RuntimeError,FloatingPointError) as exc:
        status,error='error',f'{type(exc).__name__}: {exc}'
    finally:
        complete=status=='completed'
        save_ppo_checkpoint(model,folder/'final',config,trained_rollouts,completed_stage=complete)
        if complete and model.rollout_buffer.full:
            arrays={key:np.asarray(getattr(model.rollout_buffer,key)).copy() for key in
                    ('advantages','returns','values','log_probs','rewards','actions','episode_starts')}
            arrays.update({'observation_'+k:v.copy() for k,v in model.rollout_buffer.observations.items()})
            np.savez_compressed(folder/'final/last_rollout.npz',**arrays)
        final_hashes=parameter_hashes(model)
        summary=dict(status=status,error=error,requested_timesteps=timesteps,
            actual_timesteps=int(model.num_timesteps),planned_complete_rollout_timesteps=rounded_budget,
            completed_rollout_updates=trained_rollouts,collected_complete_rollouts=audit.complete_rollouts,
            epoch_updates=int(model._n_updates),optimizer_steps=optimizer_steps(model),training=audit.summary(),
            initial_parameter_hashes=initial,final_parameter_hashes=final_hashes,
            actor_updated=initial['actor']!=final_hashes['actor'],critic_updated=initial['critic']!=final_hashes['critic'],
            timing=dict(training_wall_s=training_wall,evaluation_wall_s=evaluation_wall,
                total_wall_s=perf_counter()-started_run,
                training_steps_per_s=model.num_timesteps/training_wall if training_wall else None),
            stages=stages,note='PPO epoch_updates counts epochs; optimizer_steps is Adam parameter step count. '
                               'Environment rewards are recorded before stock timeout value bootstrapping.')
        write_json(folder/'summary.json',summary);env.close();model.logger.close()
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--task',choices=list(TASKS),default='fixed_reach')
    p.add_argument('--observation-mode',choices=OBSERVATION_MODES,
                   help='Task defaults: legacy for fixed, goal_relative_v1 for local random, goal_relative_v3 for 20cm tasks')
    p.add_argument('--reward-mode',choices=REWARD_MODES,
                   help='Defaults to reach_progress_v1 with goal-relative observations, otherwise reach_shaped_v1')
    p.add_argument('--action-scale',type=float,help='Metres per axis; defaults to .001 for 20cm, .00025 otherwise')
    p.add_argument('--init-model',type=Path,help='Initialize actor, critic and log-std from a saved goal-relative PPO')
    p.add_argument('--timesteps',type=int,default=16384);p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--eval-every',type=int,default=2048)
    p.add_argument('--eval-episodes',type=int,help='Number of episodes or multi-goal sequences; default one, or all discrete candidates')
    p.add_argument('--eval-seed-start',type=int,default=87000)
    p.add_argument('--checkpoint-every',type=int,default=2048);p.add_argument('--tensorboard',action='store_true')
    p.add_argument('--n-steps',type=int,default=512);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--n-epochs',type=int,default=10)
    a=p.parse_args()
    try:
        result=train(a.run_dir,timesteps=a.timesteps,seed=a.seed,task=a.task,reward_mode=a.reward_mode,
            action_scale=a.action_scale,eval_every=a.eval_every,eval_episodes=a.eval_episodes,
            checkpoint_every=a.checkpoint_every,eval_seed_start=a.eval_seed_start,tensorboard=a.tensorboard,
            n_steps=a.n_steps,batch_size=a.batch_size,n_epochs=a.n_epochs,observation_mode=a.observation_mode,
            initial_model=a.init_model)
        print(f"status={result['status']} requested={result['requested_timesteps']} actual={result['actual_timesteps']} error={result['error']}")
        return 0 if result['status']=='completed' else 1
    except (ValueError,RuntimeError,OSError,ImportError) as exc:
        p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
