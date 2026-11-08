#!/usr/bin/env python3
"""CPU SAC Reach experiment, using official SB3 and the unchanged six-axis env."""
import argparse
import hashlib
import inspect
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from realman_rl_simtoreal.paths import OUTPUTS
from realman_rl_simtoreal.experiments.sac_reach import (
    ENV_CONFIG, SAC_CONFIG, training_env, make_model, TrainingAudit, evaluate,
    save_checkpoint, asset_hashes, versions, write_json, make_probe)
from realman_rl_simtoreal.experiments.reach_tasks import TASKS, task_environment
from realman_rl_simtoreal.experiments.sac_monitoring import (
    MonitoredTrainingAudit, probe_q, record_evaluation)
from realman_rl_simtoreal.experiments.sac_diagnostics import td_diagnostic


def train(folder, *, timesteps=5000, seed=2026, eval_every=5000, eval_episodes=10, label='pilot',
          environment=None, eval_seed_start=10000, init_mode='scratch', bc_model=None,
          tensorboard=False, checkpoint_every=None, eval_at_start=False,
          task=None, reward_mode=None, action_scale=None):
    folder = Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Run directory must be empty; never overwrite a previous experiment')
    if min(timesteps, eval_every, eval_episodes) < 1 or min(seed,eval_seed_start) < 0:
        raise ValueError('Positive step/evaluation counts and nonnegative seed required')
    checkpoint_every = eval_every if checkpoint_every is None else checkpoint_every
    if checkpoint_every < 1 or init_mode not in ('scratch','bc_actor'):
        raise ValueError('Positive checkpoint interval and scratch/bc_actor initialization required')
    if init_mode == 'bc_actor' and bc_model is None:
        raise ValueError('bc_actor requires --bc-model; no random fallback')
    pure_run=task is not None or reward_mode is not None or action_scale is not None
    if pure_run and (init_mode != 'scratch' or bc_model is not None):
        raise ValueError('Pure RL tasks require scratch initialization and prohibit any BC reference/data')
    if tensorboard:
        from torch.utils.tensorboard import SummaryWriter  # fail clearly before creating the run
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env_config = dict(ENV_CONFIG)
    if environment is not None:
        env_config.update(environment)
    if task is not None:
        env_config=task_environment(task,env_config,reward_mode=reward_mode,action_scale=action_scale)
    else:
        if reward_mode is not None:env_config['reward_mode']=reward_mode
        if action_scale is not None:env_config['action_scale']=action_scale
    pure_run=pure_run or env_config.get('reward_mode')=='reach_shaped_v1'
    if pure_run and (init_mode!='scratch' or bc_model is not None):
        raise ValueError('Pure RL reward/task configuration prohibits BC initialization/reference')
    if env_config.get('reward_mode')=='reach_shaped_v1' and env_config.get('reward_gamma',.99)!=SAC_CONFIG['gamma']:
        raise ValueError('Reward shaping gamma must equal SAC gamma')
    env, transform = training_env(env_config, folder/'train.monitor.csv')
    model = make_model(env, seed)
    model.set_logger(configure(str(folder/'sb3'), ['csv', 'json']+(['tensorboard'] if tensorboard else [])))
    reference, initialization = None, dict(mode='scratch')
    evaluation_seeds=list(range(eval_seed_start, eval_seed_start+eval_episodes))
    try:
        probe, probe_transform=make_probe(env_config,evaluation_seeds)
        if probe_transform.config != transform.config:
            raise ValueError('Probe preprocessing differs from training')
        if bc_model is not None:
            from realman_rl_simtoreal.experiments.sac_initialization import initialize_actor_from_bc, load_reference
            if init_mode == 'bc_actor':
                reference,initialization=initialize_actor_from_bc(model,bc_model,transform,probe)
                initialization['mode']='bc_actor'
            else:
                reference,reference_metadata=load_reference(model,bc_model,transform)
                initialization.update(reference_only=reference_metadata)
            bc_train=set(reference_metadata['bc_train_seeds']+reference_metadata['bc_validation_seeds']) if init_mode=='scratch' else set(initialization['bc_train_seeds']+initialization['bc_validation_seeds'])
            if bc_train & set(evaluation_seeds):
                raise ValueError('Independent evaluation seeds overlap BC training/validation targets')
    except Exception:
        env.close();model.logger.close()
        raise
    np.savez_compressed(folder/'fixed_probe.npz',**probe)
    write_json(folder/'initialization.json',initialization)
    raw=env.venv.envs[0].unwrapped
    reward_parameters=(dict(sigma_m=raw.reward_sigma_m,gamma=raw.reward_gamma,
                           terminal_value=raw.shaped_terminal_value,rejection_cost=1-raw.reward_gamma)
                       if raw.reward_mode=='reach_shaped_v1' else None)
    config = dict(schema=1, label=label, independent_from_scratch=init_mode=='scratch', seed=seed,
                  init_mode=init_mode, bc_model=str(Path(bc_model).resolve()) if bc_model is not None else None,
                  initialization=initialization,tensorboard=tensorboard,
                  training_task=task,pure_rl=pure_run,
                  reward_parameters=reward_parameters,
                  policy='MultiInputPolicy', environment=env_config, sac=SAC_CONFIG,
                  requested_timesteps=timesteps, torch_num_threads=1, deterministic_algorithms=True,
                  observation_transform=transform.config, reward_transform='none',
                  validation_seeds=evaluation_seeds,
                  evaluation_every_steps=eval_every,checkpoint_every_steps=checkpoint_every,
                  evaluate_at_start=bool(eval_at_start or init_mode=='bc_actor'), assets=asset_hashes(),
                  experiment_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                      (Path(__file__).resolve(), Path(inspect.getfile(TrainingAudit)).resolve())},
                  checkpoint_note='Chunks continue in the same process with the same replay/env/transform. '
                                  'Each CLI invocation starts with empty replay/optimizer; no implicit resume. '
                                  +('Pure RL: no policy transfer/reference or expert replay.' if pure_run else
                                    'BC copies only the explicitly selected Actor mean path.'))
    write_json(folder/'config.json', config)
    write_json(folder/'observation_transform.json', transform.config)
    write_json(folder/'dependencies.json', versions())
    model.save(folder/'initial_model.zip')
    save_checkpoint(model,folder/'initial')
    write_json(folder/'initial/config.json',config)
    write_json(folder/'initial/observation_transform.json',transform.config)
    initial_actor = [p.detach().clone() for p in model.actor.parameters()]
    initial_critic = [p.detach().clone() for p in model.critic.parameters()]
    audit = MonitoredTrainingAudit(folder,reference=reference,transform=transform,probe=probe,
                                   collect_motion=pure_run,action_scale=raw.action_scale)
    audit.init_callback(model)  # step-0 diagnostics before the first learn()
    checkpoints, evaluations, training_wall, evaluation_wall = [], [], 0., 0.
    begin = perf_counter()
    status, error = 'completed', None
    def snapshot(evaluate_now, save_now):
        nonlocal evaluation_wall
        diagnostic=audit.record_diagnostic(scope='after_learn_checkpoint')
        diagnostic['fixed_probe_q']=probe_q(model,transform,probe)
        diagnostic['entropy_coefficient']=(float(model.log_ent_coef.detach().exp())
                                           if model.log_ent_coef is not None else float(model.ent_coef_tensor))
        model.logger.record('monitor/entropy_coefficient',diagnostic['entropy_coefficient'])
        for key,value in diagnostic['fixed_probe_q'].items():
            model.logger.record('diagnostic/'+key,value)
        if model.replay_buffer.size():
            # Indexed storage sample, RNG isolated; this is not historical minibatch loss.
            count=min(256,model.replay_buffer.size())
            terminal_labels=None
            if pure_run:
                oldest=model.num_timesteps-model.replay_buffer.size()
                terminal_labels={(e['end_step']-1)%model.replay_buffer.buffer_size:
                    ('timeout' if e['reason']=='time_limit' else 'safety_terminal' if e['safety_failure'] else 'success_terminal')
                    for e in audit.episodes if e['end_step']>oldest}
            td,_=td_diagnostic(model,indices=np.arange(count),terminal_labels=terminal_labels)
            diagnostic['soft_td_sample']=dict(transitions=count,alpha=td['alpha'],
                residual_rmse=td['categories']['all']['residual_rmse'],
                stored_dones=td['stored_dones'],timeouts=td['timeouts'],
                effective_terminations=td['effective_terminations'],sampling='first <=256 storage slots, frozen next-action draw',
                terminal_label_source='episode flags' if pure_run else 'legacy reward threshold')
            model.logger.record('diagnostic/soft_td_residual_rmse',diagnostic['soft_td_sample']['residual_rmse'])
        checkpoint=folder/'checkpoints'/f'step_{model.num_timesteps:06d}'
        if save_now:
            save_checkpoint(model,checkpoint)
            write_json(checkpoint/'config.json',config)
            write_json(checkpoint/'observation_transform.json',transform.config)
        evaluation=None
        if evaluate_now:
            started=perf_counter()
            measured=evaluate('sac',evaluation_seeds,model=model,transform=transform,
                             config=env_config,gamma=model.gamma,reference_policy=reference,collect_motion=pure_run)
            evaluation_wall+=perf_counter()-started
            evaluation=measured['summary']
            if init_mode=='bc_actor' and model.num_timesteps==0 and measured['bc_action_deviation']['max_abs']>1e-7:
                raise ValueError('Step-0 closed-loop BC/SAC action mismatch')
            destination=checkpoint if save_now else folder/'evaluations'/f'step_{model.num_timesteps:06d}'
            write_json(destination/'evaluation.json',measured)
            record_evaluation(model.logger,measured)
            evaluations.append(dict(step=int(model.num_timesteps),summary=evaluation,
                                    bc_action_deviation=measured.get('bc_action_deviation'),motion=measured.get('motion')))
            write_json(folder/'evaluations.json',evaluations)
        row=dict(step=int(model.num_timesteps),training=audit.summary(),evaluation=evaluation,
                 diagnostic=diagnostic,checkpoint_saved=save_now,
                 training_wall_s=training_wall,
                 training_steps_per_s=model.num_timesteps/training_wall if training_wall else None)
        checkpoints.append(row)
        write_json(folder/'checkpoints.json',checkpoints)
        model.logger.dump(step=int(model.num_timesteps))
        bc_text='' if pure_run else f"bc_MAE={diagnostic.get('bc_fixed_probe_deviation',{}).get('mae')} "
        print(f"[{label}] step={model.num_timesteps}/{timesteps} updates={model._n_updates} "
              f"train_success={row['training']['success_rate']} "
              f"eval_success={evaluation['success_rate'] if evaluation else None} "
              +bc_text+
              f"critic_loss={diagnostic.get('train/critic_loss')}",flush=True)
    try:
        if config['evaluate_at_start']:
            snapshot(True,True)
        while model.num_timesteps < timesteps:
            current=int(model.num_timesteps)
            next_eval=(current//eval_every+1)*eval_every
            next_checkpoint=(current//checkpoint_every+1)*checkpoint_every
            count=min(next_eval,next_checkpoint,timesteps)-current
            started = perf_counter()
            try:
                model.learn(total_timesteps=count, reset_num_timesteps=model.num_timesteps == 0,
                            callback=audit, log_interval=10)
            finally:
                training_wall += perf_counter()-started
            if audit.abort_reason is not None:
                status, error = 'safety_aborted', audit.abort_reason
                snapshot(False,True)
                break
            final_step=model.num_timesteps>=timesteps
            snapshot(model.num_timesteps%eval_every==0 or final_step,
                     model.num_timesteps%checkpoint_every==0 or final_step)
    except (ValueError, RuntimeError, FloatingPointError) as exc:
        status, error = 'error', f'{type(exc).__name__}: {exc}'
    except KeyboardInterrupt:
        status, error = 'interrupted', 'KeyboardInterrupt; partial checkpoint saved'
    finally:
        final = folder/'final'
        save_checkpoint(model, final, replay=True)
        write_json(final/'config.json', config)
        write_json(final/'observation_transform.json', transform.config)
        summary = dict(status=status, error=error, requested_timesteps=timesteps,
                       total_timesteps=int(model.num_timesteps), n_updates=int(model._n_updates),
                       buffer_size=int(model.replay_buffer.size()), training=audit.summary(),
                       actor_updated=any(not torch.equal(a, b) for a, b in zip(initial_actor, model.actor.parameters())),
                       critic_updated=any(not torch.equal(a, b) for a, b in zip(initial_critic, model.critic.parameters())),
                       timing=dict(training_wall_s=training_wall, evaluation_wall_s=evaluation_wall,
                                   total_wall_s=perf_counter()-begin,
                                   training_steps_per_s=model.num_timesteps/training_wall if training_wall else None),
                       checkpoints=checkpoints)
        write_json(folder/'summary.json', summary)
        env.close()
        model.logger.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path,
                        default=OUTPUTS/'rl/sac_reach'/datetime.now().strftime('run_%Y%m%d_%H%M%S'))
    parser.add_argument('--timesteps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--eval-every', type=int, default=5000)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--label', default='pilot')
    parser.add_argument('--boundary-mode', choices=['terminate','reject_and_hold'], default='terminate')
    parser.add_argument('--rejection-penalty',type=float,default=1.)
    parser.add_argument('--eval-seed-start',type=int,default=10000)
    parser.add_argument('--init-mode',choices=['scratch','bc_actor'],default='scratch')
    parser.add_argument('--bc-model',type=Path,help='Required for bc_actor; optional frozen reference in scratch mode')
    parser.add_argument('--tensorboard',action='store_true',help='Add SB3 TensorBoard event writer under run-dir/sb3')
    parser.add_argument('--checkpoint-every',type=int,help='Defaults to --eval-every')
    parser.add_argument('--eval-at-start',action='store_true',help='Also evaluate step 0 (always enabled for bc_actor)')
    parser.add_argument('--task',choices=list(TASKS),help='Explicit pure RL task; prohibits BC initialization/reference')
    parser.add_argument('--reward-mode',choices=['distance','safe_dense','reach_shaped_v1'])
    parser.add_argument('--action-scale',type=float,help='Metres per normalized action component; original default .001')
    args = parser.parse_args()
    if (min(args.timesteps, args.eval_every, args.eval_episodes) <= 0 or min(args.seed,args.eval_seed_start) < 0
            or not np.isfinite(args.rejection_penalty) or args.rejection_penalty<=0):
        parser.error('Positive counts and nonnegative seed required')
    try:
        result = train(args.run_dir, timesteps=args.timesteps, seed=args.seed, eval_every=args.eval_every,
                       eval_episodes=args.eval_episodes, label=args.label,
                       environment=None if args.boundary_mode=='terminate' else
                       dict(boundary_mode=args.boundary_mode,rejection_penalty=args.rejection_penalty),
                       eval_seed_start=args.eval_seed_start,init_mode=args.init_mode,bc_model=args.bc_model,
                       tensorboard=args.tensorboard,checkpoint_every=args.checkpoint_every,eval_at_start=args.eval_at_start,
                       task=args.task,reward_mode=args.reward_mode,action_scale=args.action_scale)
        print(f"status={result['status']} steps={result['total_timesteps']} error={result['error']}", flush=True)
        return 0 if result['status'] == 'completed' else 1
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f'[FAIL] {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
