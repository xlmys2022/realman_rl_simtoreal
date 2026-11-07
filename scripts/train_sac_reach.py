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
    save_checkpoint, asset_hashes, versions, write_json)


def train(folder, *, timesteps=5000, seed=2026, eval_every=5000, eval_episodes=10, label='pilot'):
    folder = Path(folder).resolve()
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('Run directory must be empty; never overwrite a previous experiment')
    if min(timesteps, eval_every, eval_episodes) < 1 or seed < 0:
        raise ValueError('Positive step/evaluation counts and nonnegative seed required')
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env, transform = training_env(ENV_CONFIG, folder/'train.monitor.csv')
    model = make_model(env, seed)
    model.set_logger(configure(str(folder/'sb3'), ['csv', 'json']))
    config = dict(schema=1, label=label, independent_from_scratch=True, seed=seed,
                  policy='MultiInputPolicy', environment=ENV_CONFIG, sac=SAC_CONFIG,
                  requested_timesteps=timesteps, torch_num_threads=1, deterministic_algorithms=True,
                  observation_transform=transform.config, reward_transform='none',
                  validation_seeds=list(range(10_000, 10_000+eval_episodes)),
                  evaluation_every_steps=eval_every, assets=asset_hashes(),
                  experiment_sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                      (Path(__file__).resolve(), Path(inspect.getfile(TrainingAudit)).resolve())},
                  checkpoint_note='Chunks continue in the same process with the same replay/env/transform. '
                                  'Each CLI invocation starts an independent experiment; no implicit resume.')
    write_json(folder/'config.json', config)
    write_json(folder/'observation_transform.json', transform.config)
    write_json(folder/'dependencies.json', versions())
    model.save(folder/'initial_model.zip')
    initial_actor = [p.detach().clone() for p in model.actor.parameters()]
    initial_critic = [p.detach().clone() for p in model.critic.parameters()]
    audit = TrainingAudit(folder)
    checkpoints, training_wall, evaluation_wall = [], 0., 0.
    begin = perf_counter()
    status, error = 'completed', None
    try:
        while model.num_timesteps < timesteps:
            count = min(eval_every, timesteps-model.num_timesteps)
            started = perf_counter()
            try:
                model.learn(total_timesteps=count, reset_num_timesteps=model.num_timesteps == 0,
                            callback=audit, log_interval=10)
            finally:
                training_wall += perf_counter()-started
            diagnostic = audit.record_diagnostic(scope='after_learn_checkpoint')
            checkpoint = folder/'checkpoints'/f'step_{model.num_timesteps:06d}'
            save_checkpoint(model, checkpoint)
            # Every checkpoint carries its own inference preprocessing/config.
            write_json(checkpoint/'config.json', config)
            write_json(checkpoint/'observation_transform.json', transform.config)
            if audit.abort_reason is not None:
                status, error = 'safety_aborted', audit.abort_reason
                break
            started = perf_counter()
            evaluation = evaluate('sac', config['validation_seeds'], model=model, transform=transform,
                                  config=ENV_CONFIG, gamma=model.gamma)
            evaluation_wall += perf_counter()-started
            write_json(checkpoint/'evaluation.json', evaluation)
            row = dict(step=int(model.num_timesteps), training=audit.summary(),
                       evaluation=evaluation['summary'], diagnostic=diagnostic,
                       training_wall_s=training_wall, training_steps_per_s=model.num_timesteps/training_wall)
            checkpoints.append(row)
            write_json(folder/'checkpoints.json', checkpoints)
            print(f"[{label}] step={model.num_timesteps}/{timesteps} updates={model._n_updates} "
                  f"train_fps={model.num_timesteps/training_wall:.2f} "
                  f"train_success={row['training']['success_rate']} eval_success={evaluation['summary']['success_rate']} "
                  f"Q=[{diagnostic.get('q_min')},{diagnostic.get('q_max')}] "
                  f"critic_loss={diagnostic.get('train/critic_loss')}", flush=True)
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
    args = parser.parse_args()
    if min(args.timesteps, args.eval_every, args.eval_episodes) <= 0 or args.seed < 0:
        parser.error('Positive counts and nonnegative seed required')
    try:
        result = train(args.run_dir, timesteps=args.timesteps, seed=args.seed, eval_every=args.eval_every,
                       eval_episodes=args.eval_episodes, label=args.label)
        print(f"status={result['status']} steps={result['total_timesteps']} error={result['error']}", flush=True)
        return 0 if result['status'] == 'completed' else 1
    except (ValueError, RuntimeError) as exc:
        print(f'[FAIL] {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
