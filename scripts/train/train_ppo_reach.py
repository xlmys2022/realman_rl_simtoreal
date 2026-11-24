#!/usr/bin/env python3
"""Pure official PPO Reach; evaluate/save only after complete rollout updates."""
import argparse
from pathlib import Path

from realman_rl_simtoreal.apps.training.reach import train
from realman_rl_simtoreal.tasks.reach.config import TASKS, OBSERVATION_MODES
from realman_rl_simtoreal.tasks.reach.reward import REWARD_MODES


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
