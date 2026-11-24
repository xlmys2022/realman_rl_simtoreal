#!/usr/bin/env python3
"""Train seven-action tabletop PPO; curriculum checkpoints are explicit."""
import argparse
from pathlib import Path

from realman_rl_simtoreal.apps.visualization.viewer import configure_backend
configure_backend()

from realman_rl_simtoreal.apps.training.tabletop_joint import train
from realman_rl_simtoreal.tasks.grasp.tabletop_joint import CURRICULUM


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path);p.add_argument('--stage',choices=list(CURRICULUM),default='full')
    p.add_argument('--init-model',type=Path);p.add_argument('--timesteps',type=int,default=65536);p.add_argument('--seed',type=int,default=2027)
    p.add_argument('--n-envs',type=int,default=8);p.add_argument('--n-steps',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--learning-rate',type=float,default=.0003);p.add_argument('--joint-std',type=float,default=.02,help='Joint-position Gaussian std in degrees')
    p.add_argument('--gripper-std',type=float,default=.3);p.add_argument('--gripper-mean',type=float,default=0.)
    p.add_argument('--reset-exploration',action='store_true');p.add_argument('--tensorboard',action='store_true')
    p.add_argument('--allow-reward-change',action='store_true')
    p.add_argument('--use-sde',action=argparse.BooleanOptionalAction,default=None,help='State dependent exploration; inherited on transfer, default on for scratch')
    p.add_argument('--sde-sample-freq',type=int,help='Exploration refresh steps; inherited on transfer, otherwise 64')
    p.add_argument('--smooth-exploration',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--eval-every',type=int,default=8192);p.add_argument('--eval-episodes',type=int,default=2)
    p.add_argument('--eval-seed-start',type=int,default=340000);p.add_argument('--initial-q-deg',type=float,nargs=6)
    for name in ('start_fraction','object_radius','pregrasp_height','initial_width','lift_height','hold_seconds',
                 'reference_smoothing','gripper_reference_speed','approach_waypoint_height','approach_waypoint_tolerance','goal_height'):
        p.add_argument('--'+name.replace('_','-'),type=float)
    p.add_argument('--max-episode-steps',type=int)
    args=p.parse_args()
    summary=train(args);print('status=',summary['status'],'error=',summary['error'],flush=True)
    return int(summary['status']!='completed')


if __name__=='__main__':raise SystemExit(main())
