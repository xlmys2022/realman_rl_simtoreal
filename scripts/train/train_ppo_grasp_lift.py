#!/usr/bin/env python3
"""Train four-action PPO from scratch or a recorded PPO curriculum checkpoint."""
import argparse
from pathlib import Path

from realman_rl_simtoreal.apps.visualization.viewer import configure_backend
configure_backend()

from realman_rl_simtoreal.apps.training.grasp_lift import train
from realman_rl_simtoreal.tasks.grasp.lift import CURRICULUM


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
