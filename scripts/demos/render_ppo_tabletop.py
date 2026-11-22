#!/usr/bin/env python3
"""Offscreen deterministic PPO replay: full desk and a gripper close-up."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
from pathlib import Path

from realman_rl_simtoreal.apps.visualization.tabletop_joint import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',type=int,default=340000)
    p.add_argument('--saved-task',action='store_true',help='Debug a saved curriculum instead of the full zero-start task')
    args=p.parse_args()
    if any(path.exists() for path in (args.output,args.output.with_suffix('.json'),args.output.with_suffix('.png'))):
        p.error('Choose a new output path')
    return run(args)


if __name__=='__main__':raise SystemExit(main())
