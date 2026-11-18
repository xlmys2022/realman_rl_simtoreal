"""Script argument handling for the Cartesian tabletop task."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    os.environ.setdefault('MUJOCO_GL','egl')
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    setup=commands.add_parser('setup');setup.add_argument('--json',type=Path,default=Path('outputs/grasp/ppo/tabletop_cartesian/validation/setup.json'))
    setup.add_argument('--output',type=Path,help='Render traditional startup to GIF')
    train=commands.add_parser('train');train.add_argument('--stage',choices=['easy','pinch','approach','lift','full'],default='full')
    train.add_argument('--run-dir',required=True,type=Path)
    initialization=train.add_mutually_exclusive_group()
    initialization.add_argument('--init-model',type=Path)
    initialization.add_argument('--init-reach-model',type=Path,help='Initialize a neural arm branch from the original PPO Reach')
    train.add_argument('--arm-speed-factor',type=float,default=.25)
    train.add_argument('--freeze-arm-steps',type=int,default=0,help='Keep the pretrained arm weights fixed during initial PPO updates')
    train.add_argument('--batch-size',type=int)
    train.add_argument('--jaw-control',choices=['absolute','delta'])
    train.add_argument('--jaw-delta-scale',type=float,help='Metres of jaw reference change per normalized action')
    train.add_argument('--reset-goal-gate',action='store_true',help='Reinitialize the learnable lift gate for a secure grip')
    train.add_argument('--timesteps',type=int,default=32768);train.add_argument('--n-envs',type=int,default=4)
    train.add_argument('--n-steps',type=int,default=256);train.add_argument('--seed',type=int,default=2027)
    train.add_argument('--learning-rate',type=float,default=.0003);train.add_argument('--std',type=float,default=.3)
    train.add_argument('--jaw-std',type=float,help='Override only the initial jaw standard deviation')
    train.add_argument('--exploration-profile',choices=['distance','grasp_lift'],default='grasp_lift',
                       help='Use grasp geometry and real grip feedback to schedule noise; network means are unchanged')
    train.add_argument('--reset-exploration',action='store_true');train.add_argument('--eval-every',type=int,default=4096)
    train.add_argument('--reset-critic',action='store_true',help='Reset only the value network when changing rewards')
    train.add_argument('--reset-jaw',action='store_true',help='Reinitialize only the jaw output at the fixed initial opening')
    train.add_argument('--adaptive-exploration',action=argparse.BooleanOptionalAction,default=True,
                       help='Reduce sampling noise near contact; learned action means are unchanged')
    train.add_argument('--max-steps',type=int)
    train.add_argument('--control-mode',choices=['auto','position','pose'])
    train.add_argument('--observation-source',choices=['state','rgbd_hsv'],
                       help='Use camera-only object estimates; visual checkpoints have a separate observation contract')
    train.add_argument('--cache-startup',action='store_true',help='Opt in to restoring a physically verified observation reset during training')
    evaluate=commands.add_parser('evaluate');evaluate.add_argument('--model',required=True,type=Path)
    evaluate.add_argument('--episodes',type=int,default=5);evaluate.add_argument('--seed-start',type=int,default=360000)
    evaluate.add_argument('--saved-task',action='store_true');evaluate.add_argument('--json',required=True,type=Path)
    render=commands.add_parser('render');render.add_argument('--model',required=True,type=Path)
    render.add_argument('--output',required=True,type=Path);render.add_argument('--seed',type=int,default=360000)
    render.add_argument('--saved-task',action='store_true')
    for command in (evaluate,render):
        command.add_argument('--object-radius',type=float,
                             help='Evaluation disk radius in metres, up to 0.05; preserves full grasp criteria')
    for command in (setup,train,evaluate,render):
        command.add_argument('--observation-pose',choices=['table_down','forward'],
                             help='Fixed task preset, independent of object location')
    args=parser.parse_args(argv)
    if args.command=='train':
        from ..rl.ppo_tabletop_cartesian import train as execute
        result=execute(args);print(result,flush=True);return int(result['status']!='completed')
    if args.command=='evaluate':
        if args.json.exists():parser.error('Choose a new JSON path')
        if args.episodes<1:parser.error('Positive episode count required')
        from ..rl.ppo_tabletop_cartesian import load,evaluate as execute,write_json
        model,config=load(args.model)
        result=execute(model,config,range(args.seed_start,args.seed_start+args.episodes),full=not args.saved_task,
                       observation_pose=args.observation_pose,object_radius=args.object_radius)
        write_json(args.json,result);print(result['summary']);return int(result['summary']['successes']!=args.episodes)
    if args.command in ('setup','render'):
        from ..visualization.tabletop_cartesian import run
        return run(args)
    return 0
