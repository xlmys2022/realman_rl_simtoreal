#!/usr/bin/env python3
"""Paired, headless random/rule policy evaluation without training or joint access."""
import argparse
import json
from pathlib import Path
import platform
import sys

import gymnasium
import mujoco
import numpy as np

from realman_rl_simtoreal.paths import OUTPUTS
from realman_rl_simtoreal.evaluation.baselines import (
    rule_action, evaluate_policy, validate_transition, reward_comparison, profile_components)
from realman_rl_simtoreal.evaluation.metrics import latency_summary, goal_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--action-seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--target-radius-mm', type=float, default=5)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--json', type=Path, default=OUTPUTS/'reach/baselines/reach_baselines.json')
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1 or args.seed_start < 0 or args.action_seed < 0:
        parser.error('Episodes/max-steps must be positive and seeds nonnegative')
    seeds = list(range(args.seed_start, args.seed_start+args.episodes))
    try:
        results = {}
        for name in ('random', 'rule'):
            results[name] = evaluate_policy(name, seeds, action_seed=args.action_seed,
                                            max_episode_steps=args.max_steps, target_radius=args.target_radius_mm/1000)
            print(name, json.dumps(results[name]['summary']), flush=True)
        paired = all(a['target_position_m'] == b['target_position_m']
                     for a, b in zip(results['random']['episodes'], results['rule']['episodes']))
        if not paired:
            raise ValueError('Policies did not receive identical goal seeds/positions')
        report = dict(configuration=dict(seeds=seeds, action_seed=args.action_seed, max_episode_steps=args.max_steps,
                                         target_radius_m=args.target_radius_mm/1000, target_mode='random', reward_mode='safe_dense'),
                      versions=dict(python=platform.python_version(), python_executable=sys.executable,
                                    gymnasium=gymnasium.__version__, mujoco=mujoco.__version__, numpy=np.__version__),
                      policies=dict(random='Independent uniform Box(-1,1) actions',
                                    rule='Observation feedback with pending-command correction; normalized norm capped at .25'),
                      paired_goals_identical=paired, results=results, reward_comparison=reward_comparison(args.max_steps))
        args.json.parent.mkdir(parents=True, exist_ok=True)
        if args.profile:
            report['profile'], text = profile_components(seeds[:10], max_episode_steps=args.max_steps,
                                                        target_radius=args.target_radius_mm/1000)
            args.json.with_suffix('.profile.txt').write_text(text)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(f'[PASS] Paired evaluation and audits completed: {args.json}', flush=True)
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f'[FAIL] {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
