#!/usr/bin/env python3
"""Independent held-out deterministic SAC/random/rule comparison in SI ReachEnv."""
import argparse
import json
from pathlib import Path

import torch
from stable_baselines3 import SAC

from realman_rl_simtoreal.rl.common.evaluation import evaluate
from realman_rl_simtoreal.rl.common.observation import checkpoint_transform, require_legacy, validate_model_observation
from realman_rl_simtoreal.rl.common.artifacts import write_json, versions


def run(model_path, *, episodes=100, seed_start=100, policies=('random', 'rule', 'sac')):
    model_path = Path(model_path).resolve()
    if episodes < 1 or seed_start < 0:
        raise ValueError('Positive episode count and nonnegative seed required')
    config = json.loads((model_path.parent/'config.json').read_text())
    require_legacy(config['environment'], 'SAC evaluation')
    transform = checkpoint_transform(config)
    torch.set_num_threads(config['torch_num_threads'])
    torch.use_deterministic_algorithms(True)
    model = SAC.load(model_path, device='cpu') if 'sac' in policies else None
    if model is not None:validate_model_observation(model, transform)
    seeds = list(range(seed_start, seed_start+episodes))
    results = {}
    for policy in policies:
        results[policy] = evaluate(policy, seeds, model=model, transform=transform,
                                   config=config['environment'], gamma=config['sac']['gamma'])
        print(policy, json.dumps(results[policy]['summary']), flush=True)
    first = next(iter(results.values()))['episodes']
    paired = all([e['target_position_m'] for e in value['episodes']]
                 == [e['target_position_m'] for e in first] for value in results.values())
    if not paired:
        raise ValueError('Policies did not receive identical target positions')
    return dict(model=str(model_path), seeds=seeds, environment=config['environment'],
                observation_transform=config['observation_transform'], paired_goals_identical=paired,
                versions=versions(), results=results,
                note='Evaluation resets its own environment; no exploration, optimizer or replay insertion.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed-start', type=int, default=100)
    parser.add_argument('--policies', nargs='+', choices=['random', 'rule', 'sac'], default=['random', 'rule', 'sac'])
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    try:
        result = run(args.model, episodes=args.episodes, seed_start=args.seed_start, policies=args.policies)
        path = args.json or args.model.parent/'evaluation.json'
        write_json(path, result)
        print(f'Saved independent evaluation: {path}', flush=True)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'[FAIL] {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
