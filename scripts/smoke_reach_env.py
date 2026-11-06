#!/usr/bin/env python3
"""Headless fixed-action smoke test; no policy, training framework or SDK."""

import argparse
import json
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env

from realman_rl_simtoreal.envs.reach_env import ReachEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action", type=float, nargs=3, metavar=("DX", "DY", "DZ"),
                        help="constant normalized world XYZ action; default: five +X actions, then hold")
    parser.add_argument("--action-scale-mm", type=float, default=1.0)
    parser.add_argument("--check", action="store_true", help="also run the official Gymnasium checker")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    rows = []
    try:
        with ReachEnv(action_scale=args.action_scale_mm/1000, max_episode_steps=args.steps) as env:
            if args.check:
                check_env(env)
                print("[PASS] Gymnasium check_env")
            obs, info = env.reset(seed=args.seed)
            print(f"reset: ee_world_m={obs['achieved_goal']} goal_world_m={obs['desired_goal']}")
            print(f"control_period={env.control_period}s, physics_steps={env.physics_steps}, reward unit=m")
            initial = obs
            for index in range(args.steps):
                action = np.array(args.action if args.action is not None else
                                  ([1, 0, 0] if index < 5 else [0, 0, 0]), dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                rows.append(dict(action=action, observation=obs, reward=reward, terminated=terminated,
                                 truncated=truncated, info=info))
                print(f"step={index+1:3d} action={action} reward={reward:.9f} "
                      f"ee_world_m={obs['achieved_goal']} distance_mm={info['distance_m']*1000:.6f} "
                      f"reason={info['termination_reason']}")
                if terminated or truncated:
                    break
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(dict(initial_observation=initial, steps=rows),
                                                default=lambda value: value.tolist(), indent=2)+"\n")
            return int(terminated and not info["is_success"])
    except (ValueError, RuntimeError) as exc:
        print(f"[FAIL] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
