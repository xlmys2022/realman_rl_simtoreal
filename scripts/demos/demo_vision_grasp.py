#!/usr/bin/env python3
"""Run the RGB-D tabletop PPO in an interactive MuJoCo window."""

import argparse
import math
from pathlib import Path

from realman_rl_simtoreal.apps.visualization.vision_grasp import DEFAULT_MODEL, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Visual PPO checkpoint")
    parser.add_argument("--seed", type=int, default=997000, help="Reset seed (default 997000)")
    parser.add_argument("--object-radius", type=float, default=0.0,
                        help="Cube position randomization radius in metres, up to 0.05")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Simulation seconds per wall-clock second when rendering can keep up (default 2)")
    parser.add_argument("--exit-on-finish", action="store_true",
                        help="Close after one episode instead of keeping the final scene open")
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if not 0 <= args.object_radius <= 0.05 or args.seed < 0:
        parser.error("--object-radius must be in [0, 0.05] and --seed must be nonnegative")

    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
