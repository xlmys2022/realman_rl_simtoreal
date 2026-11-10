#!/usr/bin/env python3
"""Collect public-observation rule demonstrations, split by whole episode."""
import argparse
from pathlib import Path
from realman_rl_simtoreal.imitation.bc import save_demonstrations


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--train-episodes',type=int,default=240)
    p.add_argument('--validation-episodes',type=int,default=60)
    p.add_argument('--seed-start',type=int,default=30000)
    a=p.parse_args()
    if min(a.train_episodes,a.validation_episodes) < 1 or a.seed_start < 0:
        p.error('Positive episode counts and nonnegative seed required')
    try:
        split=a.seed_start+a.train_episodes
        result=save_demonstrations(a.out,list(range(a.seed_start,split)),
                                   list(range(split,split+a.validation_episodes)))
        print(result['distribution'],flush=True)
        return 0
    except (ValueError,RuntimeError) as exc:
        p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
