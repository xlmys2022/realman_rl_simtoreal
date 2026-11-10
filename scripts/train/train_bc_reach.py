#!/usr/bin/env python3
"""Train a supervised BC network, never SAC or an environment rollout."""
import argparse
from pathlib import Path
from realman_rl_simtoreal.imitation.bc import train_bc


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    try:
        result=train_bc(a.data,a.out,epochs=a.epochs,batch_size=a.batch_size,
                        learning_rate=a.learning_rate,seed=a.seed)
        print(result,flush=True)
        return 0
    except (ValueError,RuntimeError) as exc:
        p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
