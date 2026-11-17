#!/usr/bin/env python3
"""train the Cartesian tabletop task in the existing realman_rl project."""
import sys
from realman_rl_simtoreal.utils.tabletop_cartesian_cli import main

if __name__=='__main__':
    raise SystemExit(main(['train',*sys.argv[1:]]))
