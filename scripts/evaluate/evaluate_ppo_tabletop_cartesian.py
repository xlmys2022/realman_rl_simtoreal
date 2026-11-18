#!/usr/bin/env python3
"""evaluate the Cartesian tabletop task in the existing realman_rl project."""
import sys
from realman_rl_simtoreal.cli.tabletop_cartesian import main

if __name__=='__main__':
    raise SystemExit(main(['evaluate',*sys.argv[1:]]))
