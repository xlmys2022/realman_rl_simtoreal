#!/usr/bin/env python3
"""Run small joint-position commands under gravity, headlessly or in MuJoCo Viewer."""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time

# MuJoCo's renderer import can load pyGLFW, so select the process-local backend
# before importing MuJoCo (the headless path still creates no graphics context).

from realman_rl_simtoreal.viewer import configure_backend, passive_viewer
configure_backend()
from realman_rl_simtoreal.simulation.model import SCENE, load_model, initialize
from realman_rl_simtoreal.simulation.control import require, validate_target
from realman_rl_simtoreal.utils.demo_helpers import joint_result_lines, run_joint_demo

import mujoco
import numpy as np


def print_result(result):
    for line in joint_result_lines(result):
        print(line, flush=True)


def run_demo(model, data, joint_ids, actuator_ids, target=None, tick=None, step_observer=None):
    return run_joint_demo(model, data, joint_ids, actuator_ids, target, tick, step_observer,
                          reporter=print_result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=SCENE)
    parser.add_argument("--viewer", action="store_true", help="show the same tests in real time")
    parser.add_argument("--duration", type=float, help="optional Viewer wall-time limit; early exit is not a full validation")
    parser.add_argument("--target-deg", nargs=6, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="absolute model joint angles, each within 5 deg of home")
    parser.add_argument("--json", type=Path, help="save completed numerical results")
    args = parser.parse_args()
    if args.duration is not None and (not args.viewer or not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration requires --viewer and a finite positive number")
    try:
        model, joint_ids, actuator_ids = load_model(args.model)
        data = initialize(model)
        target = None if args.target_deg is None else validate_target(model, joint_ids, data.ctrl[actuator_ids], np.deg2rad(args.target_deg))
        print(f"MuJoCo {mujoco.__version__}: nq={model.nq} nv={model.nv} nu={model.nu}; gravity={model.opt.gravity.tolist()}; dt={model.opt.timestep:g}")
        print("Position PD control, no gravity compensation; model gains are not real-arm parameters.", flush=True)
        for jid, aid in zip(joint_ids, actuator_ids):
            print(f"  {model.joint(int(jid)).name} <- {model.actuator(int(aid)).name}: "
                  f"kp={model.actuator_gainprm[aid, 0]:g}, kv={-model.actuator_biasprm[aid, 2]:g}, "
                  f"torque range={model.actuator_forcerange[aid].tolist()} N*m", flush=True)
        context = nullcontext(None)
        if args.viewer:
            require(bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")), "No desktop display")
            import glfw
            require(bool(glfw.init()), "GLFW could not initialize desktop display")
            context = passive_viewer(model, data)
        with context as viewer:
            start_wall = time.monotonic()
            start_sim = data.time
            last_sync = start_wall
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = [0, 0, 0.5]
                    viewer.cam.distance = 1.8
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -20
                    viewer.opt.geomgroup[0] = 0
            def tick():
                nonlocal last_sync
                now = time.monotonic()
                if not viewer.is_running() or (args.duration is not None and now - start_wall >= args.duration):
                    raise InterruptedError("Viewer closed or duration reached before the complete test sequence")
                if now - last_sync >= 1 / 60:
                    viewer.sync()
                    last_sync = now
                delay = start_wall + data.time - start_sim - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            results = run_demo(model, data, joint_ids, actuator_ids, target, tick if viewer is not None else None)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(dict(mujoco=mujoco.__version__, timestep=model.opt.timestep,
                                                gravity=model.opt.gravity.tolist(), results=results), indent=2) + "\n")
        print(f"\n[PASS] All {len(results)} segments completed; simulation time={data.time:.3f} s")
        return 0
    except (KeyboardInterrupt, InterruptedError):
        print("\n[INFO] Viewer stopped normally; unfinished segments are not counted as passed")
        return 0
    except (OSError, ValueError, KeyError, mujoco.FatalError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
