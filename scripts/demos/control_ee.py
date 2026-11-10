#!/usr/bin/env python3
"""Small world-frame end-effector translations, with home orientation constrained."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time

from realman_rl_simtoreal.viewer import configure_backend, passive_viewer
configure_backend()
from realman_rl_simtoreal.simulation import model as model_api, control
from realman_rl_simtoreal.simulation.kinematics import Kinematics, rotation_vector
from realman_rl_simtoreal.simulation.ee_control import check_motion, execute_motion
from realman_rl_simtoreal.utils.demo_helpers import joint_result_lines
import mujoco
import numpy as np


def print_result(result):
    for line in joint_result_lines(result):
        print(line, flush=True)


def result_dict(result):
    values = asdict(result)
    values["q"] = None if result.q is None else result.q.tolist()
    return values


def print_report(report):
    for key in ("initial_position_m", "motion_start_position_m", "target_position_m",
                "theoretical_position_m", "actual_position_m"):
        print(f"{key}: {np.array(report[key])}")
    print(f"Actual terminal position error: {1000 * report['actual_position_error_m']:.6f} mm")
    print(f"Actual terminal orientation error: {np.rad2deg(report['actual_orientation_error_rad']):.6f} deg")
    print(f"Maximum actual orientation error during motion/hold: {np.rad2deg(report['max_actual_orientation_error_rad']):.6f} deg")
    print(f"Nominal joint-interpolation orientation drift: {np.rad2deg(report['max_command_interpolation_rotation_error_rad']):.6f} deg")
    print("Actual site-to-world rotation:\n", np.array(report["actual_rotation"]))
    print_result(report["hold"])
    print_result(report["motion"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=model_api.SCENE)
    goal = parser.add_mutually_exclusive_group()
    goal.add_argument("--delta-mm", nargs=3, type=float, metavar=("DX", "DY", "DZ"), help="world translation from home; default +X 5 mm")
    goal.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"), help="absolute world position in metres")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, help="Viewer wall-time cap; early close does not pass the test")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.duration is not None and (not args.viewer or not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration requires --viewer and a finite positive value")
    output = dict(completed=False, motion_executed=False)
    data = None
    try:
        model, jids, aids = model_api.load_model(args.model)
        data = model_api.initialize(model)
        kin = Kinematics(model)
        target = (np.asarray(args.target) if args.target is not None else
                  kin.home_position + np.asarray(args.delta_mm if args.delta_mm is not None else [5, 0, 0]) / 1000)
        print("Temporary ee_site: link_6 local origin, not a calibrated TCP.")
        print("Home world position [m]:", kin.home_position)
        print("Fixed target site-to-world rotation:\n", kin.home_rotation)
        result = kin.solve_ik(target, data.qpos[kin.qadr])
        output["ik"] = result_dict(result)
        print(f"IK: success={result.success}, iterations={result.iterations}, position_error_m={result.position_error}, "
              f"orientation_error_rad={result.orientation_error}, reason={result.reason}", flush=True)
        check_motion(model, data, jids, aids, kin, result, target)
        print("IK joint target [deg]:", np.rad2deg(result.q), flush=True)
        context = nullcontext(None)
        if args.viewer:
            control.require(bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")), "No desktop display")
            import glfw
            control.require(bool(glfw.init()), "GLFW could not initialize display")
            context = passive_viewer(model, data)
        with context as viewer:
            start_wall, start_sim, last_sync = time.monotonic(), data.time, 0.0
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = [0, 0, 0.55]
                    viewer.cam.distance = 1.8
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -20
                    viewer.opt.geomgroup[0] = 0
                    viewer.opt.sitegroup[3] = 1
                    viewer.user_scn.ngeom = 1
                    mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                                       np.array([0.012, 0, 0]), target, np.eye(3).reshape(9),
                                       np.array([1.0, 0.1, 0.1, 0.35], dtype=np.float32))

            def tick():
                nonlocal last_sync
                now = time.monotonic()
                if not viewer.is_running() or (args.duration is not None and now - start_wall >= args.duration):
                    raise InterruptedError("Viewer closed before full validation")
                if now - last_sync >= 1 / 60:
                    viewer.sync()
                    last_sync = now
                delay = start_wall + data.time - start_sim - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

            report = execute_motion(model, data, jids, aids, kin, result, target, tick if viewer is not None else None)
        output.update(completed=True, motion_executed=True, report=report)
        print_report(report)
        print(f"[PASS] End-effector dynamics validation; simulation time={data.time:.3f} s")
        return 0
    except (KeyboardInterrupt, InterruptedError):
        output["reason"] = "Stopped early; complete validation not performed"
        print("[INFO] Viewer stopped normally; complete validation not performed")
        return 0
    except (OSError, ValueError, KeyError, mujoco.FatalError) as exc:
        output["reason"] = str(exc)
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    finally:
        if data is not None:
            output["motion_executed"] = bool(data.time > 0)
            output["simulation_time_s"] = float(data.time)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
