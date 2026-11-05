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

# Trust only this project's script directory under python -I. control_joints
# selects the GLFW backend before MuJoCo imports its renderer.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_joints as control
from kinematics import Kinematics, rotation_vector
import mujoco
import numpy as np

MAX_TRANSLATION = 0.010  # metres from nominal home; a demo envelope, not a workspace claim.
MAX_ACTUAL_POSITION_ERROR = 0.005  # metres; includes the existing PD gravity error.
MAX_ACTUAL_ORIENTATION_ERROR = np.deg2rad(1.0)


def check_motion(model, data, joint_ids, actuator_ids, kin, result, target):
    """Reject failed IK and unsafe demo goals before any integration or ctrl write."""
    control.require(result.success and result.q is not None, f"IK failed; no motion: {result.reason}")
    target = np.asarray(target, dtype=float)
    control.require(target.shape == (3,) and np.isfinite(target).all(), "Expected a finite XYZ target")
    control.require(np.linalg.norm(target - kin.home_position) <= MAX_TRANSLATION + 1e-12,
                    "Demo translation exceeds 10 mm from home")
    q = control.validate_target(model, joint_ids, kin.home_q, result.q)
    theoretical_position, theoretical_rotation = kin.fk(q)
    control.require(np.linalg.norm(theoretical_position - target) <= 1e-5
                    and np.linalg.norm(rotation_vector(kin.home_rotation, theoretical_rotation)) <= 1e-4,
                    "IK result does not satisfy the requested fixed-orientation pose")
    # Discrete collision screening in calculation-only data. This is not a
    # continuous collision guarantee and does not account for tracking error.
    probe = mujoco.MjData(model)
    start = data.qpos[kin.qadr].copy()
    max_rotation_drift = 0.0
    for alpha in np.linspace(0, 1, 51):
        angles = start + alpha * (q - start)
        probe.qpos[kin.qadr] = angles
        mujoco.mj_forward(model, probe)
        for contact in probe.contact:
            control.require(contact.dist >= -0.0001, "Joint interpolation has an unexpected collision")
        _, rotation = kin.fk(angles)
        max_rotation_drift = max(max_rotation_drift, float(np.linalg.norm(rotation_vector(kin.home_rotation, rotation))))
    return theoretical_position, theoretical_rotation, max_rotation_drift


def execute_motion(model, data, joint_ids, actuator_ids, kin, result, target, tick=None, step_observer=None):
    """Drive existing position actuators; live qpos is never assigned here."""
    theory_p, theory_r, command_rotation_drift = check_motion(model, data, joint_ids, actuator_ids, kin, result, target)
    mujoco.mj_forward(model, data)
    initial_position = data.site_xpos[kin.site_id].copy()
    max_orientation_error, min_distance = 0.0, 0.0

    def sample():
        nonlocal max_orientation_error, min_distance
        # mj_step integrates qpos; refresh site poses/contacts at that new state.
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[kin.site_id].reshape(3, 3)
        angle = float(np.linalg.norm(rotation_vector(kin.home_rotation, rotation)))
        max_orientation_error = max(max_orientation_error, angle)
        control.require(angle <= MAX_ACTUAL_ORIENTATION_ERROR, "Actual orientation exceeded the 1 degree demo envelope")
        for contact in data.contact:
            min_distance = min(min_distance, float(contact.dist))
            control.require(contact.dist >= -0.0001, "Unexpected collision after integration")
        if tick is not None:
            tick()

    hold = control.run_segment(model, data, joint_ids, actuator_ids, kin.home_q, "settle home", ramp=0, hold=3, tick=sample, step_observer=step_observer)
    motion_start = data.site_xpos[kin.site_id].copy()
    motion = control.run_segment(model, data, joint_ids, actuator_ids, result.q, "end-effector target", ramp=2, hold=3, tick=sample, step_observer=step_observer)
    actual_position = data.site_xpos[kin.site_id].copy()
    actual_rotation = data.site_xmat[kin.site_id].reshape(3, 3).copy()
    position_error = float(np.linalg.norm(np.asarray(target) - actual_position))
    orientation_error = float(np.linalg.norm(rotation_vector(kin.home_rotation, actual_rotation)))
    control.require(position_error <= MAX_ACTUAL_POSITION_ERROR, "Actual end-effector error exceeds 5 mm")
    return dict(initial_position_m=initial_position.tolist(), motion_start_position_m=motion_start.tolist(),
                target_position_m=np.asarray(target).tolist(), target_rotation=kin.home_rotation.tolist(),
                theoretical_position_m=theory_p.tolist(), theoretical_rotation=theory_r.tolist(),
                actual_position_m=actual_position.tolist(), actual_rotation=actual_rotation.tolist(),
                actual_position_error_m=position_error, actual_orientation_error_rad=orientation_error,
                max_actual_orientation_error_rad=max_orientation_error,
                max_command_interpolation_rotation_error_rad=command_rotation_drift,
                min_contact_distance_m=min_distance, joint_target_rad=result.q.tolist(),
                joint_actual_rad=data.qpos[kin.qadr].tolist(), simulation_time_s=float(data.time),
                hold=hold, motion=motion)


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
    control.print_result(report["hold"])
    control.print_result(report["motion"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=control.SCENE)
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
        model, jids, aids = control.load_model(args.model)
        data = control.initialize(model)
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
            from view_robot import passive_viewer
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
