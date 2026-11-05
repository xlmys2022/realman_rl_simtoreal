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
if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
    os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")

import mujoco
import numpy as np

SCENE = Path(__file__).resolve().parents[1] / "assets/robots/rml63/scene.xml"
JOINTS = tuple(f"joint_{i}" for i in range(1, 7))
MAX_ERROR_DEG = 0.5
MAX_TAIL_RANGE_DEG = 0.05
MAX_TAIL_SPEED_DEG_S = 0.1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_model(path=SCENE, *, gravity_compensation=False):
    path = str(Path(path).expanduser().resolve())
    if gravity_compensation:
        # Configure before compilation: merely editing body_gravcomp on an
        # already compiled baseline leaves its compiled gravity metadata disabled.
        spec = mujoco.MjSpec.from_file(path)
        for name in (*[f"link_{i}" for i in range(1, 7)], "camera_rolink", "camera_link"):
            spec.body(name).gravcomp = 1
        for name in JOINTS:
            spec.joint(name).actgravcomp = True
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(path)
    require((model.nq, model.nv, model.nu, model.njnt) == (6, 6, 6, 6), "Expected six arm hinges and six actuators")
    joint_ids = np.array([model.joint(name).id for name in JOINTS])
    actuator_ids = np.array([model.actuator(f"position_{i}").id for i in range(1, 7)])
    for joint_id, actuator_id in zip(joint_ids, actuator_ids):
        require(model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE, "Expected a hinge joint")
        require(model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[actuator_id, 0] == joint_id, "Wrong joint/actuator mapping")
        require(np.array_equal(model.actuator_gear[actuator_id], [1, 0, 0, 0, 0, 0]), "Expected direct-drive gear=1")
        require(model.jnt_limited[joint_id] and model.actuator_ctrllimited[actuator_id]
                and model.actuator_forcelimited[actuator_id], "Position and force limits are required")
        require(model.jnt_actfrclimited[joint_id], "Joint total-drive torque limit is required")
        require(np.allclose(model.jnt_range[joint_id], model.actuator_ctrlrange[actuator_id]), "Control/joint limits differ")
    base = model.body("base_link").id
    require(model.body_parentid[base] == 0 and model.body_jntnum[base] == 0
            and model.body_weldid[base] == 0, "Base must be rigidly fixed to world")
    for name in ("camera_rolink", "camera_link"):
        require(model.body_weldid[model.body(name).id] == model.body_weldid[model.body("link_6").id],
                "Camera must be rigidly attached to link_6")
    require(np.allclose(model.opt.gravity, [0, 0, -9.81]), "Expected Earth gravity")
    require(not int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT), "Contact detection disabled")
    return model, joint_ids, actuator_ids


def initialize(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def validate_target(model, joint_ids, home, target):
    target = np.asarray(target, dtype=float)
    require(target.shape == (6,) and np.isfinite(target).all(), "Target must contain six finite angles")
    require(np.max(np.abs(target - home)) <= np.deg2rad(5) + 1e-12,
            "Demo targets must stay within 5 degrees of home")
    ranges = model.jnt_range[joint_ids]
    require(np.all(target > ranges[:, 0]) and np.all(target < ranges[:, 1]), "Target violates joint limits")
    return target


def run_segment(model, data, joint_ids, actuator_ids, target, label, ramp=1.0, hold=3.0, tick=None, step_observer=None):
    """Smooth the control setpoint and measure real integrated state; never reset qpos."""
    qadr = model.jnt_qposadr[joint_ids]
    vadr = model.jnt_dofadr[joint_ids]
    start = data.ctrl[actuator_ids].copy()
    target = np.asarray(target)
    dt = model.opt.timestep
    records, velocities = [], []
    peak_torque = np.zeros(6)
    peak_speed = np.zeros(6)
    contacts = 0
    min_distance = 0.0
    expected_time = data.time
    for step in range(round((ramp + hold) / dt)):
        alpha = min((step + 1) * dt / ramp, 1.0) if ramp else 1.0
        alpha = alpha * alpha * (3 - 2 * alpha)
        data.ctrl[actuator_ids] = start + alpha * (target - start)
        mujoco.mj_step(model, data)
        if step_observer is not None:
            # Observe forces used by this step before a tick refreshes mjData.
            step_observer(data)
        expected_time += dt
        require(abs(data.time - expected_time) < 1e-8, "Simulation time reset unexpectedly")
        for name in ("qpos", "qvel", "qacc", "qfrc_actuator", "actuator_force", "ctrl", "xpos"):
            require(np.isfinite(getattr(data, name)).all(), f"Non-finite {name}")
        require(not any(warning.number for warning in data.warning), "MuJoCo numerical warning")
        contacts = max(contacts, data.ncon)
        for contact in data.contact:
            min_distance = min(min_distance, float(contact.dist))
            if contact.dist < -0.0001:
                names = [model.geom(int(i)).name for i in contact.geom]
                raise ValueError(f"Unexpected collision {names}: distance={contact.dist:.6g} m")
        q, v = data.qpos[qadr].copy(), data.qvel[vadr].copy()
        require(np.all(q >= model.jnt_range[joint_ids, 0] - 1e-4)
                and np.all(q <= model.jnt_range[joint_ids, 1] + 1e-4), "Joint limit violated")
        records.append(q)
        velocities.append(v)
        peak_torque = np.maximum(peak_torque, np.abs(data.qfrc_actuator[vadr]))
        peak_speed = np.maximum(peak_speed, np.abs(v))
        if tick is not None:
            tick()
    positions = np.rad2deg(records)
    tail_size = max(1, round(0.5 / dt))
    errors = positions[-1] - np.rad2deg(target)
    tail_range = np.ptp(positions[-tail_size:], axis=0)
    tail_speed = np.max(np.abs(np.rad2deg(velocities[-tail_size:])), axis=0)
    movement = target - start
    moving = np.flatnonzero(np.abs(movement) > 1e-10)
    overshoot = np.zeros(6)
    for i in moving:
        overshoot[i] = max(0.0, np.max(np.sign(movement[i]) * (positions[:, i] - np.rad2deg(target[i]))))
    unmoving = np.setdiff1d(np.arange(6), moving)
    other_motion = float(np.max(np.abs(positions[:, unmoving] - positions[0, unmoving]))) if unmoving.size else 0.0
    require(np.max(np.abs(errors)) <= MAX_ERROR_DEG, f"{label}: final error exceeds {MAX_ERROR_DEG} deg")
    require(np.max(tail_range) <= MAX_TAIL_RANGE_DEG, f"{label}: persistent oscillation")
    require(np.max(tail_speed) <= MAX_TAIL_SPEED_DEG_S, f"{label}: not settled")
    require(np.max(overshoot) <= MAX_ERROR_DEG, f"{label}: excessive overshoot")
    require(other_motion <= MAX_ERROR_DEG, f"{label}: uncommanded joint moved excessively")
    require(np.all(peak_torque <= np.max(np.abs(model.jnt_actfrcrange[joint_ids]), axis=1) + 1e-8),
            f"{label}: torque bound exceeded")
    for i in moving:
        require(np.sign(movement[i]) * (positions[-1, i] - positions[0, i]) > 0.5 * abs(np.rad2deg(movement[i])),
                f"{label}: commanded joint did not move toward target")
    return dict(label=label, target_deg=np.rad2deg(target).tolist(), actual_deg=positions[-1].tolist(),
                error_deg=errors.tolist(), tail_range_deg=tail_range.tolist(), tail_speed_deg_s=tail_speed.tolist(),
                overshoot_deg=overshoot.tolist(), peak_torque_nm=peak_torque.tolist(),
                peak_speed_deg_s=np.rad2deg(peak_speed).tolist(), max_contacts=contacts,
                min_contact_distance_m=min_distance, max_uncommanded_motion_deg=other_motion)


def print_result(result):
    print(f"\n[PASS] {result['label']}", flush=True)
    for i, (target, actual, error) in enumerate(zip(result["target_deg"], result["actual_deg"], result["error_deg"]), 1):
        print(f"  J{i}: target={target:9.4f} deg actual={actual:9.4f} deg error={error:+.6f} deg", flush=True)
    print(f"  max final error={max(map(abs, result['error_deg'])):.6f} deg; "
          f"overshoot={max(result['overshoot_deg']):.6f} deg; tail range={max(result['tail_range_deg']):.3e} deg; "
          f"tail speed={max(result['tail_speed_deg_s']):.3e} deg/s; contacts={result['max_contacts']}", flush=True)
    print(f"  max uncommanded joint travel={result['max_uncommanded_motion_deg']:.6f} deg", flush=True)
    print("  peak torque [N*m]: " + np.array2string(np.array(result["peak_torque_nm"]), precision=5), flush=True)


def run_demo(model, data, joint_ids, actuator_ids, target=None, tick=None, step_observer=None):
    home = data.ctrl[actuator_ids].copy()
    targets = [("hold home under gravity", home, 0.0, 3.0)]
    if target is None:
        for i in range(6):
            command = home.copy()
            command[i] += np.deg2rad(3)
            targets.extend([(f"J{i+1} +3 deg", command, 1.0, 3.0), (f"J{i+1} return", home, 1.0, 1.0)])
    else:
        targets.append(("custom six-joint target", validate_target(model, joint_ids, home, target), 1.0, 3.0))
    results = []
    for label, command, ramp, hold in targets:
        result = run_segment(model, data, joint_ids, actuator_ids, command, label, ramp, hold, tick, step_observer)
        print_result(result)
        results.append(result)
    return results


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
            # Only the trusted local scripts directory is added under python -I.
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from view_robot import passive_viewer
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
