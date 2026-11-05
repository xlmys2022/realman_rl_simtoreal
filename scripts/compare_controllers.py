#!/usr/bin/env python3
"""Matched PD versus joint-torque-limited native gravity compensation experiments."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_joints as control
import control_ee as ee
from kinematics import Kinematics, rotation_vector
import mujoco
import numpy as np


def gravity_torque(model, q):
    """g(q): zero-velocity bias in private data; live moving bias is not pure gravity."""
    q = np.asarray(q, dtype=float)
    control.require(q.shape == (model.nq,) and np.isfinite(q).all(), "Expected finite model coordinates")
    probe = mujoco.MjData(model)
    probe.qpos[:] = q
    probe.qvel[:] = 0
    mujoco.mj_forward(model, probe)
    return probe.qfrc_bias.copy()


class ForceMonitor:
    """Check the step's total drive, including compensation, before cache refresh."""

    def __init__(self, model, joint_ids, actuator_ids):
        self.model, self.jids, self.aids = model, joint_ids, actuator_ids
        self.vadr = model.jnt_dofadr[joint_ids]
        control.require(np.all(model.jnt_actfrclimited[joint_ids]), "Joint total-drive limits must be active")
        if np.any(model.body_gravcomp):
            control.require(model.ngravcomp > 0 and np.all(model.jnt_actgravcomp[joint_ids]),
                            "Gravity compensation must be compiled and routed through joint actuator limits")
        self.limits = model.jnt_actfrcrange[joint_ids].copy()
        self.steps, self.saturated_steps = 0, 0
        self.peak_total = np.zeros(6)
        self.peak_servo = np.zeros(6)
        self.peak_gravity = np.zeros(6)
        self.peak_speed = np.zeros(6)

    def observe(self, data):
        total = data.qfrc_actuator[self.vadr]
        servo = data.actuator_force[self.aids]  # This model has one gear=1 servo per hinge.
        gc = data.qfrc_gravcomp[self.vadr]
        routed_gc = np.where(self.model.jnt_actgravcomp[self.jids], gc, 0)
        requested = servo + routed_gc
        expected = np.clip(requested, self.limits[:, 0], self.limits[:, 1])
        control.require(np.isfinite(np.concatenate([total, servo, gc, data.qfrc_passive])).all(), "Non-finite force channel")
        control.require(np.all(total >= self.limits[:, 0] - 1e-9)
                        and np.all(total <= self.limits[:, 1] + 1e-9), "Total joint drive exceeds torque limits")
        control.require(np.allclose(total, expected, rtol=0, atol=1e-9), "Unexpected drive composition or gravity routing")
        # Physical damping remains passive; gravity compensation must not bypass
        # motor limits as passive force or externally applied force.
        passive_mechanical = data.qfrc_damper + data.qfrc_spring + data.qfrc_fluid
        control.require(np.allclose(data.qfrc_passive, passive_mechanical, rtol=0, atol=1e-9),
                        "Unexpected passive force (possible gravity compensation bypass)")
        control.require(not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied), "External force injection is not allowed")
        self.steps += 1
        self.saturated_steps += int(np.max(np.abs(requested - total)) > 1e-9)
        self.peak_total = np.maximum(self.peak_total, np.abs(total))
        self.peak_servo = np.maximum(self.peak_servo, np.abs(servo))
        self.peak_gravity = np.maximum(self.peak_gravity, np.abs(gc))
        self.peak_speed = np.maximum(self.peak_speed, np.abs(data.qvel[self.vadr]))

    def summary(self):
        return dict(steps=self.steps, saturated_steps=self.saturated_steps,
                    joint_total_limits_nm=self.limits.tolist(), peak_total_drive_nm=self.peak_total.tolist(),
                    peak_servo_generated_nm=self.peak_servo.tolist(), peak_raw_gravity_compensation_nm=self.peak_gravity.tolist(),
                    peak_speed_deg_s=np.rad2deg(self.peak_speed).tolist())


def diagnose(model, data, kin, joint_ids, actuator_ids):
    """Coherent snapshot; pure g(q) is computed independently with qvel=0."""
    mujoco.mj_forward(model, data)
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    g = gravity_torque(model, data.qpos)[vadr]
    bias = data.qfrc_bias[vadr].copy()
    servo = data.actuator_force[actuator_ids].copy()
    gc = data.qfrc_gravcomp[vadr].copy()
    total = data.qfrc_actuator[vadr].copy()
    kp = model.actuator_gainprm[actuator_ids, 0]
    kv = -model.actuator_biasprm[actuator_ids, 2]
    return dict(target_joint_deg=np.rad2deg(data.ctrl[actuator_ids]).tolist(),
                actual_joint_deg=np.rad2deg(data.qpos[qadr]).tolist(),
                joint_error_deg=np.rad2deg(data.qpos[qadr] - data.ctrl[actuator_ids]).tolist(),
                velocity_rad_s=data.qvel[vadr].tolist(), acceleration_rad_s2=data.qacc[vadr].tolist(),
                pure_gravity_nm=g.tolist(), moving_bias_nm=bias.tolist(), velocity_bias_nm=(bias - g).tolist(),
                pd_generated_nm=servo.tolist(), raw_gravity_compensation_nm=gc.tolist(),
                total_drive_nm=total.tolist(), passive_nm=data.qfrc_passive[vadr].tolist(),
                force_balance_residual_nm=(total + data.qfrc_passive[vadr] - bias).tolist(),
                pd_formula_nm=(kp * (data.ctrl[actuator_ids] - data.qpos[qadr]) - kv * data.qvel[vadr]).tolist())


def pose_metrics(kin, target, rotation, settled_home, actual, actual_rotation):
    """Uniform vector definitions; absolute_error = home_offset + relative_error."""
    rotation, actual_rotation = np.asarray(rotation), np.asarray(actual_rotation)
    nominal = kin.home_position
    home_offset = np.asarray(settled_home) - nominal
    expected = np.asarray(target) - nominal
    displacement = np.asarray(actual) - settled_home
    relative_error = displacement - expected
    absolute_error = np.asarray(actual) - target
    return dict(nominal_home_m=nominal.tolist(), settled_home_m=np.asarray(settled_home).tolist(),
                target_position_m=np.asarray(target).tolist(), actual_position_m=np.asarray(actual).tolist(),
                target_rotation=np.asarray(rotation).tolist(), actual_rotation=np.asarray(actual_rotation).tolist(),
                home_offset_mm=(home_offset * 1000).tolist(), home_offset_norm_mm=float(1000 * np.linalg.norm(home_offset)),
                expected_displacement_mm=(expected * 1000).tolist(), actual_displacement_mm=(displacement * 1000).tolist(),
                relative_error_mm=(relative_error * 1000).tolist(), relative_error_norm_mm=float(1000 * np.linalg.norm(relative_error)),
                absolute_error_mm=(absolute_error * 1000).tolist(), absolute_error_norm_mm=float(1000 * np.linalg.norm(absolute_error)),
                orientation_error_deg=float(np.rad2deg(np.linalg.norm(rotation_vector(rotation, actual_rotation)))))


def run_home(model, jids, aids):
    kin, data = Kinematics(model), control.initialize(model)
    monitor = ForceMonitor(model, jids, aids)
    stats = control.run_segment(model, data, jids, aids, kin.home_q, "home hold", ramp=0, hold=3,
                                tick=lambda: mujoco.mj_forward(model, data), step_observer=monitor.observe)
    snapshot = diagnose(model, data, kin, jids, aids)
    actual, rotation = data.site_xpos[kin.site_id].copy(), data.site_xmat[kin.site_id].reshape(3, 3).copy()
    metrics = pose_metrics(kin, kin.home_position, kin.home_rotation, actual, actual, rotation)
    jp, jr = kin.jacobian(kin.home_q)
    dq = data.qpos[kin.qadr] - kin.home_q
    return dict(metrics=metrics, diagnostics=snapshot, joint_stats=stats, forces=monitor.summary(),
                nominal_gravity_nm=gravity_torque(model, model.key_qpos[model.key("home").id]).tolist(),
                linearized_home_position_offset_mm=(1000 * jp @ dq).tolist(),
                linearized_home_rotation_vector_deg=np.rad2deg(jr @ dq).tolist(), simulation_time_s=float(data.time))


def run_ee_case(model, jids, aids, target, solution):
    kin, data = Kinematics(model), control.initialize(model)
    monitor = ForceMonitor(model, jids, aids)
    home_diagnostics = None
    def sample():
        nonlocal home_diagnostics
        if home_diagnostics is None and data.time >= 3 - 1e-10:
            home_diagnostics = diagnose(model, data, kin, jids, aids)
    report = ee.execute_motion(model, data, jids, aids, kin, solution, target, tick=sample, step_observer=monitor.observe)
    metrics = pose_metrics(kin, target, kin.home_rotation, report["motion_start_position_m"],
                           report["actual_position_m"], report["actual_rotation"])
    return dict(metrics=metrics, home_diagnostics=home_diagnostics,
                final_diagnostics=diagnose(model, data, kin, jids, aids),
                forces=monitor.summary(), dynamics=report,
                ik=dict(iterations=solution.iterations, position_error_m=solution.position_error,
                        orientation_error_rad=solution.orientation_error, target_joint_rad=solution.q.tolist()))


def run_joint_sequence(model, jids, aids):
    kin, data = Kinematics(model), control.initialize(model)
    monitor = ForceMonitor(model, jids, aids)
    ends = [3] + [t for i in range(6) for t in (7 + 6 * i, 9 + 6 * i)]
    snapshots, settled_home = [], None
    def sample():
        nonlocal settled_home
        mujoco.mj_forward(model, data)
        if len(snapshots) < len(ends) and abs(data.time - ends[len(snapshots)]) < 1e-8:
            actual, rotation = data.site_xpos[kin.site_id].copy(), data.site_xmat[kin.site_id].reshape(3, 3).copy()
            if settled_home is None:
                settled_home = actual.copy()
            target, target_rotation = kin.fk(data.ctrl[aids])
            snapshots.append(dict(metrics=pose_metrics(kin, target, target_rotation, settled_home, actual, rotation),
                                  diagnostics=diagnose(model, data, kin, jids, aids)))
    log = io.StringIO()
    with redirect_stdout(log):
        stats = control.run_demo(model, data, jids, aids, tick=sample, step_observer=monitor.observe)
    control.require(len(snapshots) == len(stats) == 13, "Joint sequence snapshots are incomplete")
    for snapshot, values in zip(snapshots, stats):
        snapshot["label"], snapshot["joint_stats"] = values["label"], values
    return dict(phases=snapshots, forces=monitor.summary(), simulation_time_s=float(data.time), log=log.getvalue())


def run_comparison(controllers=("pd", "gravity")):
    reference, _, _ = control.load_model()
    kin = Kinematics(reference)
    goals = {name: kin.home_position + delta for name, delta in
             (("x_plus_5mm", [.005, 0, 0]), ("x_minus_5mm", [-.005, 0, 0]), ("z_plus_5mm", [0, 0, .005]))}
    # One shared IK target per scenario: the controller is the only A/B change.
    solutions = {name: kin.solve_ik(target, kin.home_q) for name, target in goals.items()}
    for name, solution in solutions.items():
        control.require(solution.success, f"Comparison target IK failed: {name}: {solution.reason}")
    output = dict(mujoco=mujoco.__version__, numpy=np.__version__,
                  conditions=dict(timestep_s=float(reference.opt.timestep), gravity_m_s2=reference.opt.gravity.tolist(),
                                  home_joint_deg=np.rad2deg(kin.home_q).tolist(), kp=reference.actuator_gainprm[:, 0].tolist(),
                                  kv=(-reference.actuator_biasprm[:, 2]).tolist(), ee_hold_s=3, ee_ramp_s=2, ee_final_hold_s=3,
                                  joint_total_limits_nm=reference.jnt_actfrcrange.tolist()), controllers={})
    for mode in controllers:
        control.require(mode in ("pd", "gravity"), "Unknown controller")
        model, jids, aids = control.load_model(gravity_compensation=mode == "gravity")
        cases = dict(home=run_home(model, jids, aids))
        for name, target in goals.items():
            cases[name] = run_ee_case(model, jids, aids, target, solutions[name])
        output["controllers"][mode] = dict(cases=cases, joint_sequence=run_joint_sequence(model, jids, aids),
                                           body_gravcomp=model.body_gravcomp.tolist(),
                                           joint_actuatorgravcomp=model.jnt_actgravcomp.tolist(), ngravcomp=int(model.ngravcomp))
    return output


def print_summary(report):
    print("Errors are norms [mm]; relative error is measured from settled home, absolute error from nominal target.")
    print("scenario        controller   home_offset  relative_err  absolute_err  rotation_deg  peak_total_Nm  peak_speed_deg/s")
    for name in ("home", "x_plus_5mm", "x_minus_5mm", "z_plus_5mm"):
        for mode, result in report["controllers"].items():
            case = result["cases"][name]
            m, f = case["metrics"], case["forces"]
            print(f"{name:15s} {mode:10s} {m['home_offset_norm_mm']:12.6f} {m['relative_error_norm_mm']:13.6f} "
                  f"{m['absolute_error_norm_mm']:13.6f} {m['orientation_error_deg']:13.6f} "
                  f"{max(f['peak_total_drive_nm']):14.6f} {max(f['peak_speed_deg_s']):17.6f}")
    for mode, result in report["controllers"].items():
        seq = result["joint_sequence"]
        error = max(abs(e) for p in seq["phases"] for e in p["joint_stats"]["error_deg"])
        print(f"{mode}: original joint sequence 13/13 passed; max joint error={error:.9f} deg; "
              f"peak total drive={max(seq['forces']['peak_total_drive_nm']):.6f} N*m; "
              f"saturated steps={seq['forces']['saturated_steps']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", choices=("both", "pd", "gravity"), default="both")
    parser.add_argument("--json", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/control_comparison.json")
    args = parser.parse_args()
    try:
        modes = ("pd", "gravity") if args.controller == "both" else (args.controller,)
        report = run_comparison(modes)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print_summary(report)
        print(f"[PASS] Matched comparison completed; complete diagnostics: {args.json.resolve()}")
        return 0
    except (OSError, ValueError, KeyError, mujoco.FatalError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
