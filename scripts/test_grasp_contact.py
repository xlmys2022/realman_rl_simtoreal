#!/usr/bin/env python3
"""Offline dynamic cube contact/hold/release experiments (not a grasp policy)."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

from realman_rl_simtoreal.viewer import configure_backend, passive_viewer
configure_backend()

import mujoco
import numpy as np

from realman_rl_simtoreal.paths import OUTPUTS
from realman_rl_simtoreal.simulation.control import advance_synchronized_cycle, validate_target
from realman_rl_simtoreal.simulation.grasp_contact import load_experiment, ContactRecorder
from realman_rl_simtoreal.simulation.gripper import angle_to_width, width_to_angle
from realman_rl_simtoreal.simulation.kinematics import Kinematics
from realman_rl_simtoreal.simulation.model import require

CASES = dict(normal=dict(), no_close=dict(close=False), release=dict(release=True),
             low_friction=dict(friction=.05), heavy=dict(mass=.03), hull=dict(hull=True))


class ViewerStopped(Exception):
    pass


def summarize(rows):
    summary = {}
    for phase in dict.fromkeys(r['phase'] for r in rows):
        selected = [r for r in rows if r['phase'] == phase]
        tail = selected[-250:]
        bilateral = [r['left_normal_n'] > .005 and r['right_normal_n'] > .005 for r in tail]
        airborne = [r['support_normal_n'] < .001 and r['support_clearance_m'] > .001 for r in tail]
        relative = np.array([r['object_relative_to_pads_m'] for r in tail])
        slip = float(np.max(np.linalg.norm(relative-relative[0], axis=1)))
        summary[phase] = dict(duration_s=len(selected)*.002, final=selected[-1],
                              tail_bilateral_fraction=float(np.mean(bilateral)),
                              tail_airborne_fraction=float(np.mean(airborne)), tail_relative_slip_m=slip,
                              peak_left_normal_n=max(r['left_normal_n'] for r in selected),
                              peak_right_normal_n=max(r['right_normal_n'] for r in selected),
                              peak_left_tangent_n=max(r['left_tangent_n'] for r in selected),
                              peak_right_tangent_n=max(r['right_tangent_n'] for r in selected),
                              max_penetration_m=max(r['max_penetration_m'] for r in selected),
                              max_object_speed_m_s=max(np.linalg.norm(r['object_qvel'][:3]) for r in selected))
    hold = summary.get('air_hold')
    held = bool(hold and hold['duration_s'] >= 2-1e-9 and hold['tail_bilateral_fraction'] >= .95
                and hold['tail_airborne_fraction'] >= .95 and hold['tail_relative_slip_m'] < .002)
    return summary, held


def run_case(name='normal', *, mass=.01, friction=.7, close=True, release=False, hull=False,
             lift=True, viewer=False, duration=None):
    m, d, jids, aids, g, policy = load_experiment(mass=mass, friction=friction, hull=hull)
    rec = ContactRecorder(m, d, g, policy)
    kin = Kinematics(m)
    home = kin.home_q.copy()
    lift_translation = .005*d.xmat[g.base].reshape(3, 3)[:, 2].copy()
    lift_q = home.copy()
    if lift:
        lift_result = kin.solve_ik(kin.home_position+lift_translation, home, kin.home_rotation)
        require(lift_result.success, f'Optional lift IK failed: {lift_result.reason}')
        lift_q = validate_target(m, jids, home, lift_result.q)
    require(np.linalg.norm(kin.fk(lift_q)[0]-kin.home_position) <= .01, 'Lift outside 10mm experiment region')
    phases = [('settle', 1.5, .06, home), ('close' if close else 'stay_open', 6., 0 if close else .06, home),
              ('supported_hold', 1., 0 if close else .06, home)]
    if lift:
        phases += [('lift', 2., 0 if close else .06, lift_q), ('air_hold', 2., 0 if close else .06, lift_q)]
    if release and lift:
        place = kin.solve_ik(kin.home_position+.2*lift_translation, lift_q, kin.home_rotation)
        require(place.success, f'Release approach IK failed: {place.reason}')
        release_q = validate_target(m, jids, home, place.q)
        phases += [('lower_for_release', 2., 0 if close else .06, release_q)]
    else:
        release_q = home
    if release:
        phases += [('release', 6., .06, release_q),
                   ('after_release', 1., .06, release_q)]
    failure, stopped = None, False
    start_wall, last_sync = time.perf_counter(), 0.
    try:
        if viewer and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise ValueError('No desktop display for Viewer')
        with passive_viewer(m, d) if viewer else nullcontext(None) as window:
            if window is not None:
                window.cam.lookat[:] = d.xpos[m.body('target_object').id]
                window.cam.distance = .26
                window.cam.azimuth, window.cam.elevation = 145, -15
                window.sync()
            for label, seconds, width, arm_target in phases:
                rec.phase = label
                if viewer:
                    print(f'[PHASE] {label} at simulation t={d.time:.3f}s', flush=True)
                g.set_width(width)
                angle_start = float(d.ctrl[g.actuator_id])
                angle_end = width_to_angle(width)
                arm_start = d.ctrl[aids].copy()
                for cycle in range(round(seconds/.05)):
                    alpha = (cycle+1)*.05/seconds
                    alpha = alpha*alpha*(3-2*alpha)
                    if label in ('close', 'release'):
                        g.set_width(angle_to_width(angle_start+alpha*(angle_end-angle_start)))
                    advance_synchronized_cycle(m, d, jids, aids, arm_start+alpha*(arm_target-arm_start), g,
                                               physics_steps=25, step_observer=rec)
                    if window is not None:
                        elapsed = time.perf_counter()-start_wall
                        if not window.is_running() or (duration is not None and elapsed >= duration):
                            raise ViewerStopped()
                        if elapsed-last_sync >= 1/30:
                            window.sync()
                            last_sync = elapsed
                        time.sleep(max(0, d.time-elapsed))
    except (ValueError, RuntimeError, mujoco.FatalError) as exc:
        failure = str(exc)
    except (ViewerStopped, KeyboardInterrupt):
        stopped = True
    phases_summary, held = summarize(rec.rows)
    failure_state = None if failure is None else dict(
        time_s=float(d.time), gripper_q_rad=d.qpos[g.qadr].tolist(),
        gripper_velocity_rad_s=d.qvel[g.vadr].tolist(),
        mimic_error_rad=float(np.max(np.abs(d.qpos[g.qadr]-g.signs*d.qpos[g.qadr][0]))),
        contacts=[dict(category=policy.classify(*c.geom), geoms=[m.geom(int(i)).name for i in c.geom],
                       distance_m=float(c.dist)) for c in d.contact])
    result = dict(case=name, completed=failure is None and not stopped, safety_failure=failure,
                  failure_state=failure_state,
                  viewer_stopped=stopped, airborne_hold_passed=held,
                  logging=dict(scalar_period_s=.002, detailed_contact_period_s=.05,
                               forces='post mj_step, before mj_forward; state is post integration',
                               object_qvel='translation world xyz, rotation body-local xyz'),
                  parameters=dict(cube_side_m=.02, mass_kg=mass, pad_friction=friction, support_friction=1.,
                                  close=close, release=release, geometry='hull' if hull else 'measured_pad_layers',
                                  gravity=m.opt.gravity.tolist(), timestep_s=m.opt.timestep,
                                  lift_translation_world_m=lift_translation.tolist() if lift else [0., 0., 0.]),
                  counts=dict(nq=m.nq, nv=m.nv, nu=m.nu, neq=m.neq),
                  simulation_time_s=float(d.time), phases=phases_summary,
                  gripper_statistics=g.statistics(),
                  final_arm_error_deg=np.rad2deg(d.qpos[m.jnt_qposadr[jids]]-d.ctrl[aids]).tolist(),
                  records=rec.rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['all']+list(CASES), default='all')
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--duration', type=float, help='Viewer wall time; early close is explicitly incomplete')
    parser.add_argument('--no-lift', action='store_true')
    parser.add_argument('--json', type=Path, default=OUTPUTS/'grasp_contact/results.json')
    args = parser.parse_args()
    if args.viewer and args.case == 'all':
        parser.error('Viewer requires a single --case')
    if args.duration is not None and (not args.viewer or not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('duration requires viewer and positive finite seconds')
    results = []
    print('Simulation only: uncalibrated EG2 mass, mount and servos; not real gripper performance.', flush=True)
    for name in CASES if args.case == 'all' else [args.case]:
        try:
            result = run_case(name, **CASES[name], lift=not args.no_lift, viewer=args.viewer, duration=args.duration)
        except (ValueError, RuntimeError, KeyError, mujoco.FatalError) as exc:
            print(f'[FAIL] {name} initialization: {exc}', flush=True)
            return 1
        results.append(result)
        print(f"{name}: completed={result['completed']} airborne_hold={result['airborne_hold_passed']} "
              f"t={result['simulation_time_s']:.3f}s safety_failure={result['safety_failure']}", flush=True)
        for label, row in result['phases'].items():
            f = row['final']
            print(f"  {label}: gap={f['actual_width_m']*1000:.3f}mm cmd={f['command_width_m']*1000:.3f}mm "
                  f"normal L/R={f['left_normal_n']:.4f}/{f['right_normal_n']:.4f}N "
                  f"table={f['support_normal_n']:.4f}N clearance={f['support_clearance_m']*1000:.3f}mm", flush=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(results, separators=(',', ':'), allow_nan=False)+'\n')
    # A safe negative physical outcome is a result, not an execution error.
    return 1 if any(r['safety_failure'] for r in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
