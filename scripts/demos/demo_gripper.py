#!/usr/bin/env python3
"""Physical EG2-4C2 open/close demo; experimental mount and uncalibrated payload."""

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

from realman_rl_simtoreal.paths import GRIPPER_SCENE
from realman_rl_simtoreal.simulation.gripper import Gripper, MAX_WIDTH, angle_to_width
from realman_rl_simtoreal.simulation.model import initialize, load_model, require


class ViewerStopped(Exception):
    pass


def run_demo(model, data, joint_ids, gripper, *, cycles=2, tick=None):
    """Select width targets and advance physical steps; never assign live qpos."""
    home = model.key_qpos[model.key('home').id, model.jnt_qposadr[joint_ids]]
    phases = []
    for label, width, seconds in [('hold open', MAX_WIDTH, .5)]+[
            (f'{direction} {cycle+1}', target, .82/.5+1.0)
            for cycle in range(cycles) for direction, target in [('close', 0), ('open', MAX_WIDTH)]]:
        gripper.set_width(width)
        widths, arm_errors = [], []
        start, reached = float(data.time), None
        max_tracking_error, peak_width_speed = 0.0, 0.0
        previous_width = gripper.get_width()
        for _ in range(round(seconds/model.opt.timestep)):
            actual = gripper.step()
            widths.append(actual)
            arm_errors.append(np.abs(data.qpos[model.jnt_qposadr[joint_ids]]-home))
            max_tracking_error = max(max_tracking_error, abs(actual-angle_to_width(data.ctrl[gripper.actuator_id])))
            peak_width_speed = max(peak_width_speed, abs(actual-previous_width)/model.opt.timestep)
            previous_width = actual
            if reached is None and abs(actual-width) <= .0002:
                reached = float(data.time)-start
            if tick is not None:
                tick()
        tail = widths[-round(.5/model.opt.timestep):]
        final_error = abs(widths[-1]-width)
        require(final_error <= .0002, f'{label}: width did not converge within 0.2 mm')
        require(np.ptp(tail) <= .0001, f'{label}: persistent width oscillation')
        require(np.max(arm_errors) <= np.deg2rad(.5), f'{label}: arm home drift exceeded .5 degrees')
        row = dict(label=label, target_width_m=width, actual_width_m=widths[-1], error_m=final_error,
                   time_to_0_2mm_s=reached, phase_time_s=float(data.time)-start,
                   tail_range_m=float(np.ptp(tail)), peak_width_speed_m_s=peak_width_speed,
                   max_command_tracking_error_m=max_tracking_error,
                   max_arm_home_error_deg=np.rad2deg(np.max(arm_errors, axis=0)).tolist())
        phases.append(row)
        print(f"[PASS] {label}: target={width*1000:.6f}mm actual={widths[-1]*1000:.6f}mm "
              f"error={final_error*1000:.6g}mm time_to_0.2mm={reached:.3f}s", flush=True)
    return dict(completed=True, counts=dict(nq=model.nq, nv=model.nv, nu=model.nu, neq=model.neq),
                simulation_time_s=float(data.time), phases=phases, statistics=gripper.statistics())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=GRIPPER_SCENE)
    parser.add_argument('--cycles', type=int, default=2)
    parser.add_argument('--viewer', action='store_true')
    parser.add_argument('--duration', type=float, help='Viewer wall-clock limit; incomplete sequence is not passed')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.cycles < 1 or (args.duration is not None and (not args.viewer or not np.isfinite(args.duration) or args.duration <= 0)):
        parser.error('cycles must be positive; duration requires viewer and positive seconds')
    try:
        if args.viewer and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
            raise ValueError('No desktop display for Viewer')
        model, joint_ids, _ = load_model(args.model, gravity_compensation=True)
        data = initialize(model)
        gripper = Gripper(model, data)
        print(f'Exact-name upstream EG2-4C2; experimental 8mm mount, not verified on real 6F-V hardware.')
        print(f'Model gap={MAX_WIDTH*1000:.6f}mm, payload mass=94.2992g + 11g adapter; product says 70mm/231g.')
        start_wall, start_sim = time.perf_counter(), float(data.time)
        with passive_viewer(model, data) if args.viewer else nullcontext(None) as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = data.site_xpos[model.site('grasp_site').id]
                viewer.cam.distance = .55
                viewer.cam.azimuth, viewer.cam.elevation = 120, -20
                viewer.sync()
            last_sync = start_wall
            def tick():
                nonlocal last_sync
                if viewer is None:
                    return
                elapsed = time.perf_counter()-start_wall
                if not viewer.is_running() or (args.duration is not None and elapsed >= args.duration):
                    raise ViewerStopped()
                if time.perf_counter()-last_sync >= 1/30:
                    viewer.sync()
                    last_sync = time.perf_counter()
                time.sleep(max(0, (data.time-start_sim)-elapsed))
            result = run_demo(model, data, joint_ids, gripper, cycles=args.cycles, tick=tick)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(result, indent=2)+'\n')
        return 0
    except (ViewerStopped, KeyboardInterrupt):
        print('[INFO] Viewer stopped normally; complete sequence not validated', flush=True)
        return 0
    except (ValueError, RuntimeError, KeyError, mujoco.FatalError) as exc:
        print(f'[FAIL] {exc}', flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
