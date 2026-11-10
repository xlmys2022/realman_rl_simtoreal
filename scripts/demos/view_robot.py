#!/usr/bin/env python3
"""Show RML63II-6F-V at its URDF zero pose without advancing physics."""

import argparse
import math
import os
from pathlib import Path
import sys
import time

from realman_rl_simtoreal.paths import DEFAULT_URDF as DEFAULT_MODEL
from realman_rl_simtoreal.viewer import configure_backend, passive_viewer


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be finite and greater than zero")
    return seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="URDF model path")
    parser.add_argument("--duration", type=positive_seconds, help="close after this many seconds")
    args = parser.parse_args()
    path = args.model.expanduser().resolve()
    if not path.is_file():
        print(f"[FAIL] Model file not found: {path}", file=sys.stderr)
        return 1
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[FAIL] No desktop display. Run in a graphical terminal; inspect_model.py works headlessly.", file=sys.stderr)
        return 1
    # pyGLFW supports this process-local backend selection. The native Wayland
    # backend had libdecor initialization errors; use X11/XWayland if present.
    # An explicit user selection still takes precedence.
    configure_backend()
    try:
        import mujoco
        import mujoco.viewer
        import numpy as np
        import glfw
    except ImportError as exc:
        print(f"[FAIL] Missing dependency: {exc}. Install requirements.txt in realman_rl.", file=sys.stderr)
        return 1
    try:
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        initial_qpos = data.qpos.copy()
        # Fail in the calling thread if the display cannot be opened. Otherwise
        # launch_passive can wait indefinitely for its failed background thread.
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize the desktop display")
        print(f"Model: {path}", flush=True)
        print(f"GLFW backend preference: {os.environ.get('PYGLFW_LIBRARY_VARIANT', 'automatic')}", flush=True)
        print("Static URDF zero pose; no physics stepping. Real hardware match remains unverified.", flush=True)
        with passive_viewer(model, data) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = model.stat.center
                viewer.cam.distance = max(1.5 * model.stat.extent, 0.5)
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                # The URDF has matching visual/collision meshes; display only visual geoms.
                viewer.opt.geomgroup[0] = 0
                viewer.opt.geomgroup[1] = 1
            print("[PASS] Viewer initialized. Rotate: left drag; pan: right drag; zoom: scroll.", flush=True)
            start = time.monotonic()
            while viewer.is_running():
                if args.duration is not None and time.monotonic() - start >= args.duration:
                    break
                # Reassert the inspection pose if the UI changed state. Never call mj_step.
                with viewer.lock():
                    data.qpos[:] = initial_qpos
                    data.qvel[:] = 0
                    data.time = 0
                    mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(1 / 60)
            with viewer.lock():
                data.qpos[:] = initial_qpos
                data.qvel[:] = 0
                data.time = 0
                mujoco.mj_forward(model, data)
            if not np.array_equal(data.qpos, initial_qpos) or data.time != 0:
                raise RuntimeError("Static pose was not preserved")
        print("[PASS] Viewer closed normally; qpos unchanged, simulation time=0", flush=True)
    except KeyboardInterrupt:
        print("[PASS] Viewer stopped by Ctrl+C")
    except (OSError, ValueError, RuntimeError, mujoco.FatalError) as exc:
        print(f"[FAIL] Viewer/model error: {exc}. Check desktop access and OpenGL/GLFW support.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
