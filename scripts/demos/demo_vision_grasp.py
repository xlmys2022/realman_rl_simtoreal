#!/usr/bin/env python3
"""Run the RGB-D tabletop PPO in an interactive MuJoCo window."""

import argparse
from contextlib import ExitStack
from copy import copy
import math
import os
from pathlib import Path
import sys
from threading import Event
import time

from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.viewer import configure_backend, passive_viewer


DEFAULT_MODEL = PROJECT_ROOT / (
    "outputs/grasp/ppo/tabletop_cartesian/vision_rgbd_v1/"
    "full_table_down_seed2031_r4_zero_anchored/checkpoints/step_00000768/model.zip"
)
PHASE_LABELS = {
    "traditional_startup": "Moving from zero to observation pose",
    "traditional_settle": "Settling at observation pose",
    "ppo": "Vision-guided PPO",
}


class RestartRequested(Exception):
    """Restart the episode while keeping the viewer open."""


class WindowClosed(Exception):
    """The viewer was closed by the operator."""


class GraspWindow:
    def __init__(self, stack, speed):
        self.stack = stack
        self.speed = speed
        self.viewer = None
        self.paused = Event()
        self.restart = Event()
        self.last_frame = None
        self.last_viewport = None
        self.reset_episode()

    def reset_episode(self):
        self.paused.clear()
        self.restart.clear()
        self.sim_time = 0.0
        self.last_wall = None
        self.status = PHASE_LABELS["traditional_startup"]
        self.vision = None

    def on_key(self, key):
        if key == ord(" "):
            if self.paused.is_set():
                self.paused.clear()
            else:
                self.paused.set()
        elif key == ord("R"):
            self.restart.set()

    def open(self, robot):
        import mujoco

        # GUI perturbations and model edits act on a display copy.
        display_model = copy(robot.model)
        self.display_data = mujoco.MjData(display_model)
        mujoco.mj_copyData(self.display_data, robot.model, robot.data)
        self.viewer = self.stack.enter_context(
            passive_viewer(
                display_model, self.display_data, key_callback=self.on_key,
                show_left_ui=False, show_right_ui=False,
            )
        )
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = [0.25, 0.0, 1.13]
            self.viewer.cam.distance = 2.65
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -23
            self.viewer.opt.geomgroup[1] = 0
        print("MuJoCo window ready. Space pauses, R restarts.", flush=True)

    def update_panels(self):
        import cv2
        import mujoco

        if self.vision is None:
            if self.last_frame is not None:
                self.viewer.clear_images()
                self.last_frame = None
            return
        viewport = self.viewer.viewport
        if viewport is None:
            return
        bounds = (viewport.left, viewport.bottom, viewport.width, viewport.height)
        frame = self.vision["frame"]
        if frame is self.last_frame and bounds == self.last_viewport:
            return
        self.last_frame = frame
        self.last_viewport = bounds

        margin, title_height = 12, 22
        source_height, source_width = frame.rgb.shape[:2]
        max_height = (viewport.height - 3 * margin) // 2 - title_height
        width = min(300, viewport.width // 3, max_height * source_width // source_height)
        if width < 80:
            self.viewer.clear_images()
            return
        height = width * source_height // source_width
        rgb = cv2.resize(frame.rgb, (width, height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(self.vision["mask"], (width, height), interpolation=cv2.INTER_NEAREST)
        panels = [(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), "HSV mask"), (rgb, "Wrist RGB")]
        images = []
        for index, (pixels, title) in enumerate(panels):
            labeled = cv2.copyMakeBorder(
                pixels, title_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(26, 31, 39),
            )
            cv2.putText(labeled, title, (7, 16), cv2.FONT_HERSHEY_SIMPLEX,
                        0.47, (240, 244, 248), 1, cv2.LINE_AA)
            rect = mujoco.MjrRect(
                viewport.left + viewport.width - width - margin,
                viewport.bottom + margin + index * (labeled.shape[0] + margin),
                width, labeled.shape[0],
            )
            images.append((rect, labeled))
        self.viewer.set_images(images)

    def update_overlay(self):
        import mujoco

        labels = ["Phase", "Simulation"]
        status = f"{self.status} (paused)" if self.paused.is_set() else self.status
        values = [status, f"{self.sim_time:.2f} s"]
        if self.vision is not None:
            measurement = self.vision["measurement"]
            x, y, z = measurement["object_position_base_m"]
            labels.extend(("RGB-D target [m, base]", "Visual hold"))
            values.extend((
                f"({x:.3f}, {y:.3f}, {z:.3f})" if measurement["detected"] else "Target lost",
                f'{measurement["visual_hold_seconds"]:.2f} / '
                f'{measurement["visual_hold_required_seconds"]:.2f} s',
            ))
        self.viewer.set_texts([
            (None, mujoco.mjtGridPos.mjGRID_TOPLEFT, "\n".join(labels), "\n".join(values)),
            (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
             "Space  Pause/resume   R  Restart\nLeft drag  Orbit   Right drag  Pan   Scroll  Zoom", ""),
        ])
        self.update_panels()

    def check_requests(self):
        if not self.viewer.is_running():
            raise WindowClosed
        if self.restart.is_set():
            raise RestartRequested

    def wait(self, deadline=None):
        while True:
            self.check_requests()
            self.update_overlay()
            remaining = None if deadline is None else deadline - time.monotonic()
            if not self.paused.is_set() and remaining is not None:
                if remaining <= 0:
                    return
                time.sleep(min(remaining, 0.02))
            else:
                time.sleep(0.02)

    def tick(self, robot, phase):
        import mujoco

        if self.viewer is None:
            self.open(robot)
        self.check_requests()
        with self.viewer.lock():
            mujoco.mj_copyData(self.display_data, robot.model, robot.data)
        self.viewer.sync()

        sim_time = float(robot.data.time)
        status = PHASE_LABELS[phase]
        if phase == "ppo" and self.status != status:
            print("Observation pose reached. Vision-guided PPO is running.", flush=True)
        deadline = time.monotonic() if self.last_wall is None else (
            self.last_wall + (sim_time - self.sim_time) / self.speed
        )
        self.sim_time = sim_time
        self.status = status
        self.vision = getattr(robot, "vision_debug", None)
        self.wait(deadline)
        self.last_wall = time.monotonic()

    def finish(self, episode):
        if episode["success"]:
            clearance_mm = 1000 * episode["clearance_m"]
            self.status = f"Grasp complete  |  Lift {clearance_mm:.1f} mm"
            print(f"{self.status}, stable hold {episode['hold_seconds']:.2f} s.", flush=True)
        else:
            self.status = f"Stopped  |  {episode['reason']}"
            print(f"{self.status}. {episode['detail'] or ''}".rstrip(), flush=True)
        self.paused.clear()
        self.update_overlay()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Visual PPO checkpoint")
    parser.add_argument("--seed", type=int, default=997000, help="Reset seed (default 997000)")
    parser.add_argument("--object-radius", type=float, default=0.0,
                        help="Cube position randomization radius in metres, up to 0.05")
    parser.add_argument("--speed", type=float, default=2.0,
                        help="Simulation seconds per wall-clock second when rendering can keep up (default 2)")
    parser.add_argument("--exit-on-finish", action="store_true",
                        help="Close after one episode instead of keeping the final scene open")
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if not 0 <= args.object_radius <= 0.05 or args.seed < 0:
        parser.error("--object-radius must be in [0, 0.05] and --seed must be nonnegative")

    path = args.model.expanduser()
    if not path.is_file():
        print(f"[FAIL] Visual PPO checkpoint not found at {path}. Use --model to select one.", file=sys.stderr)
        return 1
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[FAIL] No desktop display. Run the demo from a graphical terminal.", file=sys.stderr)
        return 1
    configure_backend()
    if sys.platform.startswith("linux"):
        # GLFW owns the desktop window. EGL renders sensor frames on the simulation thread.
        os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        import cv2
        import glfw
        import mujoco
        from realman_rl_simtoreal.rl.ppo_tabletop_cartesian import evaluate, load
    except ImportError as exc:
        print(f"[FAIL] {exc}. Check the RL and vision dependencies in realman_rl.", file=sys.stderr)
        return 1

    status = 0
    try:
        model, config = load(path)
        if config["environment"].get("observation_source") != "rgbd_hsv":
            raise ValueError("A checkpoint trained with rgbd_hsv observations is required")
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize the desktop display")
        try:
            with ExitStack() as stack:
                window = GraspWindow(stack, args.speed)
                while True:
                    try:
                        result = evaluate(
                            model, config, [args.seed], full=True, tick=window.tick,
                            object_radius=args.object_radius,
                        )
                        episode = result["episodes"][0]
                        status = int(not episode["success"])
                        window.finish(episode)
                        if args.exit_on_finish:
                            return status
                        window.wait()
                    except RestartRequested:
                        window.reset_episode()
                        print("Restarting from zero.", flush=True)
        finally:
            glfw.terminate()
    except (WindowClosed, KeyboardInterrupt):
        return status
    except (OSError, ValueError, RuntimeError, mujoco.FatalError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
