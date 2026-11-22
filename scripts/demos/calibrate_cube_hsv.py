#!/usr/bin/env python3
"""Tune a cube's HSV thresholds with a live camera preview."""

import argparse
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread

import cv2
import numpy as np

from realman_rl_simtoreal.perception.rgbd import orange_mask
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig
from realman_rl_simtoreal.utils.artifacts import write_json


HSV_LIMITS = (179, 255, 255)


def read_color_frame(capture):
    ok, frame = capture.read()
    if ok and frame is not None and frame.size and frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8:
        return frame
    return None


def camera_choices():
    """List Linux device indexes and names without opening camera streams."""
    devices = {}
    for directory in (Path("/dev"), Path("/sys/class/video4linux")):
        for path in directory.glob("video[0-9]*"):
            suffix = path.name.removeprefix("video")
            if suffix.isdecimal():
                devices.setdefault(int(suffix), "")
                try:
                    devices[int(suffix)] = (path / "name").read_text().strip()
                except OSError:
                    pass
    return {index: f"{index} · {name}" if name else str(index)
            for index, name in sorted(devices.items())}


class CameraFeed:
    def __init__(self, index):
        self.index = index
        self.stop = Event()
        self.latest = (None, None)
        self.thread = Thread(target=self.read_frames, daemon=True)
        self.thread.start()

    def read_frames(self):
        capture = None
        try:
            capture = cv2.VideoCapture(self.index)
            if not capture.isOpened():
                raise RuntimeError(f"无法打开摄像头 {self.index}，请检查编号或占用情况。")
            failures = 0
            while not self.stop.is_set():
                frame = read_color_frame(capture)
                if frame is None:
                    failures += 1
                    if failures == 3:
                        raise RuntimeError(f"摄像头 {self.index} 无法读取彩色画面，请重新打开。")
                    self.stop.wait(0.05)
                    continue
                failures = 0
                self.latest = (frame, None)
        except (cv2.error, OSError, OverflowError, RuntimeError) as exc:
            self.latest = (None, str(exc))
        finally:
            if capture is not None:
                capture.release()


def threshold_ranges(low, high):
    hue_low, hue_high = low[0], high[0]
    hue_ranges = [(hue_low, hue_high)] if hue_low <= hue_high else [(0, hue_high), (hue_low, 179)]
    return [{"low": [lo, *low[1:]], "high": [hi, *high[1:]]} for lo, hi in hue_ranges]


def make_mask(rgb, ranges):
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for bounds in ranges:
        mask |= orange_mask(rgb, bounds["low"], bounds["high"])
    return mask


def save_calibration(output_dir, camera, ranges):
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")

    report = {"source": {"camera": camera}, "hsv_ranges": ranges}
    if len(ranges) == 1:
        report["vision_hsv_low"] = ranges[0]["low"]
        report["vision_hsv_high"] = ranges[0]["high"]

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cube_hsv_", dir=output_dir.parent) as staging:
        result = Path(staging) / "result"
        result.mkdir()
        write_json(result / "hsv.json", report)
        result.rename(output_dir)

    print(f"Saved {output_dir.resolve() / 'hsv.json'}", flush=True)


class CalibrationWindow:
    def __init__(self, window, tk, ttk, args):
        self.window = window
        self.tk = tk
        self.output_dir = args.output_dir
        self.feed = None
        self.pending_camera = None
        self.camera_error = None
        self.frame = self.rgb = None
        self.preview_size = None
        self.photos = [None, None]

        window.title("方块 HSV 实时标定")
        body = ttk.Frame(window, padding=12)
        body.pack(fill="both", expand=True)
        source_bar = ttk.Frame(body)
        source_bar.pack(fill="x", pady=(0, 8))
        self.source_status = tk.StringVar(value="选择或输入摄像头编号。")

        ttk.Label(source_bar, text="摄像头").pack(side="left", padx=(0, 6))
        self.camera_select = ttk.Combobox(source_bar, width=28)
        self.camera_select.pack(side="left", padx=(0, 6))
        devices = self.refresh_devices()
        index = args.camera if args.camera is not None else next(iter(devices), 0)
        self.camera_select.set(devices.get(index, str(index)))
        self.camera_select.bind("<<ComboboxSelected>>", self.open_camera)
        self.camera_select.bind("<Return>", self.open_camera)
        ttk.Button(source_bar, text="打开", command=self.open_camera).pack(side="left", padx=(0, 4))
        ttk.Button(source_bar, text="刷新列表", command=self.refresh_devices).pack(side="left")
        ttk.Label(source_bar, textvariable=self.source_status).pack(side="left", padx=8)

        previews = ttk.Frame(body)
        previews.pack(fill="x")
        self.canvases = []
        self.preview_items = []
        for col, title in enumerate(("彩色原图", "二值掩膜")):
            previews.columnconfigure(col, weight=1)
            panel = ttk.LabelFrame(previews, text=title, padding=4)
            panel.grid(row=0, column=col, sticky="nsew", padx=3)
            canvas = tk.Canvas(panel, width=280, height=145, background="#15181c", highlightthickness=0)
            canvas.pack()
            self.canvases.append(canvas)
            self.preview_items.append(canvas.create_image(0, 0, anchor="nw"))

        controls = ttk.LabelFrame(body, text="HSV 阈值", padding=(8, 2))
        controls.pack(fill="x", pady=6)
        defaults = (CartesianTabletopConfig.vision_hsv_low, CartesianTabletopConfig.vision_hsv_high)
        self.bounds = [[tk.IntVar(value=value) for value in bound] for bound in defaults]
        for col, bound_label in enumerate(("下限", "上限")):
            controls.columnconfigure(col, weight=1)
            for channel, (name, limit) in enumerate(zip("HSV", HSV_LIMITS)):
                tk.Scale(controls, label=f"{name} {bound_label}", from_=0, to=limit, orient="horizontal",
                         variable=self.bounds[col][channel], highlightthickness=0, length=235,
                         command=lambda _, side=col, axis=channel: self.move_slider(side, axis)
                         ).grid(row=channel, column=col, sticky="ew", padx=4)
        ttk.Label(controls, text="H 下限大于上限时，色相范围跨过 179/0。"
                  ).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 4))

        actions = ttk.Frame(body)
        actions.pack(fill="x")
        self.save_button = ttk.Button(actions, text="保存 HSV", command=self.save, state="disabled")
        self.save_button.pack(side="right")
        self.save_status = tk.StringVar()
        ttk.Label(actions, textvariable=self.save_status, wraplength=455).pack(side="left")

        if args.camera is not None:
            self.open_camera()
        window.after(35, self.poll_camera)

    def refresh_devices(self):
        devices = camera_choices()
        self.camera_select.configure(values=tuple(devices.values()))
        return devices

    def open_camera(self, _event=None):
        index = self.camera_select.get().partition(" · ")[0].strip()
        if not index.isdecimal():
            self.source_status.set("请输入非负整数摄像头编号。")
            return
        self.pending_camera = int(index)
        if self.feed is not None:
            self.feed.stop.set()
        self.clear_frame()
        self.camera_error = None
        self.source_status.set(f"正在打开摄像头 {index}…")

    def poll_camera(self):
        if self.pending_camera is not None and (self.feed is None or not self.feed.thread.is_alive()):
            self.feed = CameraFeed(self.pending_camera)
            self.pending_camera = None
        if self.feed is not None and self.pending_camera is None:
            frame, error = self.feed.latest
            if error is not None and error != self.camera_error:
                self.camera_error = error
                self.clear_frame()
                self.source_status.set(error)
            elif frame is not None and frame is not self.frame:
                if self.frame is None:
                    self.source_status.set(f"摄像头 {self.feed.index} · {frame.shape[1]} × {frame.shape[0]}")
                self.show_frame(frame)
        self.window.after(35, self.poll_camera)

    def close(self):
        if self.feed is not None:
            self.feed.stop.set()
            self.feed.thread.join(timeout=0.3)

    def clear_frame(self):
        self.frame = self.rgb = None
        self.preview_size = None
        for canvas, item in zip(self.canvases, self.preview_items):
            canvas.itemconfigure(item, image="")
        self.photos = [None, None]
        self.save_button.configure(state="disabled")

    def show_frame(self, frame):
        if self.frame is None or self.frame.shape != frame.shape:
            height, width = frame.shape[:2]
            scale = min(1, 0.44 * self.window.winfo_screenwidth() / width,
                        0.50 * self.window.winfo_screenheight() / height)
            self.preview_size = max(1, int(width * scale)), max(1, int(height * scale))
            for canvas in self.canvases:
                canvas.configure(width=self.preview_size[0], height=self.preview_size[1])
        self.frame = frame
        self.rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.draw_preview(0, frame, cv2.INTER_AREA)
        self.refresh_mask()
        self.save_button.configure(state="normal")

    def draw_preview(self, index, image, interpolation):
        if image.shape[:2][::-1] != self.preview_size:
            image = cv2.resize(image, self.preview_size, interpolation=interpolation)
        _, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        photo = self.tk.PhotoImage(master=self.window, data=png.tobytes(), format="png")
        self.canvases[index].itemconfigure(self.preview_items[index], image=photo)
        self.photos[index] = photo

    def current_ranges(self):
        low, high = ([variable.get() for variable in bound] for bound in self.bounds)
        # A camera tick may run before Tk dispatches the slider's command callback.
        for channel in (1, 2):
            low[channel], high[channel] = sorted((low[channel], high[channel]))
        return threshold_ranges(low, high)

    def move_slider(self, side, channel):
        if channel != 0:
            low, high = self.bounds[0][channel], self.bounds[1][channel]
            if low.get() > high.get():
                self.bounds[1 - side][channel].set(self.bounds[side][channel].get())
        self.refresh_mask()

    def refresh_mask(self):
        if self.rgb is not None:
            self.draw_preview(1, make_mask(self.rgb, self.current_ranges()), cv2.INTER_NEAREST)

    def save(self):
        if self.frame is None:
            return
        output_dir = self.output_dir or Path("outputs/calibration") / datetime.now().strftime("cube_hsv_%Y%m%d_%H%M%S_%f")
        try:
            save_calibration(output_dir, self.feed.index, self.current_ranges())
        except (OSError, ValueError) as exc:
            self.save_status.set(f"保存失败  {exc}")
            return
        self.output_dir = None
        self.save_status.set(f"已保存  {output_dir / 'hsv.json'}")


def run_window(args):
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as exc:
        raise RuntimeError("Tkinter is required for the HSV calibration window.") from exc
    try:
        window = tk.Tk()
    except tk.TclError as exc:
        raise RuntimeError("Cannot open the calibration window. Run the script in a desktop session.") from exc

    app = None
    try:
        app = CalibrationWindow(window, tk, ttk, args)
        window.protocol("WM_DELETE_WINDOW", window.quit)
        window.bind("<Escape>", lambda _: window.quit())
        window.mainloop()
    finally:
        if app is not None:
            app.close()
        window.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, help="Open this camera index when the window starts")
    parser.add_argument("--output-dir", type=Path, help="New directory for the first save")
    args = parser.parse_args()

    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be a non-negative index.")
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("Choose a new --output-dir.")

    try:
        run_window(args)
    except (cv2.error, OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
