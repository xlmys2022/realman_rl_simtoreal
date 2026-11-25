"""用 Tk 显示对齐后的 RGB、深度及可选的 HSV 候选掩膜。"""

import numpy as np


class RealRgbdPreview:
    def __init__(self, *, localization, depth_max_m):
        import cv2
        import tkinter as tk

        self._cv2 = cv2
        self._tk = tk
        self.depth_max_m = depth_max_m
        try:
            self.window = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError(f"无法打开显示窗口，可用 --no-preview · {exc}") from exc
        self.window.title("RealSense 方块定位" if localization else "RealSense RGB-D 检查")
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.bind("<Escape>", self.close)
        self.window.bind("<Key-q>", self.close)
        self.open = True
        self._images = []
        names = ("RGB", "深度（m）", "HSV Mask · 最大连通域") if localization else ("RGB", "深度（m）")
        for column, name in enumerate(names):
            tk.Label(self.window, text=name).grid(row=0, column=column, padx=4)
            image = tk.Label(self.window)
            image.grid(row=1, column=column, padx=4)
            self._images.append(image)
        self._status = tk.Label(self.window, anchor="w", justify="left")
        self._status.grid(row=2, column=0, columnspan=len(names), padx=8, pady=6, sticky="ew")

    def render(self, sample, status, *, mask=None, centroid=None):
        cv2 = self._cv2
        bgr = cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2BGR)
        if centroid is not None:
            cv2.circle(bgr, tuple(round(x) for x in centroid), 6, (255, 255, 255), 2)
        valid = np.isfinite(sample.depth_m) & (sample.depth_m > 0)
        depth = np.where(valid, sample.depth_m, 0)
        level = (255 * (1 - np.clip(depth / self.depth_max_m, 0, 1))).astype(np.uint8)
        color_depth = cv2.applyColorMap(level, cv2.COLORMAP_TURBO)
        color_depth[~valid] = 0
        panels = (bgr, color_depth) if mask is None else (bgr, color_depth, mask)
        height, width = depth.shape
        scale = min(1.0, 0.90 * self.window.winfo_screenwidth() / (width * len(panels)),
                    0.75 * self.window.winfo_screenheight() / height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        for label, image in zip(self._images, panels):
            if size != (width, height):
                resampling = cv2.INTER_NEAREST if image.ndim == 2 else cv2.INTER_AREA
                image = cv2.resize(image, size, interpolation=resampling)
            _, png = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            photo = self._tk.PhotoImage(master=self.window, data=png.tobytes(), format="png")
            label.configure(image=photo)
            label.photo = photo
        self._status.configure(text=status)
        self.window.update()

    def close(self, _event=None):
        if self.open:
            self.open = False
            self.window.destroy()
