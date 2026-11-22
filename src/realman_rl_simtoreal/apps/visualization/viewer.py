"""延迟初始化 MuJoCo Viewer，统一演示程序的渲染后端和线程收尾。"""

from contextlib import contextmanager
import os
import sys
import threading


def configure_backend():
    """在 Linux X11 桌面上设置当前进程的 GLFW 默认后端。"""
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")


@contextmanager
def passive_viewer(model, data, **options):
    """启动被动 Viewer，退出上下文后等待新增的渲染线程结束。"""
    import mujoco.viewer

    existing_threads = set(threading.enumerate())
    viewer_threads = []
    try:
        with mujoco.viewer.launch_passive(model, data, **options) as viewer:
            # MuJoCo 3.17.0 close() requests shutdown but does not join its
            # daemon render thread. This CLI starts no other Python workers.
            viewer_threads = [thread for thread in threading.enumerate()
                              if thread not in existing_threads]
            yield viewer
    finally:
        for thread in viewer_threads:
            thread.join()

