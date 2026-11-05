"""Lazy CLI-only Viewer lifecycle helpers; importing this creates no GUI."""

from contextlib import contextmanager
import os
import sys
import threading


def configure_backend():
    """Preserve the process-local X11 default before MuJoCo/GLFW imports."""
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
        os.environ.setdefault("PYGLFW_LIBRARY_VARIANT", "x11")


@contextmanager
def passive_viewer(model, data):
    """Wait for this standalone viewer's threads before GLFW shuts down."""
    import mujoco.viewer

    existing_threads = set(threading.enumerate())
    viewer_threads = []
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # MuJoCo 3.17.0 close() requests shutdown but does not join its
            # daemon render thread. This CLI starts no other Python workers.
            viewer_threads = [thread for thread in threading.enumerate()
                              if thread not in existing_threads]
            yield viewer
    finally:
        for thread in viewer_threads:
            thread.join()

