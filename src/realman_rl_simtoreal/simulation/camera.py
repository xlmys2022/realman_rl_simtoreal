"""Camera-frame convention, independent of any learning task."""
import numpy as np


def camera_rotation(forward, up):
    """Return camera-to-body rotation; MuJoCo optical forward is local -Z."""
    forward=np.asarray(forward,dtype=float)
    up=np.asarray(up,dtype=float)
    return np.column_stack([np.cross(forward,up),up,-forward])
