"""SE(3) helpers shared by offline kinematics comparisons."""
import numpy as np


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def inverse(t):
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ t[:3, 3]
    return result
