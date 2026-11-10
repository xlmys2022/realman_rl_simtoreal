"""Public-observation rule baseline; never used to guide pure RL training."""
import numpy as np


def rule_action(observation, action_scale):
    """Position feedback minus already-commanded tracking lag (anti-windup).

    delta = (goal-actual) - (command-actual) = goal-command. Once the command
    reaches the goal we send zero increments while the actuators catch up.
    The normalized vector norm is capped at .25 (0.25mm by default). This
    leaves margin for coupled wrist motion under the existing joint rate cap.
    Only public observations are read; all motion goes through env.step.
    """
    if not np.isfinite(action_scale) or action_scale <= 0:
        raise ValueError("action_scale must be positive")
    actual = observation['achieved_goal']
    error = observation['desired_goal']-actual
    pending = observation['observation'][18:21]-actual
    action = (error-pending)/action_scale
    action *= min(1.0, .25/max(float(np.linalg.norm(action)), 1e-30))
    return np.clip(action, -1, 1).astype(np.float32)
