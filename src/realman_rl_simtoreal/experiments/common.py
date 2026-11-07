"""Shared existing experiment fixtures, sequence and text formatting.

Simulation components never import these demo policies. Printing stays in CLIs.
"""

import numpy as np
from ..simulation import control


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def inverse(t):
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ t[:3, 3]
    return result


def sample_poses(home):
    poses = [("home", home.copy())]
    for i in range(6):
        q = home.copy()
        q[i] += np.deg2rad(3)
        poses.append((f"J{i+1}_plus_3deg", q))
    poses.extend([("combined", home + np.deg2rad([2, -2, 2, 1, -1, 2])),
                  ("diverse_fk_only", np.deg2rad([25, -45, 55, 20, 40, -30]))])
    return poses


def joint_result_lines(result):
    lines = []
    lines.append(f"\n[PASS] {result['label']}")
    for i, (target, actual, error) in enumerate(zip(result['target_deg'], result['actual_deg'], result['error_deg']), 1):
        lines.append(f'  J{i}: target={target:9.4f} deg actual={actual:9.4f} deg error={error:+.6f} deg')
    lines.append(f"  max final error={max(map(abs, result['error_deg'])):.6f} deg; "
                 f"overshoot={max(result['overshoot_deg']):.6f} deg; "
                 f"tail range={max(result['tail_range_deg']):.3e} deg; "
                 f"tail speed={max(result['tail_speed_deg_s']):.3e} deg/s; "
                 f"contacts={result['max_contacts']}")
    lines.append(f"  max uncommanded joint travel={result['max_uncommanded_motion_deg']:.6f} deg")
    lines.append('  peak torque [N*m]: ' + np.array2string(np.array(result['peak_torque_nm']), precision=5))
    return lines


def run_joint_demo(model, data, joint_ids, actuator_ids, target=None, tick=None, step_observer=None, *, reporter=None):
    home = data.ctrl[actuator_ids].copy()
    targets = [("hold home under gravity", home, 0.0, 3.0)]
    if target is None:
        for i in range(6):
            command = home.copy()
            command[i] += np.deg2rad(3)
            targets.extend([(f"J{i+1} +3 deg", command, 1.0, 3.0), (f"J{i+1} return", home, 1.0, 1.0)])
    else:
        targets.append(("custom six-joint target", control.validate_target(model, joint_ids, home, target), 1.0, 3.0))
    results = []
    for label, command, ramp, hold in targets:
        result = control.run_segment(model, data, joint_ids, actuator_ids, command, label, ramp, hold, tick, step_observer)
        if reporter is not None:
            reporter(result)
        results.append(result)
    return results



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

