"""Existing joint-position segments and total-drive audit; no CLI or Viewer."""

import mujoco
import numpy as np

from .model import require

MAX_ERROR_DEG = 0.5
MAX_TAIL_RANGE_DEG = 0.05
MAX_TAIL_SPEED_DEG_S = 0.1

def validate_target(model, joint_ids, home, target):
    target = np.asarray(target, dtype=float)
    require(target.shape == (6,) and np.isfinite(target).all(), "Target must contain six finite angles")
    require(np.max(np.abs(target - home)) <= np.deg2rad(5) + 1e-12,
            "Demo targets must stay within 5 degrees of home")
    ranges = model.jnt_range[joint_ids]
    require(np.all(target > ranges[:, 0]) and np.all(target < ranges[:, 1]), "Target violates joint limits")
    return target


def run_segment(model, data, joint_ids, actuator_ids, target, label, ramp=1.0, hold=3.0, tick=None, step_observer=None):
    """Smooth the control setpoint and measure real integrated state; never reset qpos."""
    qadr = model.jnt_qposadr[joint_ids]
    vadr = model.jnt_dofadr[joint_ids]
    start = data.ctrl[actuator_ids].copy()
    target = np.asarray(target)
    dt = model.opt.timestep
    records, velocities = [], []
    peak_torque = np.zeros(6)
    peak_speed = np.zeros(6)
    contacts = 0
    min_distance = 0.0
    expected_time = data.time
    for step in range(round((ramp + hold) / dt)):
        alpha = min((step + 1) * dt / ramp, 1.0) if ramp else 1.0
        alpha = alpha * alpha * (3 - 2 * alpha)
        data.ctrl[actuator_ids] = start + alpha * (target - start)
        mujoco.mj_step(model, data)
        if step_observer is not None:
            # Observe forces used by this step before a tick refreshes mjData.
            step_observer(data)
        expected_time += dt
        require(abs(data.time - expected_time) < 1e-8, "Simulation time reset unexpectedly")
        for name in ("qpos", "qvel", "qacc", "qfrc_actuator", "actuator_force", "ctrl", "xpos"):
            require(np.isfinite(getattr(data, name)).all(), f"Non-finite {name}")
        require(not any(warning.number for warning in data.warning), "MuJoCo numerical warning")
        contacts = max(contacts, data.ncon)
        for contact in data.contact:
            min_distance = min(min_distance, float(contact.dist))
            if contact.dist < -0.0001:
                names = [model.geom(int(i)).name for i in contact.geom]
                raise ValueError(f"Unexpected collision {names}: distance={contact.dist:.6g} m")
        q, v = data.qpos[qadr].copy(), data.qvel[vadr].copy()
        require(np.all(q >= model.jnt_range[joint_ids, 0] - 1e-4)
                and np.all(q <= model.jnt_range[joint_ids, 1] + 1e-4), "Joint limit violated")
        records.append(q)
        velocities.append(v)
        peak_torque = np.maximum(peak_torque, np.abs(data.qfrc_actuator[vadr]))
        peak_speed = np.maximum(peak_speed, np.abs(v))
        if tick is not None:
            tick()
    positions = np.rad2deg(records)
    tail_size = max(1, round(0.5 / dt))
    errors = positions[-1] - np.rad2deg(target)
    tail_range = np.ptp(positions[-tail_size:], axis=0)
    tail_speed = np.max(np.abs(np.rad2deg(velocities[-tail_size:])), axis=0)
    movement = target - start
    moving = np.flatnonzero(np.abs(movement) > 1e-10)
    overshoot = np.zeros(6)
    for i in moving:
        overshoot[i] = max(0.0, np.max(np.sign(movement[i]) * (positions[:, i] - np.rad2deg(target[i]))))
    unmoving = np.setdiff1d(np.arange(6), moving)
    other_motion = float(np.max(np.abs(positions[:, unmoving] - positions[0, unmoving]))) if unmoving.size else 0.0
    require(np.max(np.abs(errors)) <= MAX_ERROR_DEG, f"{label}: final error exceeds {MAX_ERROR_DEG} deg")
    require(np.max(tail_range) <= MAX_TAIL_RANGE_DEG, f"{label}: persistent oscillation")
    require(np.max(tail_speed) <= MAX_TAIL_SPEED_DEG_S, f"{label}: not settled")
    require(np.max(overshoot) <= MAX_ERROR_DEG, f"{label}: excessive overshoot")
    require(other_motion <= MAX_ERROR_DEG, f"{label}: uncommanded joint moved excessively")
    require(np.all(peak_torque <= np.max(np.abs(model.jnt_actfrcrange[joint_ids]), axis=1) + 1e-8),
            f"{label}: torque bound exceeded")
    for i in moving:
        require(np.sign(movement[i]) * (positions[-1, i] - positions[0, i]) > 0.5 * abs(np.rad2deg(movement[i])),
                f"{label}: commanded joint did not move toward target")
    return dict(label=label, target_deg=np.rad2deg(target).tolist(), actual_deg=positions[-1].tolist(),
                error_deg=errors.tolist(), tail_range_deg=tail_range.tolist(), tail_speed_deg_s=tail_speed.tolist(),
                overshoot_deg=overshoot.tolist(), peak_torque_nm=peak_torque.tolist(),
                peak_speed_deg_s=np.rad2deg(peak_speed).tolist(), max_contacts=contacts,
                min_contact_distance_m=min_distance, max_uncommanded_motion_deg=other_motion)


class ForceMonitor:
    """Check the step's total drive, including compensation, before cache refresh."""

    def __init__(self, model, joint_ids, actuator_ids):
        self.model, self.jids, self.aids = model, joint_ids, actuator_ids
        self.vadr = model.jnt_dofadr[joint_ids]
        require(np.all(model.jnt_actfrclimited[joint_ids]), "Joint total-drive limits must be active")
        if np.any(model.body_gravcomp):
            require(model.ngravcomp > 0 and np.all(model.jnt_actgravcomp[joint_ids]),
                            "Gravity compensation must be compiled and routed through joint actuator limits")
        self.limits = model.jnt_actfrcrange[joint_ids].copy()
        self.steps, self.saturated_steps = 0, 0
        self.peak_total = np.zeros(6)
        self.peak_servo = np.zeros(6)
        self.peak_gravity = np.zeros(6)
        self.peak_speed = np.zeros(6)

    def observe(self, data):
        total = data.qfrc_actuator[self.vadr]
        servo = data.actuator_force[self.aids]  # This model has one gear=1 servo per hinge.
        gc = data.qfrc_gravcomp[self.vadr]
        routed_gc = np.where(self.model.jnt_actgravcomp[self.jids], gc, 0)
        requested = servo + routed_gc
        expected = np.clip(requested, self.limits[:, 0], self.limits[:, 1])
        require(np.isfinite(np.concatenate([total, servo, gc, data.qfrc_passive])).all(), "Non-finite force channel")
        require(np.all(total >= self.limits[:, 0] - 1e-9)
                        and np.all(total <= self.limits[:, 1] + 1e-9), "Total joint drive exceeds torque limits")
        require(np.allclose(total, expected, rtol=0, atol=1e-9), "Unexpected drive composition or gravity routing")
        # Physical damping remains passive; gravity compensation must not bypass
        # motor limits as passive force or externally applied force.
        passive_mechanical = data.qfrc_damper + data.qfrc_spring + data.qfrc_fluid
        require(np.allclose(data.qfrc_passive, passive_mechanical, rtol=0, atol=1e-9),
                        "Unexpected passive force (possible gravity compensation bypass)")
        require(not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied), "External force injection is not allowed")
        self.steps += 1
        self.saturated_steps += int(np.max(np.abs(requested - total)) > 1e-9)
        self.peak_total = np.maximum(self.peak_total, np.abs(total))
        self.peak_servo = np.maximum(self.peak_servo, np.abs(servo))
        self.peak_gravity = np.maximum(self.peak_gravity, np.abs(gc))
        self.peak_speed = np.maximum(self.peak_speed, np.abs(data.qvel[self.vadr]))

    def summary(self):
        return dict(steps=self.steps, saturated_steps=self.saturated_steps,
                    joint_total_limits_nm=self.limits.tolist(), peak_total_drive_nm=self.peak_total.tolist(),
                    peak_servo_generated_nm=self.peak_servo.tolist(), peak_raw_gravity_compensation_nm=self.peak_gravity.tolist(),
                    peak_speed_deg_s=np.rad2deg(self.peak_speed).tolist())


class ControlCycleError(ValueError):
    """A fixed cycle stopped early; summary reports the steps actually taken."""

    def __init__(self, reason, message, summary):
        super().__init__(message)
        self.reason = reason
        self.summary = summary


def advance_synchronized_cycle(model, data, joint_ids, actuator_ids, target, gripper, *,
                               physics_steps=25, step_observer=None):
    """Arm + gripper references, ONE integration per tick, shared safety audit.

    Gripper owns its existing force/state checks and optional contact policy.
    No segment settling, no reset, no extra integration by either controller.
    Observers read the actual step's force cache before mj_forward.
    """
    require(gripper.model is model and gripper.data is data, 'Controllers must share model/data')
    require(gripper.arm_monitor is not None
            and np.array_equal(joint_ids, gripper.arm_monitor.jids)
            and np.array_equal(actuator_ids, gripper.arm_monitor.aids), 'Wrong arm mapping')
    require(isinstance(physics_steps, (int, np.integer)) and not isinstance(physics_steps, bool)
            and physics_steps > 0, 'physics_steps must be a positive integer')
    home = model.key_qpos[model.key('home').id, model.jnt_qposadr[joint_ids]]
    target = validate_target(model, joint_ids, home, target)
    start = data.ctrl[actuator_ids].copy()
    validate_target(model, joint_ids, home, start)
    command = start+np.clip(target-start, -np.deg2rad(.5), np.deg2rad(.5))
    initial_time = float(data.time)
    gripper._check_state()
    for index in range(physics_steps):
        alpha = (index+1)/physics_steps
        alpha = alpha*alpha*(3-2*alpha)
        data.ctrl[actuator_ids] = start+alpha*(command-start)
        gripper.update_control()
        mujoco.mj_step(model, data)
        gripper.observe_step()
        require(abs(data.time-(initial_time+(index+1)*model.opt.timestep)) < 1e-8,
                'Simulation time reset unexpectedly')
        if step_observer is not None:
            step_observer(data)
        mujoco.mj_forward(model, data)
        gripper._check_state()
    return dict(physics_steps=physics_steps, simulated_elapsed_s=float(data.time)-initial_time,
                commanded_joint_target_rad=command.tolist())


def advance_control_cycle(model, data, joint_ids, actuator_ids, target, *,
                          physics_steps=25, max_joint_delta=np.deg2rad(0.5),
                          max_joint_speed=np.deg2rad(30)):
    """Advance one bounded position-servo cycle, without resetting live state.

    Unlike run_segment, this has no settling phase or final convergence test.
    The *legal* target is rate-limited relative to the previous ctrl, then
    smoothstep-interpolated over exactly physics_steps (unless safety stops it).
    Invalid joint solutions are rejected, never clipped into position limits.
    Torque channels are audited before mj_forward refreshes the force cache.
    """
    require(isinstance(physics_steps, (int, np.integer)) and not isinstance(physics_steps, bool)
            and physics_steps > 0, "physics_steps must be a positive integer")
    require(np.isfinite(max_joint_delta) and max_joint_delta > 0, "Invalid joint command rate bound")
    require(np.isfinite(max_joint_speed) and max_joint_speed > 0, "Invalid joint speed bound")
    qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    limits = model.jnt_range[joint_ids]
    target = np.asarray(target, dtype=float)
    valid_target = target.shape == (6,) and np.isfinite(target).all()
    start = data.ctrl[actuator_ids].copy()
    monitor = ForceMonitor(model, joint_ids, actuator_ids)
    taken, contacts, min_distance = 0, 0, 0.0
    initial_time = float(data.time)
    command = start.copy()

    def summary():
        return dict(physics_steps=taken, simulated_elapsed_s=float(data.time)-initial_time,
                    requested_joint_target_rad=target.tolist() if valid_target else None,
                    commanded_joint_target_rad=command.tolist(),
                    command_rate_limited=bool(valid_target and np.any(np.abs(command-target) > 1e-12)),
                    max_contacts=contacts, min_contact_distance_m=min_distance, forces=monitor.summary())

    def fail(reason, message):
        raise ControlCycleError(reason, message, summary())

    def check_state():
        nonlocal contacts, min_distance
        for name in ("qpos", "qvel", "qacc", "ctrl", "xpos", "site_xpos", "qfrc_actuator", "actuator_force"):
            if not np.isfinite(getattr(data, name)).all():
                fail("numerical_failure", f"Non-finite {name}")
        if not np.isfinite(data.time) or any(warning.number for warning in data.warning):
            fail("numerical_failure", "MuJoCo numerical warning or invalid simulation time")
        q = data.qpos[qadr]
        if np.any(q < limits[:, 0]-1e-4) or np.any(q > limits[:, 1]+1e-4):
            fail("joint_limit", "Actual joint position exceeded the model limit tolerance")
        if np.any(np.abs(data.qvel[vadr]) > max_joint_speed):
            fail("velocity_limit", "Actual joint speed exceeded the experimental speed limit")
        contacts = max(contacts, data.ncon)
        for contact in data.contact:
            if not np.isfinite(contact.dist):
                fail("numerical_failure", "Non-finite contact distance")
            min_distance = min(min_distance, float(contact.dist))
            if contact.dist < -0.0001:
                names = [model.geom(int(i)).name for i in contact.geom]
                fail("collision", f"Unexpected collision {names}: distance={contact.dist:.6g} m")
        if np.any(data.qfrc_applied) or np.any(data.xfrc_applied):
            fail("drive_failure", "External force injection is not allowed")

    if target.shape != (6,) or not np.isfinite(target).all():
        fail("joint_limit", "Expected six finite joint target angles")
    if np.any(target < limits[:, 0]) or np.any(target > limits[:, 1]):
        fail("joint_limit", "Joint target outside position limits")
    if not np.isfinite(start).all() or np.any(start < limits[:, 0]) or np.any(start > limits[:, 1]):
        fail("joint_limit", "Previous control reference is invalid")
    # This bounds a valid command's rate; it does not repair an illegal IK result.
    command = start + np.clip(target-start, -max_joint_delta, max_joint_delta)
    check_state()
    try:
        for step in range(physics_steps):
            alpha = (step+1)/physics_steps
            alpha = alpha*alpha*(3-2*alpha)
            data.ctrl[actuator_ids] = start+alpha*(command-start)
            mujoco.mj_step(model, data)
            taken += 1
            check_state()
            try:
                monitor.observe(data)
            except ValueError as exc:
                fail("drive_failure", str(exc))
            if abs(data.time-(initial_time+taken*model.opt.timestep)) > 1e-8:
                fail("numerical_failure", "Simulation time reset unexpectedly")
            mujoco.mj_forward(model, data)
            check_state()
    except mujoco.FatalError as exc:
        fail("numerical_failure", str(exc))
    return summary()

