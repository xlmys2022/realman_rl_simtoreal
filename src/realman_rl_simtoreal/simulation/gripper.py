"""Physical EG2-4C2 width commands for the upstream six-hinge mimic mechanism.

The model's gap is 0..66.217mm, not the product's nominal 70mm. Dynamics and
the experimental mounting fixture are uncalibrated; this is not a force servo.
"""

import mujoco
import numpy as np

from .model import JOINTS, require
from .control import ForceMonitor

SIGNS = {'4C2_Joint1': 1, '4C2_Joint2': 1, '4C2_Joint3': -1,
         '4C2_Joint4': -1, '4C2_Joint5': 1, '4C2_Joint6': 1}
LINK_X, LINK_Z, MAX_ANGLE = .0223, .035591, .82


def angle_to_width(angle):
    angle = float(angle)
    require(np.isfinite(angle) and 0 <= angle <= MAX_ANGLE, 'Gripper angle outside [0, .82] rad')
    return float(2*(LINK_X*(1-np.cos(angle))+LINK_Z*np.sin(angle)))


MAX_WIDTH = angle_to_width(MAX_ANGLE)


def width_to_angle(width):
    width = float(width)
    require(np.isfinite(width) and 0 <= width <= MAX_WIDTH,
            f'Width must be in [0, {MAX_WIDTH:.12g}] m; upstream model does not reach 70 mm')
    if width in (0, MAX_WIDTH):
        return 0.0 if width == 0 else MAX_ANGLE
    phi = np.arctan2(LINK_X, LINK_Z)
    return float(phi+np.arcsin((width/2-LINK_X)/np.hypot(LINK_X, LINK_Z)))


class Gripper:
    """Control only eg2_position; arm ctrl entries are never assigned.

    Call initialize(model) before constructing. step advances shared physics,
    so the arm evolves too, under its existing control references. set_width,
    open and close only select a target; normal motion never assigns qpos.
    """

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.joint_ids = np.array([model.joint(name).id for name in SIGNS])
        self.qadr = model.jnt_qposadr[self.joint_ids]
        self.vadr = model.jnt_dofadr[self.joint_ids]
        self.signs = np.array(list(SIGNS.values()))
        self.primary = model.joint('4C2_Joint1').id
        self.actuator_id = model.actuator('eg2_position').id
        self.base = model.body('4C2_baselink').id
        self.left, self.right = model.site('eg2_left_pad').id, model.site('eg2_right_pad').id
        require(np.all(model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_HINGE)
                and np.all(model.jnt_limited[self.joint_ids]), 'Expected six limited gripper hinges')
        self.equality_ids = []
        for joint_id, sign in zip(self.joint_ids, self.signs):
            expected = [0, MAX_ANGLE] if sign == 1 else [-MAX_ANGLE, 0]
            require(np.array_equal(model.jnt_range[joint_id], expected), 'Unexpected gripper limits')
            if joint_id != self.primary:
                matching = [i for i in range(model.neq) if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT
                            and model.eq_obj1id[i] == joint_id and model.eq_obj2id[i] == self.primary
                            and np.array_equal(model.eq_data[i, :5], [0, sign, 0, 0, 0])
                            and model.eq_active0[i]]
                require(len(matching) == 1, 'Missing or ambiguous gripper mimic constraint')
                self.equality_ids.append(matching[0])
        aid = self.actuator_id
        require(model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[aid, 0] == self.primary
                and np.array_equal(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]), 'Wrong gripper actuator mapping')
        require(model.actuator_ctrllimited[aid] and model.actuator_forcelimited[aid]
                and model.jnt_actfrclimited[self.primary], 'Gripper limits must be active')
        if np.any(model.body_gravcomp):
            require(model.ngravcomp > 0 and np.all(model.jnt_actgravcomp[self.joint_ids]),
                    'Gripper gravity compensation must be compiled and routed to actuator limits')
        self._probe = mujoco.MjData(model)
        self._target_angle = float(data.ctrl[aid])
        self.target_width = angle_to_width(self._target_angle)
        self.arm_monitor = None
        if all(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0 for name in JOINTS):
            jids = np.array([model.joint(name).id for name in JOINTS])
            aids = np.array([model.actuator(f'position_{i}').id for i in range(1, 7)])
            self.arm_monitor = ForceMonitor(model, jids, aids)
            ancestor = self.base
            while ancestor and model.body(ancestor).name != 'link_6':
                require(model.body_jntnum[ancestor] == 0, 'Gripper base must be rigidly attached')
                ancestor = int(model.body_parentid[ancestor])
            require(ancestor != 0, 'Gripper base must descend from link_6')
        self.steps = 0
        self.peak_drive = 0.0
        self.peak_speed = 0.0
        self.max_mimic_error = 0.0
        self.min_contact_distance = 0.0

    def set_width(self, width_m):
        self._target_angle = width_to_angle(width_m)
        self.target_width = float(width_m)

    def open(self):
        self.set_width(MAX_WIDTH)

    def close(self):
        self.set_width(0)

    def pad_positions(self):
        """Read actual pad sites using independent calculation data (no live cache writes)."""
        require(np.isfinite(self.data.qpos).all(), 'Non-finite gripper calculation state')
        self._probe.qpos[:] = self.data.qpos
        mujoco.mj_forward(self.model, self._probe)
        return self._probe.site_xpos[self.left].copy(), self._probe.site_xpos[self.right].copy()

    def get_width(self):
        left, right = self.pad_positions()
        axis = self._probe.xmat[self.base].reshape(3, 3)[:, 0]
        return float((right-left) @ axis)  # Signed gap; tiny contact compression is reported.

    def _check_state(self):
        m, d = self.model, self.data
        require(np.all(d.eq_active[self.equality_ids]), 'Gripper mimic constraints must remain active')
        for name in ('qpos', 'qvel', 'qacc', 'ctrl', 'qfrc_actuator', 'actuator_force', 'site_xpos'):
            require(np.isfinite(getattr(d, name)).all(), f'Non-finite {name}')
        require(np.isfinite(d.time) and not any(w.number for w in d.warning), 'MuJoCo numerical warning/time failure')
        require(not np.any(d.qfrc_applied) and not np.any(d.xfrc_applied), 'External force injection is not allowed')
        if self.arm_monitor is not None:
            arm_q = d.qpos[m.jnt_qposadr[self.arm_monitor.jids]]
            arm_limits = m.jnt_range[self.arm_monitor.jids]
            require(np.all(arm_q >= arm_limits[:, 0]-1e-4) and np.all(arm_q <= arm_limits[:, 1]+1e-4),
                    'Arm position limit exceeded')
            require(np.max(np.abs(d.qvel[self.arm_monitor.vadr])) <= np.deg2rad(30), 'Arm speed limit exceeded')
        q = d.qpos[self.qadr]
        ranges = m.jnt_range[self.joint_ids]
        require(np.all(q >= ranges[:, 0]-1e-4) and np.all(q <= ranges[:, 1]+1e-4), 'Gripper position limit exceeded')
        speed = float(np.max(np.abs(d.qvel[self.vadr])))
        require(speed <= 1.001, 'Gripper speed exceeds the upstream 1 rad/s bound')
        self.peak_speed = max(self.peak_speed, speed)
        error = float(np.max(np.abs(q-self.signs*q[0])))
        self.max_mimic_error = max(self.max_mimic_error, error)
        require(error <= .001, 'Gripper mimic constraint drift exceeds .001 rad')
        for contact in d.contact:
            self.min_contact_distance = min(self.min_contact_distance, float(contact.dist))
            require(np.isfinite(contact.dist) and contact.dist >= -.0001,
                    f'Unexpected collision: {tuple(m.geom(int(i)).name for i in contact.geom)}, {contact.dist:.6g}m')

    def step(self, physics_steps=1):
        require(isinstance(physics_steps, (int, np.integer)) and not isinstance(physics_steps, bool)
                and physics_steps > 0, 'physics_steps must be a positive integer')
        m, d, aid = self.model, self.data, self.actuator_id
        self._check_state()
        expected_time = float(d.time)
        for _ in range(physics_steps):
            current = float(d.ctrl[aid])
            require(np.isfinite(current) and 0 <= current <= MAX_ANGLE, 'Invalid gripper control reference')
            # Conservative .5rad/s reference slew, below the URDF 1rad/s field.
            d.ctrl[aid] = current+np.clip(self._target_angle-current, -.5*m.opt.timestep, .5*m.opt.timestep)
            mujoco.mj_step(m, d)
            expected_time += m.opt.timestep
            require(abs(d.time-expected_time) < 1e-8, 'Simulation time reset unexpectedly')
            if self.arm_monitor is not None:
                self.arm_monitor.observe(d)
            else:
                require(np.allclose(d.qfrc_passive, d.qfrc_damper+d.qfrc_spring+d.qfrc_fluid, rtol=0, atol=1e-9),
                        'Unexpected passive force bypass')
            self._check_state()
            total = d.qfrc_actuator[m.jnt_dofadr[self.primary]]
            limits = m.jnt_actfrcrange[self.primary]
            gravity = d.qfrc_gravcomp[m.jnt_dofadr[self.primary]] if m.jnt_actgravcomp[self.primary] else 0
            expected = np.clip(d.actuator_force[aid]+gravity, *limits)
            require(limits[0]-1e-9 <= total <= limits[1]+1e-9 and abs(total-expected) < 1e-9,
                    'Gripper total drive composition/limit failure')
            self.peak_drive = max(self.peak_drive, abs(float(total)))
            self.steps += 1
            mujoco.mj_forward(m, d)
            self._check_state()
        return self.get_width()

    def statistics(self):
        return dict(physics_steps=self.steps, target_width_m=self.target_width, actual_width_m=self.get_width(),
                    peak_primary_total_drive_nm=self.peak_drive, peak_joint_speed_rad_s=self.peak_speed,
                    max_mimic_error_rad=self.max_mimic_error, min_contact_distance_m=self.min_contact_distance,
                    arm_forces=None if self.arm_monitor is None else self.arm_monitor.summary())
