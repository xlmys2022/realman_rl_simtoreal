"""MuJoCo site FK and local, full-pose damped least-squares IK (SI units)."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: np.ndarray | None  # Six radians on success; never an executable candidate on failure.
    iterations: int
    position_error: float | None  # metres
    orientation_error: float | None  # radians, shortest rotation angle
    reason: str | None = None


def rotation_vector(target, current):
    """Log(target @ current.T), in the world frame used by mj_jacSite's jacr."""
    quat, vector = np.empty(4), np.empty(3)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(target @ current.T).reshape(9))
    if quat[0] < 0:  # Choose the shortest quaternion representative, including near pi.
        quat *= -1
    mujoco.mju_quat2Vel(vector, quat, 1.0)
    return vector


class Kinematics:
    """Owns private MjData; never reads or writes a live simulation's state.

    q is in joint_1..joint_6 order. Rotation matrices map site-frame vectors to
    world coordinates. A single instance is intended for one calling thread.
    """

    def __init__(self, model):
        self.model = model
        self.joint_ids = np.array([model.joint(f"joint_{i}").id for i in range(1, 7)])
        if (model.nq < 6 or model.nv < 6
                or not np.all(model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_HINGE)
                or not np.all(model.jnt_limited[self.joint_ids])):
            raise ValueError("Kinematics requires six named, limited arm hinges")
        self.qadr = model.jnt_qposadr[self.joint_ids].copy()
        self.vadr = model.jnt_dofadr[self.joint_ids].copy()
        self.limits = model.jnt_range[self.joint_ids].copy()
        try:
            self.site_id = model.site("ee_site").id
        except KeyError as exc:
            raise ValueError("Missing ee_site; rerun build_dynamics_model.py") from exc
        if model.site_bodyid[self.site_id] != model.body("link_6").id:
            raise ValueError("ee_site must be attached to link_6")
        self._data = mujoco.MjData(model)
        self.home_q = model.key_qpos[model.key("home").id, self.qadr].copy()
        self.home_position, self.home_rotation = self.fk(self.home_q)
        # Triangle inequality along the fixed-base chain: conservative reach
        # sphere about the world origin, not an assertion of actual workspace.
        body = int(model.site_bodyid[self.site_id])
        self.reach_bound = float(np.linalg.norm(model.site_pos[self.site_id]))
        while body:
            self.reach_bound += float(np.linalg.norm(model.body_pos[body]))
            body = int(model.body_parentid[body])

    def _angles(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint angles in radians")
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError("Joint angle outside position limits")
        return q.copy()

    def _forward(self, q):
        # This is private calculation data, never the dynamics MjData.
        self._data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self._data)
        return (self._data.site_xpos[self.site_id].copy(),
                self._data.site_xmat[self.site_id].reshape(3, 3).copy())

    def fk(self, q):
        """Return (world position [m], site-to-world rotation [3,3]), as copies."""
        return self._forward(self._angles(q))

    def jacobian(self, q):
        """Return world-frame translational and angular Jacobians, each [3,6]."""
        self._forward(self._angles(q))
        jacp = np.empty((3, self.model.nv))
        jacr = np.empty_like(jacp)
        mujoco.mj_jacSite(self.model, self._data, jacp, jacr, self.site_id)
        return jacp[:, self.vadr].copy(), jacr[:, self.vadr].copy()

    def solve_ik(self, target_position, initial_q, target_rotation=None, *, max_iterations=150,
                 position_tolerance=1e-5, orientation_tolerance=1e-4,
                 rotation_weight=0.2, damping=0.01, max_step=0.05, seed_weight=1e-4):
        """Solve a full pose; missing target_rotation means the home orientation.

        Bounded local solve, without random restarts or angular wrapping. The
        seed regularizer weakly prefers nearby configurations. Failure does not
        prove global unreachability, except for the conservative distance bound.
        """
        iteration, pos_error, rot_error = 0, None, None
        try:
            q = self._angles(initial_q)
            seed = q.copy()
            target = np.asarray(target_position, dtype=float)
            rotation = self.home_rotation.copy() if target_rotation is None else np.asarray(target_rotation, dtype=float)
            if target.shape != (3,) or not np.isfinite(target).all():
                raise ValueError("Target position must be three finite metres")
            if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
                    or not np.isclose(np.linalg.det(rotation), 1, atol=1e-8, rtol=0)):
                raise ValueError("Target rotation must be a proper orthonormal 3x3 matrix")
            if not isinstance(max_iterations, (int, np.integer)) or not 0 <= max_iterations <= 1000:
                raise ValueError("max_iterations must be an integer in [0, 1000]")
            params = np.array([position_tolerance, orientation_tolerance, rotation_weight, damping, max_step, seed_weight])
            if not np.isfinite(params).all() or np.any(params[:5] <= 0) or seed_weight < 0 or damping < 1e-6:
                raise ValueError("Invalid IK tolerances, weights, damping or step bound")
            if np.max(np.abs(target)) > self.reach_bound or np.linalg.norm(target) > self.reach_bound:
                return IKResult(False, None, 0, None, None, "Target exceeds conservative reach bound")

            def evaluate(angles):
                position, actual_rotation = self._forward(angles)
                ep = target - position
                er = rotation_vector(rotation, actual_rotation)
                e = np.concatenate([ep, rotation_weight * er])
                cost = e @ e + seed_weight**2 * np.sum((angles - seed)**2)
                if not np.isfinite(e).all() or not np.isfinite(cost):
                    raise ValueError("Non-finite IK residual or cost")
                return e, float(np.linalg.norm(ep)), float(np.linalg.norm(er)), float(cost)

            for iteration in range(max_iterations + 1):
                error, pos_error, rot_error, cost = evaluate(q)
                if pos_error <= position_tolerance and rot_error <= orientation_tolerance:
                    return IKResult(True, q.copy(), iteration, pos_error, rot_error)
                if iteration == max_iterations:
                    break
                jp, jr = self.jacobian(q)
                jac = np.vstack([jp, rotation_weight * jr])
                hessian = jac.T @ jac + (damping**2 + seed_weight**2) * np.eye(6)
                rhs = jac.T @ error + seed_weight**2 * (seed - q)
                step = np.linalg.solve(hessian, rhs)
                if not np.isfinite(step).all():
                    raise ValueError("Non-finite IK step")
                step *= min(1.0, max_step / max(float(np.max(np.abs(step))), 1e-30))
                accepted = False
                for scale in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125):
                    candidate = np.clip(q + scale * step, self.limits[:, 0], self.limits[:, 1])
                    if np.max(np.abs(candidate - q)) < 1e-12:
                        continue
                    if evaluate(candidate)[3] < cost - 1e-18:
                        q, accepted = candidate, True
                        break
                if not accepted:
                    return IKResult(False, None, iteration, pos_error, rot_error,
                                    "Stalled: singular, joint-limited or locally unreachable target")
            return IKResult(False, None, iteration, pos_error, rot_error, "Maximum iterations reached")
        except (ValueError, TypeError, OverflowError, FloatingPointError, np.linalg.LinAlgError, mujoco.FatalError) as exc:
            return IKResult(False, None, iteration, pos_error, rot_error, str(exc))
