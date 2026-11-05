"""Checked offline RM_API2 IK. Results are calculations, never motion commands.

Shares RealManFK's single active native Algo configuration. No parallel sessions,
DH edits, robot connection, actuator writes or automatic solver fallbacks.
"""

from dataclasses import dataclass, replace
from time import perf_counter

import mujoco
import numpy as np

from realman_fk import RealManFK
from kinematics import rotation_vector

POSITION_TOLERANCE_M = 1e-5
ORIENTATION_TOLERANCE_RAD = 1e-4
RETURN_CODES = {0: "success", 1: "SDK inverse kinematics failed",
                -1: "SDK invalid/empty reference joints (all-solve also documents limit violation)",
                -2: "SDK invalid target quaternion"}


@dataclass(frozen=True)
class VendorIKResult:
    success: bool
    q: np.ndarray | None  # Six radians, only after range AND independent FK checks.
    sdk_code: int | None
    reason: str | None
    position_error_m: float | None = None
    orientation_error_rad: float | None = None
    within_vendor_limits: bool | None = None
    within_model_limits: bool | None = None
    raw_within_vendor_limits: bool | None = None
    raw_within_model_limits: bool | None = None
    periodic_turns: tuple[int, ...] | None = None
    elapsed_s: float | None = None  # Complete checked single call; None for all-solve candidates.
    native_elapsed_s: float | None = None  # SDK method call including ctypes, not per-branch time.


@dataclass(frozen=True)
class AllIKResult:
    success: bool
    sdk_code: int | None
    reported_count: int
    candidates: tuple[VendorIKResult, ...]
    reason: str | None
    elapsed_s: float
    native_elapsed_s: float


def quaternion_pose(position, rotation):
    """SI position + proper active rotation -> [x,y,z,w,x,y,z], wxyz unit q."""
    p, r = np.asarray(position, dtype=float), np.asarray(rotation, dtype=float)
    if p.shape != (3,) or not np.isfinite(p).all() or np.max(np.abs(p)) > 1e6:
        raise ValueError("Expected three finite metres, magnitude <=1e6 (numeric input guard)")
    if (r.shape != (3, 3) or not np.isfinite(r).all()
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-8, rtol=0)):
        raise ValueError("Expected a proper orthonormal 3x3 target rotation")
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(r).reshape(9))
    quat /= np.linalg.norm(quat)
    if quat[0] < 0:
        quat *= -1
    if not np.isfinite(quat).all() or not np.isclose(quat @ quat, 1, atol=1e-12, rtol=0):
        raise ValueError("Could not encode a unit wxyz target quaternion")
    return np.concatenate([p, quat]).tolist()


def periodic_equivalent(q, seed, limits):
    """Choose nearest seed representation using ONLY integer 2*pi turns.

    Never clip joints or add a calibration/zero offset. None means no equivalent
    angle lies within the supplied limits. FK must still verify the result.
    """
    q, seed, limits = (np.asarray(value, dtype=float) for value in (q, seed, limits))
    out, turns = q.copy(), []
    for i, angle in enumerate(q):
        low = int(np.ceil((limits[i, 0] - angle) / (2*np.pi)))
        high = int(np.floor((limits[i, 1] - angle) / (2*np.pi)))
        if low > high:
            return None, None
        k = min(range(low, high+1), key=lambda n: abs(angle + 2*np.pi*n - seed[i]))
        out[i] = angle + 2*np.pi*k
        turns.append(k)
    return out, tuple(turns)


class RealManIK(RealManFK):
    def __init__(self, model_limits, model="II", sensor="SF"):
        limits = np.asarray(model_limits, dtype=float)
        if limits.shape != (6, 2) or not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Expected six ordered finite model joint ranges in radians")
        super().__init__(model, sensor)
        self.model_limits = limits.copy()
        self.vendor_limits = np.deg2rad(np.column_stack([
            self._algo.rm_algo_get_joint_min_limit(), self._algo.rm_algo_get_joint_max_limit()]))
        if (self.vendor_limits.shape != (6, 2) or not np.isfinite(self.vendor_limits).all()
                or np.any(self.vendor_limits[:, 0] >= self.vendor_limits[:, 1])):
            raise ValueError("Invalid joint limits read from the manufacturer Algo")
        self.limits = np.column_stack([np.maximum(limits[:, 0], self.vendor_limits[:, 0]),
                                      np.minimum(limits[:, 1], self.vendor_limits[:, 1])])
        if np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError("Manufacturer and model joint ranges do not overlap")
        self._algo.rm_algo_set_redundant_parameter_traversal_mode(True)
        self.info.update(ik_traversal_mode=True, vendor_limits_rad=self.vendor_limits.tolist(),
                         model_limits_rad=self.model_limits.tolist(), effective_limits_rad=self.limits.tolist(),
                         position_tolerance_m=POSITION_TOLERANCE_M,
                         orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD)

    @staticmethod
    def _inside(q, limits):
        return bool(np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1]))

    def _params(self, position, rotation, seed):
        self._check_active()
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (6,) or not np.isfinite(seed).all() or not self._inside(seed, self.limits):
            raise ValueError("Initial joints must be six finite radians within BOTH joint ranges")
        pose = quaternion_pose(position, rotation)
        return self._sdk.rm_inverse_kinematics_params_t(np.rad2deg(seed).tolist(), pose, 0), seed.copy()

    def _check_solution(self, raw_degrees, position, rotation, seed):
        q = np.asarray(raw_degrees, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all() or np.max(np.abs(q)) > 1e6:
            return VendorIKResult(False, None, 0, "SDK success returned invalid joint count or values")
        raw = np.deg2rad(q)
        rv, rm = self._inside(raw, self.vendor_limits), self._inside(raw, self.model_limits)
        normalized, turns = periodic_equivalent(raw, seed, self.limits)
        # Audit every geometrically meaningful candidate, even one rejected by
        # limits. This remains calculation-only FK; no invalid q is exposed.
        p, r = self.fk(raw if normalized is None else normalized)
        ep = float(np.linalg.norm(np.asarray(position) - p))
        er = float(np.linalg.norm(rotation_vector(np.asarray(rotation), r)))
        v_ok = self._inside(raw, self.vendor_limits) if normalized is None else self._inside(normalized, self.vendor_limits)
        m_ok = self._inside(raw, self.model_limits) if normalized is None else self._inside(normalized, self.model_limits)
        reason = None
        if not np.isfinite([ep, er]).all() or ep > POSITION_TOLERANCE_M or er > ORIENTATION_TOLERANCE_RAD:
            reason = "SDK success failed independent manufacturer FK pose audit"
        elif normalized is None:
            reason = "No periodic equivalent satisfies BOTH manufacturer/model joint limits"
        return VendorIKResult(reason is None, normalized.copy() if reason is None else None, 0, reason,
                              ep, er, v_ok, m_ok, rv, rm, turns)

    def solve_ik(self, target_position, initial_q, target_rotation):
        """Same SI argument order as Kinematics.solve_ik; mandatory full rotation."""
        start, native_time, code = perf_counter(), 0.0, None
        try:
            params, seed = self._params(target_position, target_rotation, initial_q)
            native_start = perf_counter()
            code, raw = self._algo.rm_algo_inverse_kinematics(params)
            native_time = perf_counter() - native_start
            if code != 0:
                # SDK fills q_out even on failure. Deliberately discard it.
                result = VendorIKResult(False, None, int(code), RETURN_CODES.get(code, f"Unknown SDK code {code}"))
            else:
                result = self._check_solution(raw, target_position, target_rotation, seed)
        except (ValueError, TypeError, OverflowError, RuntimeError, mujoco.FatalError) as exc:
            result = VendorIKResult(False, None, None if code is None else int(code), str(exc))
        return replace(result, elapsed_s=perf_counter()-start, native_elapsed_s=native_time)

    def solve_all(self, target_position, initial_q, target_rotation):
        """Audit all native branches. Rejected candidates have q=None, not clipped q."""
        start, native_time, code, count = perf_counter(), 0.0, None, 0
        candidates, reason = (), None
        try:
            params, seed = self._params(target_position, target_rotation, initial_q)
            native_start = perf_counter()
            native = self._algo.rm_algo_inverse_kinematics_all(params)
            native_time = perf_counter()-native_start
            code, count = int(native.result), int(native.num)
            if code != 0:
                reason = RETURN_CODES.get(code, f"Unknown SDK code {code}")
            elif not 1 <= count <= 8:
                reason = "SDK success returned invalid all-solution count (expected 1..8)"
            else:
                candidates = tuple(self._check_solution(list(native.q_solve[i])[:6], target_position,
                                                        target_rotation, seed) for i in range(count))
                if not any(c.success for c in candidates):
                    reason = "No all-solution candidate passed joint ranges and FK audit"
        except (ValueError, TypeError, OverflowError, RuntimeError, mujoco.FatalError) as exc:
            reason = str(exc)
        return AllIKResult(reason is None, code, count, candidates, reason, perf_counter()-start, native_time)
