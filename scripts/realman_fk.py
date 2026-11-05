"""Optional, offline-only RM_API2 FK adapter. No robot connection or motion API.

RM_API2 Algo uses process-global native configuration. Only the most recently
created adapter may be used; keep configurations sequential, in one thread.
"""

from importlib import import_module, metadata
from pathlib import Path
import hashlib

import numpy as np

MODELS = {"II": "RM_MODEL_RM_63_II_E", "III": "RM_MODEL_RM_63_III_E"}
SENSORS = {name: f"RM_MODEL_RM_{name}_E" for name in ("B", "ZF", "SF", "ISF", "BV", "ISFV")}


class SDKUnavailable(RuntimeError):
    """The optional package or its native library is unavailable."""


def quaternion_matrix(wxyz):
    """Validate/normalize a scalar-first quaternion and return an active R."""
    q = np.asarray(wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError("Expected four finite quaternion components [w,x,y,z]")
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-12 or abs(norm - 1) > 1e-3:
        raise ValueError("SDK quaternion is not approximately unit length")
    w, x, y, z = q / norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def euler_matrix(rpy):
    """Fixed-axis XYZ roll/pitch/yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    values = np.asarray(rpy, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected three finite Euler angles in radians")
    r, p, y = values
    cr, cp, cy = np.cos([r, p, y])
    sr, sp, sy = np.sin([r, p, y])
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


class RealManFK:
    _active = None

    def __init__(self, model="II", sensor="SF"):
        if model not in MODELS or sensor not in SENSORS:
            raise ValueError("Unsupported model or sensor candidate")
        try:
            sdk = import_module("Robotic_Arm.rm_robot_interface")
        except (ImportError, OSError) as exc:
            raise SDKUnavailable("Optional Robotic_Arm SDK unavailable; install requirements-vendor.txt "
                                 f"in realman_rl. Detail: {exc}") from exc
        self._sdk = sdk
        arm = getattr(sdk.rm_robot_arm_model_e, MODELS[model])
        force = getattr(sdk.rm_force_type_e, SENSORS[sensor])
        self._algo = sdk.Algo(arm, force)
        # Official standalone Algo creates a zero-valued handle, never connects.
        if self._algo.handle.id != 0 or self._algo.arm_dof != 6:
            raise RuntimeError("Expected standalone six-axis Algo with handle id=0")
        RealManFK._active = self
        defaults = self._frames()
        self._algo.rm_algo_set_angle(0, 0, 0)
        self._algo.rm_algo_set_workframe(sdk.rm_frame_t("base", [0.0]*6))
        self._algo.rm_algo_set_toolframe(sdk.rm_frame_t("flange", [0.0]*6))
        configured = self._frames()
        if (not np.array_equal(configured["installation_deg"], [0, 0, 0])
                or any(not np.array_equal(configured[key]["pose"], [0]*6)
                       for key in ("work", "tool"))):
            raise RuntimeError("Could not establish identity installation/work/tool frames")
        dh = self._algo.rm_algo_get_dh()
        if set(dh) != {"d", "a", "alpha", "offset"} or any(
                np.asarray(value).shape != (6,) or not np.isfinite(value).all() for value in dh.values()):
            raise ValueError("Invalid six-axis DH parameters returned by SDK")
        package = Path(sdk.__file__).parent
        files = [package / "rm_robot_interface.py", package / "rm_ctypes_wrap.py"]
        # Record the actually loaded native library, not another architecture's file.
        wrap = import_module("Robotic_Arm.rm_ctypes_wrap")
        native = Path(wrap.dll_path) / wrap.libname
        files.append(native)
        self.info = dict(sdk_version=metadata.version("Robotic_Arm"), api_version=str(sdk.rm_api_version()),
                         algorithm_version=str(self._algo.rm_algo_version()), model=arm.name, model_value=int(arm),
                         sensor=force.name, sensor_value=int(force), sensor_mapping_confirmed=False,
                         dh=dh, dh_units=dict(d="m", a="m", alpha="deg", offset="deg"),
                         default_frames=defaults, configured_frames=configured, handle_id=0,
                         package_path=str(package), file_sha256={str(p.relative_to(package)):
                         hashlib.sha256(p.read_bytes()).hexdigest() for p in files})

    def _frames(self):
        return dict(installation_deg=list(self._algo.rm_algo_get_angle()),
                    work=self._algo.rm_algo_get_curr_workframe(),
                    tool=self._algo.rm_algo_get_curr_toolframe())

    def _check_active(self):
        if RealManFK._active is not self:
            raise RuntimeError("Another Algo configuration replaced this process-global session")

    def fk(self, q_rad):
        """Six radians -> (base position [m], flange-to-base R [3,3]).

        Limits are checked against the MuJoCo model by the comparison caller.
        This adapter has no joint commands and never advances a simulation.
        """
        self._check_active()
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint angles in radians")
        degrees = np.rad2deg(q)
        if not np.isfinite(degrees).all() or np.max(np.abs(degrees)) > 1e6:
            raise ValueError("Joint angle cannot be represented safely by the SDK")
        pose = np.asarray(self._algo.rm_algo_forward_kinematics(degrees.tolist(), flag=0), dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("Invalid FK pose returned by SDK")
        return pose[:3].copy(), quaternion_matrix(pose[3:])

    def euler_fk(self, q_rad):
        """Independent representation audit; comparison itself uses quaternions."""
        self._check_active()
        q = np.asarray(q_rad, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite joint angles in radians")
        pose = np.asarray(self._algo.rm_algo_forward_kinematics(np.rad2deg(q).tolist(), flag=1))
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError("Invalid Euler FK pose returned by SDK")
        return pose[:3].copy(), euler_matrix(pose[3:])
