"""RML63II-6FB 和控制器内置 EG2-4C2 接口，连接和查询均为只读。"""

from ctypes import byref
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from ipaddress import IPv4Address
from threading import Lock, RLock
from time import monotonic, sleep

import numpy as np

from ..paths import ROBOT_MODEL
from ..utils.geometry import euler_matrix, modified_dh_transform, transform
from . import HardwareError, HardwareTimeout
from .realman_config import REALMAN_HARDWARE
from .safety import MotionLimits, MotionSafetyError


@dataclass(frozen=True)
class ArmState:
    joint_deg: np.ndarray
    controller_pose_m_rad: np.ndarray
    T_base_flange: np.ndarray
    error_codes: tuple[int, ...]
    request_start_s: float
    request_end_s: float

    @property
    def timestamp_s(self):
        """控制器查询收发中点，使用主机单调时钟，单位秒。"""
        return (self.request_start_s + self.request_end_s) / 2


@dataclass(frozen=True)
class GripperState:
    online: bool
    enabled: bool
    opening_raw: int
    force_g: int
    temperature_c: int
    mode: int
    error: int
    timestamp_s: float


def _check_status(operation, status):
    if status == -2:
        raise HardwareTimeout(f"{operation} 接收失败或超时（SDK {status}）")
    if status != 0:
        raise HardwareError(f"{operation} 失败（SDK {status}）")


@lru_cache(maxsize=1)
def _load_sdk():
    try:
        sdk = import_module("Robotic_Arm.rm_robot_interface")
    except (ImportError, OSError) as exc:
        raise HardwareError(f"无法加载 RealMan SDK，请安装 requirements-vendor.txt · {exc}") from exc
    # 单线程模式逐条等待回执，由适配器管理采样与运动超时。
    _check_status("初始化 RealMan SDK", sdk.rm_init(sdk.rm_thread_mode_e.RM_SINGLE_MODE_E))
    return sdk


def _six(values, label):
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise HardwareError(f"{label} 需要六个有限数值") from exc
    if array.shape != (6,) or not np.isfinite(array).all():
        raise HardwareError(f"{label} 需要六个有限数值")
    return array


class RealManArm:
    """with 块管理只读连接。运动须显式启用并提供现场 MotionLimits。"""

    # SDK 的返回超时是进程级设置，一个进程只维护一个受管连接。
    _session_lock = Lock()

    def __init__(self, ip, *, port=8080, timeout_ms=500, enable_motion=False, limits=None):
        self.ip = str(IPv4Address(ip))
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("控制器端口必须为 1–65535")
        if type(timeout_ms) is not int or not 1 <= timeout_ms <= 60_000:
            raise ValueError("控制器通信超时必须为 1–60000 ms")
        if enable_motion is True and not isinstance(limits, MotionLimits):
            raise ValueError("启用运动需要显式提供现场 MotionLimits")
        self.port = port
        self.timeout_ms = timeout_ms
        self._motion_enabled = enable_motion is True
        self._limits = limits
        self._lock = RLock()
        self._owns_session = False
        self._connected = False
        self._client = None
        self._faulted = False
        self._mdh = None
        self.identity = None

    def __enter__(self):
        with self._lock:
            if not self._session_lock.acquire(blocking=False):
                raise HardwareError("本进程已有 RealMan 连接")
            self._owns_session = True
            try:
                sdk = _load_sdk()
                self._client = sdk.RoboticArm()
                self._client.rm_set_timeout(self.timeout_ms)
                try:
                    handle = self._client.rm_create_robot_arm(self.ip, self.port)
                except (ValueError, OSError) as exc:
                    raise HardwareError(f"连接 RealMan {self.ip}:{self.port} 失败 · {exc}") from exc
                if handle.id <= 0:
                    raise HardwareError(f"连接 RealMan {self.ip}:{self.port} 失败（句柄 {handle.id}）")
                self._connected = True
                self.identity = self._read_identity(sdk)
                self._require_real_mode()
                dh = self._read("rm_get_DH_data")
                if not isinstance(dh, dict):
                    raise HardwareError("控制器 DH 回包格式无效")
                # 使用本次从控制器读回的 DH，避免依赖切换中的工具或工作坐标系。
                self._mdh = np.column_stack([
                    _six(dh.get("a"), "控制器 DH a"), _six(dh.get("d"), "控制器 DH d"),
                    np.deg2rad(_six(dh.get("alpha"), "控制器 DH alpha")),
                    np.deg2rad(_six(dh.get("offset"), "控制器 DH offset")),
                ])
            except BaseException as exc:
                self.__exit__(type(exc), exc, exc.__traceback__)
                raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as close_error:
            if exc is None:
                raise
            exc.add_note(str(close_error))

    def close(self):
        with self._lock:
            if not self._owns_session:
                return
            try:
                if self._client is not None and getattr(self._client, "handle", None):
                    _check_status("断开 RealMan", self._client.rm_delete_robot_arm())
            finally:
                self._connected = False
                self._client = None
                self._owns_session = False
                self._session_lock.release()

    def _read_identity(self, sdk):
        # SDK 的字符串 RML_63 合并了多代型号，需核对原始枚举。
        info = sdk.rm_robot_info_t()
        _check_status("rm_get_robot_info", sdk.rm_get_robot_info(self._client.handle, byref(info)))
        identity = {
            "arm_model_value": info.arm_model,
            "force_type_value": info.force_type,
            "arm_dof": info.arm_dof,
            "controller_generation": info.robot_controller_version,
        }
        expected = REALMAN_HARDWARE
        if (info.arm_model != expected.sdk_arm_model_value
                or info.force_type != int(sdk.rm_force_type_e.RM_MODEL_RM_ISF_E)
                or info.arm_dof != expected.arm_dof
                or info.robot_controller_version != expected.controller_generation):
            raise HardwareError(f"控制器与项目已核实的 {ROBOT_MODEL} 构型不匹配 · {identity}")
        return {"model": ROBOT_MODEL, **identity}

    def _require_real_mode(self):
        if self._read("rm_get_arm_run_mode") != 1:
            raise HardwareError("控制器未处于真实模式，请在示教器检查当前模式")

    def _read(self, operation):
        if not self._connected:
            raise HardwareError("RealMan 未连接")
        status, data = getattr(self._client, operation)()
        _check_status(operation, status)
        return data

    def read_frames(self):
        """读取控制器当前工作系和工具系配置，保留 SDK 的 m/rad 表示。"""
        with self._lock:
            return {
                "work": self._read("rm_get_current_work_frame"),
                "tool": self._read("rm_get_current_tool_frame"),
            }

    def read_state(self):
        with self._lock:
            began = monotonic()
            raw = self._read("rm_get_current_arm_state")
            ended = monotonic()
            if not isinstance(raw, dict):
                raise HardwareError("控制器当前状态回包格式无效")
            joint = _six(raw.get("joint"), "控制器关节角")
            pose = _six(raw.get("pose"), "控制器末端位姿")
            try:
                codes = raw["err"]["err"]
                if not isinstance(codes, (list, tuple)) or any(type(c) not in (int, str) for c in codes):
                    raise ValueError("err")
                errors = tuple(int(code) for code in codes)
                if len(errors) != raw["err"]["err_len"]:
                    raise ValueError("err_len")
            except (KeyError, TypeError, ValueError) as exc:
                raise HardwareError("当前状态回包缺少有效的机械臂错误码") from exc
            flange = np.eye(4)
            for (a, d, alpha, offset), angle in zip(self._mdh, np.deg2rad(joint)):
                flange = flange @ modified_dh_transform(a, d, alpha, offset + angle)
            return ArmState(joint, pose, flange, errors, began, ended)

    def _trajectory(self):
        data = self._client.rm_get_arm_current_trajectory()
        _check_status("rm_get_arm_current_trajectory", data["return_code"])
        return data["trajectory_type"]

    def _check_motion_state(self, state, joint_low, joint_high):
        if any(state.error_codes):
            raise MotionSafetyError(f"机械臂报错 · {state.error_codes}")
        if np.any(state.joint_deg < joint_low) or np.any(state.joint_deg > joint_high):
            raise MotionSafetyError("实测关节角超出控制器关节限位")
        pose = state.controller_pose_m_rad
        rotation = euler_matrix(pose[3:])
        # 实际运行必须使用同一基座和法兰，交叉核对控制器位姿与实时 FK。
        if (np.linalg.norm(pose[:3] - state.T_base_flange[:3, 3]) > 0.001
                or not np.allclose(rotation, state.T_base_flange[:3, :3], atol=0.005, rtol=0)):
            raise MotionSafetyError("控制器位姿与实时基座系法兰位姿不一致，请检查坐标系")
        tcp = (transform(pose[:3], rotation) @ self._limits.T_flange_tcp)[:3, 3]
        self._limits.require_in_workspace(tcp, "实测 TCP")
        return tcp, rotation

    def move_linear_tcp(self, target_xyz_m, *, speed_pct):
        """显式的、小范围 TCP 直线平移，保持当前姿态，等待到位并监测状态。"""
        with self._lock:
            if not self._motion_enabled or self._faulted:
                raise MotionSafetyError("运动已禁用或当前连接已因运动异常锁定")
            limits = self._limits
            target = limits.require_in_workspace(target_xyz_m, "目标 TCP")
            self._require_real_mode()
            for frame in self.read_frames().values():
                if not np.allclose(_six(frame.get("pose"), "活动坐标系"), 0, atol=1e-6, rtol=0):
                    raise MotionSafetyError("直线平移要求控制器工作系为 Base、工具系为法兰")
            limits.require_speed(speed_pct, self._read("rm_get_arm_max_line_speed"))
            joint_low = _six(self._read("rm_get_joint_min_pos"), "控制器关节下限")
            joint_high = _six(self._read("rm_get_joint_max_pos"), "控制器关节上限")
            if np.any(joint_low >= joint_high):
                raise MotionSafetyError("控制器关节限位无效")
            if self._trajectory() != 0:  # RM_NO_PLANNING_E
                raise MotionSafetyError("控制器已有活动轨迹")
            start = self.read_state()
            initial_tcp, initial_rotation = self._check_motion_state(start, joint_low, joint_high)
            delta = target - initial_tcp
            distance = np.linalg.norm(delta)
            if distance > limits.max_step_m:
                raise MotionSafetyError("单次 TCP 位移超出现场限幅")
            if distance <= 0.001:
                return start
            pose = start.controller_pose_m_rad.copy()
            pose[:3] += delta
            deadline = monotonic() + limits.completion_timeout_s
            previous_tcp, previous_started = initial_tcp, start.request_start_s
            try:
                # 发送开始即视为可能已执行。丢失回执时也要尝试轨迹停止。
                _check_status("rm_movel", self._client.rm_movel(pose.tolist(), speed_pct, 0, 0, 0))
                while monotonic() < deadline:
                    state = self.read_state()
                    tcp, rotation = self._check_motion_state(state, joint_low, joint_high)
                    # 以两次查询的最长可能间隔作保守比较，0.1 mm 留给回包量化和抖动。
                    sample_span = state.request_end_s - previous_started
                    travel_m = np.linalg.norm(tcp - previous_tcp)
                    if sample_span <= 0 or travel_m > limits.max_linear_speed_m_s * sample_span + 0.0001:
                        raise MotionSafetyError("相邻回包检测到实测 TCP 线速度超限")
                    previous_tcp, previous_started = tcp, state.request_start_s
                    if not np.allclose(rotation, initial_rotation, atol=0.005, rtol=0):
                        raise MotionSafetyError("直线平移中检测到法兰姿态变化")
                    trajectory = self._trajectory()
                    if trajectory not in (0, 2):  # 无规划、直线规划
                        raise MotionSafetyError("运动中检测到非预期轨迹")
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    if trajectory == 0 and np.linalg.norm(target - tcp) <= 0.001:
                        return state
                    sleep(min(0.03, remaining))
                raise HardwareTimeout("直线平移超时")
            except BaseException as exc:
                self._faulted = True
                try:
                    _check_status("rm_set_arm_stop", self._client.rm_set_arm_stop())
                except Exception as stop_error:
                    raise MotionSafetyError(
                        f"运动异常且控制器未确认轨迹停止，请使用现场急停 · {exc} · {stop_error}"
                    ) from exc
                raise


class RealManGripper:
    """通过同一个 RealMan 连接读取内置协议返回的 EG2-4C2 状态。"""

    def __init__(self, arm: RealManArm):
        self._arm = arm

    def read_state(self):
        with self._arm._lock:
            began = monotonic()
            raw = self._arm._read("rm_get_gripper_state")
            ended = monotonic()
        try:
            online, enabled = raw["status"], raw["enable_state"]
            if online not in (0, 1) or enabled not in (0, 1):
                raise ValueError("status / enable_state")
            # actpos 是原始开口量。毫米换算需要单独的实测开口标定。
            return GripperState(bool(online), bool(enabled), int(raw["actpos"]),
                                int(raw["current_force"]), int(raw["temperature"]),
                                int(raw["mode"]), int(raw["error"]), (began + ended) / 2)
        except (KeyError, ValueError, TypeError) as exc:
            raise HardwareError("夹爪回包缺少有效状态字段") from exc
