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
from .realman_config import REALMAN_HARDWARE, REALMAN_TABLETOP
from .safety import GripperLimits, MotionLimits, MotionSafetyError


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


@dataclass(frozen=True)
class _MotionSnapshot:
    state: ArmState
    tcp: np.ndarray
    rotation: np.ndarray
    controller_tool: np.ndarray
    joint_low: np.ndarray
    joint_high: np.ndarray


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
    """with 块管理连接。动作须显式启用、提供现场限值并逐条人工确认。"""

    # SDK 的返回超时是进程级设置，一个进程只维护一个受管连接。
    _session_lock = Lock()

    def __init__(self, ip, *, port=8080, timeout_ms=500, enable_motion=False, limits=None, confirm=None):
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
        self._confirm = confirm
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

    def read_kinematic_parameters(self):
        """只读查询 DH、方向定义所用的关节限位和控制器速度配置。"""
        with self._lock:
            return dict(dh=self._read("rm_get_DH_data"),
                        joint_min_deg=self._read("rm_get_joint_min_pos"),
                        joint_max_deg=self._read("rm_get_joint_max_pos"),
                        joint_max_speed_deg_s=self._read("rm_get_joint_max_speed"))

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

    def _require_motion_enabled(self):
        if not self._motion_enabled or self._faulted:
            raise MotionSafetyError("运动已禁用或当前连接已因运动异常锁定")

    def _confirm_motion(self, action, details):
        if not callable(self._confirm) or self._confirm(action, details) is not True:
            raise MotionSafetyError(f"未获人工确认 · {action}")

    def _controller_tool(self):
        frames = self.read_frames()
        work, tool = frames["work"], frames["tool"]
        work_pose = _six(work.get("pose"), "活动工作系")
        if work.get("name") != "Base" or not np.allclose(work_pose, 0, atol=1e-6, rtol=0):
            raise MotionSafetyError("直线及夹爪调试要求控制器工作系为 Base")
        active = _six(tool.get("pose"), "活动工具系")
        T_flange_active = transform(active[:3], euler_matrix(active[3:]))
        if tool.get("name") == "Arm_Tip" and np.allclose(T_flange_active, np.eye(4), atol=1e-6, rtol=0):
            return T_flange_active
        if (tool.get("name") == REALMAN_TABLETOP.tcp_name
                and np.allclose(T_flange_active, self._limits.T_flange_tcp, atol=1e-6, rtol=0)):
            return T_flange_active
        raise MotionSafetyError("活动工具系必须为法兰 Arm_Tip 或与实测 TCP 一致的 Grip_plus")

    def read_collision_stage(self):
        """只读查询控制器的动力学碰撞等级，0 表示关闭。"""
        with self._lock:
            stage = self._read("rm_get_collision_stage")
            if type(stage) is not int or not 0 <= stage <= 8:
                raise HardwareError("控制器碰撞防护等级回包无效")
            return stage

    def read_max_line_speed(self):
        """控制器末端线速度上限，单位 m/s。"""
        with self._lock:
            return self._read("rm_get_arm_max_line_speed")

    def read_joint_bounds(self):
        """当前控制器的关节限位，单位度。"""
        with self._lock:
            low = _six(self._read("rm_get_joint_min_pos"), "控制器关节下限")
            high = _six(self._read("rm_get_joint_max_pos"), "控制器关节上限")
            if np.any(low >= high):
                raise MotionSafetyError("控制器关节限位无效")
            return low, high

    def _require_collision_protection(self):
        if self.read_collision_stage() == 0:
            raise MotionSafetyError("控制器的动力学碰撞防护关闭")

    def _check_motion_state(self, state, joint_low, joint_high, controller_tool):
        if any(state.error_codes):
            raise MotionSafetyError(f"机械臂报错 · {state.error_codes}")
        if np.any(state.joint_deg < joint_low) or np.any(state.joint_deg > joint_high):
            raise MotionSafetyError("实测关节角超出控制器关节限位")
        pose = state.controller_pose_m_rad
        rotation = euler_matrix(pose[3:])
        # 先用当前工具系核对控制器回包，再从法兰 FK 计算真正要控制的 Grip_plus TCP。
        expected = state.T_base_flange @ controller_tool
        if (np.linalg.norm(pose[:3] - expected[:3, 3]) > 0.001
                or not np.allclose(rotation, expected[:3, :3], atol=0.005, rtol=0)):
            raise MotionSafetyError("控制器活动工具位姿与实时基座系 FK 不一致，请检查坐标系")
        tcp = state.T_base_flange @ self._limits.T_flange_tcp
        self._limits.require_in_workspace(tcp[:3, 3], "实测 TCP")
        return tcp[:3, 3], tcp[:3, :3]

    def _motion_snapshot(self):
        if not isinstance(self._limits, MotionLimits):
            raise MotionSafetyError("检查运动条件需要现场 MotionLimits")
        self._require_real_mode()
        tool = self._controller_tool()
        self._require_collision_protection()
        low, high = self.read_joint_bounds()
        if self._trajectory() != 0:  # RM_NO_PLANNING_E
            raise MotionSafetyError("控制器已有活动轨迹")
        state = self.read_state()
        tcp, rotation = self._check_motion_state(state, low, high, tool)
        return _MotionSnapshot(state, tcp, rotation, tool, low, high)

    def check_motion_ready(self):
        """读取并核对有待人工确认的运动条件，不向控制器发送动作。"""
        with self._lock:
            return self._motion_snapshot().state

    @staticmethod
    def _require_same_start(preview, live):
        if (not np.allclose(preview.controller_tool, live.controller_tool, atol=1e-6, rtol=0)
                or np.max(np.abs(preview.state.joint_deg - live.state.joint_deg)) > 0.05
                or np.linalg.norm(preview.tcp - live.tcp) > 0.001):
            raise MotionSafetyError("人工确认期间关节、TCP 或活动工具系变化，请重新规划")

    def move_linear_tcp(self, target_xyz_m, *, speed_pct, action="move", preflight=None, interlock=None):
        """显式的、小范围 TCP 直线平移，保持当前姿态，等待到位并监测状态。"""
        with self._lock:
            self._require_motion_enabled()
            limits = self._limits
            target = limits.require_in_workspace(target_xyz_m, "目标 TCP")
            preview = self._motion_snapshot()
            limits.require_speed(speed_pct, self.read_max_line_speed())
            if limits.require_segment(preview.tcp, target) <= 0.001:
                return preview.state
            self._confirm_motion(action, {"start_tcp_m": preview.tcp.tolist(), "target_tcp_m": target.tolist(),
                                          "speed_pct": speed_pct, "max_linear_speed_m_s": limits.max_linear_speed_m_s})
            if preflight is not None:
                preflight()
            if interlock is not None:
                interlock()
            start = self._motion_snapshot()
            self._require_same_start(preview, start)
            limits.require_speed(speed_pct, self.read_max_line_speed())
            distance = limits.require_segment(start.tcp, target)
            if distance <= 0.001:
                return start.state
            delta = target - start.tcp
            direction = delta / distance
            pose = start.state.controller_pose_m_rad.copy()
            # rm_movel 的目标是活动工具。Grip_plus 活动时这里的差为零，法兰活动时仅换算一次。
            tool_delta = start.controller_tool[:3, 3] - limits.T_flange_tcp[:3, 3]
            pose[:3] = target + start.state.T_base_flange[:3, :3] @ tool_delta
            deadline = monotonic() + limits.completion_timeout_s
            previous_tcp, previous_started = start.tcp, start.state.request_start_s
            try:
                # 发送开始即视为可能已执行。丢失回执时也要尝试轨迹停止。
                _check_status("rm_movel", self._client.rm_movel(pose.tolist(), speed_pct, 0, 0, 0))
                while monotonic() < deadline:
                    state = self.read_state()
                    tcp, rotation = self._check_motion_state(state, start.joint_low, start.joint_high,
                                                             start.controller_tool)
                    # 以两次查询的最长可能间隔作保守比较，0.1 mm 留给回包量化和抖动。
                    sample_span = state.request_end_s - previous_started
                    travel_m = np.linalg.norm(tcp - previous_tcp)
                    if sample_span <= 0 or travel_m > limits.max_linear_speed_m_s * sample_span + 0.0001:
                        raise MotionSafetyError("相邻回包检测到实测 TCP 线速度超限")
                    previous_tcp, previous_started = tcp, state.request_start_s
                    if not np.allclose(rotation, start.rotation, atol=0.005, rtol=0):
                        raise MotionSafetyError("直线平移中检测到法兰姿态变化")
                    reach = tcp - start.tcp
                    nearest = direction * np.clip(reach @ direction, 0., distance)
                    if np.linalg.norm(reach - nearest) > limits.max_path_error_m:
                        raise MotionSafetyError("实测 TCP 偏离已确认的直线段")
                    trajectory = self._trajectory()
                    if trajectory not in (0, 2):  # 无规划、直线规划
                        raise MotionSafetyError("运动中检测到非预期轨迹")
                    self._require_collision_protection()
                    if interlock is not None:
                        interlock()
                    if trajectory == 0 and np.linalg.norm(target - tcp) <= 0.001:
                        self._require_real_mode()
                        if not np.allclose(self._controller_tool(), start.controller_tool, atol=1e-6, rtol=0):
                            raise MotionSafetyError("运动期间活动工具系变化")
                        if monotonic() < deadline:
                            return state
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
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
    """EG2-4C2 内置协议，复用机械臂连接、会话锁和人工确认。"""

    def __init__(self, arm: RealManArm, *, limits=None):
        self._arm = arm
        self._limits = limits

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

    def _require_calibrated(self):
        if not isinstance(self._limits, GripperLimits):
            raise MotionSafetyError("夹爪动作需要现场实测的 GripperLimits")
        return self._limits

    def _require_healthy(self):
        state = self.read_state()
        if not state.online or not state.enabled or state.error or state.mode not in range(1, 7):
            raise MotionSafetyError(f"夹爪状态异常 · {state}")
        return state

    def _at_open(self, state):
        lo, hi = self._limits.open_feedback_raw
        return state.mode == 1 and lo <= state.opening_raw <= hi

    def _has_grip(self, state):
        lo, hi = self._limits.held_feedback_raw
        force_lo, force_hi = self._limits.held_force_g
        return state.mode == 6 and lo <= state.opening_raw <= hi and force_lo <= state.force_g <= force_hi

    def require_open(self):
        """复核张满的原始位置值和厂商空闲状态。"""
        self._require_calibrated()
        state = self._require_healthy()
        if not self._at_open(state):
            raise MotionSafetyError(f"夹爪未张满至已验收的状态 · {state}")
        return state

    def require_holding(self):
        """力控停止、夹指开口和力反馈须同时匹配实物夹持。"""
        self._require_calibrated()
        state = self._require_healthy()
        if not self._has_grip(state):
            raise MotionSafetyError(f"夹爪回读未确认实物夹持 · {state}")
        return state

    def open_max(self, *, preflight=None):
        with self._arm._lock:
            self._arm._require_motion_enabled()
            limits = self._require_calibrated()
            return self._act("open", "rm_set_gripper_release", (limits.open_speed_raw,), self._at_open,
                             preflight)

    def close_on_object(self, *, preflight=None):
        with self._arm._lock:
            self._arm._require_motion_enabled()
            limits = self._require_calibrated()
            # 单次力控闭合在力阈值处停止，后续抬升持续检查实际开口和力回读。
            return self._act("close", "rm_set_gripper_pick", (limits.close_speed_raw, limits.close_force_raw),
                             self._has_grip, preflight)

    def _act(self, action, sdk_command, command_args, reached, preflight):
        arm, limits = self._arm, self._limits
        preview = arm._motion_snapshot()
        jaw = self._require_healthy()
        if action == "open" and reached(jaw):
            return jaw
        if jaw.mode in (4, 5) or (action == "close" and not self._at_open(jaw)):
            raise MotionSafetyError("夹爪必须静止，夹取前须确认已张满")
        details = {"tcp_m": preview.tcp.tolist(), "opening_raw": jaw.opening_raw,
                   "speed_raw": command_args[0]}
        if action == "close":
            details["force_command_raw"] = command_args[1]
        arm._confirm_motion(action, details)
        if preflight is not None:
            preflight()
        start = arm._motion_snapshot()
        arm._require_same_start(preview, start)
        fresh = self._require_healthy()
        if fresh.mode != jaw.mode or fresh.opening_raw != jaw.opening_raw:
            raise MotionSafetyError("人工确认期间夹爪状态变化")
        deadline = monotonic() + limits.completion_timeout_s
        try:
            # False/1 等待设置回执，不以该回执判定动作结束。1001 一类读回值从不作为参数发送。
            _check_status(sdk_command, getattr(arm._client, sdk_command)(*command_args, False, 1))
            good_reads = 0
            while monotonic() < deadline:
                state = self._require_healthy()
                actual = arm.read_state()
                if (any(actual.error_codes) or np.max(np.abs(actual.joint_deg - start.state.joint_deg)) > 0.05
                        or arm._trajectory() != 0):
                    raise MotionSafetyError("夹爪动作期间机械臂出现运动或报错")
                good_reads = good_reads + 1 if reached(state) else 0
                if good_reads >= 2 and monotonic() < deadline:
                    return state
                if action == "close" and (state.mode == 2 or (state.mode == 6 and not reached(state))):
                    raise MotionSafetyError(f"闭合结束时未确认实物夹持 · {state}")
                sleep(max(0., min(0.03, deadline - monotonic())))
            raise HardwareTimeout("夹爪动作超时")
        except BaseException as exc:
            arm._faulted = True
            # 内置协议没有独立的夹爪停止指令。保持故障锁定，现场人员处理夹爪状态。
            raise MotionSafetyError(f"夹爪动作中止且本会话已锁定，请用现场停止装置并核查夹爪 · {exc or type(exc).__name__}") from exc
