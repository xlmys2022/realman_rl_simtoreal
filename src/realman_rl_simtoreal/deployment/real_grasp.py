"""固定俯视位姿的传统定位、直线接近、力控夹持和垂直抬升。"""

from dataclasses import asdict, dataclass
from math import ceil

import numpy as np

from ..hardware import HardwareError
from ..hardware.safety import GripperLimits, MotionSafetyError
from ..paths import TABLETOP_CARTESIAN_REALMAN_SCENE
from ..simulation.motion.tool_kinematics import ToolKinematics
from ..simulation.physics.model import load_model

from .grasp_config import GRASP_STAGES


def requested_grasp_stages(until, start_after):
    if until not in GRASP_STAGES or start_after not in (None, "pregrasp", "descend", "close"):
        raise ValueError("无效抓取阶段")
    begin = 0 if start_after is None else GRASP_STAGES.index(start_after) + 1
    stages = GRASP_STAGES[begin:GRASP_STAGES.index(until) + 1]
    if not stages:
        raise ValueError("停止阶段必须在起始检查点之后")
    return stages


def _require_same_pose(actual, expected):
    if (np.linalg.norm(actual[:3, 3] - expected[:3, 3]) > 0.001
            or not np.allclose(actual[:3, :3], expected[:3, :3], atol=0.003, rtol=0)):
        raise MotionSafetyError("TCP 与已呈现的抓取位姿不一致，请重新定位规划")


def _tool_tilt_deg(rotation):
    return float(np.rad2deg(np.arccos(np.clip(-rotation[2, 2], -1., 1.))))


def _jaw_angle(rotation, yaws, axis):
    jaw_yaw = np.arctan2(rotation[1, axis], rotation[0, axis])
    # 方块四边等价，取与夹紧方向最近的一条，无需在接近时旋转。
    d_yaw = (np.asarray(yaws) - jaw_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    if not d_yaw.size or not np.isfinite(d_yaw).all():
        raise MotionSafetyError("缺少可靠的方块边方向")
    return float(np.max(abs(d_yaw)))


def inspect_grasp_observation(estimate, table_z_m, edge_m):
    """只读记录验收所需的高度、XY 偏移、朝向与定位离散。"""
    tcp = estimate.T_base_tcp
    rotation = tcp[:3, :3]
    jaw_errors = {axis: float(np.rad2deg(_jaw_angle(rotation, estimate.yaw_rad, i)))
                  for i, axis in enumerate(("x", "y"))} if estimate.yaw_rad else None
    return dict(cube_center_m=estimate.center_m.tolist(), detections=estimate.detections, frames=estimate.frames,
                spread_m=estimate.spread_m, tcp_m=tcp[:3, 3].tolist(), tcp_rotation_base=rotation.tolist(),
                vertical_tcp_tilt_deg=_tool_tilt_deg(rotation),
                signed_cube_z_error_m=float(estimate.center_m[2] - (table_z_m + edge_m / 2)),
                target_minus_tcp_xy_m=(estimate.center_m[:2] - tcp[:2, 3]).tolist(),
                gripper_edge_error_deg_by_tcp_axis=jaw_errors)


def _check_visual_target(estimate, geometry, edge_m, *, check_yaw=True):
    if np.shape(estimate.center_m) != (3,) or not np.isfinite(estimate.center_m).all():
        raise MotionSafetyError("视觉方块中心无效")
    expected_z = geometry.table_z_base_m + edge_m / 2
    z_error = estimate.center_m[2] - expected_z
    if abs(z_error) > geometry.max_visual_z_error_m:
        raise MotionSafetyError(f"视觉绝对 Z 偏差 {z_error * 1000:+.2f} mm 超限，须先验收真机 Z 基准")
    rotation = estimate.T_base_tcp[:3, :3]
    tilt_deg = _tool_tilt_deg(rotation)
    if tilt_deg > geometry.max_tool_tilt_deg:
        raise MotionSafetyError(f"TCP 倾角 {tilt_deg:.2f}° 超出固定俯视限制")
    if not check_yaw:
        return float(z_error)
    angle = _jaw_angle(rotation, estimate.yaw_rad, 0 if geometry.closing_axis_tcp == "x" else 1)
    if np.rad2deg(angle) > geometry.max_jaw_yaw_error_deg:
        raise MotionSafetyError(f"夹指与方块边的夹角 {np.rad2deg(angle):.2f}° 超限")
    required_gap = edge_m * (np.cos(angle) + np.sin(angle)) + 2 * geometry.max_target_shift_m
    if geometry.jaw_open_gap_m <= required_gap:
        raise MotionSafetyError("实测张满间距无法覆盖方块的投影宽度及定位余量")
    return float(z_error)


@dataclass(frozen=True)
class GraspPlan:
    start_tcp: np.ndarray
    cube_center_m: np.ndarray
    waypoints: dict[str, tuple[float, float, float]]
    z_error_m: float

    def report(self):
        return dict(cube_center_m=self.cube_center_m.tolist(), start_tcp_m=self.start_tcp[:3, 3].tolist(),
                    visual_z_error_m=self.z_error_m, waypoints_m=self.waypoints)


def plan_grasp(estimate, site, edge_m, *, until="lift", start_after=None):
    """用视觉确定 XY。下降终点取经实物验收的 TCP 夹持高度，并校核视觉绝对 Z。"""
    geometry = site.geometry
    z_error = _check_visual_target(estimate, geometry, edge_m, check_yaw=start_after != "close")
    start = estimate.T_base_tcp
    x, y = map(float, estimate.center_m[:2])
    above_z = float(start[2, 3])
    grasp_z = geometry.grasp_tcp_z_base_m
    pregrasp_z = grasp_z + geometry.pregrasp_clearance_m
    requested = requested_grasp_stages(until, start_after)
    if start_after is not None:
        checkpoint_z = pregrasp_z if start_after == "pregrasp" else grasp_z
        if abs(above_z - checkpoint_z) > 0.001:
            raise MotionSafetyError("实测 TCP 高度与请求的抓取检查点相差超过 1 mm")
        if np.linalg.norm(estimate.center_m[:2] - start[:2, 3]) > geometry.max_target_shift_m:
            raise MotionSafetyError("当前对准误差超限，检查点续行禁止低位横移")
        # 已对准后固定当前 XY，分别核验下降及抬升路径。
        x, y = map(float, start[:2, 3])
    elif above_z < pregrasp_z:
        raise MotionSafetyError("观察 TCP 必须在预抓取高度之上，横移阶段保持该高度")
    waypoints = {"align": (x, y, above_z), "pregrasp": (x, y, pregrasp_z),
                 "descend": (x, y, grasp_z), "lift": (x, y, grasp_z + geometry.lift_height_m)}
    waypoints = {stage: point for stage, point in waypoints.items() if stage in requested}
    previous = start[:3, 3]
    for point in waypoints.values():
        site.motion.require_segment(previous, point)
        previous = point
    return GraspPlan(start.copy(), estimate.center_m.copy(), waypoints, z_error)


def check_tcp_path(state, motion, points, *, joint_bounds_deg=None):
    """复用已验证的 TCP 运动学，沿拟执行直线检查逆解和关节分支。"""
    model, _, _ = load_model(TABLETOP_CARTESIAN_REALMAN_SCENE)
    kin = ToolKinematics(model, tcp_site="grasp_site")
    base = model.body("base_link").pos.copy()
    q = np.deg2rad(state.joint_deg)
    lower, upper = kin.limits.T
    if joint_bounds_deg is not None:
        physical_low, physical_high = np.deg2rad(joint_bounds_deg)
        lower, upper = np.maximum(lower, physical_low), np.minimum(upper, physical_high)
    if np.any(q < lower) or np.any(q > upper):
        raise MotionSafetyError("实际起点关节角超出预检关节范围")
    tcp = state.T_base_flange @ motion.T_flange_tcp
    orientation = tcp[:3, :3]
    p, r = kin.fk(q)
    if (np.linalg.norm(p - base - tcp[:3, 3]) > 0.001
            or not np.allclose(r, orientation, atol=0.003, rtol=0)):
        raise MotionSafetyError("部署 TCP 与用于路径预检的模型不一致")
    previous = tcp[:3, 3]
    for point in points:
        target = np.asarray(point, dtype=float)
        distance = motion.require_segment(previous, target)
        for fraction in np.linspace(0., 1., max(1, ceil(distance / 0.005)) + 1)[1:]:
            probe = previous + fraction * (target - previous)
            result = kin.solve_ik(probe + base, q, orientation)
            if not result.success or np.max(abs(result.q - q)) > np.deg2rad(10.):
                raise MotionSafetyError(f"直线路径逆解失败或切换关节分支 · {probe.tolist()} · {result.reason}")
            if np.any(result.q < lower) or np.any(result.q > upper):
                raise MotionSafetyError(f"直线路径逆解超出控制器关节限位 · {probe.tolist()}")
            q = result.q
        previous = target


def execute_grasp(plan, site, edge_m, arm, gripper, observe, *, until, report, start_after=None, observe_lift=None):
    """按已显示的计划执行。每个物理指令在硬件层独立请求人工确认。"""
    requested = requested_grasp_stages(until, start_after)
    if any(stage not in plan.waypoints for stage in requested if stage not in ("open", "close")):
        raise MotionSafetyError("规划仅覆盖较早阶段，必须按请求的阶段重新预检")
    if any(stage in requested for stage in ("close", "lift")) and not isinstance(site.gripper, GripperLimits):
        raise MotionSafetyError("夹取和抬升需要完整的现场夹爪验收参数")
    geometry = site.geometry
    expected = plan.start_tcp.copy()

    def check_start():
        state = arm.check_motion_ready()
        _require_same_pose(state.T_base_flange @ site.motion.T_flange_tcp, expected)
        return state

    def recheck_cube():
        estimate = observe()
        _require_same_pose(estimate.T_base_tcp, expected)
        _check_visual_target(estimate, geometry, edge_m)
        if np.linalg.norm(estimate.center_m - plan.cube_center_m) > geometry.max_target_shift_m:
            raise MotionSafetyError("二次观察检测到方块移动，请重新规划")

    if start_after is not None:
        check_start()
        if start_after == "close":
            gripper.require_holding()
        else:
            gripper.require_open()
    for stage in requested:
        check_start()
        if stage == "open":
            completed = {"gripper": asdict(gripper.open_max(preflight=check_start))}
        elif stage == "close":
            completed = {"gripper": asdict(gripper.close_on_object(preflight=check_start))}
        else:
            goal = plan.waypoints[stage]
            holding = stage == "lift"

            def check_segment():
                # 工具位置相同也可能对应不同关节分支，确认后从真实关节起点重新预检本段。
                check_tcp_path(check_start(), site.motion, [goal], joint_bounds_deg=arm.read_joint_bounds())
                if not holding:
                    recheck_cube()

            state = arm.move_linear_tcp(goal, speed_pct=geometry.speed_pct, action=stage,
                                        preflight=check_segment,
                                        interlock=gripper.require_holding if holding else gripper.require_open)
            expected[:3, 3] = goal
            check_start()
            tcp = state.T_base_flange @ site.motion.T_flange_tcp
            completed = {"tcp_m": tcp[:3, 3].tolist(), "target_tcp_m": list(goal)}
        report(dict(stage=stage, **completed))

    if until != "lift":
        return dict(status="stopped_at_requested_stage", stage=until)
    gripper.require_holding()
    try:
        after = (observe if observe_lift is None else observe_lift)()
    except HardwareError as exc:
        check_start()
        return dict(status="lift_needs_visual_confirmation", gripper=asdict(gripper.require_holding()), detail=str(exc))
    _require_same_pose(after.T_base_tcp, expected)
    # 视觉采样结束后复查夹持，避免用采样开始前的夹爪状态报告成功。
    checked_grip = asdict(gripper.require_holding())
    # 抬升后的视觉确认核对三维位移，遮挡或误检时保留待人工核对的状态。
    delta = after.center_m - plan.cube_center_m
    verified = np.linalg.norm(delta - [0., 0., geometry.lift_height_m]) <= geometry.max_target_shift_m
    return dict(status="lift_verified" if verified else "lift_needs_visual_confirmation",
                gripper=checked_grip, visual_delta_m=delta.tolist())
