"""用实测棋盘格分开诊断相机光轴深度与基座系 Z。"""

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class BoardReference:
    """棋盘格首末内角点跨度及图案表面相对桌面的实测高度，单位 m。"""

    cols: int
    rows: int
    span_x_m: float
    span_y_m: float
    top_above_table_m: float
    table_z_base_m: float

    def __post_init__(self):
        if any(type(n) is not int or not 3 <= n <= 64 for n in (self.cols, self.rows)):
            raise ValueError("横向和竖向内角点数均应为 3–64")
        if (not all(isfinite(x) for x in (self.span_x_m, self.span_y_m, self.top_above_table_m, self.table_z_base_m))
                or min(self.span_x_m, self.span_y_m) <= 0 or self.top_above_table_m < 0):
            raise ValueError("实测棋盘跨度须为正，图案面距桌面高度须非负，且所有尺寸须为有限值")

    @property
    def top_z_base_m(self):
        return self.table_z_base_m + self.top_above_table_m

    def points_m(self):
        # 两个轴独立量取跨度，可反映打印比例差。原点在首末内角点构成矩形的中心。
        x = np.linspace(-0.5, 0.5, self.cols) * self.span_x_m
        y = np.linspace(-0.5, 0.5, self.rows) * self.span_y_m
        xx, yy = np.meshgrid(x, y)
        return np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.size)]


def board_corners(rgb, reference):
    """在已归一到 RGB 光学图像的帧中查找棋盘格。"""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, (reference.cols, reference.rows),
                                                 flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        raise ValueError("RGB 未识别完整的棋盘格内角点")
    return np.ascontiguousarray(corners.reshape(-1, 2))


def board_pose(points, pixels, intrinsics, distortion, *, max_reprojection_px=1.0):
    """返回棋盘到 RGB 光学坐标系的变换和该图像的重投影 RMS。"""
    import cv2

    from ..utils.geometry import transform

    ok, rotation, translation = cv2.solvePnP(points, pixels, intrinsics, distortion)
    if not ok:
        raise ValueError("棋盘格 PnP 失败")
    R_color_board = cv2.Rodrigues(rotation)[0]
    offset = translation.ravel()
    if (not np.isfinite(R_color_board).all() or not np.isfinite(offset).all()
            or np.any((points @ R_color_board.T + offset)[:, 2] <= 0)):
        raise ValueError("PnP 棋盘姿态无效")
    projected = cv2.projectPoints(points, rotation, translation, intrinsics, distortion)[0].reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
    if not np.isfinite(error) or error > max_reprojection_px:
        raise ValueError(f"棋盘重投影误差 {error:.3f} px 超出 {max_reprojection_px:g} px")
    return transform(offset, R_color_board), error


def fit_depth_plane(cloud):
    """在棋盘图像的内区独立拟合深度平面。利用 MAD 去掉深度飞点。"""
    # 先用点到平面的最小中位距离选初值，避免飞点把全点 SVD 拉成近竖直面。
    rng = np.random.default_rng(0)
    triplets = cloud[rng.integers(len(cloud), size=(64, 3))]
    normals = np.cross(triplets[:, 1] - triplets[:, 0], triplets[:, 2] - triplets[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    non_collinear = lengths > 1e-8
    if not non_collinear.any():
        raise ValueError("深度有效点不足以约束棋盘面")
    normals = normals[non_collinear] / lengths[non_collinear, None]
    offsets = np.einsum("ij,ij->i", triplets[non_collinear, 0], normals)
    trials = cloud @ normals.T - offsets
    residual = trials[:, np.argmin(np.median(abs(trials), axis=0))]
    for _ in range(4):
        deviation = abs(residual - np.median(residual))
        cutoff = max(0.002, 3 * 1.4826 * np.median(deviation))
        mask = deviation <= cutoff
        if mask.sum() < max(80, 0.7 * len(cloud)):
            raise ValueError("棋盘深度平面的内点覆盖率不足")
        selected = cloud[mask]
        center = selected.mean(axis=0)
        _, singular, vh = np.linalg.svd(selected - center, full_matrices=False)
        if singular[1] < 1e-6:
            raise ValueError("深度有效点不足以约束棋盘面")
        normal = vh[-1]
        residual = (cloud - center) @ normal
    rms_mm = 1000 * float(np.sqrt(np.mean(residual[mask] ** 2)))
    return center, normal, rms_mm, float(mask.mean())


def upward_normal(normal):
    """选取与基座 +Z 方向同侧的单位法向量。"""
    unit = normal / np.linalg.norm(normal)
    return unit if unit[2] >= 0 else -unit


def measure_board(frame, reference, *, max_reprojection_px=1.0, pixels=None):
    """比较同一块板的 RGB PnP 与对齐深度。返回统计量，不生成校正或动作参数。"""
    import cv2

    if pixels is None:
        pixels = board_corners(frame.rgb, reference)
    distortion = np.asarray(frame.distortion) if frame.distortion else None
    optical_board, error = board_pose(reference.points_m(), pixels, frame.intrinsics, distortion,
                                      max_reprojection_px=max_reprojection_px)
    offset = optical_board[:3, 3]
    normal = optical_board[:3, 2]

    # 仅比较两种方法共同看到的像素，避开棋盘外边缘和深度孔洞。
    hull = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
    mask = np.zeros(frame.depth_m.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    interior = mask[::4, ::4] != 0
    grid_depth = frame.depth_m[::4, ::4]
    valid = interior & np.isfinite(grid_depth) & (grid_depth > 0)
    count = int(valid.sum())
    if count < 80 or count < 0.25 * int(interior.sum()):
        raise ValueError("棋盘范围内有效对齐深度不足")
    v, u = np.nonzero(valid)
    uv = 4 * np.c_[u, v].astype(np.float64)
    normalized = cv2.undistortPoints(uv.reshape(-1, 1, 2), frame.intrinsics, distortion).reshape(-1, 2)
    rays = np.c_[normalized, np.ones(count)]
    den = rays @ normal
    if np.any(abs(den) < 1e-6):
        raise ValueError("棋盘与视线夹角过小，无法可靠比较轴深度")
    rgb_depth = (offset @ normal) / den
    if not np.isfinite(rgb_depth).all() or np.any(rgb_depth <= 0):
        raise ValueError("RGB 棋盘平面深度无效")

    R_base_color = frame.camera_to_base[:3, :3]
    color_base_z = frame.camera_to_base[2, 3]
    ray_base_z = rays @ R_base_color[2]
    measured_depth = grid_depth[valid]
    plane_center, plane_normal, plane_rms_mm, plane_inliers = fit_depth_plane(rays * measured_depth[:, None])
    center_ray = offset / offset[2]
    center_ray_dot = plane_normal @ center_ray
    if abs(center_ray_dot) < 1e-6:
        raise ValueError("深度拟合平面与棋盘中心视线近似平行")
    measured_center_depth = float((plane_normal @ plane_center) / center_ray_dot)
    if measured_center_depth <= 0:
        raise ValueError("深度拟合所得棋盘中心无效")
    depth_delta = measured_depth - rgb_depth
    rgb_z = rgb_depth * ray_base_z + color_base_z
    rgbd_z = measured_depth * ray_base_z + color_base_z
    center_base = R_base_color @ offset + frame.camera_to_base[:3, 3]
    normal_base = upward_normal(R_base_color @ normal)
    depth_normal_base = upward_normal(R_base_color @ plane_normal)
    if abs(normal_base[2]) < 1e-6:
        raise ValueError("基座系棋盘平面无法表示为 Z 高度，需核查外参与观察姿态")
    slope = -normal_base[:2] / normal_base[2]
    q10, median, q90 = np.percentile(1000 * depth_delta, (10, 50, 90))
    return dict(pnp_rms_px=error, paired_depth_pixels=count, valid_depth_fraction=count / int(interior.sum()),
                median_rgb_distance_m=float(np.median(rgb_depth)),
                median_aligned_depth_m=float(np.median(measured_depth)),
                depth_minus_rgb_axial_mm=float(median), depth_minus_rgb_p10_mm=float(q10),
                depth_minus_rgb_p90_mm=float(q90),
                rgb_paired_base_z_error_mm=1000 * float(np.median(rgb_z) - reference.top_z_base_m),
                rgbd_paired_base_z_error_mm=1000 * float(np.median(rgbd_z) - reference.top_z_base_m),
                paired_depth_contribution_base_z_mm=1000 * float(np.median(ray_base_z * depth_delta)),
                rgb_board_center_x_m=float(center_base[0]), rgb_board_center_y_m=float(center_base[1]),
                rgb_board_center_base_z_error_mm=1000 * float(center_base[2] - reference.top_z_base_m),
                rgb_board_center_axial_m=float(offset[2]), rgbd_board_center_axial_m=measured_center_depth,
                rgbd_board_center_base_z_error_mm=1000 * float(
                    color_base_z + measured_center_depth * (R_base_color[2] @ center_ray) - reference.top_z_base_m),
                rgb_normal_x_base=float(normal_base[0]), rgb_normal_y_base=float(normal_base[1]),
                rgb_normal_z_base=float(normal_base[2]),
                rgbd_normal_x_base=float(depth_normal_base[0]), rgbd_normal_y_base=float(depth_normal_base[1]),
                rgbd_normal_z_base=float(depth_normal_base[2]),
                rgbd_plane_rms_mm=plane_rms_mm, rgbd_plane_inlier_fraction=plane_inliers,
                rgb_rgbd_normal_difference_deg=float(np.rad2deg(np.arccos(np.clip(normal_base @ depth_normal_base, -1, 1)))),
                apparent_dz_dx=float(slope[0]), apparent_dz_dy=float(slope[1]),
                apparent_z_at_base_xy_zero_m=float(center_base[2] - slope @ center_base[:2]),
                apparent_board_tilt_deg=float(np.rad2deg(np.arccos(np.clip(abs(normal_base[2]), 0, 1)))))
