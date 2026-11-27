"""离线复现眼在手上标定。结果只用于比较，不能写入部署标定。"""

from itertools import combinations

import cv2
import numpy as np

from ..utils.geometry import euler_matrix, inverse, transform, validate_transform
from .z_reference import board_pose, upward_normal


HAND_EYE_METHODS = ("Tsai", "Park", "Horaud", "Andreff", "Daniilidis")


def read_calibration_views(directory, reference):
    """按编号一对一读取旧图像和 Base 下的 TCP 示教器位姿。"""
    poses = np.loadtxt(directory / "poses.txt", delimiter=",", ndmin=2)
    if poses.shape[1] != 6 or len(poses) < 4 or not np.isfinite(poses).all():
        raise ValueError("poses.txt 至少需要四行有限的 XYZ（m）和固定轴 RPY（rad）")
    numbers = range(1, len(poses) + 1)
    if {p.name for p in directory.glob("*.jpg")} != {f"{i}.jpg" for i in numbers}:
        raise ValueError("器材位姿与图像必须按 1.jpg 至 N.jpg 一一对应")
    tcp = np.stack([transform(row[:3], euler_matrix(row[3:])) for row in poses])
    observations = []
    size = None
    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    for i in numbers:
        path = directory / f"{i}.jpg"
        color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if color is None or (size is not None and color.shape[1::-1] != size):
            raise ValueError(f"无法读取 {path} 或图像分辨率不一致")
        size = color.shape[1::-1]
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (reference.cols, reference.rows), None)
        if not found:
            # 复现脚本静默跳过角点失败会打乱照片与位姿配对。这里直接报出编号。
            raise ValueError(f"{path} 未识别全部内角点")
        observations.append(cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria).reshape(-1, 2))
    return tcp, np.stack(observations), size


def resized_intrinsics(intrinsics, from_size, to_size):
    """复现 cv2.resize 的像素中心约定，每个轴保留半像素平移。"""
    sx, sy = np.asarray(to_size) / from_size
    pixel_map = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
    return pixel_map @ intrinsics


def fit_intrinsics(points, pixels, size):
    # 恢复旧脚本使用的左上内角点原点，保证可以重现原解。
    origin = points[0]
    objects = (points - origin).astype(np.float32)
    rms, K, d, rotations, positions = cv2.calibrateCamera([objects] * len(pixels), list(pixels), size, None, None)
    centered = []
    for rvec, tvec in zip(rotations, positions):
        rotation = cv2.Rodrigues(rvec)[0]
        centered.append(transform(tvec.ravel() - rotation @ origin, rotation))
    return K, d.ravel(), float(rms), np.stack(centered)


def rotation_deg(rotation):
    cosine = (np.trace(rotation, axis1=-2, axis2=-1) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))


def _average_target(poses):
    u, _, vh = np.linalg.svd(poses[:, :3, :3].mean(axis=0))
    # 投影到 SO(3)，均值只用于度量同一实体在不同视角下的离散。
    u[:, -1] *= np.linalg.det(u @ vh)
    return transform(poses[:, :3, 3].mean(axis=0), u @ vh)


def _target_error(target, consensus):
    d = 1000 * (target[:3, 3] - consensus[:3, 3])
    normal = upward_normal(target[:3, 2]) @ upward_normal(consensus[:3, 2])
    return dict(center_xyz_mm=d.tolist(), center_error_mm=float(np.linalg.norm(d)),
                rotation_deg=float(rotation_deg(consensus[:3, :3].T @ target[:3, :3])),
                normal_deg=float(np.rad2deg(np.arccos(np.clip(normal, -1, 1)))))


def _summarize_errors(rows):
    errors = np.array([row["center_error_mm"] for row in rows])
    z = np.array([row["center_xyz_mm"][2] for row in rows])
    return dict(center_rms_mm=float(np.sqrt(np.mean(errors ** 2))),
                center_z_rms_mm=float(np.sqrt(np.mean(z ** 2))),
                normal_rms_deg=float(np.sqrt(np.mean([row["normal_deg"] ** 2 for row in rows]))),
                rotation_rms_deg=float(np.sqrt(np.mean([row["rotation_deg"] ** 2 for row in rows]))),
                worst_view=int(np.argmax(errors)) + 1, center_max_mm=float(errors.max()), views=rows)


def closure_metrics(tcp, color_board, hand_eye):
    """固定目标闭环和相对运动 AX=XB 残差，以标定板中心为平移点。"""
    targets = tcp @ hand_eye @ color_board
    consensus = _average_target(targets)
    errors = _summarize_errors([_target_error(target, consensus) for target in targets])
    translation_m, rotations = [], []
    for i, j in combinations(range(len(tcp)), 2):
        ax = inverse(tcp[j]) @ tcp[i] @ hand_eye
        xb = hand_eye @ color_board[j] @ inverse(color_board[i])
        translation_m.append(np.linalg.norm(ax[:3, 3] - xb[:3, 3]))
        rotations.append(rotation_deg(ax[:3, :3].T @ xb[:3, :3]))
    return dict(**errors, mean_center_base_m=consensus[:3, 3].tolist(),
                mean_up_normal_base=upward_normal(consensus[:3, 2]).tolist(),
                relative_ax_xb_translation_rms_mm=1000 * float(np.sqrt(np.mean(np.square(translation_m)))),
                relative_ax_xb_rotation_rms_deg=float(np.sqrt(np.mean(np.square(rotations)))))


def solve_hand_eye(tcp, color_board, method):
    rotation, offset = cv2.calibrateHandEye(tcp[:, :3, :3], tcp[:, :3, 3], color_board[:, :3, :3],
                                           color_board[:, :3, 3], method=getattr(cv2, f"CALIB_HAND_EYE_{method.upper()}"))
    return validate_transform(transform(offset.ravel(), rotation), method)


def _reprojection_rms(points, pixels, K, d, poses):
    residuals = []
    for observed, pose in zip(pixels, poses):
        rvec = cv2.Rodrigues(pose[:3, :3])[0]
        expected = cv2.projectPoints(points, rvec, pose[:3, 3], K, d)[0].reshape(-1, 2)
        residuals.append(float(np.sqrt(np.mean(np.sum((expected - observed) ** 2, axis=1)))))
    return np.array(residuals)


def audit_hand_eye(tcp, pixels, img_size, points, factory_K, factory_size, saved):
    """对比全部算法，逐图留出时重新求解器材外参与自由内参。"""
    if not callable(getattr(cv2, "calibrateHandEye", None)):
        raise ValueError("当前 OpenCV 没有 calibrateHandEye。请使用原 handeye 环境的 OpenCV 4.x")
    factory_high_K = resized_intrinsics(factory_K, factory_size, img_size)
    # 历史 JPG 是显示流放大 2 倍保存。误差统一换算为 640×480 原始 RGB 像素量级。
    pixel_scale = factory_size[0] / img_size[0]
    if not np.isclose(pixel_scale, factory_size[1] / img_size[1]):
        raise ValueError("历史图像与现有 RGB 流的缩放比在两轴不一致")
    K_fit, d_fit, _, poses_fit = fit_intrinsics(points, pixels, img_size)
    zero = np.zeros(5)
    poses_factory = np.stack([board_pose(points, uv, factory_high_K, zero,
                                         max_reprojection_px=1 / pixel_scale)[0] for uv in pixels])
    report = {}
    external_models = {}
    for name, K, d, color_board in (("fitted_rgb", K_fit, d_fit, poses_fit),
                                     ("factory_rgb", factory_high_K, zero, poses_factory)):
        train_reprojection = pixel_scale * _reprojection_rms(points, pixels, K, d, color_board)
        leave_one_out = {m: [] for m in HAND_EYE_METHODS}
        held_reprojection = []
        lenses = []
        for view in range(len(tcp)):
            train = np.arange(len(tcp)) != view
            if name == "fitted_rgb":
                fold_K, fold_d, _, train_board = fit_intrinsics(points, pixels[train], img_size)
                held_board, img_error = board_pose(points, pixels[view], fold_K, fold_d,
                                                    max_reprojection_px=1 / pixel_scale)
            else:
                fold_K = K
                train_board, held_board = color_board[train], color_board[view]
                img_error = train_reprojection[view] / pixel_scale
            lenses.append(resized_intrinsics(fold_K, img_size, factory_size))
            held_reprojection.append(img_error * pixel_scale)
            for method in HAND_EYE_METHODS:
                x = solve_hand_eye(tcp[train], train_board, method)
                seen_board = _average_target(tcp[train] @ x @ train_board)
                leave_one_out[method].append(_target_error(tcp[view] @ x @ held_board, seen_board))

        solutions = {m: solve_hand_eye(tcp, color_board, m) for m in HAND_EYE_METHODS}
        models = {}
        for method, hand in {"stored": saved, **solutions}.items():
            models[method] = dict(T_tcp_color=hand.tolist(),
                                  delta_tcp_translation_mm=(1000 * (hand[:3, 3] - saved[:3, 3])).tolist(),
                                  delta_tcp_rotation_deg=float(rotation_deg(saved[:3, :3].T @ hand[:3, :3])),
                                  in_sample=closure_metrics(tcp, color_board, hand),
                                  leave_one_out=_summarize_errors(leave_one_out[method]) if method in solutions else None)
        native_K = resized_intrinsics(K, img_size, factory_size)
        report[name] = dict(K_rgb=native_K.tolist(), distortion=d.tolist(),
                            native_reprojection_rms_px=float(np.sqrt(np.mean(train_reprojection ** 2))),
                            native_reprojection_by_view_px=train_reprojection.tolist(),
                            native_reprojection_loo_rms_px=float(np.sqrt(np.mean(np.square(held_reprojection)))),
                            native_reprojection_loo_by_view_px=held_reprojection,
                            loo_K_min=np.min(lenses, axis=0).tolist(), loo_K_max=np.max(lenses, axis=0).tolist(),
                            models=models)
        external_models[name] = (native_K, d, {"stored": saved, **solutions})
    return report, external_models
