"""摄像头只读诊断，比较原生点云和彩色像素对齐后的光轴距离。"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from realman_rl_simtoreal.deployment.z_reference import BoardReference, board_corners, board_pose


def extrinsic(a, b):
    raw = a.get_extrinsics_to(b)
    return np.asarray(raw.rotation).reshape(3, 3, order="F"), np.asarray(raw.translation)


def distribution(x):
    p10, med, p90 = np.percentile(1000 * x, [10, 50, 90])
    return dict(n=len(x), p10_mm=float(p10), median_mm=float(med), p90_mm=float(p90))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--board-corners", type=int, nargs=2, required=True)
    parser.add_argument("--board-spans-mm", type=float, nargs=2, required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not 3 <= args.frames <= 30:
        parser.error("需要新的输出路径，采样帧数为 3–30")
    # 光学相对量只使用棋盘跨度，测试无需连接机械臂或指定桌面高度。
    reference = BoardReference(*args.board_corners, *(mm / 1000 for mm in args.board_spans_mm), 0., 0.)
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    try:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        if any(intr.coeffs):
            raise ValueError("只读同帧诊断需要已验证的零畸变 RGB profile")
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1.]])
        rotation, offset = extrinsic(depth_stream, color_stream)
        reverse, reverse_offset = extrinsic(color_stream, depth_stream)
        v, u = np.indices((intr.height, intr.width))
        rays = np.stack([(u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy, np.ones_like(u)], axis=-1)
        factor = rays @ reverse[2]
        raw_pc = rs.pointcloud()
        aligner = rs.align(rs.stream.color)
        for _ in range(16):
            pipeline.wait_for_frames(2000)
        repeat, rejected = [], []
        for _ in range(args.frames):
            f = pipeline.wait_for_frames(2000)
            color = f.get_color_frame()
            depth = f.get_depth_frame()
            rgb = np.asanyarray(color.get_data())
            try:
                corner = board_corners(rgb, reference)
                pose, rms = board_pose(reference.points_m(), corner, K, None)
            except ValueError as exc:
                rejected.append(dict(color_number=color.get_frame_number(), reason=str(exc)))
                continue
            n, t = pose[:3, 2], pose[:3, 3]
            expected = (t @ n) / (rays @ n)
            hull = cv2.convexHull(corner.astype(np.float32)).reshape(-1, 2)
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.rint(hull).astype(np.int32), 1)
            mask = cv2.erode(mask, np.ones((9, 9), dtype=np.uint8)).astype(bool)

            # 由 SDK 输出原生深度点云，直接换到 RGB 光学系，只用 RGB 掩膜选对应区域。
            points = raw_pc.calculate(depth)
            raw_cloud = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            rgb_cloud = raw_cloud[raw_cloud[:, 2] > .02] @ rotation.T + offset
            uv = rgb_cloud[:, :2] / rgb_cloud[:, 2, None]
            u2 = intr.fx * uv[:, 0] + intr.ppx
            v2 = intr.fy * uv[:, 1] + intr.ppy
            inside = (u2 >= 0) & (u2 < intr.width - 1) & (v2 >= 0) & (v2 < intr.height - 1)
            xy = np.rint(np.c_[u2[inside], v2[inside]]).astype(int)
            selected = mask[xy[:, 1], xy[:, 0]]
            pc = rgb_cloud[inside][selected]
            raw_expected = (t @ n) / ((pc / pc[:, 2, None]) @ n)

            # 复算部署接口对 SDK 对齐图中原始 Z16 的光轴修正。
            aligned = np.asanyarray(aligner.process(f).get_depth_frame().get_data()) * depth.get_units()
            valid = mask & (aligned > 0)
            converted = (aligned - reverse_offset[2]) / factor
            if len(pc) < 80 or np.count_nonzero(valid) < 80:
                rejected.append(dict(color_number=color.get_frame_number(), reason="棋盘有效点数量不足"))
                continue
            repeat.append(dict(color_number=color.get_frame_number(), depth_number=depth.get_frame_number(),
                               color_timestamp_ms=color.get_timestamp(), depth_timestamp_ms=depth.get_timestamp(),
                               rgb_pnp_rms_px=rms, rgb_board_center_axial_m=float(t[2]),
                               native_minus_rgb_plane=distribution(pc[:, 2] - raw_expected),
                               aligned_minus_rgb_plane=distribution((converted - expected)[valid]),
                               color_axis_conversion_change=distribution((converted - aligned)[valid])))
        report = dict(date_local=datetime.now().astimezone().date().isoformat(), serial=args.serial,
                      rgb_intrinsics=K.tolist(), requested_frames=args.frames,
                      board_corners=args.board_corners, board_spans_mm=args.board_spans_mm,
                      depth_to_color_R=rotation.tolist(), depth_to_color_translation_m=offset.tolist(),
                      rgb_distortion=list(intr.coeffs), native_depth_scale_m=depth.get_units(),
                      usable_frames=len(repeat), frames=repeat, rejected_frames=rejected)
        if repeat:
            report["median_by_frame_mm"] = {
                key: float(np.median([x[key]["median_mm"] for x in repeat])) for key in (
                    "native_minus_rgb_plane", "aligned_minus_rgb_plane", "color_axis_conversion_change")}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as result:
            json.dump(report, result, ensure_ascii=False, indent=2)
            result.write("\n")
        print(json.dumps(dict(output=str(args.output), usable_frames=len(repeat),
                              medians_mm=report.get("median_by_frame_mm")), ensure_ascii=False))
        return 0 if len(repeat) >= 3 else 2
    finally:
        pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())
