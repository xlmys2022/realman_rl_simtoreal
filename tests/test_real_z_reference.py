"""检验棋盘参考能分辨公共基座偏差和同像素深度偏差。"""

import json
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

from realman_rl_simtoreal.deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from realman_rl_simtoreal.deployment.hand_eye_audit import closure_metrics, resized_intrinsics
from realman_rl_simtoreal.deployment.z_reference import BoardReference, fit_base_leveling, measure_board
from realman_rl_simtoreal.hardware.realman import ArmState, GripperState
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.perception.rgbd import RgbdFrame
from realman_rl_simtoreal.utils.geometry import euler_matrix, inverse, transform


def render_board(board, intrinsics, z_color_m):
    """用独立棋盘纹理投影生成 RGB，保留非等距的横竖打印尺寸。"""
    square_px = 50
    rows, cols = board.rows + 1, board.cols + 1
    raster_y, raster_x = np.indices((rows * square_px, cols * square_px))
    texture = (((raster_y // square_px + raster_x // square_px) % 2) * 230 + 12).astype(np.uint8)
    height, width = texture.shape
    source = np.array([[-.5, -.5], [width - .5, -.5], [width - .5, height - .5], [-.5, height - .5]],
                      dtype=np.float32)
    x = .5 * board.span_x_m * cols / (board.cols - 1)
    y = .5 * board.span_y_m * rows / (board.rows - 1)
    extent = np.array([[-x, -y, 0.], [x, -y, 0.], [x, y, 0.], [-x, y, 0.]])
    camera = extent + np.array([.016, -.012, z_color_m])
    projection = camera @ intrinsics.T
    target = (projection[:, :2] / projection[:, 2:]).astype(np.float32)
    warp = cv2.getPerspectiveTransform(source, target)
    gray = cv2.warpPerspective(texture, warp, (640, 480), borderValue=220)
    return np.repeat(gray[:, :, None], 3, axis=2)


class RealZReferenceTests(unittest.TestCase):
    def setUp(self):
        self.board = BoardReference(7, 5, .180, .112, .002, -.040)
        self.K = np.array([[600., 0, 312.], [0, 603., 233.], [0, 0, 1.]])
        self.optical = transform([.30, 0, .400], np.diag([1., -1., -1.]))
        self.rgb = render_board(self.board, self.K, .435)
        self.depth = np.full(self.rgb.shape[:2], .429, dtype=np.float32)

    def test_rgb_and_matched_depth_report_independent_signed_errors(self):
        report = measure_board(RgbdFrame(self.rgb, self.depth, self.K, self.optical, 1.), self.board)
        self.assertLess(report["pnp_rms_px"], .4)
        self.assertAlmostEqual(report["depth_minus_rgb_axial_mm"], -6., delta=.6)
        self.assertAlmostEqual(report["rgb_paired_base_z_error_mm"], 3., delta=.6)
        self.assertAlmostEqual(report["rgbd_paired_base_z_error_mm"], 9., delta=.01)
        self.assertAlmostEqual(report["paired_depth_contribution_base_z_mm"], 6., delta=.6)

    def test_depth_normal_is_fitted_independently_with_outliers(self):
        v, u = np.indices(self.depth.shape)
        dx, dy = (u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1]
        depth = .429 / (1 + .04 * dx - .025 * dy)
        depth[::9] = .55
        report = measure_board(RgbdFrame(self.rgb, depth, self.K, self.optical, 1.), self.board)
        expected = np.array([-.04, -.025, 1.])
        actual = [report[f"rgbd_normal_{axis}_base"] for axis in "xyz"]
        np.testing.assert_allclose(actual, expected / np.linalg.norm(expected), atol=1e-6)
        self.assertGreater(report["rgb_rgbd_normal_difference_deg"], 1.)
        self.assertLess(report["rgbd_plane_inlier_fraction"], .95)

    def test_hand_eye_closure_uses_camera_to_tcp_direction_and_pixel_centers(self):
        X = transform([.05, -.04, -.25], euler_matrix([.13, .12, 1.5]))
        fixed = transform([.46, .03, -.0365], euler_matrix([.01, -.04, .1]))
        poses = np.stack([transform([.24, .08 * i, .3], euler_matrix([i * .1, i / 5, i / 6]))
                          for i in range(5)])
        observations = np.stack([inverse(X) @ inverse(tool) @ fixed for tool in poses])
        closed = closure_metrics(poses, observations, X)
        self.assertLess(closed["center_rms_mm"], 1e-8)
        self.assertLess(closed["relative_ax_xb_translation_rms_mm"], 1e-8)
        self.assertLess(closed["relative_ax_xb_rotation_rms_deg"], 1e-5)
        self.assertGreater(closure_metrics(poses, observations, inverse(X))["center_rms_mm"], 10.)
        enlarged = resized_intrinsics(self.K, (640, 480), (1280, 960))
        self.assertAlmostEqual(enlarged[0, 2], self.K[0, 2] * 2 + .5)
        np.testing.assert_allclose(resized_intrinsics(enlarged, (1280, 960), (640, 480)), self.K)

    def test_requires_measured_dimensions_and_enough_valid_depth(self):
        with self.assertRaisesRegex(ValueError, "跨度"):
            BoardReference(7, 5, np.nan, .112, .002, -.040)
        with self.assertRaisesRegex(ValueError, "有效对齐深度不足"):
            measure_board(RgbdFrame(self.rgb, np.zeros_like(self.depth), self.K, self.optical, 1.), self.board)

    def test_leveling_recovers_absolute_height_and_preserves_real_height_differences(self):
        expected = transform([0., 0., .001], cv2.Rodrigues(np.array([.008, .031, 0.]))[0])
        raw = inverse(expected)
        z = self.board.top_z_base_m
        centers = np.array([[.26, -.08, z], [.47, .04, z], [.52, -.06, z]])
        raw_centers = centers @ raw[:3, :3].T + raw[:3, 3]
        raw_normals = np.tile(raw[:3, 2], (len(centers), 1))
        leveling = fit_base_leveling(raw_centers, raw_normals, z)
        np.testing.assert_allclose(leveling, expected, atol=1e-12)

        raised = np.r_[.41, .02, z + .020, 1.]
        np.testing.assert_allclose(leveling @ (raw @ raised), raised, atol=1e-12)

    def test_batch_reports_bad_frames_and_requires_stationary_provenance(self):
        audit = run_path(str(PROJECT_ROOT / "scripts/diagnostics/audit_realman_vision.py"))
        analyze = audit["analyze_board_capture"]
        calibration = load_calibration(DEFAULT_CALIBRATION.with_name("calibration.20261126.json"))
        flange = self.optical @ inverse(calibration.camera_to_base(np.eye(4)))
        flanges = np.repeat(flange[None], 4, axis=0)
        depths = np.repeat(self.depth[None], 4, axis=0)
        depths[0] = 0
        meta = {"camera": {"serial": calibration.camera_serial}, "calibration_sha256": calibration.checksum()}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.npz"

            def save_recording():
                np.savez(path, rgb=np.repeat(self.rgb[None], 4, axis=0), depth_m=depths, K=self.K,
                         T_base_flange=flanges, host_received_s=np.arange(4), metadata=json.dumps(meta))

            save_recording()
            sample = analyze(path, calibration, calibration, self.board)
            self.assertTrue(sample["enough_frames"])
            self.assertEqual(sample["usable_frames"], 3)
            self.assertEqual(sample["rejected_frames"][0]["frame"], 0)
            flanges[3, 0, 3] += .003
            save_recording()
            with self.assertRaisesRegex(ValueError, "未保持静止"):
                analyze(path, calibration, calibration, self.board)
            meta.pop("calibration_sha256")
            save_recording()
            with self.assertRaisesRegex(ValueError, "校验值"):
                analyze(path, calibration, calibration, self.board)

    def test_onsite_gripper_burst_keeps_readbacks_and_uses_read_only_session(self):
        check = run_path(str(PROJECT_ROOT / "scripts/diagnostics/check_realman_arm.py"))["main"]
        now = ArmState(np.zeros(6), np.zeros(6), np.eye(4), (0,), 2., 2.001)
        arm = MagicMock()
        arm.__enter__.return_value = arm
        arm.read_state.return_value = now
        arm.read_collision_stage.return_value = 4
        arm.read_max_line_speed.return_value = .25
        device = Mock(return_value=arm)
        jaw = Mock()
        jaw.read_state.side_effect = [GripperState(True, True, 241, 715, 26, 6, 0, 2.05),
                                      GripperState(True, True, 241, 718, 26, 6, 0, 2.10)]
        output = Mock()
        with patch.dict(check.__globals__, dict(RealManArm=device, RealManGripper=Mock(return_value=jaw),
                                                 sleep=Mock(), print_json=output)):
            self.assertEqual(check(["--ip", "192.0.2.1", "--collision", "--gripper", "--gripper-samples", "2"]), 0)
        self.assertFalse(device.call_args.kwargs.get("enable_motion", False))
        surveyed = output.call_args.args[0]["gripper_observed"]
        self.assertEqual(surveyed["force_g"], [715, 718])
        self.assertEqual(surveyed["opening_raw"], [241, 241])
        self.assertEqual(surveyed["modes"], [6])
        arm.move_linear_tcp.assert_not_called()
        jaw.open_max.assert_not_called()


if __name__ == "__main__":
    unittest.main()
