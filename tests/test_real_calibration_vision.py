"""用合成绿色方块验证真机配置、变换方向和静止腕部定位边界。"""

from dataclasses import replace
import json
from pathlib import Path
from runpy import run_path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from realman_rl_simtoreal.deployment.calibration import (
    CalibrationError, EXAMPLE_CALIBRATION, load_calibration,
)
from realman_rl_simtoreal.deployment.vision import RealCubeLocalizer
from realman_rl_simtoreal.hardware import HardwareError
from realman_rl_simtoreal.hardware.realman import ArmState
from realman_rl_simtoreal.hardware.realsense import FrameTiming, RgbdSample
from realman_rl_simtoreal.paths import PROJECT_ROOT
from realman_rl_simtoreal.utils.geometry import euler_matrix, inverse, transform


def box_sample(center, edge_m, camera_to_base):
    """用解析射线和盒体相交生成 RGB 与彩色光轴深度。"""
    height, width, focal = 120, 160, 200.
    cx, cy = (width - 1) / 2, (height - 1) / 2
    v, u = np.indices((height, width))
    rays = np.stack(((u - cx) / focal, (v - cy) / focal, np.ones_like(u)), axis=-1)
    rays = rays @ camera_to_base[:3, :3].T
    origin = camera_to_base[:3, 3]
    with np.errstate(divide="ignore"):
        low = (center - edge_m / 2 - origin) / rays
        high = (center + edge_m / 2 - origin) / rays
    near = np.minimum(low, high).max(axis=-1)
    far = np.maximum(low, high).min(axis=-1)
    hit = (near <= far) & (near > 0)
    image = np.full((height, width, 3), [100, 90, 80], dtype=np.uint8)
    image[hit] = [30, 200, 50]
    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.]])
    timing = FrameTiming(10.04, 456., 455., "hardware_clock", "hardware_clock", 31, 30, None)
    return RgbdSample(image, np.where(hit, near, 0).astype(np.float32), K, timing)


class CalibrationVisionTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "calibration.json"
        self.hsv_path = self.path.with_name("hsv.json")
        self.hsv_path.write_text((EXAMPLE_CALIBRATION.parent / "cube_hsv.json").read_text())
        self.center = np.array([0.28, -0.04, 0.055])
        self.edge = 0.035
        self.base_color = np.eye(4)
        self.base_color[:3, :3] = np.diag([1., -1., -1.])
        self.base_color[:3, 3] = [0.25, 0.01, 0.36]
        self.flange_tcp = np.array([[-1., 0., 0., 0.01], [0., -1., 0., 0.02],
                                    [0., 0., 1., 0.11], [0., 0., 0., 1.]])
        self.base_flange = np.array([[0., -1., 0., 0.2], [1., 0., 0., -0.03],
                                     [0., 0., 1., 0.57], [0., 0., 0., 1.]])
        self.camera = Mock(info={"serial": "known-camera"})
        self.camera.capture.return_value = box_sample(self.center, self.edge, self.base_color)
        self.arm = Mock()
        self.before = ArmState(np.zeros(6), np.zeros(6), self.base_flange, (0,), 10.0, 10.005)
        self.after = replace(self.before, request_start_s=10.050, request_end_s=10.055)
        self.arm.read_state.side_effect = (self.before, self.after)
        clock = patch("realman_rl_simtoreal.deployment.vision.monotonic", return_value=10.10)
        clock.start()
        self.addCleanup(clock.stop)

    def write_calibration(self, *, parent="base", parent_color=None, **fields):
        data = {
            "hsv_file": "hsv.json",
            "hand_eye": {"parent_frame": parent, "T_parent_color":
                         (self.base_color if parent_color is None else parent_color).tolist()},
            "T_flange_tcp": self.flange_tcp.tolist(),
            "cube_edge_m": self.edge,
            "camera_serial": "known-camera",
        }
        data.update(fields)
        self.path.write_text(json.dumps(data))
        return self.path

    def test_real_template_refuses_missing_values_and_does_not_fallback(self):
        with self.assertRaisesRegex(CalibrationError, r"T_parent_color.*T_flange_tcp.*cube_edge_m"):
            load_calibration(EXAMPLE_CALIBRATION)
        self.write_calibration(hsv_file=None)
        with self.assertRaisesRegex(CalibrationError, "hsv_file"):
            load_calibration(self.path)
        self.write_calibration(hsv_file="absent.json")
        with self.assertRaisesRegex(CalibrationError, "absent.json"):
            load_calibration(self.path)

    def test_bad_measurements_are_rejected_before_capture(self):
        reflection = np.diag([-1., 1., 1., 1.])
        self.write_calibration(parent_color=reflection)
        with self.assertRaisesRegex(CalibrationError, "右手刚体"):
            load_calibration(self.path)
        self.write_calibration(base_leveling=reflection.tolist())
        with self.assertRaisesRegex(CalibrationError, "base_leveling"):
            load_calibration(self.path)
        self.write_calibration(cube_edge_m=True)
        with self.assertRaisesRegex(CalibrationError, "cube_edge_m"):
            load_calibration(self.path)
        self.write_calibration()
        self.hsv_path.write_text('{"hsv_ranges":[{"low":[49,80,66],"high":[180,255,255]}]}')
        with self.assertRaisesRegex(CalibrationError, "HSV"):
            load_calibration(self.path)

    def test_mount_compositions_include_tcp_only_for_tcp_calibration(self):
        for parent, base_to_parent in (("base", np.eye(4)), ("flange", self.base_flange),
                                       ("tcp", self.base_flange @ self.flange_tcp)):
            with self.subTest(parent=parent):
                parent_color = np.linalg.inv(base_to_parent) @ self.base_color
                cal = load_calibration(self.write_calibration(parent=parent, parent_color=parent_color))
                np.testing.assert_allclose(cal.camera_to_base(self.base_flange), self.base_color, atol=1e-12)
                if parent != "base":
                    with self.assertRaisesRegex(CalibrationError, "T_base_flange"):
                        cal.camera_to_base()

    def test_offline_replay_checks_capture_calibration_and_camera_serial(self):
        audit = run_path(str(PROJECT_ROOT / "scripts/diagnostics/audit_realman_vision.py"))
        analyze_capture = audit["analyze_capture"]
        capture_path = self.path.with_name("capture_calibration.json")
        flange_color = np.linalg.inv(self.base_flange) @ self.base_color
        capture_path.write_bytes(self.write_calibration(parent="flange", parent_color=flange_color).read_bytes())
        capture_calibration = load_calibration(capture_path)
        leveling = transform([0., 0., .001], euler_matrix([0., .035, 0.]))
        tcp_color = inverse(self.base_flange @ self.flange_tcp) @ inverse(leveling) @ self.base_color
        replay_calibration = load_calibration(self.write_calibration(parent="tcp", parent_color=tcp_color,
                                                                     base_leveling=leveling.tolist()))
        sample = self.camera.capture.return_value
        record = self.path.with_name("capture.npz")
        metadata = {"calibration_sha256": capture_calibration.checksum(), "camera": self.camera.info}
        np.savez(record, rgb=sample.rgb[None], depth_m=sample.depth_m[None], K=sample.intrinsics,
                 T_base_flange=self.base_flange[None], host_received_s=[10.04], metadata=json.dumps(metadata))

        with self.assertRaisesRegex(ValueError, "校验值不符"):
            analyze_capture(record, replay_calibration)
        result = analyze_capture(record, replay_calibration, capture_calibration=capture_calibration)
        np.testing.assert_allclose(result["candidates"]["tcp"]["upright"]["median_base_m"], self.center, atol=.0018)
        np.testing.assert_allclose(result["T_base_tcp"], self.base_flange @ self.flange_tcp, atol=1e-12)
        with self.assertRaisesRegex(ValueError, "序列号"):
            analyze_capture(record, replace(replay_calibration, camera_serial="other-camera"),
                            capture_calibration=capture_calibration)

    def test_fixed_camera_measures_real_size_green_and_hides_lost_target(self):
        cal = load_calibration(self.write_calibration())
        self.assertEqual(cal.hsv_ranges, (((49, 80, 66), (84, 255, 255)),))
        localizer = RealCubeLocalizer(cal)
        first = localizer.capture(self.camera, self.arm)
        np.testing.assert_allclose(first.tracking["position_m"], self.center, atol=0.0018)
        self.assertGreater(np.count_nonzero(first.mask), 25)
        self.arm.read_state.assert_not_called()

        empty = replace(first.sample, depth_m=np.zeros_like(first.sample.depth_m),
                        timing=replace(first.sample.timing, host_received_s=10.09))
        self.camera.capture.return_value = empty
        self.assertIsNone(localizer.capture(self.camera).tracking)
        self.assertFalse(localizer.tracker.visible)

    def test_multiple_hsv_ranges_are_combined_and_camera_identity_checked(self):
        self.hsv_path.write_text(json.dumps({"hsv_ranges": [
            {"low": [3, 175, 50], "high": [25, 255, 255]},
            {"low": [49, 80, 66], "high": [84, 255, 255]},
        ]}))
        localizer = RealCubeLocalizer(load_calibration(self.write_calibration()))
        self.assertIsNotNone(localizer.capture(self.camera).tracking)
        self.camera.info["serial"] = "different-camera"
        with self.assertRaisesRegex(HardwareError, "序列号"):
            localizer.capture(self.camera)
        self.assertIsNone(localizer.tracker.measurement)

    def test_wrist_matches_confirmed_stationary_sample_and_skips_queued_exposure(self):
        parent_color = np.linalg.inv(self.base_flange @ self.flange_tcp) @ self.base_color
        cal = load_calibration(self.write_calibration(parent="tcp", parent_color=parent_color))
        sample = self.camera.capture.return_value
        queued = replace(sample, timing=replace(sample.timing, host_capture_interval_s=(9.99, 10.01)))
        fresh = replace(sample, timing=replace(sample.timing, host_capture_interval_s=(10.02, 10.03)))
        self.camera.capture.side_effect = (queued, fresh)
        observation = RealCubeLocalizer(cal).capture(self.camera, self.arm)
        np.testing.assert_allclose(observation.tracking["position_m"], self.center, atol=0.0018)
        self.assertIs(observation.sample, fresh)
        self.assertEqual(observation.arm_timestamps_s, (self.before.timestamp_s, self.after.timestamp_s))
        self.assertEqual(self.camera.capture.call_args_list[0].kwargs, {"fresh": True})

    def test_wrist_leveling_applies_once_to_camera_and_leaves_arm_readback_in_base(self):
        leveling = transform([0., 0., .001], euler_matrix([0., .035, 0.]))
        tcp = self.base_flange @ self.flange_tcp
        raw_optical = inverse(leveling) @ self.base_color
        cal = load_calibration(self.write_calibration(parent="tcp", parent_color=inverse(tcp) @ raw_optical,
                                                       base_leveling=leveling.tolist()))
        np.testing.assert_allclose(cal.camera_to_flange(), inverse(self.base_flange) @ raw_optical, atol=1e-12)
        observation = RealCubeLocalizer(cal).capture(self.camera, self.arm)
        np.testing.assert_allclose(observation.T_base_color, self.base_color, atol=1e-12)
        np.testing.assert_allclose(observation.T_base_tcp, tcp, atol=1e-12)
        np.testing.assert_allclose(observation.tracking["position_m"], self.center, atol=.0018)

    def test_wrist_rejects_motion_and_long_pose_window(self):
        parent_color = np.linalg.inv(self.base_flange) @ self.base_color
        cal = load_calibration(self.write_calibration(parent="flange", parent_color=parent_color))
        for after, message in ((replace(self.after, joint_deg=np.ones(6)), "关节变化"),
                               (replace(self.after, request_end_s=10.4), "时间窗")):
            with self.subTest(message=message):
                self.arm.read_state.side_effect = (self.before, after)
                localizer = RealCubeLocalizer(cal)
                with self.assertRaisesRegex(HardwareError, message):
                    localizer.capture(self.camera, self.arm)
                self.assertIsNone(localizer.tracker.measurement)


if __name__ == "__main__":
    unittest.main()
