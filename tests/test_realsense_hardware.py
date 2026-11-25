"""用离线帧和相机 SDK 替身验证 RGB-D、校正、时间戳和资源释放。"""

from enum import Enum
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from realman_rl_simtoreal.hardware import HardwareTimeout
from realman_rl_simtoreal.hardware.realsense import RealSenseCamera


class Distortion(Enum):
    none = 0
    brown_conrady = 1
    inverse_brown_conrady = 2
    modified_brown_conrady = 3


class Clock(Enum):
    hardware_clock = 0
    global_time = 1
    system_time = 2


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.zeros((2, 4, 3), dtype=np.uint8)
        self.rgb[:, 1] = [5, 219, 12]
        self.z16 = np.array([[100, 120, 0, 170], [200, 300, 400, 500]], dtype=np.uint16)
        self.intr = SimpleNamespace(width=4, height=2, fx=100., fy=80., ppx=1., ppy=0.5,
                                    coeffs=[0]*5, model=Distortion.none)
        self.color_stream = Mock()
        self.color_stream.as_video_stream_profile.return_value = self.color_stream
        self.color_stream.get_intrinsics.return_value = self.intr
        self.color_stream.fps.return_value = 30
        self.depth_stream = Mock()
        self.depth_stream.as_video_stream_profile.return_value = self.depth_stream
        self.depth_stream.fps.return_value = 30
        self.extr = SimpleNamespace(rotation=np.eye(3).flatten(order="F"), translation=[0., 0., 0.])
        self.color_stream.get_extrinsics_to.return_value = self.extr

        devices = [Mock()]
        devices[0].get_info.side_effect = {"serial": "A", "name": "Fake Camera"}.__getitem__
        self.context = Mock()
        self.context.query_devices.return_value = devices
        self.profile = Mock()
        self.profile.get_stream.side_effect = {"color": self.color_stream, "depth": self.depth_stream}.__getitem__
        self.profile.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value = 0.002
        self.pipeline = Mock()
        self.pipeline.start.return_value = self.profile
        self.cfg = Mock()
        self.align = Mock()
        self.align.process.return_value.get_depth_frame.return_value.get_data.return_value = self.z16
        self.rs = SimpleNamespace(
            context=Mock(return_value=self.context), config=Mock(return_value=self.cfg),
            pipeline=Mock(return_value=self.pipeline), align=Mock(return_value=self.align),
            stream=SimpleNamespace(color="color", depth="depth"),
            format=SimpleNamespace(rgb8="rgb8", z16="z16"),
            camera_info=SimpleNamespace(serial_number="serial", name="name"),
            distortion=Distortion, rs2_project_point_to_pixel=Mock(),
        )
        import_patch = patch("realman_rl_simtoreal.hardware.realsense.import_module", return_value=self.rs)
        import_patch.start()
        self.addCleanup(import_patch.stop)
        self.frames = self.pair()
        self.pipeline.try_wait_for_frames.return_value = (True, self.frames)

    def pair(self, *, number=1, color_ms=111., depth_ms=110., clock=Clock.hardware_clock):
        frames = Mock()
        rgb, depth = frames.get_color_frame.return_value, frames.get_depth_frame.return_value
        rgb.get_data.return_value = self.rgb
        depth.get_units.return_value = 0.002
        for frame, timestamp, sequence in ((rgb, color_ms, number), (depth, depth_ms, number + 3)):
            frame.get_timestamp.return_value = timestamp
            frame.get_frame_timestamp_domain.return_value = clock
            frame.get_frame_number.return_value = sequence
        return frames

    def test_alignment_intrinsics_optical_z_and_timestamp(self):
        angle = 0.05
        c, s = np.cos(angle), np.sin(angle)
        self.extr.rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]).flatten(order="F")
        self.extr.translation = [0, 0, 0.02]
        with RealSenseCamera(serial="A", width=4, height=2) as camera:
            sample = camera.capture(fresh=True)
            np.testing.assert_array_equal(sample.rgb, self.rgb)
            self.assertIsNot(sample.rgb, self.rgb)
            factor = c - s * (np.arange(4) - 1) / 100
            expected_z = np.where(self.z16 > 0, (self.z16 * 0.002 - 0.02) / factor, 0.)
            np.testing.assert_allclose(sample.depth_m, expected_z, atol=1e-7)
            np.testing.assert_array_equal(camera.intrinsics, [[100, 0, 1], [0, 80, 0.5], [0, 0, 1]])
            self.assertEqual(camera.info["serial"], "A")
            self.assertEqual(sample.timing.color_clock, "hardware_clock")
            self.assertIsNone(sample.timing.host_capture_interval_s)
            self.assertEqual(sample.timing.depth_frame_number, 4)
            self.assertEqual(sample.timing.depth_timestamp_ms, 110.)
            frame = sample.to_frame(np.eye(4))
            self.assertEqual(frame.timestamp, sample.timing.host_received_s)
        self.cfg.enable_device.assert_called_once_with("A")
        self.rs.align.assert_called_once_with("color")
        self.align.process.assert_called_once_with(self.frames)
        self.pipeline.poll_for_frames.assert_called_once()
        self.pipeline.stop.assert_called_once()

    def test_nonlinear_color_map_rectifies_both_streams(self):
        self.intr.model = Distortion.inverse_brown_conrady
        self.intr.coeffs[0] = 0.1
        # 测试 remap 的方向和配对，SDK 映射替身将原图索引向右移一列。
        self.rs.rs2_project_point_to_pixel.side_effect = lambda intr, ray: (
            ray[0] * intr.fx + intr.ppx + 1, ray[1] * intr.fy + intr.ppy)
        with RealSenseCamera() as camera:
            sample = camera.capture()
        np.testing.assert_array_equal(sample.rgb[:, 0], self.rgb[:, 1])
        np.testing.assert_allclose(sample.depth_m[:, 0], self.z16[:, 1] * 0.002)
        np.testing.assert_array_equal(sample.depth_m[:, -1], [0, 0])
        self.assertEqual(self.rs.rs2_project_point_to_pixel.call_count, 8)

    def test_global_clock_interval_and_duplicate_frame_is_dropped(self):
        first = self.pair(color_ms=10_005, depth_ms=10_001, clock=Clock.global_time)
        newer = self.pair(number=2, color_ms=10_023, depth_ms=10_019, clock=Clock.global_time)
        self.pipeline.try_wait_for_frames.side_effect = [(True, first), (True, first), (True, newer)]
        with patch("realman_rl_simtoreal.hardware.realsense.time", return_value=10.03):
            with RealSenseCamera() as camera:
                a = camera.capture().timing
                b = camera.capture().timing
        np.testing.assert_allclose(a.host_capture_interval_s,
                                   np.array([-0.029, -0.025]) + a.host_received_s)
        self.assertEqual(b.color_frame_number, 2)
        self.assertEqual(self.pipeline.try_wait_for_frames.call_count, 3)

    def test_bad_pair_skew_and_timeout_close_stream(self):
        bad = self.pair(color_ms=10_000, depth_ms=10_200, clock=Clock.global_time)
        self.pipeline.try_wait_for_frames.side_effect = [(True, bad), (False, None), (False, None)]
        with self.assertRaisesRegex(HardwareTimeout, "时间差"), RealSenseCamera() as camera:
            camera.capture()
        with self.assertRaises(HardwareTimeout), RealSenseCamera(timeout_ms=22) as camera:
            camera.capture()
        self.assertEqual(self.pipeline.stop.call_count, 2)

    def test_startup_pair_with_old_depth_is_skipped_until_recovery(self):
        bad = self.pair(color_ms=9400, depth_ms=9000, clock=Clock.global_time)
        good = self.pair(number=2, color_ms=10_000, depth_ms=10_003, clock=Clock.global_time)
        self.pipeline.try_wait_for_frames.side_effect = [(True, bad), (True, good)]
        with RealSenseCamera() as camera:
            sample = camera.capture()
        self.assertEqual(sample.timing.color_frame_number, 2)
        self.align.process.assert_called_once_with(good)


if __name__ == "__main__":
    unittest.main()
