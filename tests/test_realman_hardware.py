"""使用模拟 SDK 回包验证 RML63II-6FB 默认只读与运动保护。"""

from ctypes import Structure, c_int
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from realman_rl_simtoreal.hardware import HardwareError, HardwareTimeout
from realman_rl_simtoreal.hardware.realman import RealManArm, RealManGripper
from realman_rl_simtoreal.hardware.realman_config import REALMAN_HARDWARE
from realman_rl_simtoreal.hardware.safety import MotionLimits, MotionSafetyError


class RobotInfo(Structure):
    _fields_ = [(name, c_int) for name in ("arm_model", "force_type", "arm_dof", "robot_controller_version")]


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.handle = object()
        self.client.rm_create_robot_arm.return_value = SimpleNamespace(id=1)
        self.client.rm_delete_robot_arm.return_value = 0
        self.client.rm_get_arm_run_mode.return_value = (0, 1)
        self.client.rm_get_current_work_frame.return_value = (0, {"name": "Base", "pose": [0]*6})
        self.client.rm_get_current_tool_frame.return_value = (0, {"name": "Arm_Tip", "pose": [0]*6})
        dh = REALMAN_HARDWARE.controller_dh
        self.client.rm_get_DH_data.return_value = (
            0, dict(a=dh.a_m, d=dh.d_m, alpha=dh.alpha_deg, offset=dh.offset_deg))
        self.client.rm_get_current_arm_state.return_value = (
            0, {"joint": [0]*6, "pose": [-0.017, 0., 1.0892, 0., 0., 0.],
                "err": {"err_len": 1, "err": ["0"]}})
        self.client.rm_get_gripper_state.return_value = (
            0, {"status": 1, "enable_state": 1, "actpos": 421, "current_force": 90,
                "temperature": 25, "mode": 3, "error": 0})
        self.client.rm_get_joint_min_pos.return_value = (0, REALMAN_HARDWARE.joint_min_deg)
        self.client.rm_get_joint_max_pos.return_value = (0, REALMAN_HARDWARE.joint_max_deg)
        self.client.rm_get_arm_max_line_speed.return_value = (0, 0.5)
        self.client.rm_get_arm_current_trajectory.return_value = {"return_code": 0, "trajectory_type": 0}
        self.client.rm_movel.return_value = 0
        self.client.rm_set_arm_stop.return_value = 0
        self.robot_model = 3

        def identify(_handle, ptr):
            info = ptr._obj
            info.arm_model = self.robot_model
            info.force_type = info.robot_controller_version = 3
            info.arm_dof = 6
            return 0

        sdk = SimpleNamespace(RoboticArm=Mock(return_value=self.client), rm_robot_info_t=RobotInfo,
                              rm_get_robot_info=identify, rm_force_type_e=SimpleNamespace(RM_MODEL_RM_ISF_E=3))
        sdk_patch = patch("realman_rl_simtoreal.hardware.realman._load_sdk", return_value=sdk)
        sdk_patch.start()
        self.addCleanup(sdk_patch.stop)

        tcp = np.eye(4)
        tcp[2, 3] = 0.1
        self.limits = MotionLimits((-0.10, -0.10, 1.10), (0.10, 0.10, 1.30),
                                   tcp, 10, 0.025, 0.02, 0.5)
        self.target = np.array([-0.012, 0., 1.1892])

    def arm(self, *, motion=False, limits=None):
        return RealManArm("192.0.2.1", timeout_ms=180, enable_motion=motion,
                          limits=limits if limits is not None else self.limits)

    def test_read_only_returns_live_dh_flange_gripper_and_time(self):
        with self.arm() as arm:
            self.assertEqual(arm.identity["model"], "RML63II-6FB")
            state = arm.read_state()
            np.testing.assert_allclose(state.T_base_flange[:3, 3], [-0.017, 0, 1.0892], atol=1e-9)
            np.testing.assert_allclose(state.T_base_flange[:3, :3], np.eye(3), atol=1e-9)
            self.assertLessEqual(state.request_start_s, state.timestamp_s)
            self.assertLessEqual(state.timestamp_s, state.request_end_s)
            gripper = RealManGripper(arm).read_state()
            self.assertTrue(gripper.online)
            self.assertEqual(gripper.opening_raw, 421)
            self.assertGreaterEqual(gripper.timestamp_s, state.timestamp_s)
            with self.assertRaises(MotionSafetyError):
                arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_set_timeout.assert_called_once_with(180)
        self.client.rm_movel.assert_not_called()
        self.client.rm_set_arm_stop.assert_not_called()
        self.client.rm_delete_robot_arm.assert_called_once()

    def test_wrong_model_and_wrong_run_mode_close_the_handle(self):
        self.robot_model = 4
        with self.assertRaisesRegex(HardwareError, "构型不匹配"), self.arm():
            self.fail("Unexpected valid connection")
        self.robot_model = 3
        self.client.rm_get_arm_run_mode.return_value = (0, 0)
        with self.assertRaisesRegex(HardwareError, "真实模式"), self.arm():
            self.fail("Unexpected valid connection")
        self.assertEqual(self.client.rm_delete_robot_arm.call_count, 2)
        self.client.rm_movel.assert_not_called()
        self.client.rm_set_arm_stop.assert_not_called()

    def test_read_timeout_never_sends_stop(self):
        self.client.rm_get_current_arm_state.return_value = (-2, {})
        with self.assertRaises(HardwareTimeout), self.arm() as arm:
            arm.read_state()
        self.client.rm_set_arm_stop.assert_not_called()
        self.client.rm_delete_robot_arm.assert_called_once()

    def test_enabling_motion_requires_site_limits(self):
        with self.assertRaisesRegex(ValueError, "MotionLimits"):
            RealManArm("192.0.2.1", enable_motion=True)

    def test_workspace_speed_increment_and_frame_reject_before_motion(self):
        with self.arm(motion=True) as arm:
            for target, speed, message in (
                ((0.4, 0, 1.1892), 3, "工作空间"),
                (self.target, 8, "现场限速"),
                (self.target, 11, "速度百分比"),
                ((0.05, 0, 1.1892), 3, "单次 TCP 位移"),
            ):
                with self.subTest(message=message), self.assertRaisesRegex(MotionSafetyError, message):
                    arm.move_linear_tcp(target, speed_pct=speed)
            self.client.rm_get_current_work_frame.return_value = (0, {"pose": [0.1, 0, 0, 0, 0, 0]})
            with self.assertRaisesRegex(MotionSafetyError, "工作系为 Base"):
                arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_movel.assert_not_called()
        self.client.rm_set_arm_stop.assert_not_called()

    def translated_state(self, first, *, time_offset):
        pose = first.controller_pose_m_rad.copy()
        pose[0] += 0.005
        flange = first.T_base_flange.copy()
        flange[0, 3] += 0.005
        return replace(first, controller_pose_m_rad=pose, T_base_flange=flange,
                       request_start_s=first.request_start_s + time_offset,
                       request_end_s=first.request_end_s + time_offset)

    def test_guarded_linear_motion_sends_only_fixed_pose_tcp_translation(self):
        with self.arm(motion=True) as arm:
            first = arm.read_state()
            last = self.translated_state(first, time_offset=0.3)
            with patch.object(arm, "read_state", side_effect=(first, last)):
                self.assertIs(arm.move_linear_tcp(self.target, speed_pct=3), last)
        args = self.client.rm_movel.call_args.args
        np.testing.assert_allclose(args[0], [-0.012, 0., 1.0892, 0., 0., 0.])
        self.assertEqual(args[1:], (3, 0, 0, 0))
        self.client.rm_set_arm_stop.assert_not_called()

    def test_measured_overspeed_attempts_stop(self):
        with self.arm(motion=True) as arm:
            first = arm.read_state()
            fast = self.translated_state(first, time_offset=0.01)
            with patch.object(arm, "read_state", side_effect=(first, fast)):
                with self.assertRaisesRegex(MotionSafetyError, "线速度超限"):
                    arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_set_arm_stop.assert_called_once()

    def test_uncertain_motion_ack_stops_and_latches_guard(self):
        self.client.rm_movel.return_value = -2
        with self.arm(motion=True) as arm:
            with self.assertRaises(HardwareTimeout):
                arm.move_linear_tcp(self.target, speed_pct=3)
            with self.assertRaisesRegex(MotionSafetyError, "锁定"):
                arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_movel.assert_called_once()
        self.client.rm_set_arm_stop.assert_called_once()

    def test_motion_completion_deadline_stops(self):
        limits = replace(self.limits, completion_timeout_s=0.005)
        with self.arm(motion=True, limits=limits) as arm:
            with self.assertRaisesRegex(HardwareTimeout, "直线平移超时"):
                arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_movel.assert_called_once()
        self.client.rm_set_arm_stop.assert_called_once()

    def test_unsafe_readback_attempts_stop_and_reports_failed_stop(self):
        self.client.rm_set_arm_stop.return_value = -1
        with self.arm(motion=True) as arm:
            first = arm.read_state()
            unsafe = replace(first, error_codes=(5,))
            with patch.object(arm, "read_state", side_effect=(first, unsafe)):
                with self.assertRaisesRegex(MotionSafetyError, "未确认轨迹停止"):
                    arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_set_arm_stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
