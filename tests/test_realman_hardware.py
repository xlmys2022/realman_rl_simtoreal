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
from realman_rl_simtoreal.hardware.safety import GripperLimits, GripperOpeningLimits, MotionLimits, MotionSafetyError


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
        self.client.rm_get_collision_stage.return_value = (0, 4)
        self.client.rm_get_tool_voltage.return_value = (0, 3)
        self.client.rm_get_arm_current_trajectory.return_value = {"return_code": 0, "trajectory_type": 0}
        self.client.rm_movel.return_value = 0
        self.client.rm_set_arm_stop.return_value = 0
        self.client.rm_set_gripper_release.return_value = 0
        self.client.rm_set_gripper_pick.return_value = 0
        self.confirm = Mock(return_value=True)
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
        self.gripper_limits = GripperLimits(30, 40, 120, (998, 1002), (239, 243), -2,
                                            (70, 200), .5)
        self.target = np.array([-0.012, 0., 1.1892])

    def arm(self, *, motion=False, limits=None):
        return RealManArm("192.0.2.1", timeout_ms=180, enable_motion=motion,
                          limits=limits if limits is not None else self.limits, confirm=self.confirm)

    def test_read_only_returns_live_dh_flange_gripper_and_time(self):
        with self.arm() as arm:
            self.assertEqual(arm.identity["model"], "RML63II-6FB")
            state = arm.read_state()
            self.assertEqual(arm.read_tool_voltage(), 24)
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
            with self.assertRaises(MotionSafetyError):
                RealManGripper(arm, limits=self.gripper_limits).open_max()
        self.client.rm_set_timeout.assert_called_once_with(180)
        self.client.rm_movel.assert_not_called()
        self.client.rm_set_arm_stop.assert_not_called()
        self.client.rm_set_gripper_release.assert_not_called()
        self.client.rm_set_tool_voltage.assert_not_called()
        self.confirm.assert_not_called()
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

    def test_enabling_motion_alone_does_not_approve_arm_or_gripper(self):
        with RealManArm("192.0.2.1", enable_motion=True, limits=self.limits) as arm:
            with self.assertRaisesRegex(MotionSafetyError, "未获人工确认"):
                arm.move_linear_tcp(self.target, speed_pct=3)
            with self.assertRaisesRegex(MotionSafetyError, "未获人工确认"):
                RealManGripper(arm, limits=self.gripper_limits).open_max()
        self.client.rm_movel.assert_not_called()
        self.client.rm_set_gripper_release.assert_not_called()

    def test_collision_disabled_and_changed_pose_after_confirmation_prevent_movement(self):
        with self.arm(motion=True) as arm:
            self.client.rm_get_collision_stage.return_value = (0, 0)
            with self.assertRaisesRegex(MotionSafetyError, "碰撞防护关闭"):
                arm.move_linear_tcp(self.target, speed_pct=3)
            self.confirm.assert_not_called()
            self.client.rm_get_collision_stage.return_value = (0, 4)
            first = arm.read_state()
            with patch.object(arm, "read_state", side_effect=(first, self.translated_state(first, time_offset=.3))):
                with self.assertRaisesRegex(MotionSafetyError, "人工确认期间"):
                    arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_movel.assert_not_called()
        self.client.rm_set_arm_stop.assert_not_called()

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
            with patch.object(arm, "read_state", side_effect=(first, first, last)):
                self.assertIs(arm.move_linear_tcp(self.target, speed_pct=3), last)
        args = self.client.rm_movel.call_args.args
        np.testing.assert_allclose(args[0], [-0.012, 0., 1.0892, 0., 0., 0.], atol=1e-12)
        self.assertEqual(args[1:], (3, 0, 0, 0))
        self.confirm.assert_called_once()
        self.client.rm_set_arm_stop.assert_not_called()

    def test_live_grip_plus_pose_is_used_without_applying_tcp_twice(self):
        self.client.rm_get_current_tool_frame.return_value = (0, {"name": "Grip_plus", "pose": [0, 0, .10, 0, 0, 0]})
        raw = self.client.rm_get_current_arm_state.return_value[1]
        raw["pose"][2] += .10
        with self.arm(motion=True) as arm:
            first = arm.read_state()
            last = self.translated_state(first, time_offset=.3)
            with patch.object(arm, "read_state", side_effect=(first, first, last)):
                arm.move_linear_tcp(self.target, speed_pct=3)
        np.testing.assert_allclose(self.client.rm_movel.call_args.args[0], [-.012, 0, 1.1892, 0, 0, 0], atol=1e-12)

    def test_measured_overspeed_attempts_stop(self):
        with self.arm(motion=True) as arm:
            first = arm.read_state()
            fast = self.translated_state(first, time_offset=0.01)
            with patch.object(arm, "read_state", side_effect=(first, first, fast)):
                with self.assertRaisesRegex(MotionSafetyError, "线速度超限"):
                    arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_set_arm_stop.assert_called_once()

    def test_path_deviation_or_lost_hold_attempts_stop(self):
        with self.arm(motion=True) as arm:
            first = arm.read_state()
            last = self.translated_state(first, time_offset=.3)
            last.T_base_flange[1, 3] += .003
            last.controller_pose_m_rad[1] += .003
            with patch.object(arm, "read_state", side_effect=(first, first, last)):
                with self.assertRaisesRegex(MotionSafetyError, "偏离"):
                    arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_set_arm_stop.assert_called_once()
        self.client.rm_set_arm_stop.reset_mock()
        with self.arm(motion=True) as arm:
            first = arm.read_state()
            last = self.translated_state(first, time_offset=.3)
            guard = Mock(side_effect=(None, MotionSafetyError("lost grip")))
            with patch.object(arm, "read_state", side_effect=(first, first, last)):
                with self.assertRaisesRegex(MotionSafetyError, "lost grip"):
                    arm.move_linear_tcp(self.target, speed_pct=3, action="lift", interlock=guard)
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
            with patch.object(arm, "read_state", side_effect=(first, first, unsafe)):
                with self.assertRaisesRegex(MotionSafetyError, "未确认轨迹停止"):
                    arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_set_arm_stop.assert_called_once()

    def gripper_reply(self, mode, opening, force=0):
        return 0, dict(self.client.rm_get_gripper_state.return_value[1], mode=mode,
                       actpos=opening, current_force=force)

    def test_gripper_uses_speed_and_force_commands_and_checks_readback(self):
        closed = self.gripper_reply(3, 350)
        opened = self.gripper_reply(1, 1001)
        holding = self.gripper_reply(6, 241, 111)
        self.client.rm_get_gripper_state.side_effect = (closed, closed, opened, opened,
                                                        opened, opened, holding, holding, holding)
        with self.arm(motion=True) as arm:
            jaw = RealManGripper(arm, limits=self.gripper_limits)
            self.assertEqual(jaw.open_max().opening_raw, 1001)
            self.assertEqual(jaw.close_on_object().opening_raw, 241)
            self.assertEqual(jaw.require_holding().force_g, 111)
        self.assertEqual([call.args[0] for call in self.confirm.call_args_list], ["open", "close"])
        self.client.rm_set_gripper_release.assert_called_once_with(30, False, 1)
        self.client.rm_set_gripper_pick.assert_called_once_with(40, 120, False, 1)
        self.client.rm_set_gripper_position.assert_not_called()

    def test_opening_only_requires_confirmation_and_does_not_allow_gripping(self):
        closed = self.gripper_reply(6, -2, 702)
        opened = self.gripper_reply(1, 999)
        self.client.rm_get_gripper_state.side_effect = (closed, closed, opened, opened)
        limits = GripperOpeningLimits(30, (999, 1001), .5)
        with self.arm(motion=True) as arm:
            jaw = RealManGripper(arm, limits=limits)
            with self.assertRaisesRegex(MotionSafetyError, "GripperLimits"):
                jaw.close_on_object()
            with self.assertRaisesRegex(MotionSafetyError, "GripperLimits"):
                jaw.require_holding()
            self.assertEqual(jaw.open_max().opening_raw, 999)
        self.confirm.assert_called_once()
        self.assertEqual(self.confirm.call_args.args[0], "open")
        self.client.rm_set_gripper_release.assert_called_once_with(30, False, 1)
        self.client.rm_set_gripper_pick.assert_not_called()

    def test_empty_grip_or_lost_ack_latches_the_session_without_auto_open(self):
        opened = self.gripper_reply(1, 1001)
        empty = self.gripper_reply(2, 1)
        empty_under_force = self.gripper_reply(6, -2, 111)
        for answer, state in ((0, empty), (0, empty_under_force), (-2, empty)):
            with self.subTest(sdk_answer=answer, mode=state[1]["mode"]), self.arm(motion=True) as arm:
                self.client.rm_get_gripper_state.side_effect = (opened, opened, state)
                self.client.rm_set_gripper_pick.return_value = answer
                with self.assertRaisesRegex(MotionSafetyError, "本会话已锁定"):
                    RealManGripper(arm, limits=self.gripper_limits).close_on_object()
                with self.assertRaisesRegex(MotionSafetyError, "锁定"):
                    arm.move_linear_tcp(self.target, speed_pct=3)
        self.client.rm_set_gripper_release.assert_not_called()
        self.client.rm_movel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
