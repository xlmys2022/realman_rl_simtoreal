"""以 SDK 替身、实机模型和合成观测覆盖只读规划及分段抓取。"""

from dataclasses import asdict, replace
from runpy import run_path
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from realman_rl_simtoreal.deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from realman_rl_simtoreal.deployment.grasp_config import (
    GraspGeometry, GraspSite, grasp_site, gripper_limits, gripper_opening_limits,
)
from realman_rl_simtoreal.deployment.real_grasp import GRASP_STAGES, check_tcp_path, execute_grasp, plan_grasp
from realman_rl_simtoreal.deployment.vision import RealCubeLocalizer, StationaryCube, cube_outline_yaw
from realman_rl_simtoreal.hardware import HardwareError
from realman_rl_simtoreal.hardware.realman import ArmState, GripperState, RealManGripper
from realman_rl_simtoreal.hardware.safety import GripperLimits, MotionLimits, MotionSafetyError
from realman_rl_simtoreal.paths import PROJECT_ROOT, TABLETOP_CARTESIAN_REALMAN_SCENE
from realman_rl_simtoreal.perception.rgbd import RgbdFrame
from realman_rl_simtoreal.simulation.motion.tool_kinematics import ToolKinematics
from realman_rl_simtoreal.simulation.physics.model import load_model
from realman_rl_simtoreal.utils.geometry import inverse, transform


class VisualGraspTests(unittest.TestCase):
    def setUp(self):
        self.calibration = load_calibration(DEFAULT_CALIBRATION.with_name("calibration.20261126.json"))
        self.edge = self.calibration.cube_edge_m
        motion = MotionLimits((.1, -.2, -.03), (.55, .2, .5), self.calibration.T_flange_tcp,
                              8, .03, .11, 10.)
        gripper = GripperLimits(45, 45, 100, (997, 1002), (237, 245), 2, (750, 820), 2.)
        geometry = GraspGeometry(-.04, -.025, .06, .04, .05, "x", 4., 12., .004, .004, 3)
        self.site = GraspSite(motion, gripper, geometry)
        self.pose = transform([.36, .02, .10], np.diag([1., -1., -1.]))
        self.center = np.array([.4, .0, -.029])
        self.estimate = StationaryCube(self.center, (0.01, 0.0, np.pi / 2), self.pose, 7, 8, .0004)

    def test_plan_preserves_fixed_orientation_and_uses_measured_descend_height(self):
        plan = plan_grasp(self.estimate, self.site, self.edge)
        expected = ((.4, 0., .10), (.4, 0., .015), (.4, 0., -.025), (.4, 0., .025))
        np.testing.assert_allclose(list(plan.waypoints.values()), expected, atol=1.e-12)
        self.assertAlmostEqual(plan.z_error_m, .001)

    def test_pregrasp_plan_stops_at_approved_height_with_only_opening_limits(self):
        motion = asdict(self.site.motion)
        del motion["T_flange_tcp"]
        motion["workspace_min_m"] = (.1, -.2, .013)
        data = {"motion": motion, "grasp": asdict(self.site.geometry),
                "gripper": {"open_speed_raw": 45, "open_feedback_raw": [997, 1002],
                            "completion_timeout_s": 2., "close_force_raw": None}}
        site = grasp_site(data, self.calibration, until="pregrasp")
        plan = plan_grasp(self.estimate, site, self.edge, until="pregrasp")
        self.assertEqual(tuple(plan.waypoints), ("align", "pregrasp"))
        with self.assertRaisesRegex(ValueError, r"gripper.close_force_raw"):
            grasp_site(data, self.calibration)
        data["max_grasp_stage"] = "pregrasp"
        with self.assertRaisesRegex(ValueError, "只批准执行到 pregrasp"):
            grasp_site(data, self.calibration, until="descend")
        arm, gripper = Mock(), Mock()
        with self.assertRaisesRegex(MotionSafetyError, "重新预检"):
            execute_grasp(plan, site, self.edge, arm, gripper, Mock(), until="lift", report=Mock())
        arm.check_motion_ready.assert_not_called()
        gripper.open_max.assert_not_called()

    def test_resume_from_pregrasp_rechecks_height_and_never_moves_xy(self):
        pose = self.pose.copy()
        pose[:3, 3] = [.401, .001, .015]
        estimate = replace(self.estimate, T_base_tcp=pose)
        plan = plan_grasp(estimate, self.site, self.edge, start_after="pregrasp", until="descend")
        self.assertEqual(tuple(plan.waypoints), ("descend",))
        self.assertEqual(plan.waypoints["descend"], (.401, .001, -.025))
        changed = pose.copy()
        changed[2, 3] += .002
        with self.assertRaisesRegex(MotionSafetyError, "检查点"):
            plan_grasp(replace(estimate, T_base_tcp=changed), self.site, self.edge,
                       start_after="pregrasp", until="descend")
        changed = pose.copy()
        changed[1, 3] += .004
        with self.assertRaisesRegex(MotionSafetyError, "低位横移"):
            plan_grasp(replace(estimate, T_base_tcp=changed), self.site, self.edge,
                       start_after="pregrasp", until="descend")

    def test_closed_checkpoint_only_lifts_after_holding_is_rechecked(self):
        live, commands, arm, gripper = self.fake_devices()
        live[:3, 3] = [.4, 0., self.site.geometry.grasp_tcp_z_base_m]
        estimate = replace(self.estimate, T_base_tcp=live.copy(), yaw_rad=())
        plan = plan_grasp(estimate, self.site, self.edge, start_after="close", until="lift")
        self.assertEqual(tuple(plan.waypoints), ("lift",))

        gripper.require_holding.side_effect = MotionSafetyError("夹持丢失")
        with self.assertRaisesRegex(MotionSafetyError, "夹持丢失"):
            execute_grasp(plan, self.site, self.edge, arm, gripper, Mock(), until="lift",
                          start_after="close", report=Mock())
        self.assertEqual(commands, [])

        gripper.require_holding.side_effect = None

        def observe():
            return replace(estimate, T_base_tcp=live.copy(),
                           center_m=estimate.center_m + [0., 0., self.site.geometry.lift_height_m])

        result = execute_grasp(plan, self.site, self.edge, arm, gripper, observe, until="lift",
                               start_after="close", report=Mock())
        self.assertEqual(result["status"], "lift_verified")
        self.assertEqual(commands, ["lift"])
        gripper.open_max.assert_not_called()
        gripper.close_on_object.assert_not_called()

    def test_visual_z_bias_unaligned_jaw_and_workspace_are_rejected(self):
        samples = (
            (replace(self.estimate, center_m=np.array([.4, 0., -.0175])), "绝对 Z"),
            (replace(self.estimate, yaw_rad=(np.deg2rad(22.),) * 3), "夹角"),
            (replace(self.estimate, center_m=np.array([.59, .0, -.029])), "工作空间"),
        )
        for sample, message in samples:
            with self.subTest(message=message), self.assertRaisesRegex(MotionSafetyError, message):
                plan_grasp(sample, self.site, self.edge)

    def test_yaw_uses_four_projected_edges_and_rejects_oversized_mask(self):
        rgb = np.zeros((160, 160, 3), dtype=np.uint8)
        mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        K = np.array([[900., 0., 80.], [0., 900., 80.], [0., 0., 1.]])
        optical = transform([0., 0., .40], np.diag([1., -1., -1.]))
        frame = RgbdFrame(rgb, np.full(mask.shape, .42), K, optical, 1.)
        angle_deg = 17.
        width_px = 900 * self.edge / .42
        corners = cv2.boxPoints(((80., 80.), (width_px, width_px), -angle_deg))
        cv2.fillConvexPoly(mask, np.rint(corners).astype(np.int32), 255)
        self.assertAlmostEqual(np.rad2deg(cube_outline_yaw(frame, mask, -.020, self.edge)), angle_deg, delta=1.)
        mask.fill(255)
        self.assertIsNone(cube_outline_yaw(frame, mask, -.020, self.edge))

    def fake_devices(self):
        live = self.pose.copy()
        commands = []
        # 替身关节只用于测试流程。真实 MuJoCo FK/IK 由独立用例覆盖。
        path_preflight = patch("realman_rl_simtoreal.deployment.real_grasp.check_tcp_path")
        self.check_segment = path_preflight.start()
        self.addCleanup(path_preflight.stop)

        def state():
            flange = live @ inverse(self.site.motion.T_flange_tcp)
            return ArmState(np.zeros(6), np.zeros(6), flange, (0,), 1., 1.)

        def move(goal, *, speed_pct, action, preflight, interlock):
            preflight()
            interlock()
            commands.append(action)
            live[:3, 3] = goal
            return state()

        open_state = GripperState(True, True, 1001, 0, 25, 1, 0, 1.)
        held = replace(open_state, opening_raw=241, force_g=787, mode=6)

        def jaw_action(action, reply, preflight):
            preflight()
            commands.append(action)
            return reply

        arm = Mock(check_motion_ready=Mock(side_effect=state), move_linear_tcp=Mock(side_effect=move))
        gripper = Mock(open_max=Mock(side_effect=lambda preflight: jaw_action("open", open_state, preflight)),
                       close_on_object=Mock(side_effect=lambda preflight: jaw_action("close", held, preflight)),
                       require_open=Mock(return_value=open_state), require_holding=Mock(return_value=held))
        return live, commands, arm, gripper

    def test_execute_stages_and_verify_lift_by_visual_delta(self):
        live, commands, arm, gripper = self.fake_devices()
        plan = plan_grasp(self.estimate, self.site, self.edge)

        def observe():
            delta = self.site.geometry.lift_height_m if "lift" in commands else 0.
            return replace(self.estimate, T_base_tcp=live.copy(), center_m=self.center + [0., 0., delta])

        trace = []
        result = execute_grasp(plan, self.site, self.edge, arm, gripper, observe, until="lift", report=trace.append)
        self.assertEqual(result["status"], "lift_verified")
        self.assertEqual(tuple(commands), GRASP_STAGES)
        self.assertEqual(tuple(item["stage"] for item in trace), GRASP_STAGES)
        self.assertEqual(gripper.require_open.call_count, 3)
        self.assertEqual(gripper.require_holding.call_count, 3)
        self.assertEqual(self.check_segment.call_count, 4)

    def test_recheck_of_joint_branch_prevents_executing_a_segment(self):
        _, commands, arm, gripper = self.fake_devices()
        plan = plan_grasp(self.estimate, self.site, self.edge)
        self.check_segment.side_effect = MotionSafetyError("实际起点关节分支不满足预检")
        with self.assertRaisesRegex(MotionSafetyError, "关节分支"):
            execute_grasp(plan, self.site, self.edge, arm, gripper, Mock(), until="lift", report=Mock())
        self.assertEqual(commands, ["open"])

    def test_optional_stop_stage_and_target_shift_do_not_grasp(self):
        live, commands, arm, gripper = self.fake_devices()
        plan = plan_grasp(self.estimate, self.site, self.edge)
        result = execute_grasp(plan, self.site, self.edge, arm, gripper, Mock(), until="open", report=Mock())
        self.assertEqual(result["stage"], "open")
        self.assertEqual(commands, ["open"])
        observe = Mock(return_value=replace(self.estimate, T_base_tcp=live, center_m=self.center + [.01, 0., 0.]))
        with self.assertRaisesRegex(MotionSafetyError, "方块移动"):
            execute_grasp(plan, self.site, self.edge, arm, gripper, observe, until="lift", report=Mock())
        self.assertNotIn("align", commands)
        gripper.close_on_object.assert_not_called()

    def test_lift_without_visible_cube_requires_manual_confirmation(self):
        live, commands, arm, gripper = self.fake_devices()
        plan = plan_grasp(self.estimate, self.site, self.edge)

        def observe():
            if "lift" in commands:
                raise HardwareError("目标被夹指遮挡")
            return replace(self.estimate, T_base_tcp=live.copy())

        result = execute_grasp(plan, self.site, self.edge, arm, gripper, observe, until="lift", report=Mock())
        self.assertEqual(result["status"], "lift_needs_visual_confirmation")
        self.assertEqual(commands.count("open"), 1)

    def test_lift_cannot_pass_when_grip_is_lost_during_visual_verification(self):
        live, commands, arm, gripper = self.fake_devices()
        plan = plan_grasp(self.estimate, self.site, self.edge)

        def observe():
            delta = 0.
            if "lift" in commands:
                delta = self.site.geometry.lift_height_m
                gripper.require_holding.side_effect = MotionSafetyError("停稳后夹持丢失")
            return replace(self.estimate, T_base_tcp=live.copy(), center_m=self.center + [0., 0., delta])

        with self.assertRaisesRegex(MotionSafetyError, "夹持丢失"):
            execute_grasp(plan, self.site, self.edge, arm, gripper, observe, until="lift", report=Mock())

    def test_stationary_estimate_rejects_missing_detections_and_pose_changes(self):
        locator = RealCubeLocalizer(self.calibration)
        detection = Mock(T_base_tcp=self.pose, tracking={"position_m": self.center}, yaw_rad=0.)
        missing = Mock(T_base_tcp=self.pose, tracking=None, yaw_rad=None)
        with patch.object(locator, "capture", side_effect=[detection] * 6 + [missing] * 2):
            self.assertEqual(locator.estimate_stationary(None, None, max_spread_m=.003).detections, 6)
        with patch.object(locator, "capture", return_value=missing):
            with self.assertRaisesRegex(HardwareError, "0/8"):
                locator.estimate_stationary(None, None, max_spread_m=.003)
        moved = self.pose.copy()
        moved[1, 3] += .01
        mismatch = Mock(T_base_tcp=moved, tracking=None)
        with patch.object(locator, "capture", side_effect=[detection] * 7 + [mismatch]):
            with self.assertRaisesRegex(HardwareError, "不静止"):
                locator.estimate_stationary(None, None, max_spread_m=.003)

    def test_read_only_entry_does_not_dispatch_jaw_command(self):
        script = run_path(str(PROJECT_ROOT / "scripts/demos/grasp_real_cube.py"))
        main = script["main"]
        _, _, arm, gripper = self.fake_devices()
        device = Mock()
        device.return_value.__enter__ = Mock(return_value=arm)
        device.return_value.__exit__ = Mock(return_value=False)
        arm.identity = {"model": "test"}
        arm.read_frames.return_value = {}
        arm.read_collision_stage.return_value = 4
        gripper.read_state.return_value = gripper.require_open.return_value
        gripper.verify_ready.side_effect = RealManGripper.verify_ready
        with patch.dict(main.__globals__, {"read_site": Mock(return_value={}), "RealManArm": device,
                                           "RealManGripper": Mock(return_value=gripper),
                                           "motion_limits": Mock(return_value=self.site.motion),
                                           "gripper_opening_limits": Mock(return_value=self.site.gripper), "print_json": Mock()}):
            self.assertEqual(main(["open", "--site", "fixture.json", "--ip", "192.0.2.1"]), 0)
            gripper.read_state.return_value = replace(gripper.require_open.return_value, online=False, enabled=False)
            with patch("sys.stderr") as diagnostic:
                self.assertEqual(main(["open", "--site", "fixture.json", "--ip", "192.0.2.1"]), 1)
            self.assertIn("夹爪状态异常", diagnostic.write.call_args_list[0].args[0])
        self.assertFalse(device.call_args.kwargs["enable_motion"])
        gripper.open_max.assert_not_called()

    def test_opening_config_requires_experimental_speed_feedback_and_timeout(self):
        data = {"gripper": {"open_speed_raw": 45, "open_feedback_raw": [1001, 1001],
                            "completion_timeout_s": 2.0, "close_force_raw": None}}
        self.assertEqual(gripper_opening_limits(data).open_feedback_raw, (1001, 1001))
        with self.assertRaisesRegex(ValueError, r"gripper.close_force_raw"):
            gripper_limits(data)
        del data["gripper"]["completion_timeout_s"]
        with self.assertRaisesRegex(ValueError, r"gripper.completion_timeout_s"):
            gripper_opening_limits(data)

    def test_survey_reads_tool_and_target_with_only_the_measured_table_height(self):
        main = run_path(str(PROJECT_ROOT / "scripts/demos/grasp_real_cube.py"))["main"]
        _, commands, arm, gripper = self.fake_devices()
        arm.read_state.return_value = arm.check_motion_ready()
        arm.identity = {"model": "test"}
        arm.read_frames.return_value = {}
        arm.read_collision_stage.return_value = 4
        gripper.read_state.return_value = gripper.require_open.return_value
        arm_device = Mock()
        arm_device.return_value.__enter__ = Mock(return_value=arm)
        arm_device.return_value.__exit__ = Mock(return_value=False)
        camera_device = Mock()
        camera = Mock(info={"color_fps": 30})
        camera_device.return_value.__enter__ = Mock(return_value=camera)
        camera_device.return_value.__exit__ = Mock(return_value=False)
        localizer = Mock()
        localizer.return_value.estimate_stationary.return_value = self.estimate
        output = Mock()
        with patch.dict(main.__globals__, {"read_site": Mock(return_value={"grasp": {"table_z_base_m": -.04}}),
                                           "RealManArm": arm_device, "RealManGripper": Mock(return_value=gripper),
                                           "RealSenseCamera": camera_device, "RealCubeLocalizer": localizer,
                                           "print_json": output}):
            self.assertEqual(main(["survey", "--site", "fixture.json", "--ip", "192.0.2.1"]), 0)
        self.assertFalse(arm_device.call_args.kwargs.get("enable_motion", False))
        arm.move_linear_tcp.assert_not_called()
        gripper.open_max.assert_not_called()
        self.assertEqual(camera.capture.call_count, 120)
        self.assertEqual(commands, [])
        measured = output.call_args.args[0]
        self.assertEqual(measured["status"], "survey_complete")
        self.assertAlmostEqual(measured["signed_cube_z_error_m"], .001)
        self.assertAlmostEqual(measured["vertical_tcp_tilt_deg"], 0.)

    def test_path_check_reuses_realman_tcp_kinematics(self):
        model, _, _ = load_model(TABLETOP_CARTESIAN_REALMAN_SCENE)
        kin = ToolKinematics(model, "grasp_site")
        joints = np.array([0., -15.6176, 72.9939, 0., 122.6237, 90.])
        position, rotation = kin.body_pose(np.deg2rad(joints), "link_6")
        flange = transform(position - model.body("base_link").pos, rotation)
        state = ArmState(joints, np.zeros(6), flange, (0,), 1., 1.)
        current = (flange @ self.site.motion.T_flange_tcp)[:3, 3]
        check_tcp_path(state, self.site.motion, [current + [.006, 0., 0.]])
        flange[2, 3] += .015
        with self.assertRaisesRegex(MotionSafetyError, "模型不一致"):
            check_tcp_path(state, self.site.motion, [current])


if __name__ == "__main__":
    unittest.main()
