"""用已配对的真机数据构造影子策略输入，输出只读的米制动作请求。"""

import numpy as np

from ..hardware.realman_config import REALMAN_TABLETOP
from ..perception.rgbd import RgbdFrame, UprightCubeDetector, VisualTracker
from ..simulation.runtime.observation_presets import REALMAN_TRANSFER_PROFILE
from ..tasks.grasp.tabletop_policy_io import VisualPolicyObserver, VisualPolicyProprioception, decode_cartesian_action


class RealVisualPolicyIO:
    """接收 Base/Grip_plus 配对的 RGB-D 与本体数据，用于离线或影子运行。

    RGB-D 可来自 RealCubeLocalizer 的对齐帧。TCP 参考、米制夹距及速度、IK 接受比例
    由调用端提供。当前原始开口反馈及单段直线 SDK 回执无法提供这些通道。
    """

    def __init__(self, config):
        if config.scene_profile != REALMAN_TRANSFER_PROFILE:
            raise ValueError('RealVisualPolicyIO requires the realman_transfer scene contract')
        self.config = config
        detector = UprightCubeDetector(edge_m=config.vision_cube_size,
                                       hsv_low=config.vision_hsv_low, hsv_high=config.vision_hsv_high)
        self.tracker = VisualTracker(detector, timeout=config.vision_timeout)
        self.observer = VisualPolicyObserver(config)

    def reset(self):
        self.tracker.reset()
        self.observer.reset()

    def ingest_frame(self, frame: RgbdFrame):
        """加入已按实测标定变换到 Base 的对齐帧。时间戳与本体数据共用主机时钟。"""
        if (frame.depth_m.shape != (self.config.vision_height, self.config.vision_width)
                or not np.allclose(frame.intrinsics, REALMAN_TABLETOP.camera_intrinsics, atol=.001, rtol=0)):
            raise ValueError('RGB-D stream does not match the calibrated D435 profile')
        if self.tracker.last_capture is not None and frame.timestamp <= self.tracker.last_capture:
            raise ValueError('RGB-D frame timestamps must increase')
        return self.tracker.update(frame)

    def observation(self, sample: VisualPolicyProprioception):
        """用完整、同域的本体读数和未过期的跟踪估计构造 29 维输入。"""
        if not isinstance(sample, VisualPolicyProprioception):
            raise ValueError('Complete SI proprioception is required before real policy inference')
        open_gap = REALMAN_TABLETOP.gripper_measured_open_gap_m
        if not (0 <= sample.gripper_width_m <= open_gap and 0 <= sample.jaw_reference_m <= open_gap):
            raise ValueError('Real gripper telemetry must use the measured gap range in metres')
        capture = self.tracker.last_capture
        if capture is None or capture > sample.timestamp_s:
            raise ValueError('Policy observation requires an earlier RGB-D frame in the same clock domain')
        visual = self.tracker.state(sample.timestamp_s)
        if visual['expired']:
            raise ValueError('RGB-D target expired; acquire a new valid observation')
        return self.observer.observe(sample, visual)

    def decode_action(self, action, previous_jaw_reference_m=None):
        """将动作解释成 Base 增量和米制开口请求。请求需经独立的执行保护与人工确认。"""
        if (self.config.jaw_control == 'delta' and (previous_jaw_reference_m is None
                or not 0 <= previous_jaw_reference_m <= REALMAN_TABLETOP.gripper_measured_open_gap_m)):
            raise ValueError('Incremental jaw request requires a previous metric reference, not raw SDK feedback')
        return decode_cartesian_action(action, self.config, previous_jaw_reference_m)
