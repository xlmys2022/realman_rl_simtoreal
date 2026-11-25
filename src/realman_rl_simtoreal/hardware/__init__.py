"""真实机械臂、夹爪和相机的硬件适配边界。"""


class HardwareError(RuntimeError):
    """设备连接、通信或状态异常。"""


class HardwareTimeout(HardwareError):
    """硬件通信或等待设备响应超时。"""
