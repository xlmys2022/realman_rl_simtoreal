"""加载已有的真机只读 RGB-D 录像，并核对采样标定来源。"""

import json

import numpy as np


def load_recording(path, calibration, original):
    with np.load(path, allow_pickle=False) as source:
        rgb, depth, K, flanges = (source[k] for k in ('rgb', 'depth_m', 'K', 'T_base_flange'))
        stamps = source['host_received_s']
        metadata = json.loads(source['metadata'].item()) if 'metadata' in source else {}
        reported = source['controller_pose_m_rad'] if 'controller_pose_m_rad' in source else None
    # 离线替换外参时仍按原配置核对采样哈希。
    if metadata.get('calibration_sha256') not in (None, original.checksum()):
        raise ValueError(f'{path} 与采样标定文件校验值不符，使用 --capture-calibration 指定原配置')
    serial = metadata.get('camera', {}).get('serial')
    if calibration.camera_serial is not None and serial not in (None, calibration.camera_serial):
        raise ValueError(f'{path} 的采样相机序列号 {serial} 与回放标定不符')
    return rgb, depth, K, flanges, stamps, metadata, reported
