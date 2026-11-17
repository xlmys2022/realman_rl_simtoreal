#!/usr/bin/env python3
"""Capture existing wrist RGB-D, HSV mask and a cube position in robot base."""
import argparse
import json
import os
from pathlib import Path


def main():
    os.environ.setdefault('MUJOCO_GL','egl')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=990000)
    parser.add_argument('--object-radius',type=float,default=0.)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Choose a new output directory')
    import cv2
    import numpy as np
    from realman_rl_simtoreal.tasks.tabletop_cartesian import CartesianTabletopConfig
    from realman_rl_simtoreal.envs.factory import make_tabletop_cartesian_env
    env=make_tabletop_cartesian_env(CartesianTabletopConfig(observation_source='rgbd_hsv',object_radius=args.object_radius))
    try:
        _,info=env.reset(seed=args.seed);frame=env.camera.frame
        args.output_dir.mkdir(parents=True)
        cv2.imwrite(str(args.output_dir/'rgb.png'),cv2.cvtColor(frame.rgb,cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.output_dir/'mask.png'),env.tracker.detector.mask)
        np.save(args.output_dir/'depth_m.npy',frame.depth_m)
        millimetres=np.where(np.isfinite(frame.depth_m)&(frame.depth_m>0)&(frame.depth_m<65.535),frame.depth_m*1000,0).astype(np.uint16)
        cv2.imwrite(str(args.output_dir/'depth_mm.png'),millimetres)
        report=dict(vision=info['vision'],intrinsics=frame.intrinsics.tolist(),
            camera_to_base=frame.camera_to_base.tolist(),coordinate_units='metres',
            depth_convention='aligned optical-Z; depth_m.npy float32 metres; depth_mm.png uint16 millimetres',
            camera_body='camera_link',camera_local_position_m=env.robot.model.cam_pos[env.camera.camera_id].tolist(),
            calibration_source='MuJoCo camera intrinsics and robot base/camera frame; hardware calibration not supplied',
            hsv_low=list(env.config.vision_hsv_low),hsv_high=list(env.config.vision_hsv_high),cube_edge_m=env.config.vision_cube_size,
            startup=info['startup'],simulator_object_state_in_perception=False)
        (args.output_dir/'detection.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(info['vision'],ensure_ascii=False),flush=True)
    finally:env.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
