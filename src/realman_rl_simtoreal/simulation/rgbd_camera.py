"""MuJoCo RGB-D adapter at the existing camera_link optical center.

Only the named camera, fixed base frame and robot frames are read. Depth is a
camera measurement; object pose, segmentation IDs and contact data are absent.
"""
import mujoco
import numpy as np
from ..perception.rgbd import RgbdFrame


class MujocoRgbdCamera:
    def __init__(self,robot,*,width=640,height=480):
        self.robot=robot;self.width=width;self.height=height;self.renderer=None
        self.camera_id=robot.model.camera('wrist_rgb').id
        self.base_id=robot.model.body('base_link').id
        self.options=mujoco.MjvOption()
        # The CAD has duplicate visual/collision meshes. Draw the collision
        # mesh copy plus the actual table/cube, retaining physical occlusion.
        self.options.geomgroup[1]=0
        f=height/(2*np.tan(np.deg2rad(robot.model.cam_fovy[self.camera_id])/2))
        self.intrinsics=np.array([[f,0,(width-1)/2],[0,f,(height-1)/2],[0,0,1.]])
        self.frame=None

    def base_pose(self):
        data=self.robot.data
        return data.xpos[self.base_id].copy(),data.xmat[self.base_id].reshape(3,3).copy()

    def capture(self):
        robot=self.robot
        if self.renderer is None:
            robot.model.vis.global_.offwidth=max(robot.model.vis.global_.offwidth,self.width)
            robot.model.vis.global_.offheight=max(robot.model.vis.global_.offheight,self.height)
            self.renderer=mujoco.Renderer(robot.model,height=self.height,width=self.width)
        self.renderer.update_scene(robot.data,camera='wrist_rgb',scene_option=self.options)
        self.renderer.disable_depth_rendering();rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:depth=self.renderer.render().copy()
        finally:self.renderer.disable_depth_rendering()
        position,rotation=self.base_pose()
        camera_rotation=robot.data.cam_xmat[self.camera_id].reshape(3,3)@np.diag([1.,-1.,-1.])
        transform=np.eye(4);transform[:3,:3]=rotation.T@camera_rotation
        transform[:3,3]=rotation.T@(robot.data.cam_xpos[self.camera_id]-position)
        self.frame=RgbdFrame(rgb,depth,self.intrinsics.copy(),transform,float(robot.data.time))
        return self.frame

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
