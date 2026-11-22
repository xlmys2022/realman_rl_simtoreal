"""从 MuJoCo 腕部相机采样 RGB 和米制深度，生成感知模块共用的 RgbdFrame。"""
import mujoco
import numpy as np
from ...perception.rgbd import RgbdFrame


class MujocoRgbdCamera:
    """从 robot 的腕部相机按 width、height（像素）渲染 RGB-D。

    intrinsics 为像素内参，frame 为最新采样。capture 返回包含相机到基座变换的帧，
    close 释放惰性创建的渲染器。
    """
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
        """返回机械臂基座在世界系的位置（m）及基座到世界系的旋转矩阵。"""
        data=self.robot.data
        return data.xpos[self.base_id].copy(),data.xmat[self.base_id].reshape(3,3).copy()

    def capture(self):
        """采集 RGB、深度（m）、内参、相机到基座位姿和仿真时间戳（秒）。"""
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
        # OpenCV 光学系向右、向下、向前。翻转 MuJoCo 的 Y/Z 轴后，再从世界系变换到机器人基座系。
        camera_rotation=robot.data.cam_xmat[self.camera_id].reshape(3,3)@np.diag([1.,-1.,-1.])
        transform=np.eye(4);transform[:3,:3]=rotation.T@camera_rotation
        transform[:3,3]=rotation.T@(robot.data.cam_xpos[self.camera_id]-position)
        self.frame=RgbdFrame(rgb,depth,self.intrinsics.copy(),transform,float(robot.data.time))
        return self.frame

    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
