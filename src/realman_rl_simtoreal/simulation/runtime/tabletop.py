"""组合桌面 MuJoCo 模型、工具运动学、位置伺服、夹爪和接触测量，为桌面环境提供机器人接口。"""
import copy
import mujoco
import numpy as np
from ..devices.camera_geometry import OPTICAL_TO_MUJOCO, camera_rotation
from ...paths import TABLETOP_CARTESIAN_SCENE as DEFAULT_SCENE, TABLETOP_CARTESIAN_REALMAN_SCENE
from ...deployment.calibration import DEFAULT_CALIBRATION, load_calibration
from ...hardware.realman_config import REALMAN_TABLETOP
from ..physics.model import load_model,initialize,require
from ..devices.mujoco_gripper import Gripper,SIGNS,MAX_WIDTH,width_to_angle
from ..physics.grasp_contact import ContactPolicy,ContactRecorder,PAD_PENETRATION,SUPPORT_PENETRATION,UNEXPECTED_PENETRATION
from ..physics.joint_control import advance_synchronized_cycle
from ..motion.kinematics import rotation_vector
from ..motion.tool_kinematics import ToolKinematics
from ..physics.grasp_metrics import grasp_forces, CONTACT_MEASUREMENT


class TabletopRuntime:
    """按 config 提供的时间步、相机约定和夹爪参数构建桌面机器人。

    prepare 初始化同一份活跃 MjData 并交接观察位，advance 推进 50 个物理步。
    get_joint_state、tcp_pose 和 forces 提供关节、末端和接触读数。
    """
    def __init__(self,config):
        self.config=config
        self.scene_path = DEFAULT_SCENE if config.scene_profile == "legacy" else TABLETOP_CARTESIAN_REALMAN_SCENE
        self.model,self.joint_ids,self.actuator_ids=load_model(self.scene_path,gravity_compensation=True)
        self.model.opt.timestep=self.config.physics_timestep
        self._camera_source = self._configure_camera()
        self.kinematics=ToolKinematics(self.model,tcp_site="grasp_site")
        self.contact_policy=ContactPolicy(self.model)
        self.object_qadr=int(self.model.joint('target_free').qposadr[0])
        self.object_vadr=int(self.model.joint('target_free').dofadr[0])
        self.object_body=self.model.body('target_object').id
        self.cube_half_edge = float(self.model.geom('target_box').size[2])
        self.table_height=float(self.model.body_pos[self.model.body('support').id,2]+
                                self.model.geom_size[self.model.geom('support_top').id,2])
        self.base_position=self.model.body_pos[self.model.body('base_link').id].copy()
        self._probe=mujoco.MjData(self.model);self._cache=None;self._prepared=False

    def _configure_camera(self):
        camera = self.model.camera('wrist_rgb').id
        rotation = camera_rotation(self.config.camera_forward_local, self.config.camera_up_local)
        mujoco.mju_mat2Quat(self.model.cam_quat[camera], rotation.ravel())
        if self.config.scene_profile == "legacy":
            self.model.cam_fovy[camera] = self.config.camera_fovy_deg
            return {}
        source = {"intrinsics": "RealSense D435 640×480", "extrinsics": "legacy_mount"}
        if self.config.scene_profile == "realman_hand_eye":
            calibrated = load_calibration(self.config.hand_eye_file or DEFAULT_CALIBRATION)
            expected_tcp = np.eye(4)
            expected_tcp[2, 3] = REALMAN_TABLETOP.flange_tcp_m
            if (calibrated.camera_parent == "base"
                    or not np.allclose(calibrated.T_flange_tcp, expected_tcp, atol=1e-7, rtol=0)):
                raise ValueError("RealMan wrist scene requires the measured Grip_plus TCP and a flange/TCP parent")
            if calibrated.camera_serial not in (None, REALMAN_TABLETOP.camera_serial):
                raise ValueError("手眼标定的序列号与场景 D435 内参的来源不符")
            # 相机局部 +Y/+Z 与 OpenCV 光学轴相反。先拼接一次 TCP，再换轴。
            flange_color = calibrated.camera_to_base(np.eye(4))
            body = self.model.body('camera_link').id
            self.model.body_pos[body] = flange_color[:3, 3]
            mujoco.mju_mat2Quat(self.model.body_quat[body],
                                (flange_color[:3, :3] @ OPTICAL_TO_MUJOCO).ravel())
            self.model.cam_quat[camera] = [1, 0, 0, 0]
            source.update(extrinsics="hand_eye_candidate", parent_frame=calibrated.camera_parent,
                          calibration_sha256=calibrated.checksum())
        return source

    rotation_error=staticmethod(rotation_vector)

    def reset_zero(self,object_xy):
        self._prepared=False
        self.data=initialize(self.model)
        self.data.qpos[self.kinematics.qadr]=0
        self.data.ctrl[self.actuator_ids]=0
        angle=width_to_angle(self.config.initial_width)
        for name,sign in SIGNS.items():self.data.qpos[self.model.joint(name).qposadr[0]]=sign*angle
        self.data.ctrl[self.model.actuator('eg2_position').id]=angle
        self.data.qpos[self.object_qadr:self.object_qadr+7]=[*object_xy,self.table_height+.01,1,0,0,0]
        self.data.qvel[:]=0
        mujoco.mj_forward(self.model,self.data)
        self._new_monitors()
        self.contact_policy.check(self.data)

    def _new_monitors(self):
        self.gripper=Gripper(self.model,self.data,contact_policy=self.contact_policy,reference_speed=.5)
        self.recorder=ContactRecorder(self.model,self.data,self.gripper,self.contact_policy,keep_history=False)
        self._grasp_forces=np.zeros(3)

    def get_joint_state(self):
        """读取六轴实际位置（rad）、速度（rad/s）和上次控制参考（rad）的副本。"""
        return dict(position_rad=self.data.qpos[self.kinematics.qadr].copy(),
                    velocity_rad_s=self.data.qvel[self.kinematics.vadr].copy(),
                    reference_rad=self.data.ctrl[self.actuator_ids].copy())

    def get_gripper_state(self):
        """读取实际开口（m）及开口速度（m/s）。"""
        return dict(width_m=self.gripper.get_width(),velocity_m_s=self.gripper.get_velocity())

    def tcp_pose(self):
        """返回工具中心点在世界系的位置（m）及工具到世界系的姿态矩阵。"""
        site=self.kinematics.site_id
        return self.data.site_xpos[site].copy(),self.data.site_xmat[site].reshape(3,3).copy()

    def tcp_velocity(self):
        """通过 site 线速度雅可比与广义速度，返回世界系 TCP 线速度（m/s）。"""
        jac=np.empty((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self.data,jac,None,self.kinematics.site_id)
        return jac@self.data.qvel

    def tcp_angular_velocity(self):
        """通过 site 角速度雅可比，返回世界系 TCP 角速度（rad/s）。"""
        jac=np.empty((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self.data,None,jac,self.kinematics.site_id)
        return jac@self.data.qvel

    def camera_frame(self):
        camera=self.model.camera('wrist_rgb').id
        rotation=self.data.cam_xmat[camera].reshape(3,3)
        return dict(position_m=self.data.cam_xpos[camera].tolist(),forward=(-rotation[:,2]).tolist(),
                    up=rotation[:,1].tolist(),optical_frame_calibrated=False, **self._camera_source)

    def forces(self):
        """返回左、右夹指向内夹持力与桌面支撑力（N）的副本。"""
        return self._grasp_forces.copy()

    def raw_forces(self):
        row=self.recorder.last
        return np.zeros(3) if row is None else np.array([row['left_normal_n'],row['right_normal_n'],row['support_normal_n']])

    def reference_is_safe(self,target,*,held=None):
        q=np.asarray(target)
        limits=self.kinematics.limits
        if q.shape!=(6,) or not np.isfinite(q).all() or np.any(q<=limits[:,0]) or np.any(q>=limits[:,1]):
            return False,'joint_limit'
        self._probe.qpos[:]=self.data.qpos;self._probe.qpos[self.kinematics.qadr]=q
        mujoco.mj_forward(self.model,self._probe)
        if held is None:held=min(self.forces()[:2])>.005
        for contact in self._probe.contact:
            category=self.contact_policy.classify(*contact.geom)
            if held and category.endswith('_pad_target'):continue
            limit=(PAD_PENETRATION if category.endswith('_pad_target') else
                   SUPPORT_PENETRATION if category=='target_support' else UNEXPECTED_PENETRATION)
            if not np.isfinite(contact.dist) or contact.dist < -limit:return False,category
        return True,None

    def path_is_safe(self,start,end):
        samples=max(51,int(np.ceil(np.max(abs(end-start))/self.config.joint_step))+1)
        return all(self.reference_is_safe(start+alpha*(end-start),held=False)[0]
                   for alpha in np.linspace(0,1,samples))

    def advance(self,target,width,observer=None):
        # target 是本周期关节参考（rad），width 是目标开口（m）。接触监控按每个物理步更新。
        target=np.asarray(target,dtype=float)
        start=self.data.ctrl[self.actuator_ids]
        require(np.max(abs(target-start))<=self.config.joint_step+1e-9,'Unbounded joint-reference step')
        self.gripper.set_width(float(np.clip(width,self.config.jaw_margin,MAX_WIDTH-self.config.jaw_margin)))
        def observe(data):
            self.recorder(data)
            self._grasp_forces=grasp_forces(self.model,data,self.gripper,self.contact_policy,self.recorder)
            self.recorder.last.update(left_grasp_n=float(self._grasp_forces[0]),
                                      right_grasp_n=float(self._grasp_forces[1]))
            if observer:observer(self.recorder.last)
        return advance_synchronized_cycle(self.model,self.data,self.joint_ids,self.actuator_ids,target,self.gripper,
            physics_steps=50,step_observer=observe,max_home_delta=2*np.pi)

    def prepare(self,object_xy,tick=None):
        """Evaluation always simulates startup; training may restore a verified robot reset state."""
        from .observation_startup import move_to_observation
        self.reset_zero(object_xy)
        signature=(self.config.observation_pose,self.config.initial_width,
                   tuple(self.config.camera_forward_local),tuple(self.config.camera_up_local))
        if self.config.cache_startup and self._cache is not None and self._cache.get('signature')==signature:
            cache=self._cache
            # 缓存只恢复已验证的机器人初始化状态，保留本回合重新采样的自由物体。
            self.data.qpos[:self.object_qadr]=cache['qpos']
            self.data.qvel[:self.object_vadr]=cache['qvel']
            self.data.ctrl[:]=cache['ctrl']
            mujoco.mj_forward(self.model,self.data);self._new_monitors();self.contact_policy.check(self.data)
            report=copy.deepcopy(cache['report'])
            report.update(mode='cached_verified_observation_reset',simulated_seconds=0.,cycles=0)
            rotation=cache['rotation'].copy()
        else:
            report,rotation=move_to_observation(self,tick=tick)
            if self.config.cache_startup:
                self._cache=dict(signature=signature,qpos=self.data.qpos[:self.object_qadr].copy(),qvel=self.data.qvel[:self.object_vadr].copy(),
                                 ctrl=self.data.ctrl.copy(),report=copy.deepcopy(report),rotation=rotation.copy())
        self._prepared=True
        return report,rotation
