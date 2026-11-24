"""将桌面抓取环境给出的 TCP 增量转为受工作空间、关节速率和碰撞约束的关节参考值。"""
import numpy as np
import mujoco
from .kinematics import rotation_vector


def rotation_matrix(vector):
    """将世界系轴角向量（rad）转换为 3×3 旋转矩阵。"""
    vector=np.asarray(vector,dtype=float)
    angle=float(np.linalg.norm(vector))
    if angle<1e-15:return np.eye(3)
    quaternion=np.empty(4);matrix=np.empty(9)
    mujoco.mju_axisAngle2Quat(quaternion,vector/angle,angle)
    mujoco.mju_quat2Mat(matrix,quaternion)
    return matrix.reshape(3,3)


class CartesianController:
    """基于已完成观察位启动的 runtime 和初始工具姿态创建控制器。

    reference 和 orientation 是上一轮接受的 TCP 命令。execute 根据实际位姿限制
    参考滞后，选择本周期可执行的增量，再调用仿真执行器。
    """
    def __init__(self,runtime,orientation):
        if not runtime._prepared:raise RuntimeError('Traditional setup must finish before Cartesian control')
        self.robot=runtime;self.orientation=np.array(orientation,dtype=float)
        self.reference=runtime.tcp_pose()[0]
        self.last_fraction=1.;self.last_reason=None

    def execute(self,delta_position,width,observer=None,*,delta_rotation=None):
        """执行一周期的世界系平移（m）、旋转向量（rad）和夹爪开口目标（m）。

        observer 可接收物理步记录。返回请求、实际发出的参考、IK 接受比例和物理步数。
        遇到不合法输入时抛出 ValueError，无法通过预检时保留机械臂的上一参考。
        """
        robot=self.robot;c=robot.config
        delta=np.asarray(delta_position,dtype=float)
        if delta.shape!=(3,) or not np.isfinite(delta).all():raise ValueError('A finite Cartesian delta is required')
        turn=np.zeros(3) if delta_rotation is None else np.asarray(delta_rotation,dtype=float)
        if turn.shape!=(3,) or not np.isfinite(turn).all():raise ValueError('A finite Cartesian rotation delta is required')
        actual,actual_rotation=robot.tcp_pose()
        requested=self.reference+delta
        requested_rotation=rotation_matrix(turn)@self.orientation
        candidate_rotation=rotation_matrix(c.cartesian_smoothing*turn)@self.orientation
        angular_lag=rotation_vector(candidate_rotation,actual_rotation)
        angle=np.linalg.norm(angular_lag)
        if delta_rotation is not None and angle>c.max_rotation_lag:
            candidate_rotation=rotation_matrix(angular_lag*c.max_rotation_lag/angle)@actual_rotation
        candidate=(1-c.cartesian_smoothing)*self.reference+c.cartesian_smoothing*requested
        lag=candidate-actual;norm=np.linalg.norm(lag)
        if norm>c.max_reference_lag:candidate=actual+lag*(c.max_reference_lag/norm)
        clipped=np.clip(candidate,np.asarray(c.workspace_low),np.asarray(c.workspace_high))
        workspace_limited=not np.array_equal(candidate,clipped)
        candidate=clipped
        self.last_fraction=0.;self.last_reason='no_bounded_solution'
        start=robot.get_joint_state()['reference_rad']
        target=start.copy();accepted=self.reference.copy();accepted_rotation=self.orientation.copy()
        if (workspace_limited and np.linalg.norm(candidate-self.reference)<1e-12
                and np.linalg.norm(rotation_vector(candidate_rotation,self.orientation))<1e-12):
            self.last_reason='workspace_limit'
        else:
            # 沿同一笛卡尔增量逐次减半，寻找能同时满足 IK、关节限速与碰撞预检的一步。
            for index in range(12):
                fraction=.5**index
                point=self.reference+fraction*(candidate-self.reference)
                orientation=rotation_matrix(fraction*rotation_vector(candidate_rotation,self.orientation))@self.orientation
                result=robot.kinematics.solve_ik(point,start,orientation,max_iterations=80,damping=.002,
                                                max_step=.02,seed_weight=1e-6,position_tolerance=2e-6)
                if not result.success:
                    self.last_reason=result.reason;continue
                if np.max(abs(result.q-start))>c.joint_step+1e-9:
                    self.last_reason='joint_reference_rate';continue
                safe,reason=robot.reference_is_safe(result.q)
                if not safe:self.last_reason=reason;continue
                target=result.q;accepted=point;accepted_rotation=orientation
                self.last_fraction=fraction;self.last_reason=None;break
        before=float(robot.data.time)
        robot.advance(target,width,observer=observer)
        self.reference=accepted;self.orientation=accepted_rotation
        return dict(ik_fraction=self.last_fraction,ik_reason=self.last_reason,
                    physics_steps=round((float(robot.data.time)-before)/c.physics_timestep),
                    requested_tcp_m=requested.tolist(),commanded_tcp_m=accepted.tolist(),
                    workspace_limited=workspace_limited,
                    requested_rotation=requested_rotation.tolist(),commanded_rotation=accepted_rotation.tolist())
