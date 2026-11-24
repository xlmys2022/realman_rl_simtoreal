"""局部夹取与抬升的传统分段控制器，供演示、抓取评估基线和物理对照使用。"""
import numpy as np

from ..devices.mujoco_gripper import MAX_WIDTH, angle_to_width, width_to_angle


class GraspLiftController:
    """由 reset_info 的世界系参考点依次生成接近、闭合、预紧、抬升和保持动作。

    action_scale 为单周期最大位置增量（m），control_period 为秒。close 和 lift
    控制实验分支，phase 保存当前阶段以供评估记录。
    """
    def __init__(self, reset_info, *, action_scale=.001, control_period=.05,
                 close=True, lift=True):
        self.action_scale=action_scale
        self.control_period=control_period
        self.close_gripper=bool(close)
        pre=np.asarray(reset_info['pregrasp_reference_m'],dtype=float)
        grasp=np.asarray(reset_info['grasp_reference_m'],dtype=float)
        high=np.asarray(reset_info['lift_reference_m'],dtype=float) if lift else grasp
        self.phases=[('settle',1.5,pre,pre),('approach',3.,pre,grasp),
                     ('close',6.,grasp,grasp),('preload',1.,grasp,grasp),
                     ('lift',10.,grasp,high),('hold',3.,high,high)]
        self.index=0
        self.phase_step=0
        self.phase='settle'
        self.support_normal=np.asarray(reset_info['support_normal'],dtype=float)
        self.centering_offset=np.zeros(3)

    def predict(self,info):
        # 双侧接触后只在支撑面内修正偏心，物体抬升仍由夹持接触随机械臂运动产生。
        """从环境 info 读取状态，输出三维归一化平移和一个归一化绝对开口命令。"""
        if self.close_gripper and self.index>=2 and info['bilateral_contact']:
            n=self.support_normal
            error=np.asarray(info['object_position_m'])-info['grasp_position_m']
            delta=.05*(error-n*(error@n))
            delta*=min(1.,.00005/max(np.linalg.norm(delta),1e-30))
            self.centering_offset+=delta
            self.centering_offset*=min(1.,.002/max(np.linalg.norm(self.centering_offset),1e-30))
        label,seconds,start,end=self.phases[self.index]
        start=start+self.centering_offset
        end=end+self.centering_offset
        self.phase=label
        self.phase_step+=1
        fraction=min(1.,self.phase_step*self.control_period/seconds)
        blend=fraction*fraction*(3-2*fraction)
        position=start+blend*(end-start)
        if not self.close_gripper or self.index<2:width=.06
        elif label=='close':width=angle_to_width(width_to_angle(.06)*(1-blend))
        else:width=0.
        action=np.r_[np.clip((position-np.asarray(info['command_position_m']))/self.action_scale,-1,1),
                     np.clip(2*width/MAX_WIDTH-1,-1,1)].astype(np.float32)
        if self.phase_step>=round(seconds/self.control_period) and self.index<len(self.phases)-1:
            self.index+=1
            self.phase_step=0
        return action
