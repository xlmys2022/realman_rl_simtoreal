"""Explicit staged Cartesian controller for the local pick/lift task.

This is a physical-control baseline, not a learned policy. It sends the same
four normalized actions exposed to a reinforcement-learning agent.
"""
import numpy as np

from .gripper import MAX_WIDTH, angle_to_width, width_to_angle


class GraspLiftController:
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
        # Correct small lateral tracking/contact offsets while both pads grip.
        # The lift component is untouched; the cube still has to leave support
        # through real contact forces and follow the arm's commanded motion.
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
