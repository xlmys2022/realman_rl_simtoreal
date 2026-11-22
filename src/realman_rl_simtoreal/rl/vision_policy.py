"""用视觉状态替换目标对象特征，保留 Reach 臂和夹爪 PPO 的网络与迁移接口。"""
import torch
from .cartesian_reach_policy import CartesianGoalFeatures,ReachGraspPPOPolicy,ReachGraspNetworks


class VisualCartesianGoalFeatures(CartesianGoalFeatures):
    """在目标特征中使用基座系 RGB-D 位移估计及视觉夹持状态。

    visual_xy_deadband 和 visual_z_deadband 单位为 m，用于减小相机估计的小幅抖动。
    """
    def __init__(self,*args,visual_xy_deadband=.001,visual_z_deadband=.0005,**kwargs):
        if not (0<=visual_xy_deadband<=.002 and 0<=visual_z_deadband<=.001):raise ValueError('Invalid visual precision deadband')
        self.visual_xy_deadband=visual_xy_deadband;self.visual_z_deadband=visual_z_deadband
        super().__init__(*args,**kwargs)

    def forward(self,obs):
        tcp=obs[...,:3]*.5
        object_offset=torch.sinh(obs[...,6:9].clamp(-10,10))*.01
        goal_offset=torch.sinh(obs[...,9:12].clamp(-10,10))*.01
        tracking=obs[...,25:28]*.01
        # This channel is explicitly visual enclosure + encoder compression,
        # not a synthetic contact force and not simulator grasp state.
        grip=torch.sigmoid(self.grip_gate(obs[...,21:22]))
        lateral=torch.linalg.vector_norm(object_offset[...,:2],dim=-1,keepdim=True)
        clearance=self.approach_clearance*(1-torch.exp(-(lateral/self.approach_xy_scale).square()))
        approach=torch.cat([object_offset[...,:2],object_offset[...,2:3]+self.grasp_offset_z+clearance],dim=-1)
        precise=(torch.linalg.vector_norm(goal_offset[...,:2],dim=-1,keepdim=True)<self.visual_xy_deadband)&(goal_offset[...,2:3].abs()<self.visual_z_deadband)
        goal_offset=torch.where(precise,tracking,goal_offset)
        # Once grasped, use camera object-height error. A fixed TCP height
        # cannot compensate for different grasp insertion depths or slip.
        offset=(1-grip)*approach+grip*goal_offset
        achieved=self.source_home+tcp-self.observation_anchor
        desired=achieved+offset
        relative=torch.asinh(torch.cat([offset-tracking,tracking],dim=-1)/.001)
        return torch.cat([achieved,desired,relative,obs],dim=-1)


class VisualReachGraspNetworks(ReachGraspNetworks):
    """从视觉编码构造臂、夹爪和价值分支，保留各分支的原输入通道。"""
    def forward_actor(self,features):
        tracking=torch.sinh(features[...,9:12])*.001
        hold=torch.cat([features[...,:3],features[...,:3]+tracking,
                        torch.zeros_like(tracking),features[...,9:12]],dim=-1)
        # Shared learned network, evaluated relative to its no-error response.
        # This removes inherited drift and anchors "hold reference" at zero,
        # without a task action override, scripted phase or joint target.
        arm=self.policy_net['arm'](features[...,:12])-self.policy_net['arm'](hold)
        return torch.cat([arm,self.policy_net['jaw'](features[...,12:])],dim=-1)


class VisualReachGraspPPOPolicy(ReachGraspPPOPolicy):
    """从视觉编码和机器人本体量预测笛卡尔与夹爪动作。

    构造参数继承 ReachGraspPPOPolicy，沿用原 Actor/Critic 共享接口和独立网络结构。
    """
    def __init__(self,*args,**kwargs):
        kwargs['features_extractor_class']=VisualCartesianGoalFeatures
        # The inherited distance profile reads camera-derived position only;
        # the force-based profile has different channel semantics.
        kwargs['exploration_profile']='distance'
        super().__init__(*args,**kwargs)

    def _build_mlp_extractor(self):
        if self.action_space.shape!=(4,) or self.observation_space.shape!=(29,):
            raise ValueError('Visual Reach/grasp supports the camera-only 29-field position contract')
        self.mlp_extractor=VisualReachGraspNetworks(self.features_dim,29).to(self.device)


def transfer_visual_observation_inputs(policy):
    """清除旧接触力、保持和角速度观测对应的夹爪输入权重。"""
    with torch.no_grad():
        first=policy.mlp_extractor.policy_net['jaw'][0]
        first.weight[:,15:18]=0
        first.weight[:,21:25]=0
