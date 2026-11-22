"""定义笛卡尔 PPO 的连续高斯动作分布，按抓取观测调整各通道探索幅度。"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution


class CartesianPPOPolicy(ActorCriticPolicy):
    """Actor 输出笛卡尔与夹爪动作均值，Critic 估计状态价值。

    构造参数沿用 SB3。adaptive_exploration 启用按距离或夹持状态缩放的标准差，
    exploration_distance 单位为 m，动作尺度参数与环境归一化动作一致。
    Actor 的均值由网络输出，探索调整进入取样与概率计算。
    """

    def __init__(self,*args,adaptive_exploration=True,exploration_distance=.04,
                 motion_std_floor=.03,jaw_std_floor=.2,jaw_exploration_distance=None,
                 exploration_profile='distance',cage_xy_scale=.004,cage_z_scale=.005,
                 grasp_offset_z=.003,jaw_far_std_floor=.1,jaw_held_std_floor=.1,
                 grip_force_scale=.1,lift_exploration_gain=2.,**kwargs):
        if not (0 < exploration_distance <= .1 and 0 < motion_std_floor <= 1 and 0 < jaw_std_floor <= 1):
            raise ValueError('Invalid Cartesian exploration profile')
        self.adaptive_exploration=adaptive_exploration
        self.exploration_distance=exploration_distance
        self.motion_std_floor=motion_std_floor
        self.jaw_std_floor=jaw_std_floor
        self.jaw_exploration_distance=exploration_distance if jaw_exploration_distance is None else jaw_exploration_distance
        if not 0<self.jaw_exploration_distance<=.1:raise ValueError('Invalid jaw exploration distance')
        if exploration_profile not in ('distance','grasp_lift'):raise ValueError('Invalid exploration profile')
        if not (0<cage_xy_scale<=.05 and 0<cage_z_scale<=.05 and abs(grasp_offset_z)<=.02
                and 0<jaw_far_std_floor<=1 and 0<jaw_held_std_floor<=1
                and 0<grip_force_scale<=1 and 1<=lift_exploration_gain<=4):
            raise ValueError('Invalid grasp exploration geometry')
        self.exploration_profile=exploration_profile
        self.cage_xy_scale=cage_xy_scale;self.cage_z_scale=cage_z_scale
        self.grasp_offset_z=grasp_offset_z
        self.jaw_far_std_floor=jaw_far_std_floor;self.jaw_held_std_floor=jaw_held_std_floor
        self.grip_force_scale=grip_force_scale;self.lift_exploration_gain=lift_exploration_gain
        self._distribution_observation=None
        super().__init__(*args,**kwargs)
        if not isinstance(self.action_dist,DiagGaussianDistribution):
            raise ValueError('Cartesian PPO uses a diagonal Gaussian')

    def _get_constructor_parameters(self):
        values=super()._get_constructor_parameters()
        values.update(adaptive_exploration=self.adaptive_exploration,exploration_distance=self.exploration_distance,
                      motion_std_floor=self.motion_std_floor,jaw_std_floor=self.jaw_std_floor,
                      jaw_exploration_distance=self.jaw_exploration_distance,
                      exploration_profile=self.exploration_profile,cage_xy_scale=self.cage_xy_scale,
                      cage_z_scale=self.cage_z_scale,grasp_offset_z=self.grasp_offset_z,
                      jaw_far_std_floor=self.jaw_far_std_floor,jaw_held_std_floor=self.jaw_held_std_floor,
                      grip_force_scale=self.grip_force_scale,lift_exploration_gain=self.lift_exploration_gain)
        return values

    def forward(self,obs,deterministic=False):
        self._distribution_observation=obs
        return super().forward(obs,deterministic)

    def get_distribution(self,obs):
        self._distribution_observation=obs
        return super().get_distribution(obs)

    def evaluate_actions(self,obs,actions):
        self._distribution_observation=obs
        return super().evaluate_actions(obs,actions)

    def _get_action_dist_from_latent(self,latent_pi):
        distribution=super()._get_action_dist_from_latent(latent_pi)
        if not self.adaptive_exploration:return distribution
        obs=self._distribution_observation
        if obs is None:raise RuntimeError('An observation is required for Cartesian exploration')
        offset=torch.sinh(obs[...,6:9].clamp(-10,10))*.01
        distance=torch.linalg.vector_norm(offset,dim=-1,keepdim=True)
        fraction=(distance/self.exploration_distance).clamp(0,1)
        motion=fraction.clamp_min(self.motion_std_floor)
        jaw_fraction=(distance/self.jaw_exploration_distance).clamp(0,1)
        jaw=self.jaw_std_floor+(1-self.jaw_std_floor)*jaw_fraction
        motion_axes=motion.expand(-1,self.action_space.shape[0]-1)
        if self.exploration_profile=='grasp_lift':
            lateral=torch.linalg.vector_norm(offset[...,:2],dim=-1,keepdim=True)
            height_error=-offset[...,2:3]-self.grasp_offset_z
            readiness=torch.exp(-(lateral/self.cage_xy_scale).square()-(height_error/self.cage_z_scale).square())
            if self.observation_space.shape[0]==38:
                angle=torch.linalg.vector_norm(obs[...,29:32]*np.pi,dim=-1,keepdim=True)
                readiness=readiness*torch.exp(-angle/.5)
            # Forces are the observed inward normals, not the old top-pressure sum.
            inward_force=torch.minimum(obs[...,21:22],obs[...,22:23])*.2
            holding=(inward_force/self.grip_force_scale).clamp(0,1)
            jaw=(self.jaw_far_std_floor+(1-self.jaw_far_std_floor)*readiness)*(1-holding)+self.jaw_held_std_floor*holding
            goal_offset=torch.sinh(obs[...,9:12].clamp(-10,10))*.01
            lift=(goal_offset.abs()/self.exploration_distance).clamp(0,1)*self.lift_exploration_gain*holding
            translation=torch.maximum(motion.expand(-1,3),lift)
            motion_axes=translation if self.action_space.shape[0]==4 else torch.cat([translation,motion.expand(-1,3)],dim=-1)
        scales=torch.cat([motion_axes,jaw],dim=-1)
        distribution.distribution=Normal(distribution.distribution.mean,distribution.distribution.stddev*scales)
        return distribution


def reset_critic(policy):
    """任务奖励变化时只重新初始化价值分支及价值输出头。"""
    with torch.no_grad():
        for module in policy.mlp_extractor.value_net.modules():
            if isinstance(module,nn.Linear):policy.init_weights(module,gain=np.sqrt(2))
        policy.init_weights(policy.value_net,gain=1)


def reset_jaw_head(policy,normalized_opening):
    """将夹爪动作头设置为给定的归一化初始开口。"""
    if not np.isfinite(normalized_opening) or not -1<=normalized_opening<=1:
        raise ValueError('A normalized jaw opening is required')
    with torch.no_grad():
        head=getattr(policy,'jaw_action_head',policy.action_net)
        head.weight[-1].zero_()
        head.bias[-1]=float(normalized_opening)
