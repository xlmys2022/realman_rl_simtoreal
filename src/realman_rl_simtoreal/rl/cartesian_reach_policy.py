"""Goal-conditioned PPO arm and jaw branches; kinematics stays in the executor."""
import hashlib
import numpy as np
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from .cartesian_policy import CartesianPPOPolicy


class CartesianGoalFeatures(BaseFeaturesExtractor):
    """Expose Cartesian grasp/lift goals to an existing neural Reach controller.

    The two force-gate parameters are learned. Frame anchors are fixed at the
    selected observation preset and the original Reach model's nominal home.
    No joint angles, solved joint targets or action labels enter these features.
    """

    def __init__(self,observation_space,*,source_home,observation_anchor,
                 grasp_offset_z=.003,approach_clearance=.05,approach_xy_scale=.015,
                 initial_gate_weight=80.,initial_gate_bias=-40.):
        if observation_space.shape!=(29,):raise ValueError('Reach transfer requires the 29-field position observation')
        home=np.asarray(source_home,dtype=float);anchor=np.asarray(observation_anchor,dtype=float)
        if home.shape!=(3,) or anchor.shape!=(3,) or not np.isfinite([home,anchor]).all():
            raise ValueError('Finite three-dimensional frame anchors are required')
        if not (0<=grasp_offset_z<=.02 and 0<=approach_clearance<=.1 and 0<approach_xy_scale<=.1
                and np.isfinite([initial_gate_weight,initial_gate_bias]).all()):
            raise ValueError('Invalid Cartesian goal geometry')
        super().__init__(observation_space,41)
        self.register_buffer('source_home',torch.tensor(home,dtype=torch.float32))
        self.register_buffer('observation_anchor',torch.tensor(anchor,dtype=torch.float32))
        self.grasp_offset_z=grasp_offset_z;self.approach_clearance=approach_clearance
        self.approach_xy_scale=approach_xy_scale
        self.initial_gate_weight=initial_gate_weight;self.initial_gate_bias=initial_gate_bias
        self.grip_gate=nn.Linear(1,1)
        self.initialize_lift_gate()

    def initialize_lift_gate(self):
        with torch.no_grad():
            self.grip_gate.weight.fill_(self.initial_gate_weight)
            self.grip_gate.bias.fill_(self.initial_gate_bias)

    def forward(self,obs):
        tcp=obs[...,:3]*.5
        object_offset=torch.sinh(obs[...,6:9].clamp(-10,10))*.01
        object_goal_offset=torch.sinh(obs[...,9:12].clamp(-10,10))*.01
        tracking=obs[...,25:28]*.01
        inward_force=torch.minimum(obs[...,21:22],obs[...,22:23])
        grip=torch.sigmoid(self.grip_gate(inward_force))
        lateral=torch.linalg.vector_norm(object_offset[...,:2],dim=-1,keepdim=True)
        clearance=self.approach_clearance*(1-torch.exp(-(lateral/self.approach_xy_scale).square()))*(1-grip)
        offset=object_offset+grip*object_goal_offset
        offset=torch.cat([offset[...,:2],offset[...,2:3]+self.grasp_offset_z+clearance],dim=-1)
        achieved=self.source_home+tcp-self.observation_anchor
        desired=achieved+offset
        relative=torch.asinh(torch.cat([offset-tracking,tracking],dim=-1)/.001)
        return torch.cat([achieved,desired,relative,obs],dim=-1)


class ReachGraspNetworks(nn.Module):
    """Separate neural arm/jaw means and an independent task-value branch."""

    def __init__(self,feature_dim,raw_dim):
        super().__init__();self.latent_dim_pi=4;self.latent_dim_vf=128
        self.policy_net=nn.ModuleDict(dict(
            arm=nn.Sequential(nn.Linear(12,256),nn.Tanh(),nn.Linear(256,256),nn.Tanh(),nn.Linear(256,3)),
            jaw=nn.Sequential(nn.Linear(raw_dim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))))
        self.value_net=nn.Sequential(nn.Linear(feature_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())

    def forward_actor(self,features):
        return torch.cat([self.policy_net['arm'](features[...,:12]),self.policy_net['jaw'](features[...,12:])],dim=-1)

    def forward_critic(self,features):return self.value_net(features)

    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class ReachGraspPPOPolicy(CartesianPPOPolicy):
    """Four learned Cartesian means with an optional pretrained Reach branch."""

    def __init__(self,*args,**kwargs):
        kwargs.pop('net_arch',None)
        kwargs.setdefault('features_extractor_class',CartesianGoalFeatures)
        super().__init__(*args,net_arch=[],**kwargs)

    def _build_mlp_extractor(self):
        if self.action_space.shape!=(4,) or self.observation_space.shape!=(29,):
            raise ValueError('Reach initialization supports Cartesian position control only')
        self.mlp_extractor=ReachGraspNetworks(self.features_dim,29).to(self.device)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.action_net=nn.Identity()
        self.features_extractor.initialize_lift_gate()
        # Rebuild after replacing the unused SB3 projection with the branch means.
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    @property
    def jaw_action_head(self):return self.mlp_extractor.policy_net['jaw'][-1]

    def set_arm_trainable(self,trainable):
        self.mlp_extractor.policy_net['arm'].requires_grad_(bool(trainable))


def validate_reach_policy(source_policy):
    source=source_policy.mlp_extractor.policy_net
    if (source_policy.action_space.shape!=(3,) or source_policy.features_dim!=12
            or not isinstance(source,nn.Sequential) or len(source)!=4
            or not isinstance(source[1],nn.Tanh) or not isinstance(source[3],nn.Tanh)
            or source[0].weight.shape!=(256,12) or source[2].weight.shape!=(256,256)
            or source_policy.action_net.weight.shape!=(3,256)):
        raise ValueError('Reach transfer requires the original 12-feature, 256/256 Tanh PPO')
    return source


def initialize_reach_arm(policy,source_policy,action_gain):
    """Copy only the original PPO Reach mean network, with an explicit SI gain."""
    if not isinstance(policy,ReachGraspPPOPolicy):raise ValueError('Expected a ReachGraspPPOPolicy')
    if not np.isfinite(action_gain) or not 0<action_gain<=5:raise ValueError('Invalid arm action gain')
    source=validate_reach_policy(source_policy);target=policy.mlp_extractor.policy_net['arm']
    with torch.no_grad():
        for index in (0,2):target[index].load_state_dict(source[index].state_dict())
        target[4].weight.copy_(source_policy.action_net.weight*action_gain)
        target[4].bias.copy_(source_policy.action_net.bias*action_gain)


def branch_hashes(policy):
    result={}
    groups=dict(arm=policy.mlp_extractor.policy_net['arm'],jaw=policy.mlp_extractor.policy_net['jaw'],
                goal_gate=policy.features_extractor.grip_gate)
    for name,module in groups.items():
        digest=hashlib.sha256()
        for key,value in module.state_dict().items():
            digest.update(key.encode()+value.detach().cpu().contiguous().numpy().tobytes())
        result[name]=digest.hexdigest()
    return result
