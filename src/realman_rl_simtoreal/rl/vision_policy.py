"""PPO Cartesian control from RGB-D object estimates and robot proprioception."""
import torch
from .cartesian_reach_policy import CartesianGoalFeatures,ReachGraspPPOPolicy,ReachGraspNetworks


class VisualCartesianGoalFeatures(CartesianGoalFeatures):
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
    """Discard old force/hold/angular-velocity jaw weights after contract change."""
    with torch.no_grad():
        first=policy.mlp_extractor.policy_net['jaw'][0]
        first.weight[:,15:18]=0
        first.weight[:,21:25]=0
