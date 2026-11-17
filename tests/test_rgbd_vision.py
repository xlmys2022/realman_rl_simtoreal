"""Camera geometry, true-state isolation, lost-target handling and visual PPO."""
import ast
import copy
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

from realman_rl_simtoreal.perception.rgbd import RgbdFrame,OrangeCubeDetector,VisualTracker,orange_mask,backproject
from realman_rl_simtoreal.paths import PROJECT_ROOT as ROOT


def synthetic_cube(center=(.04,0,.05),timestamp=0.,occlude=False):
    """Analytical RGB-D ray/box rendering; truth is fixture generation only."""
    w,h=160,120;f=200.
    intrinsic=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
    pose=np.eye(4);pose[:3,:3]=np.diag([1.,-1.,-1.]);pose[:3,3]=[.02,.04,.3]
    v,u=np.mgrid[:h,:w];rays=np.stack([(u-intrinsic[0,2])/f,(v-intrinsic[1,2])/f,np.ones((h,w))],axis=-1)@pose[:3,:3].T
    origin=pose[:3,3];lo=np.asarray(center)-.01;hi=np.asarray(center)+.01
    with np.errstate(divide='ignore',invalid='ignore'):
        lower=(lo-origin)/rays;upper=(hi-origin)/rays
    near=np.minimum(lower,upper).max(axis=-1);far=np.maximum(lower,upper).min(axis=-1)
    hit=(far>=near)&(near>0)
    rgb=np.full((h,w,3),[122,82,46],dtype=np.uint8);rgb[hit]=[230,89,25]
    depth=np.full((h,w),2.,dtype=np.float32);depth[hit]=near[hit]
    if occlude:
        positions=origin+rays*depth[...,None]
        hidden=hit&(positions[...,0]>center[0]+.001)
        rgb[hidden]=[150,150,150];depth[hidden]=.15
    return RgbdFrame(rgb,depth,intrinsic,pose,timestamp)


class RgbdPerceptionTests(unittest.TestCase):
    def test_hsv_mask_has_white_orange_target_and_black_background(self):
        rgb=np.array([[[230,89,25],[122,82,46],[255,255,255],[25,89,230]]],dtype=np.uint8)
        mask=orange_mask(rgb)
        np.testing.assert_array_equal(mask,[[255,0,0,0]])
        self.assertEqual(mask.dtype,np.uint8)
        for low,high in [((-1,0,0),(25,255,255)),((0,0,0),(180,255,255)),((30,1,1),(10,255,255))]:
            with self.assertRaises(ValueError):orange_mask(rgb,low,high)

    def test_backprojection_respects_cv_axes_and_base_extrinsics(self):
        pose=np.eye(4);pose[:3,:3]=np.diag([1.,-1.,-1.]);pose[:3,3]=[.1,.2,.5]
        frame=RgbdFrame(np.zeros((5,5,3),dtype=np.uint8),np.full((5,5),2.),
            np.array([[100.,0,2],[0,100.,2],[0,0,1.]]),pose,0.)
        np.testing.assert_allclose(backproject(frame,[[3,2],[2,3]]),[[.12,.2,-1.5],[.1,.18,-1.5]],atol=1e-12)
        with self.assertRaises(ValueError):backproject(frame,[[9,0]])

    def test_invalid_calibration_is_rejected(self):
        frame=synthetic_cube()
        wrong=frame.camera_to_base.copy();wrong[0,0]=2
        with self.assertRaises(ValueError):replace(frame,camera_to_base=wrong)
        wrong=frame.intrinsics.copy();wrong[0,0]=0
        with self.assertRaises(ValueError):replace(frame,intrinsics=wrong)

    def test_depth_cube_center_is_recovered_at_several_heights(self):
        detector=OrangeCubeDetector()
        for position in ((.04,0,.05),(.035,-.008,.08),(.03,.005,.035)):
            found=detector.detect(synthetic_cube(position))
            self.assertIsNotNone(found)
            # At this low fixture resolution the visible edge can fall
            # between pixels; tolerance is below one projected pixel.
            np.testing.assert_allclose(found.position_m,position,atol=.0008)
            self.assertGreater(found.confidence,.8)
            self.assertGreater(found.normal[2],.99)

    def test_visible_face_geometry_handles_partial_finger_occlusion(self):
        full=synthetic_cube();occluded=synthetic_cube(occlude=True)
        self.assertLess(np.count_nonzero(orange_mask(occluded.rgb)),np.count_nonzero(orange_mask(full.rgb)))
        found=OrangeCubeDetector().detect(occluded)
        self.assertIsNotNone(found)
        np.testing.assert_allclose(found.position_m,[.04,0,.05],atol=.0008)

    def test_missing_colour_or_depth_does_not_fabricate_a_pose(self):
        detector=OrangeCubeDetector();frame=synthetic_cube()
        self.assertIsNone(detector.detect(replace(frame,rgb=np.zeros_like(frame.rgb))))
        self.assertIsNone(detector.detect(replace(frame,depth_m=np.zeros_like(frame.depth_m))))
        self.assertIsNone(detector.detect(replace(frame,depth_m=np.full_like(frame.depth_m,np.nan))))

    def test_velocity_and_staleness_come_only_from_camera_history(self):
        tracker=VisualTracker(OrangeCubeDetector(),timeout=.25)
        tracker.update(synthetic_cube(timestamp=0.))
        tracker.update(synthetic_cube(center=(.041,0,.05),timestamp=.1))
        state=tracker.state(.1)
        np.testing.assert_allclose(state['velocity_m_s'],[.0025,0,0],atol=.001)
        last=state['position_m'].copy();frame=synthetic_cube(timestamp=.2)
        tracker.update(replace(frame,rgb=np.zeros_like(frame.rgb)))
        state=tracker.state(.4)
        self.assertFalse(state['visible']);self.assertTrue(state['expired']);self.assertEqual(state['confidence'],0)
        np.testing.assert_array_equal(state['position_m'],last)
        tracker.reset()
        with self.assertRaisesRegex(ValueError,'not detected'):tracker.state(.4)

    def test_perception_and_camera_adapter_do_not_read_simulator_object_state(self):
        pure=(ROOT/'src/realman_rl_simtoreal/perception/rgbd.py').read_text()
        adapter=(ROOT/'src/realman_rl_simtoreal/simulation/rgbd_camera.py').read_text()
        for forbidden in ('object_qadr','target_free','target_object','qpos','qvel','segmentation_rendering'):
            self.assertNotIn(forbidden,pure);self.assertNotIn(forbidden,adapter)
        self.assertNotIn('mujoco',pure.lower())


class VisualPolicyTests(unittest.TestCase):
    def test_submillimetre_camera_noise_does_not_chase_a_completed_goal(self):
        import torch
        from gymnasium import spaces
        from realman_rl_simtoreal.rl.vision_policy import VisualCartesianGoalFeatures
        features=VisualCartesianGoalFeatures(spaces.Box(-np.inf,np.inf,(29,),np.float32),
            source_home=[.2,0,.5],observation_anchor=[.1,0,.35])
        obs=torch.zeros(2,29);obs[:,21]=1.;obs[:,8]=np.arcsinh(-.003/.01)
        obs[0,9:12]=torch.asinh(torch.tensor([.0003,.0002,.0004])/.01)
        obs[1,11]=np.arcsinh(.002/.01)
        with torch.no_grad():f=features(obs)
        torch.testing.assert_close(f[0,3:6]-f[0,:3],torch.zeros(3),atol=1e-7,rtol=0)
        self.assertGreater((f[1,3:6]-f[1,:3])[2].item(),.0019)

    def test_lift_features_follow_visual_object_error_instead_of_fixed_tcp_height(self):
        import torch
        from gymnasium import spaces
        from realman_rl_simtoreal.rl.vision_policy import VisualCartesianGoalFeatures
        features=VisualCartesianGoalFeatures(spaces.Box(-np.inf,np.inf,(29,),np.float32),
            source_home=[.2,0,.5],observation_anchor=[.1,0,.35])
        obs=torch.zeros(2,29);obs[:,:3]=torch.tensor([.8,0,.08])
        obs[:,21]=1.;obs[:,11]=np.arcsinh(.007/.01)
        obs[0,8]=np.arcsinh(-.003/.01);obs[1,8]=np.arcsinh(-.011/.01)
        with torch.no_grad():actual=features(obs)
        np.testing.assert_allclose((actual[:,3:6]-actual[:,:3]).numpy(),[[0,0,.007],[0,0,.007]],atol=1e-7)

    def test_camera_contract_transfer_discards_force_weights_and_remains_ppo_trainable(self):
        import torch
        from gymnasium import spaces
        from realman_rl_simtoreal.rl.vision_policy import VisualReachGraspPPOPolicy,transfer_visual_observation_inputs
        torch.set_num_threads(1)
        policy=VisualReachGraspPPOPolicy(spaces.Box(-np.inf,np.inf,(29,),np.float32),
            spaces.Box(-1,1,(4,),np.float32),lambda _:1e-4,
            features_extractor_kwargs=dict(source_home=[.2,0,.5],observation_anchor=[.1,0,.35]))
        before=copy.deepcopy(policy.mlp_extractor.policy_net['arm'].state_dict())
        transfer_visual_observation_inputs(policy)
        weight=policy.mlp_extractor.policy_net['jaw'][0].weight
        self.assertEqual(weight[:,15:18].abs().sum().item(),0)
        self.assertEqual(weight[:,21:25].abs().sum().item(),0)
        for name,value in before.items():torch.testing.assert_close(value,policy.mlp_extractor.policy_net['arm'].state_dict()[name])
        obs=torch.zeros(2,29);obs[:,21]=.5;obs[:,11]=np.arcsinh(.03/.01)
        actions,values,probability=policy(obs)
        checked,logp,_=policy.evaluate_actions(obs,actions)
        torch.testing.assert_close(values,checked);torch.testing.assert_close(probability,logp)
        self.assertEqual(policy.exploration_profile,'distance')
        policy.zero_grad();policy.get_distribution(obs).distribution.mean.sum().backward()
        self.assertGreater(policy.features_extractor.grip_gate.bias.grad.abs().sum().item(),0)

    def test_neural_zero_error_response_holds_the_issued_reference(self):
        import torch
        from gymnasium import spaces
        from realman_rl_simtoreal.rl.vision_policy import VisualReachGraspPPOPolicy
        policy=VisualReachGraspPPOPolicy(spaces.Box(-np.inf,np.inf,(29,),np.float32),
            spaces.Box(-1,1,(4,),np.float32),lambda _:1e-4,
            features_extractor_kwargs=dict(source_home=[.2,0,.5],observation_anchor=[.1,0,.35]))
        obs=torch.zeros(2,29);obs[:,:3]=torch.tensor([.8,0,.08]);obs[:,21]=1.
        obs[:,8]=np.arcsinh(-.003/.01);obs[:,11]=np.arcsinh(.0002/.01)
        obs[:,25:28]=torch.tensor([.1,-.03,.02])
        with torch.no_grad():mean=policy.get_distribution(obs).distribution.mean
        torch.testing.assert_close(mean[:,:3],torch.zeros(2,3),atol=1e-6,rtol=0)


class VisualPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from realman_rl_simtoreal.tasks.tabletop_cartesian import CartesianTabletopConfig
        from realman_rl_simtoreal.envs.factory import make_tabletop_cartesian_env
        cls.env=make_tabletop_cartesian_env(CartesianTabletopConfig(observation_source='rgbd_hsv',cache_startup=True,
            jaw_control='delta',jaw_delta_scale=.0005))
        cls.env.reset(seed=990100)

    @classmethod
    def tearDownClass(cls):cls.env.close()

    def setUp(self):self.obs,self.info=self.env.reset(seed=990100)

    def test_existing_camera_origin_and_real_rgbd_mask_are_preserved(self):
        env=self.env;cid=env.camera.camera_id
        self.assertEqual(env.robot.model.cam_bodyid[cid],env.robot.model.body('camera_link').id)
        np.testing.assert_array_equal(env.robot.model.cam_pos[cid],[0,0,0])
        self.assertEqual(env.camera.frame.rgb.shape,(480,640,3))
        self.assertEqual(set(np.unique(env.tracker.detector.mask)),{0,255})
        np.testing.assert_allclose(self.info['vision']['object_position_base_m'],[.4,0,.01],atol=1e-5)
        np.testing.assert_allclose(self.info['vision']['object_goal_base_m'],[.4,0,.04],atol=1e-5)

    def test_all_privileged_object_info_is_excluded_from_actor_observation(self):
        altered=copy.deepcopy(self.info)
        altered.update(object_position_m=[99]*3,object_velocity_m_s=[88]*3,object_angular_velocity_rad_s=[77]*3,
            object_goal_m=[-99]*3,contact_forces_n=[999]*3,hold_seconds=100,max_hold_seconds=100,
            clearance_m=99,world_z_lift_m=99,max_bilateral_force_n=999)
        saved=self.env.goal.copy();self.env.goal[:]=999
        try:np.testing.assert_array_equal(self.env._observation(altered),self.obs)
        finally:self.env.goal[:]=saved
        self.assertFalse(altered['vision']['simulator_object_state_in_observation'])

    def test_lost_camera_target_stops_without_using_true_object_position(self):
        original=self.env.camera.capture
        def black():
            frame=original();return replace(frame,rgb=np.zeros_like(frame.rgb))
        with patch.object(self.env.camera,'capture',side_effect=black):
            for _ in range(10):
                obs,_,done,truncated,info=self.env.step(np.zeros(4,dtype=np.float32))
                if done or truncated:break
        self.assertTrue(done);self.assertFalse(truncated)
        self.assertEqual(info['termination_reason'],'vision_lost');self.assertFalse(info['is_success'])
        self.assertFalse(info['vision']['detected']);self.assertEqual(info['vision']['confidence'],0)
        self.assertGreater(info['vision']['age_seconds'],self.env.config.vision_timeout)
        self.assertTrue(np.isfinite(obs).all())

    def test_privileged_success_cannot_end_visual_execution(self):
        from realman_rl_simtoreal.envs.tabletop_cartesian_env import TabletopCartesianEnv
        info=copy.deepcopy(self.info);info.update(is_success=True,termination_reason='success')
        self.env._stable_ticks=self.env._required_ticks
        self.env._visual_hold=2.
        with patch.object(TabletopCartesianEnv,'step',return_value=(self.obs,100.,True,False,info)):
            _,reward,done,truncated,result=self.env.step(np.zeros(4,dtype=np.float32))
        self.assertFalse(done);self.assertFalse(truncated);self.assertFalse(result['is_success'])
        self.assertIsNone(result['termination_reason']);self.assertLess(reward,100.)

    def test_camera_completion_is_audited_without_oracle_continuation(self):
        from realman_rl_simtoreal.envs.tabletop_cartesian_env import TabletopCartesianEnv
        self.env._stable_ticks=0;self.env._visual_hold=3.
        info=copy.deepcopy(self.info)
        with patch.object(TabletopCartesianEnv,'step',return_value=(self.obs,0.,False,False,info)):
            _,_,done,truncated,result=self.env.step(np.zeros(4,dtype=np.float32))
        self.assertTrue(done);self.assertFalse(truncated)
        self.assertEqual(result['termination_reason'],'visual_false_success')
        self.assertFalse(result['physical_audit_success']);self.assertFalse(result['is_success'])


if __name__=='__main__':unittest.main()
