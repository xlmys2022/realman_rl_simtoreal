"""Read-only state facade and unchanged latest-runtime physical clock."""
import unittest
import numpy as np
from realman_rl_simtoreal.simulation.runtime.tabletop import TabletopRuntime
from realman_rl_simtoreal.simulation.devices.mujoco_gripper import LINK_X,LINK_Z,angle_to_width
from realman_rl_simtoreal.tasks.grasp.tabletop_cartesian import CartesianTabletopConfig as Config


class RuntimeStateTests(unittest.TestCase):
    def setUp(self):
        self.robot=TabletopRuntime(Config());self.robot.reset_zero((.4,0.))

    def test_state_reads_preserve_live_storage_and_return_independent_arrays(self):
        robot=self.robot;data=robot.data
        data.qvel[robot.kinematics.vadr]=np.linspace(-.01,.01,6)
        before=[getattr(data,k).copy() for k in ('qpos','qvel','ctrl')]
        joints=robot.get_joint_state();gripper=robot.get_gripper_state()
        np.testing.assert_array_equal(joints['position_rad'],data.qpos[robot.kinematics.qadr])
        np.testing.assert_array_equal(joints['velocity_rad_s'],data.qvel[robot.kinematics.vadr])
        np.testing.assert_array_equal(joints['reference_rad'],data.ctrl[robot.actuator_ids])
        joints['position_rad'][:]=10;joints['velocity_rad_s'][:]=10;joints['reference_rad'][:]=10
        for key,expected in zip(('qpos','qvel','ctrl'),before):np.testing.assert_array_equal(getattr(data,key),expected)
        self.assertEqual(set(gripper),{'width_m','velocity_m_s'})
        self.assertEqual(data.time,0.)

    def test_jaw_helpers_match_original_encoder_formula_and_applied_reference(self):
        robot=self.robot;g=robot.gripper;robot.data.qvel[g.vadr[0]]=.013
        angle=float(robot.data.qpos[g.qadr[0]])
        expected=2*(LINK_X*np.sin(angle)+LINK_Z*np.cos(angle))*robot.data.qvel[g.vadr[0]]
        self.assertEqual(g.get_velocity(),expected)
        self.assertEqual(g.get_reference_width(),angle_to_width(float(robot.data.ctrl[g.actuator_id])))
        g.set_width(.03)
        # Setting a requested target must not silently turn it into the actual ctrl reference.
        self.assertNotEqual(g.get_reference_width(),g.target_width)

    def test_advance_retains_fifty_one_millisecond_contact_observations(self):
        robot=self.robot;times=[]
        robot.advance(robot.get_joint_state()['reference_rad'],.06,observer=lambda row:times.append(float(robot.data.time)))
        self.assertEqual(len(times),50);self.assertEqual(robot.gripper.steps,50)
        np.testing.assert_allclose(times,np.arange(1,51)*.001,atol=1e-15,rtol=0)
        self.assertAlmostEqual(robot.data.time,.05,places=14)
        self.assertEqual(set(robot.recorder.last).intersection({'left_grasp_n','right_grasp_n'}),{'left_grasp_n','right_grasp_n'})


if __name__=='__main__':unittest.main()
