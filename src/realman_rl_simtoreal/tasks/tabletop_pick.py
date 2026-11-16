"""Full desktop task, initially using zero pose at the user's request.

Zero is the official ROS reference, not the controller's packing posture.
"""
TABLETOP_PICK_CONFIG=dict(
    initial_q_deg=[0.,0.,0.,0.,0.,0.],
    initial_pose_name='zero_development',
    initial_pose_source='https://github.com/RealManRobot/ros2_rm_robot/blob/humble/rm_moveit2_config/rm_63_config/config/initial_positions.yaml',
    object_xy=[.4,0.],object_radius=0.,mass=.01,friction=.7,
    initial_width=.06,lift_height=.03,goal_height=None,height_tolerance=.002,hold_seconds=2.,
    joint_action_scale=.008726646259971648,control_period=.05,physics_timestep=.001,
    max_episode_steps=1200,gripper_reference_speed=.5,gripper_command_margin=.002,record_physics=False,
    reference_smoothing=1.,
    start_fraction=1.,pregrasp_height=.06,reject_unsafe_targets=True,
    approach_waypoint_height=0.,approach_waypoint_tolerance=.015,
)
