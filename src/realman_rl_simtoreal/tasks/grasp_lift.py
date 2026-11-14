"""Local, contact-driven pick and lift with the EG2-4C2 gripper."""

GRASP_LIFT_CONFIG = dict(
    mass=.01, friction=.7, object_radius=.002, approach_distance=.02,
    lift_height=.03, height_tolerance=.002, hold_seconds=2.,
    action_scale=.001, control_period=.05, physics_timestep=.001, max_episode_steps=700,
    workspace_radius=.06, max_home_joint_delta=.4363323129985824,
    gripper_reference_speed=.5, record_physics=False,
)
