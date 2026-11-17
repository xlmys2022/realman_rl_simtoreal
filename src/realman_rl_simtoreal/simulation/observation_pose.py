"""Traditional zero-to-fixed-observation motion, before policy execution."""
import math
import numpy as np
from .model import require
from .observation_profiles import OBSERVATION_POSES, startup_contract


def observation_orientation(kinematics, config):
    return np.array(OBSERVATION_POSES[config.observation_pose].tool_rotation)


def observation_solution(runtime):
    """Validate a predefined target. Collisions abort; they never retarget it."""
    profile=OBSERVATION_POSES[runtime.config.observation_pose]
    q=np.deg2rad(profile.joints_deg)
    rotation=np.array(profile.tool_rotation)
    actual,actual_rotation=runtime.kinematics.fk(q)
    require(np.linalg.norm(actual-profile.tcp_m)<1e-4,'Observation preset does not match robot geometry')
    require(np.linalg.norm(runtime.rotation_error(rotation,actual_rotation))<1e-3,
            'Observation preset orientation does not match robot geometry')
    require(runtime.path_is_safe(np.zeros(6),q),'Fixed observation path is obstructed; startup aborted')
    return q,rotation


def move_to_observation(runtime, tick=None):
    q,rotation=observation_solution(runtime)
    start=runtime.data.ctrl[runtime.actuator_ids].copy()
    initial_q=runtime.data.qpos[runtime.kinematics.qadr].copy()
    require(np.max(abs(start))<1e-12 and np.max(abs(initial_q))<1e-12,
            'Traditional setup must start from the zero pose')
    cfg=runtime.config;period=cfg.control_period
    duration=max(1.,1.875*float(np.max(abs(q-start)))/(cfg.joint_step/period))
    cycles=math.ceil(duration/period)
    initial_time=float(runtime.data.time)
    if tick:tick(runtime,'traditional_startup')
    for index in range(cycles):
        t=(index+1)/cycles;alpha=t*t*t*(10+t*(-15+6*t))
        runtime.advance(start+alpha*(q-start),cfg.initial_width)
        if tick:tick(runtime,'traditional_startup')
    stable=0
    for index in range(80):
        runtime.advance(q,cfg.initial_width)
        if tick:tick(runtime,'traditional_settle')
        p,r=runtime.tcp_pose()
        good=(np.linalg.norm(p-cfg.observation_tcp)<.001
              and np.linalg.norm(runtime.rotation_error(rotation,r))<.01
              and np.max(abs(runtime.data.qvel[runtime.kinematics.vadr]))<np.deg2rad(.1))
        stable=stable+1 if good else 0
        if stable>=4:break
    require(stable>=4,'Observation pose did not settle')
    return dict(mode='simulated_traditional_control',initial_q_deg=np.rad2deg(initial_q).tolist(),
                target_q_deg=np.rad2deg(q).tolist(),actual_q_deg=np.rad2deg(runtime.data.qpos[runtime.kinematics.qadr]).tolist(),
                simulated_seconds=float(runtime.data.time)-initial_time,cycles=cycles+index+1,
                actual_tcp_m=runtime.tcp_pose()[0].tolist(),tool_forward=runtime.tcp_pose()[1][:,2].tolist(),
                camera=runtime.camera_frame(),camera_calibrated=False,policy_actions=0,
                startup_contract=startup_contract(cfg.observation_pose)),rotation
