"""Small, fixed-goal Cartesian reaching task in the MuJoCo world frame."""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ..simulation.control import advance_control_cycle, ControlCycleError, validate_target
from ..simulation.ee_control import check_motion, MAX_TRANSLATION, MAX_ACTUAL_ORIENTATION_ERROR
from ..simulation.kinematics import Kinematics, rotation_vector
from ..simulation.model import load_model, initialize


class ReachEnv(gym.Env):
    """Fixed home+X5mm goal, full-pose IK and existing position actuators.

    action: float32 XYZ in [-1,1], each component scaled to metres in world.
    observation: float64 [q(6), qvel(6), ee(3), goal-ee(3), command_xyz(3), ctrl(6)].
    Goal fields are world XYZ metres. No SDK, scripts or Viewer are imported.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    def __init__(self, *, action_scale=0.001, control_period=0.05,
                 max_episode_steps=200, success_threshold=0.001,
                 gravity_compensation=True, render_mode=None):
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        if not np.isfinite(action_scale) or not 0 < action_scale <= MAX_TRANSLATION:
            raise ValueError("action_scale must be in (0, 0.010] metres per axis")
        if not np.isfinite(success_threshold) or not 0 < success_threshold < 0.005:
            raise ValueError("success_threshold must be in (0, 0.005) metres")
        if (not isinstance(max_episode_steps, (int, np.integer))
                or isinstance(max_episode_steps, bool) or max_episode_steps < 1):
            raise ValueError("max_episode_steps must be a positive integer")
        self.model, self.joint_ids, self.actuator_ids = load_model(gravity_compensation=gravity_compensation)
        dt = float(self.model.opt.timestep)
        if not np.isfinite(control_period) or control_period < dt:
            raise ValueError("control_period must be finite and at least one physics timestep")
        self.physics_steps = round(control_period/dt)
        if not np.isclose(self.physics_steps*dt, control_period, rtol=0, atol=1e-12):
            raise ValueError("control_period must be an integer multiple of the physics timestep")
        self.control_period = self.physics_steps*dt
        self.metadata = dict(self.metadata, render_fps=max(1, round(1/self.control_period)))
        self.action_scale = float(action_scale)
        self.max_episode_steps = int(max_episode_steps)
        self.success_threshold = float(success_threshold)
        self.render_mode = render_mode
        self.data = initialize(self.model)
        self.kinematics = Kinematics(self.model)
        self._goal = self.kinematics.home_position+np.array([0.005, 0, 0])
        # Validate the fixed goal against this exact model, not a workspace guess.
        goal_result = self.kinematics.solve_ik(self._goal, self.kinematics.home_q)
        check_motion(self.model, self.data, self.joint_ids, self.actuator_ids,
                     self.kinematics, goal_result, self._goal)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # Unbounded spaces keep terminal safety states representable; values are
        # checked for finiteness separately. Bounds are not physical guarantees.
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf, shape=(27,), dtype=np.float64),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        })
        self._closed = False
        self._episode_done = True
        self._elapsed_steps = 0
        self._command_position = self.kinematics.home_position.copy()
        self._joint_reference = self.kinematics.home_q.copy()
        self._last_observation = self._observation()

    def _observation(self):
        p = self.data.site_xpos[self.kinematics.site_id].copy()
        result = dict(observation=np.concatenate([
            self.data.qpos[self.kinematics.qadr], self.data.qvel[self.kinematics.vadr],
            p, self._goal-p, self._command_position, self.data.ctrl[self.actuator_ids]]),
            achieved_goal=p, desired_goal=self._goal.copy())
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError("Non-finite environment observation")
        return result

    def _info(self, observation, reason=None, **extra):
        distance = float(np.linalg.norm(observation["achieved_goal"]-self._goal))
        rotation = self.data.site_xmat[self.kinematics.site_id].reshape(3, 3)
        orientation_error = (float(np.linalg.norm(rotation_vector(self.kinematics.home_rotation, rotation)))
                             if np.isfinite(rotation).all() else None)
        return dict(distance_m=distance, is_success=reason == "success",
                    orientation_error_rad=orientation_error,
                    termination_reason=reason, elapsed_steps=self._elapsed_steps,
                    simulation_time_s=float(self.data.time), physics_steps=0,
                    command_position_m=self._command_position.copy(),
                    joint_reference_rad=self._joint_reference.copy(), observation_valid=True, **extra)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("Environment is closed")
        if options:
            raise ValueError("This fixed-goal environment has no reset options")
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self._command_position = self.kinematics.home_position.copy()
        self._joint_reference = self.kinematics.home_q.copy()
        self._elapsed_steps = 0
        self._episode_done = False
        obs = self._observation()
        self._last_observation = {k: v.copy() for k, v in obs.items()}
        return obs, self._info(obs)

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Dense reward in metres: -||achieved-desired||; also supports batches."""
        achieved, desired = np.asarray(achieved_goal), np.asarray(desired_goal)
        if (achieved.shape[-1:] != (3,) or desired.shape[-1:] != (3,)
                or not np.isfinite(achieved).all() or not np.isfinite(desired).all()):
            raise ValueError("Expected finite XYZ goal arrays")
        reward = -np.linalg.norm(achieved-desired, axis=-1)
        return float(reward) if reward.ndim == 0 else reward

    def step(self, action):
        if self._closed:
            raise RuntimeError("Environment is closed")
        if self._episode_done:
            raise gym.error.ResetNeeded("Call reset() before step() or after an episode ends")
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Action must be three finite normalized values in [-1, 1]")
        self._elapsed_steps += 1
        target = self._command_position+action*self.action_scale
        reason, detail, cycle, ik_info = None, None, None, None
        actual_q = self.data.qpos[self.kinematics.qadr]
        # The local experiment envelope is NOT a characterization of workspace.
        if not np.isfinite(actual_q).all() or not np.isfinite(self.data.qvel).all():
            reason, detail = "numerical_failure", "Non-finite actual joint state"
        elif (np.any(actual_q < self.kinematics.limits[:, 0]-1e-4)
              or np.any(actual_q > self.kinematics.limits[:, 1]+1e-4)):
            reason, detail = "joint_limit", "Actual joint position exceeded the model limit tolerance"
        elif np.linalg.norm(target-self.kinematics.home_position) > MAX_TRANSLATION+1e-12:
            reason, detail = "workspace_limit", "Outside the 10 mm experimental command envelope"
        elif np.linalg.norm(target) > self.kinematics.reach_bound:
            reason, detail = "unreachable_target", "Outside the conservative geometric reach bound"
        else:
            try:
                validate_target(self.model, self.joint_ids, self.kinematics.home_q, self._joint_reference)
            except ValueError as exc:
                reason, detail = "joint_limit", str(exc)
            if reason is None:
                result = self.kinematics.solve_ik(target, self._joint_reference,
                                                self.kinematics.home_rotation)
                ik_info = dict(success=result.success, iterations=result.iterations,
                               position_error_m=result.position_error, orientation_error_rad=result.orientation_error,
                               reason=result.reason, joint_target_rad=None if result.q is None else result.q.copy())
                if not result.success or result.q is None:
                    reason = ("unreachable_target" if result.reason == "Target exceeds conservative reach bound"
                              else "ik_failure")
                    detail = result.reason
                else:
                    try:
                        validate_target(self.model, self.joint_ids, self.kinematics.home_q, result.q)
                    except ValueError as exc:
                        reason, detail = "joint_limit", str(exc)
                    if reason is None:
                        try:
                            check_motion(self.model, self.data, self.joint_ids, self.actuator_ids,
                                         self.kinematics, result, target)
                        except ValueError as exc:
                            reason = "collision" if "collision" in str(exc).lower() else "ik_failure"
                            detail = str(exc)
                    if reason is None:
                        try:
                            cycle = advance_control_cycle(self.model, self.data, self.joint_ids,
                                                          self.actuator_ids, result.q,
                                                          physics_steps=self.physics_steps,
                                                          max_joint_delta=np.deg2rad(10)*self.control_period)
                            self._command_position = target.copy()
                            self._joint_reference = self.data.ctrl[self.actuator_ids].copy()
                            actual_r = self.data.site_xmat[self.kinematics.site_id].reshape(3, 3)
                            if np.linalg.norm(rotation_vector(self.kinematics.home_rotation, actual_r)) > MAX_ACTUAL_ORIENTATION_ERROR:
                                reason, detail = "orientation_limit", "Actual orientation exceeded the 1 degree envelope"
                        except ControlCycleError as exc:
                            reason, detail, cycle = exc.reason, str(exc), exc.summary
        valid_observation = True
        try:
            obs = self._observation()
            self._last_observation = {k: v.copy() for k, v in obs.items()}
        except ValueError as exc:
            # Report a stale terminal snapshot explicitly; never repair live qpos.
            obs = {k: v.copy() for k, v in self._last_observation.items()}
            valid_observation, reason, detail = False, "numerical_failure", str(exc)
        distance = float(np.linalg.norm(obs["achieved_goal"]-self._goal))
        if reason is None and distance <= self.success_threshold:
            reason = "success"
        terminated = reason is not None
        truncated = not terminated and self._elapsed_steps >= self.max_episode_steps
        if truncated:
            reason = "time_limit"
        self._episode_done = terminated or truncated
        info = self._info(obs, reason, detail=detail, attempted_position_m=target.copy(), ik=ik_info, control=cycle)
        info["observation_valid"] = valid_observation
        if not valid_observation:
            info["orientation_error_rad"] = None
        info["physics_steps"] = 0 if cycle is None else cycle["physics_steps"]
        return obs, self.compute_reward(obs["achieved_goal"], obs["desired_goal"]), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._closed:
            raise RuntimeError("Environment is closed")
        obs = self._observation()
        return (f"t={self.data.time:.3f}s ee_world_m={obs['achieved_goal']} "
                f"goal_world_m={obs['desired_goal']} distance_m={-self.compute_reward(obs['achieved_goal'], self._goal):.6f}")

    def close(self):
        self._closed = True
        self._episode_done = True
