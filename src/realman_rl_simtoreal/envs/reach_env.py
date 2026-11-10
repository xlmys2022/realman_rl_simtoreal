"""Local Cartesian reaching with fixed or validated random world-frame goals."""

from collections import Counter

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ..simulation.control import advance_control_cycle, ControlCycleError, validate_target
from ..simulation.ee_control import check_motion, MAX_TRANSLATION, MAX_ACTUAL_ORIENTATION_ERROR
from ..simulation.kinematics import Kinematics, rotation_vector
from ..simulation.model import load_model, initialize


class ReachEnv(gym.Env):
    """Fixed home+X5mm or seeded random goal, using existing IK and actuators.

    action: float32 XYZ in [-1,1], each component scaled to metres in world.
    observation: float64 [q(6), qvel(6), ee(3), goal-ee(3), command_xyz(3), ctrl(6)].
    Goal fields are world XYZ metres. No SDK, scripts or Viewer are imported.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    def __init__(self, *, action_scale=0.001, control_period=0.05,
                 max_episode_steps=200, success_threshold=0.001,
                 gravity_compensation=True, render_mode=None, target_mode="fixed",
                 target_radius=0.005, max_target_attempts=64, reward_mode=None,
                 boundary_mode="terminate", rejection_penalty=1.0,
                 reward_sigma_m=0.005, reward_gamma=0.99):
        if boundary_mode not in ("terminate", "reject_and_hold"):
            raise ValueError("boundary_mode must be 'terminate' or 'reject_and_hold'")
        if not np.isfinite(rejection_penalty) or rejection_penalty <= 0:
            raise ValueError("rejection_penalty must be finite and positive")
        self.boundary_mode = boundary_mode
        self.rejection_penalty = float(rejection_penalty)
        if target_mode not in ("fixed", "random"):
            raise ValueError("target_mode must be 'fixed' or 'random'")
        if reward_mode is None:
            reward_mode = "distance" if target_mode == "fixed" else "safe_dense"
        if reward_mode not in ("distance", "safe_dense", "reach_shaped_v1"):
            raise ValueError("reward_mode must be distance, safe_dense or reach_shaped_v1")
        if reward_mode == 'reach_shaped_v1':
            if not np.isfinite(reward_sigma_m) or reward_sigma_m <= 0:
                raise ValueError('reward_sigma_m must be finite and positive (metres)')
            if not np.isfinite(reward_gamma) or not 0 < reward_gamma < 1:
                raise ValueError('reward_gamma must be finite and in (0,1)')
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")
        if not np.isfinite(action_scale) or not 0 < action_scale <= MAX_TRANSLATION:
            raise ValueError("action_scale must be in (0, 0.010] metres per axis")
        if not np.isfinite(success_threshold) or not 0 < success_threshold < 0.005:
            raise ValueError("success_threshold must be in (0, 0.005) metres")
        if (not isinstance(max_episode_steps, (int, np.integer))
                or isinstance(max_episode_steps, bool) or max_episode_steps < 1):
            raise ValueError("max_episode_steps must be a positive integer")
        if not np.isfinite(target_radius) or not success_threshold < target_radius <= MAX_TRANSLATION:
            raise ValueError("target_radius must exceed success_threshold and be at most 10 mm")
        if (not isinstance(max_target_attempts, (int, np.integer))
                or isinstance(max_target_attempts, bool) or not 1 <= max_target_attempts <= 1000):
            raise ValueError("max_target_attempts must be an integer in [1, 1000]")
        self.target_mode = target_mode
        self.target_radius = float(target_radius)
        self.max_target_attempts = int(max_target_attempts)
        self.reward_mode = reward_mode
        self.reward_sigma_m, self.reward_gamma = float(reward_sigma_m), float(reward_gamma)
        if reward_mode == 'reach_shaped_v1':
            # Bounded potential and rejection costs imply strict discounted
            # success > timeout > safety failure for the declared finite H.
            exponent = -(max_episode_steps-1)*np.log(self.reward_gamma)
            if not np.isfinite(exponent) or exponent > np.log(1e6/1.01):
                raise ValueError('Shaped terminal scale is unbounded for this gamma/horizon')
            terminal = np.exp(exponent)*1.01
            self.shaped_terminal_value = float(terminal)
        self.reward_distance_scale = 2*MAX_TRANSLATION
        self._goal_sampling = dict(attempts=0, rejections={})
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
        result = dict(distance_m=distance, is_success=reason == "success",
                    is_safety_failure=reason not in (None, "success", "time_limit"),
                    target_mode=self.target_mode, reward_mode=self.reward_mode,
                    goal_sampling=dict(attempts=self._goal_sampling['attempts'],
                                       rejections=self._goal_sampling['rejections'].copy()),
                    orientation_error_rad=orientation_error,
                    termination_reason=reason, elapsed_steps=self._elapsed_steps,
                    simulation_time_s=float(self.data.time), physics_steps=0,
                    command_position_m=self._command_position.copy(),
                    joint_reference_rad=self._joint_reference.copy(), observation_valid=True, **extra)
        if self.boundary_mode == "reject_and_hold":
            result.update(is_action_rejected=False, rejection_reason=None,
                          boundary_mode=self.boundary_mode)
        if self.reward_mode == 'reach_shaped_v1':
            result.setdefault('is_action_rejected',False)
            result.setdefault('previous_achieved_goal',observation['achieved_goal'].copy())
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("Environment is closed")
        if options:
            raise ValueError("This environment has no reset options")
        self._episode_done = True  # Sampling failure must leave the environment unsteppable.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self._command_position = self.kinematics.home_position.copy()
        self._joint_reference = self.kinematics.home_q.copy()
        self._elapsed_steps = 0
        if self.target_mode == "random":
            self._goal, self._goal_sampling = self._sample_goal()
        else:
            self._goal = self.kinematics.home_position+np.array([0.005, 0, 0])
            self._goal_sampling = dict(attempts=0, rejections={})
        obs = self._observation()
        self._last_observation = {k: v.copy() for k, v in obs.items()}
        self._episode_done = False
        return obs, self._info(obs)

    def _sample_goal(self):
        """Uniform-volume spherical shell, conditioned on the original checks.

        Only np_random is consumed; all FK/IK and path probes use private data.
        With defaults the shell is 3..5 mm from home, outside the success ball.
        """
        minimum = (self.success_threshold+self.target_radius)/2
        rejected = Counter()
        for attempt in range(1, self.max_target_attempts+1):
            direction = self.np_random.normal(size=3)
            length = float(np.linalg.norm(direction))
            if not np.isfinite(length) or length == 0:
                rejected['invalid_direction'] += 1
                continue
            radius = np.cbrt(self.np_random.uniform(minimum**3, self.target_radius**3))
            target = self.kinematics.home_position+direction*(radius/length)
            result = self.kinematics.solve_ik(target, self.kinematics.home_q,
                                            self.kinematics.home_rotation)
            if not result.success or result.q is None:
                rejected['ik_failure'] += 1
                continue
            try:
                validate_target(self.model, self.joint_ids, self.kinematics.home_q, result.q)
            except ValueError:
                rejected['joint_limit'] += 1
                continue
            try:
                check_motion(self.model, self.data, self.joint_ids, self.actuator_ids,
                             self.kinematics, result, target)
            except ValueError as exc:
                rejected['collision' if 'collision' in str(exc).lower() else 'pose_check'] += 1
                continue
            return target, dict(attempts=attempt, rejections=dict(rejected))
        raise RuntimeError(f"No validated random goal after {self.max_target_attempts} attempts; "
                           f"rejections={dict(rejected)}")

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Goal-recomputable reward; safety flags are independent of desired_goal.

        distance preserves 03-A units/behavior. safe_dense is dimensionless:
        -min(distance / 20mm, 1) - (horizon+1)*is_safety_failure.
        With undiscounted returns any safe episode is in [-horizon,0], while
        any safety-failed episode is <=-(horizon+1). No success flag is reused
        under goal relabeling. safe_dense requires info, also for batch calls.
        """
        achieved, desired = np.asarray(achieved_goal), np.asarray(desired_goal)
        if (achieved.shape[-1:] != (3,) or desired.shape[-1:] != (3,)
                or not np.isfinite(achieved).all() or not np.isfinite(desired).all()):
            raise ValueError("Expected finite XYZ goal arrays")
        distance = np.linalg.norm(achieved-desired, axis=-1)
        if self.reward_mode == 'reach_shaped_v1':
            return self._shaped_reward(achieved,desired,info)
        if self.reward_mode == "distance":
            reward = -distance
        else:
            if isinstance(info, dict):
                flags = np.asarray(info.get('is_safety_failure', None))
            elif info is not None:
                records = np.asarray(info, dtype=object)
                if not all(isinstance(record, dict) and 'is_safety_failure' in record for record in records.flat):
                    raise ValueError("safe_dense requires is_safety_failure in every info record")
                flags = np.array([record['is_safety_failure'] for record in records.flat]).reshape(records.shape)
            else:
                raise ValueError("safe_dense requires info with is_safety_failure")
            if flags.dtype != np.bool_:
                raise ValueError("is_safety_failure must contain boolean values")
            flags = np.broadcast_to(flags, distance.shape)
            reward = -np.minimum(distance/self.reward_distance_scale, 1)-(self.max_episode_steps+1)*flags
        if self.boundary_mode == "reject_and_hold":
            if isinstance(info, dict):
                rejected = np.asarray(info.get('is_action_rejected', None))
            elif info is not None:
                records = np.asarray(info, dtype=object)
                if not all(isinstance(r, dict) and 'is_action_rejected' in r for r in records.flat):
                    raise ValueError("reject_and_hold requires is_action_rejected in every info record")
                rejected = np.array([r['is_action_rejected'] for r in records.flat]).reshape(records.shape)
            else:
                raise ValueError("reject_and_hold requires info with is_action_rejected")
            if rejected.dtype != np.bool_:
                raise ValueError("is_action_rejected must contain boolean values")
            # One maximum ordinary-distance cost per rejected action. The old
            # metre-valued distance mode uses the corresponding 20mm cost.
            unit = 1.0 if self.reward_mode == 'safe_dense' else self.reward_distance_scale
            reward = reward-self.rejection_penalty*unit*np.broadcast_to(rejected, distance.shape)
        return float(reward) if reward.ndim == 0 else reward

    def _shaped_reward(self, achieved, desired, info):
        """Discount-consistent exponential potential; info contains raw previous XYZ.

        Phi=-[1-exp(-d/sigma)], r=gamma*Phi_next-Phi_prev
        +K*(success-failure)-(1-gamma)*rejected. True terminals have Phi_next=0;
        truncation retains it. Success is recomputed for the supplied goal.
        K=1.01/gamma**(H-1); no old is_success/distance fields are reused.
        """
        def field(key):
            if isinstance(info,dict):
                if key not in info:raise ValueError('reach_shaped_v1 requires '+key)
                return np.asarray(info[key])
            if info is None:raise ValueError('reach_shaped_v1 requires info')
            rows=np.asarray(info,dtype=object)
            if not all(isinstance(r,dict) and key in r for r in rows.flat):
                raise ValueError('reach_shaped_v1 requires '+key+' in every info record')
            values=np.asarray([r[key] for r in rows.flat])
            return values.reshape(rows.shape+values.shape[1:])
        failure,rejected,previous=field('is_safety_failure'),field('is_action_rejected'),field('previous_achieved_goal')
        if failure.dtype != np.bool_ or rejected.dtype != np.bool_:
            raise ValueError('Shaped safety/rejection flags must be boolean')
        if previous.shape[-1:] != (3,) or not np.isfinite(previous).all():
            raise ValueError('previous_achieved_goal must contain finite world XYZ metres')
        distance=np.linalg.norm(achieved-desired,axis=-1)
        failure=np.broadcast_to(failure,distance.shape);rejected=np.broadcast_to(rejected,distance.shape)
        success=(distance <= self.success_threshold) & ~failure
        phi_previous=np.broadcast_to(np.expm1(-np.linalg.norm(previous-desired,axis=-1)/self.reward_sigma_m),distance.shape)
        phi_next=np.where(success|failure,0.,np.expm1(-distance/self.reward_sigma_m))
        reward=(self.reward_gamma*phi_next-phi_previous
                +self.shaped_terminal_value*(success.astype(float)-failure.astype(float))
                -(1-self.reward_gamma)*rejected)
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
        previous_achieved=self._last_observation['achieved_goal'].copy()
        target = self._command_position+action*self.action_scale
        reason, detail, cycle, ik_info = None, None, None, None
        rejected = None

        def advance(reference, command=None):
            nonlocal reason, detail, cycle
            try:
                cycle = advance_control_cycle(self.model, self.data, self.joint_ids,
                                              self.actuator_ids, reference,
                                              physics_steps=self.physics_steps,
                                              max_joint_delta=np.deg2rad(10)*self.control_period)
                if command is not None:
                    self._command_position = command.copy()
                self._joint_reference = self.data.ctrl[self.actuator_ids].copy()
                actual_r = self.data.site_xmat[self.kinematics.site_id].reshape(3, 3)
                if np.linalg.norm(rotation_vector(self.kinematics.home_rotation, actual_r)) > MAX_ACTUAL_ORIENTATION_ERROR:
                    reason, detail = "orientation_limit", "Actual orientation exceeded the 1 degree envelope"
            except ControlCycleError as exc:
                reason, detail, cycle = exc.reason, str(exc), exc.summary
        actual_q = self.data.qpos[self.kinematics.qadr]
        # The local experiment envelope is NOT a characterization of workspace.
        if not np.isfinite(actual_q).all() or not np.isfinite(self.data.qvel).all():
            reason, detail = "numerical_failure", "Non-finite actual joint state"
        elif (np.any(actual_q < self.kinematics.limits[:, 0]-1e-4)
              or np.any(actual_q > self.kinematics.limits[:, 1]+1e-4)):
            reason, detail = "joint_limit", "Actual joint position exceeded the model limit tolerance"
        elif np.linalg.norm(target-self.kinematics.home_position) > MAX_TRANSLATION+1e-12:
            detail = "Outside the 10 mm experimental command envelope"
            if self.boundary_mode == 'reject_and_hold':
                rejected = 'workspace_limit'
            else:
                reason = 'workspace_limit'
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
                        legal_model_q = False
                        if self.boundary_mode == 'reject_and_hold':
                            try:
                                q = np.asarray(result.q,dtype=float)
                                legal_model_q = (q.shape == (6,) and np.isfinite(q).all()
                                                 and np.all(q > self.kinematics.limits[:, 0])
                                                 and np.all(q < self.kinematics.limits[:, 1]))
                            except (TypeError,ValueError):
                                pass
                        if (self.boundary_mode == 'reject_and_hold' and legal_model_q
                                and np.max(np.abs(q-self.kinematics.home_q)) > np.deg2rad(5)+1e-12):
                            rejected, detail = 'joint_command_limit', str(exc)
                        else:
                            reason, detail = "joint_limit", str(exc)
                    if reason is None and rejected is None:
                        try:
                            check_motion(self.model, self.data, self.joint_ids, self.actuator_ids,
                                         self.kinematics, result, target)
                        except ValueError as exc:
                            reason = "collision" if "collision" in str(exc).lower() else "ik_failure"
                            detail = str(exc)
                    if reason is None and rejected is None:
                        advance(result.q, target)
        if rejected is not None:
            # Reject the NEW task command, never clip an IK result. Keep the
            # last legal Cartesian reference and issued joint ctrl reference;
            # the unchanged controller audits/advances exactly one real cycle.
            try:
                validate_target(self.model, self.joint_ids, self.kinematics.home_q, self._joint_reference)
            except ValueError as exc:
                reason, detail = 'joint_limit', str(exc)
            if reason is None:
                advance(self._joint_reference)
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
        if self.boundary_mode == 'reject_and_hold':
            info.update(is_action_rejected=rejected is not None, rejection_reason=rejected)
        if self.reward_mode == 'reach_shaped_v1':
            info['previous_achieved_goal']=previous_achieved
        info["observation_valid"] = valid_observation
        if not valid_observation:
            info["orientation_error_rad"] = None
        info["physics_steps"] = 0 if cycle is None else cycle["physics_steps"]
        return obs, self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._closed:
            raise RuntimeError("Environment is closed")
        obs = self._observation()
        return (f"t={self.data.time:.3f}s ee_world_m={obs['achieved_goal']} "
                f"goal_world_m={obs['desired_goal']} distance_m={np.linalg.norm(obs['achieved_goal']-self._goal):.6f}")

    def close(self):
        self._closed = True
        self._episode_done = True
