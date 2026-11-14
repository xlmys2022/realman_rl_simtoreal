"""Local Cartesian reaching with fixed or validated random world-frame goals."""

from collections import Counter
import copy

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ..simulation.control import advance_control_cycle, ControlCycleError, validate_target
from ..simulation.ee_control import check_motion, MAX_TRANSLATION, MAX_ACTUAL_ORIENTATION_ERROR
from ..simulation.kinematics import Kinematics, IKResult, rotation_vector
from ..simulation.model import load_model, initialize
from ..tasks.reach import OBSERVATION_MODES
from ..tasks.rewards import REWARD_MODES, REWARD_COMPONENT_KEYS, progress_parameters


class ReachEnv(gym.Env):
    """Fixed home+X5mm or seeded random goal, using existing IK and actuators.

    action: float32 XYZ in [-1,1], each component scaled to metres in world.
    legacy observation: [q(6), qvel(6), ee(3), goal-ee(3), command_xyz(3), ctrl(6)].
    fetch_like_v1 observation: [ee_world_m(3), ee_world_velocity * control_period(3)].
    The latter motion feature is displacement per control period in metres,
    not unscaled m/s. goal_relative_v1 observation: [goal-command, command-ee]
    in world-frame metres. All raw observation modes are float64.
    Goal fields are world XYZ metres. No SDK, scripts or Viewer are imported.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    def __init__(self, *, action_scale=0.001, control_period=0.05,
                 max_episode_steps=200, success_threshold=0.001,
                 gravity_compensation=True, render_mode=None, target_mode="fixed",
                 target_radius=0.005, max_target_attempts=64, reward_mode=None,
                 boundary_mode="terminate", rejection_penalty=1.0,
                 reward_sigma_m=0.005, reward_gamma=0.99,target_offsets=None,
                 observation_mode="legacy", reward_progress_parameters=None,
                 workspace_radius=MAX_TRANSLATION, max_home_joint_delta=np.deg2rad(5),
                 target_min_radius=None, synchronized_joint_rate=False,
                 max_orientation_error=MAX_ACTUAL_ORIENTATION_ERROR, cartesian_rate_limit=False,
                 control_ik_iterations=150, track_issued_command=False,
                 control_ik_damping=.01, control_ik_seed_weight=1e-4):
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}")
        self.observation_mode = observation_mode
        if boundary_mode not in ("terminate", "reject_and_hold"):
            raise ValueError("boundary_mode must be 'terminate' or 'reject_and_hold'")
        if not np.isfinite(rejection_penalty) or rejection_penalty <= 0:
            raise ValueError("rejection_penalty must be finite and positive")
        self.boundary_mode = boundary_mode
        self.rejection_penalty = float(rejection_penalty)
        if target_mode not in ("fixed", "random", "discrete"):
            raise ValueError("target_mode must be 'fixed', 'random' or 'discrete'")
        if reward_mode is None:
            reward_mode = "distance" if target_mode == "fixed" else "safe_dense"
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if reward_mode == 'reach_progress_v1':
            self.reward_progress_parameters = progress_parameters(reward_progress_parameters)
        elif reward_progress_parameters is not None:
            raise ValueError('reward_progress_parameters requires reach_progress_v1')
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
        if not np.isfinite(workspace_radius) or workspace_radius <= 0:
            raise ValueError("workspace_radius must be finite and positive")
        if not np.isfinite(max_home_joint_delta) or not 0 < max_home_joint_delta <= 2 * np.pi:
            raise ValueError("max_home_joint_delta must be finite and in (0, 2*pi] radians")
        if not isinstance(synchronized_joint_rate, bool):
            raise ValueError("synchronized_joint_rate must be boolean")
        if not isinstance(cartesian_rate_limit, bool):
            raise ValueError("cartesian_rate_limit must be boolean")
        if not isinstance(track_issued_command,bool) or (track_issued_command and not cartesian_rate_limit):
            raise ValueError('track_issued_command must be boolean and requires cartesian_rate_limit when enabled')
        if (isinstance(control_ik_iterations,bool) or not isinstance(control_ik_iterations,(int,np.integer))
                or not 1<=control_ik_iterations<=1000):
            raise ValueError('control_ik_iterations must be an integer in [1, 1000]')
        if (not np.isfinite(control_ik_damping) or control_ik_damping<1e-6
                or not np.isfinite(control_ik_seed_weight) or control_ik_seed_weight<0):
            raise ValueError('Control IK requires finite damping >= 1e-6 and seed weight >= 0')
        if not np.isfinite(max_orientation_error) or not 0 < max_orientation_error <= np.pi:
            raise ValueError("max_orientation_error must be finite and in (0, pi] radians")
        if not np.isfinite(target_radius) or not success_threshold < target_radius <= workspace_radius:
            raise ValueError("target_radius must exceed success_threshold and fit workspace_radius")
        minimum = (success_threshold + target_radius) / 2 if target_min_radius is None else target_min_radius
        if not np.isfinite(minimum) or not success_threshold < minimum <= target_radius:
            raise ValueError("target_min_radius must exceed success_threshold and fit target_radius")
        self.workspace_radius = float(workspace_radius)
        self.max_home_joint_delta = float(max_home_joint_delta)
        self.target_min_radius = float(minimum)
        self.synchronized_joint_rate = synchronized_joint_rate
        self.max_orientation_error = float(max_orientation_error)
        self.cartesian_rate_limit = cartesian_rate_limit
        self.track_issued_command = track_issued_command
        self.control_ik_iterations = int(control_ik_iterations)
        self.control_ik_damping = float(control_ik_damping)
        self.control_ik_seed_weight = float(control_ik_seed_weight)
        if (not isinstance(max_target_attempts, (int, np.integer))
                or isinstance(max_target_attempts, bool) or not 1 <= max_target_attempts <= 1000):
            raise ValueError("max_target_attempts must be an integer in [1, 1000]")
        self.target_mode = target_mode
        self.target_radius = float(target_radius)
        self._target_offsets=None
        self._goal_candidates=None
        self._candidate_validation=[]
        self._goal_index=None
        if target_mode=='discrete':
            offsets=np.asarray(target_offsets,dtype=float)
            if (offsets.ndim!=2 or offsets.shape[1:]!=(3,) or len(offsets)==0
                    or not np.isfinite(offsets).all()):
                raise ValueError('Discrete target_offsets must be a nonempty finite Nx3 array (metres)')
            radii=np.linalg.norm(offsets,axis=1)
            if (np.any(radii<=success_threshold) or np.any(radii>target_radius+1e-12)
                    or len(np.unique(offsets,axis=0))!=len(offsets)):
                raise ValueError('Discrete offsets must be unique, outside success radius and within target_radius')
            self._target_offsets=offsets.copy()
        elif target_offsets is not None:
            raise ValueError('target_offsets requires target_mode=discrete')
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
        self.reward_distance_scale = 2*self.workspace_radius
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
        if target_mode=='discrete':
            self._validate_candidates()
            self._goal=self._goal_candidates[0].copy()
        else:
            goal_result = self.kinematics.solve_ik(self._goal, self.kinematics.home_q)
            self._check_motion(goal_result, self._goal)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # Unbounded spaces keep terminal safety states representable; values are
        # checked for finiteness separately. Bounds are not physical guarantees.
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-np.inf, np.inf,
                                      shape=(27 if observation_mode == "legacy" else 6,), dtype=np.float64),
            "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
            "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float64),
        })
        self._closed = False
        self._episode_done = True
        self._elapsed_steps = 0
        self._command_position = self.kinematics.home_position.copy()
        self._joint_reference = self.kinematics.home_q.copy()
        self._last_observation = self._observation()

    @property
    def goal_candidates(self):
        """Validated world positions; return a copy, never a mutable command."""
        return None if self._goal_candidates is None else self._goal_candidates.copy()

    @property
    def candidate_validation(self):
        return copy.deepcopy(self._candidate_validation)

    def _validate_joint_target(self, target):
        return validate_target(self.model, self.joint_ids, self.kinematics.home_q, target,
                               max_home_delta=self.max_home_joint_delta)

    def _check_motion(self, result, target):
        return check_motion(self.model, self.data, self.joint_ids, self.actuator_ids,
                            self.kinematics, result, target, max_translation=self.workspace_radius,
                            max_home_delta=self.max_home_joint_delta)

    def _solve_control_ik(self, target):
        result,waypoint=self._solve_rate_limited_ik(target)
        if result.success or not self.track_issued_command:
            return result,waypoint
        # A Cartesian increment can cross a local IK boundary even when a
        # shorter, rate-limited step remains valid. Backtrack in private IK
        # data; each accepted pose still passes the original motion checks.
        origin=self.kinematics.fk(self._joint_reference)[0]
        for fraction in (.5,.25,.125,.0625,.03125,.015625):
            candidate,point=self._solve_rate_limited_ik(origin+fraction*(target-origin))
            if candidate.success:
                return candidate,point
        return result,waypoint

    def _solve_rate_limited_ik(self, target):
        kin=self.kinematics
        seed=self._joint_reference
        parameters=dict(max_iterations=self.control_ik_iterations,damping=self.control_ik_damping,
                        seed_weight=self.control_ik_seed_weight)
        result=kin.solve_ik(target,seed,kin.home_rotation,**parameters)
        if not self.cartesian_rate_limit or not result.success:
            return result,target
        try:
            self._validate_joint_target(result.q)
        except ValueError:
            return result,target
        maximum=np.deg2rad(10)*self.control_period
        fraction=min(1.,maximum/max(float(np.max(np.abs(result.q-seed))),1e-30))
        if fraction==1.:
            return result,target
        origin=kin.fk(seed)[0]
        # Solve each issued pose on the Cartesian path. Limiting a distant
        # joint solution directly can distort the tool orientation between poses.
        for _ in range(16):
            waypoint=origin+fraction*(target-origin)
            candidate=kin.solve_ik(waypoint,seed,kin.home_rotation,position_tolerance=1e-6,**parameters)
            if candidate.success and np.max(np.abs(candidate.q-seed))<=maximum+1e-12:
                return candidate,waypoint
            fraction*=.5
        failure=IKResult(False,None,result.iterations,result.position_error,result.orientation_error,
                         'No Cartesian IK step satisfies the joint command rate')
        return failure,target

    def _validate_candidates(self):
        self._goal_candidates=self.kinematics.home_position+self._target_offsets
        failures=[]
        for i,goal in enumerate(self._goal_candidates):
            result=self.kinematics.solve_ik(goal,self.kinematics.home_q,self.kinematics.home_rotation)
            record=dict(goal_index=i,offset_m=self._target_offsets[i].tolist(),position_m=goal.tolist(),
                        ik_success=result.success,position_error_m=result.position_error,
                        orientation_error_rad=result.orientation_error,valid=False)
            try:
                if not result.success or result.q is None:raise ValueError(result.reason or 'IK returned no legal target')
                self._validate_joint_target(result.q)
                self._check_motion(result, goal)
                record.update(valid=True,joint_target_rad=result.q.tolist())
            except ValueError as exc:
                record['reason']=str(exc)
                failures.append(f'candidate {i} offset={record["offset_m"]}: {exc}')
            self._candidate_validation.append(record)
        if failures:raise ValueError('Discrete targets rejected; none may be silently removed: '+'; '.join(failures))

    def _observation(self):
        if self.observation_mode == "legacy":
            return self._legacy_observation()
        p = self.data.site_xpos[self.kinematics.site_id].copy()
        if self.observation_mode in ("goal_relative_v1", "goal_relative_v2", "goal_relative_v3"):
            ordinary = np.concatenate([self._goal - self._command_position, self._command_position - p])
        else:
            # World-frame site Jacobian, with one column per generalized velocity.
            jacp = np.empty((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, None, self.kinematics.site_id)
            displacement = (jacp @ self.data.qvel) * self.control_period
            ordinary = np.concatenate([p, displacement])
        result = dict(observation=ordinary, achieved_goal=p, desired_goal=self._goal.copy())
        if not all(np.isfinite(v).all() for v in result.values()):
            raise ValueError("Non-finite environment observation")
        return result

    def _legacy_observation(self):
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
        if self.target_mode=='discrete':
            result.update(goal_index=self._goal_index,goal_count=len(self._goal_candidates))
        if self.reward_mode in ('reach_shaped_v1', 'reach_progress_v1'):
            result.setdefault('is_action_rejected',False)
            result.setdefault('previous_achieved_goal',observation['achieved_goal'].copy())
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._closed:
            raise RuntimeError("Environment is closed")
        goal_index=None
        if options:
            if self.target_mode!='discrete' or not isinstance(options,dict) or set(options)!={'goal_index'}:
                raise ValueError("Only discrete tasks support reset options={'goal_index': index}")
            goal_index=options['goal_index']
            if (not isinstance(goal_index,(int,np.integer)) or isinstance(goal_index,bool)
                    or not 0<=goal_index<len(self._goal_candidates)):
                raise ValueError('goal_index must be a valid integer candidate index')
        self._episode_done = True  # Sampling failure must leave the environment unsteppable.
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self._command_position = self.kinematics.home_position.copy()
        self._joint_reference = self.kinematics.home_q.copy()
        self._elapsed_steps = 0
        if self.target_mode == "random":
            self._goal, self._goal_sampling = self._sample_goal()
        elif self.target_mode=='discrete':
            self._goal_index=int(self.np_random.integers(len(self._goal_candidates))) if goal_index is None else int(goal_index)
            self._goal=self._goal_candidates[self._goal_index].copy()
            self._goal_sampling=dict(attempts=1,rejections={})
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
        minimum = self.target_min_radius
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
                self._validate_joint_target(result.q)
            except ValueError:
                rejected['joint_limit'] += 1
                continue
            try:
                self._check_motion(result, target)
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
        if self.reward_mode == 'reach_progress_v1':
            return self.reward_components(achieved, desired, info)['total_reward']
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

    def reward_components(self, achieved_goal, desired_goal, info):
        """Signed progress-reward terms, recomputed from real SI GoalEnv states.

        Pure query: never trusts historical success/distance/components and never
        changes info. Dicts of batched arrays and arrays/lists of info work alike.
        The six signed reward terms sum to total_reward; distance metadata is in m.
        """
        if self.reward_mode != 'reach_progress_v1':
            raise ValueError('Reward components require reach_progress_v1')
        def field(key):
            if isinstance(info, dict):
                if key not in info:raise ValueError('reach_progress_v1 requires '+key)
                return np.asarray(info[key])
            if info is None:raise ValueError('reach_progress_v1 requires info')
            rows=np.asarray(info,dtype=object)
            if not all(isinstance(r,dict) and key in r for r in rows.flat):
                raise ValueError('reach_progress_v1 requires '+key+' in every info record')
            values=np.asarray([r[key] for r in rows.flat])
            return values.reshape(rows.shape+values.shape[1:])
        achieved,desired=np.asarray(achieved_goal,dtype=float),np.asarray(desired_goal,dtype=float)
        previous=field('previous_achieved_goal')
        failure,rejected=field('is_safety_failure'),field('is_action_rejected')
        if any(v.shape[-1:]!=(3,) or not np.isfinite(v).all() for v in (achieved,desired,previous)):
            raise ValueError('Expected finite world XYZ metres for current/previous/desired goals')
        if failure.dtype!=np.bool_ or rejected.dtype!=np.bool_:
            raise ValueError('Progress safety/rejection flags must be boolean')
        try:
            current_distance,previous_distance,failure,rejected=np.broadcast_arrays(
                np.linalg.norm(achieved-desired,axis=-1),np.linalg.norm(previous-desired,axis=-1),failure,rejected)
        except ValueError as exc:
            raise ValueError('Progress reward goal/info batch shapes cannot broadcast') from exc
        p=self.reward_progress_parameters
        change=previous_distance-current_distance
        success=(current_distance<=self.success_threshold)&~failure
        terms=dict(progress_reward=p['w_progress']*change/p['progress_scale'],
            distance_penalty=-p['w_distance']*current_distance/p['distance_scale'],
            step_penalty=np.full(current_distance.shape,-p['step_cost']),
            success_bonus=p['success_bonus']*success,
            safety_penalty=-p['safety_penalty']*failure,
            rejection_penalty=-p['rejection_cost']*rejected)
        terms['total_reward']=sum(terms[key] for key in REWARD_COMPONENT_KEYS)
        terms.update(previous_distance_m=previous_distance,current_distance_m=current_distance,distance_change_m=change)
        if not all(np.isfinite(v).all() for v in terms.values()):
            raise ValueError('Non-finite reach_progress_v1 reward component')
        return {k:float(v) if np.ndim(v)==0 else v for k,v in terms.items()}

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
                                              max_joint_delta=np.deg2rad(10)*self.control_period,
                                              synchronized=self.synchronized_joint_rate)
                if command is not None:
                    self._command_position = command.copy()
                self._joint_reference = self.data.ctrl[self.actuator_ids].copy()
                actual_r = self.data.site_xmat[self.kinematics.site_id].reshape(3, 3)
                if np.linalg.norm(rotation_vector(self.kinematics.home_rotation, actual_r)) > self.max_orientation_error:
                    reason, detail = "orientation_limit", f"Actual orientation exceeded the {np.rad2deg(self.max_orientation_error):g} degree envelope"
            except ControlCycleError as exc:
                reason, detail, cycle = exc.reason, str(exc), exc.summary
        actual_q = self.data.qpos[self.kinematics.qadr]
        # The local experiment envelope is NOT a characterization of workspace.
        if not np.isfinite(actual_q).all() or not np.isfinite(self.data.qvel).all():
            reason, detail = "numerical_failure", "Non-finite actual joint state"
        elif (np.any(actual_q < self.kinematics.limits[:, 0]-1e-4)
              or np.any(actual_q > self.kinematics.limits[:, 1]+1e-4)):
            reason, detail = "joint_limit", "Actual joint position exceeded the model limit tolerance"
        elif np.linalg.norm(target-self.kinematics.home_position) > self.workspace_radius+1e-12:
            detail = f"Outside the {1000 * self.workspace_radius:g} mm experimental command envelope"
            if self.boundary_mode == 'reject_and_hold':
                rejected = 'workspace_limit'
            else:
                reason = 'workspace_limit'
        elif np.linalg.norm(target) > self.kinematics.reach_bound:
            reason, detail = "unreachable_target", "Outside the conservative geometric reach bound"
        else:
            try:
                self._validate_joint_target(self._joint_reference)
            except ValueError as exc:
                reason, detail = "joint_limit", str(exc)
            if reason is None:
                result, control_target = self._solve_control_ik(target)
                ik_info = dict(success=result.success, iterations=result.iterations,
                               position_error_m=result.position_error, orientation_error_rad=result.orientation_error,
                               reason=result.reason, joint_target_rad=None if result.q is None else result.q.copy())
                if self.cartesian_rate_limit:
                    ik_info['control_target_m']=control_target.copy()
                if not result.success or result.q is None:
                    reason = ("unreachable_target" if result.reason == "Target exceeds conservative reach bound"
                              else "ik_failure")
                    detail = result.reason
                else:
                    try:
                        self._validate_joint_target(result.q)
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
                                and np.max(np.abs(q-self.kinematics.home_q)) > self.max_home_joint_delta+1e-12):
                            rejected, detail = 'joint_command_limit', str(exc)
                        else:
                            reason, detail = "joint_limit", str(exc)
                    if reason is None and rejected is None:
                        try:
                            self._check_motion(result, control_target)
                        except ValueError as exc:
                            reason = "collision" if "collision" in str(exc).lower() else "ik_failure"
                            detail = str(exc)
                    if reason is None and rejected is None:
                        advance(result.q, control_target if self.track_issued_command else target)
        if rejected is not None:
            # Reject the NEW task command, never clip an IK result. Keep the
            # last legal Cartesian reference and issued joint ctrl reference;
            # the unchanged controller audits/advances exactly one real cycle.
            try:
                self._validate_joint_target(self._joint_reference)
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
        if self.reward_mode in ('reach_shaped_v1', 'reach_progress_v1'):
            info['previous_achieved_goal']=previous_achieved
        info["observation_valid"] = valid_observation
        if not valid_observation:
            info["orientation_error_rad"] = None
        info["physics_steps"] = 0 if cycle is None else cycle["physics_steps"]
        if self.reward_mode == 'reach_progress_v1':
            info['reward_components']=self.reward_components(obs['achieved_goal'],obs['desired_goal'],info)
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
