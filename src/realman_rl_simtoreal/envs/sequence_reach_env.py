"""Seeded random starts and physically continuous sequences of Reach goals."""
from collections import Counter
import copy

import mujoco
import numpy as np

from .reach_env import ReachEnv
from ..simulation.ee_control import check_motion


class SequenceReachEnv(ReachEnv):
    """One episode contains a seeded start pose and a fixed plan of goals.

    A handover changes the desired goal and its time budget. The transition's
    reward refers to transition_desired_goal, including the successful goal
    when the returned observation already contains the next desired goal.
    """

    def __init__(self, *, start_mode='random', goals_per_episode=5,
                 min_goal_distance=None, **kwargs):
        if start_mode not in ('home', 'random'):
            raise ValueError("start_mode must be 'home' or 'random'")
        if (isinstance(goals_per_episode, bool) or not isinstance(goals_per_episode, (int, np.integer))
                or goals_per_episode < 1):
            raise ValueError('goals_per_episode must be a positive integer')
        radius=kwargs.get('target_radius',.005)
        threshold=kwargs.get('success_threshold',.001)
        if min_goal_distance is None:
            min_goal_distance=min(.01,(threshold+radius)/2)
        if not np.isfinite(min_goal_distance) or not threshold < min_goal_distance < 2*radius:
            raise ValueError('min_goal_distance must exceed success_threshold and fit the region diameter')
        if kwargs.get('target_mode','random') != 'random':
            raise ValueError('SequenceReachEnv requires random target_mode')
        self.start_mode=start_mode
        self.goals_per_episode=int(goals_per_episode)
        self.min_goal_distance=float(min_goal_distance)
        self._goal_plan=None
        self._sequence_goal_index=0
        super().__init__(**dict(kwargs,target_mode='random'))

    @property
    def episode_step_limit(self):
        return self.goals_per_episode*self.max_episode_steps

    @property
    def goal_plan(self):
        return None if self._goal_plan is None else self._goal_plan.copy()

    def _sample_pose(self, data, seed_q):
        source_position=self.kinematics.fk(seed_q)[0]
        rejected=Counter()
        for attempt in range(1,self.max_target_attempts+1):
            direction=self.np_random.normal(size=3)
            length=float(np.linalg.norm(direction))
            if length==0 or not np.isfinite(length):
                rejected['invalid_direction']+=1
                continue
            radius=np.cbrt(self.np_random.uniform(self.target_min_radius**3,self.target_radius**3))
            target=self.kinematics.home_position+direction*(radius/length)
            if np.linalg.norm(target-source_position)<self.min_goal_distance:
                rejected['too_close']+=1
                continue
            result=self.kinematics.solve_ik(target,seed_q,self.kinematics.home_rotation)
            if not result.success or result.q is None:
                rejected['ik_failure']+=1
                continue
            try:
                self._validate_joint_target(result.q)
            except ValueError:
                rejected['joint_limit']+=1
                continue
            try:
                check_motion(self.model,data,self.joint_ids,self.actuator_ids,self.kinematics,result,target,
                             max_translation=self.workspace_radius,max_home_delta=self.max_home_joint_delta)
            except ValueError as exc:
                rejected['collision' if 'collision' in str(exc).lower() else 'pose_check']+=1
                continue
            return target,result.q.copy(),dict(attempts=attempt,rejections=dict(rejected))
        raise RuntimeError(f'No validated sequence pose after {self.max_target_attempts} attempts; '
                           f'rejections={dict(rejected)}')

    def _sample_goal(self):
        # All planning uses private data. The complete plan depends on the reset
        # RNG, independently of the policy and subsequent tracking errors.
        self._goal_plan=None
        probe=mujoco.MjData(self.model)
        probe.qpos[:]=self.data.qpos
        seed=self.kinematics.home_q.copy()
        self._start_sampling=dict(attempts=0,rejections={})
        if self.start_mode=='random':
            _,seed,self._start_sampling=self._sample_pose(probe,seed)
        self._start_q=seed.copy()
        positions=[]
        sampling=[]
        for _ in range(self.goals_per_episode):
            probe.qpos[self.kinematics.qadr]=seed
            mujoco.mj_forward(self.model,probe)
            position,seed,record=self._sample_pose(probe,seed)
            positions.append(position)
            sampling.append(record)
        self._goal_plan=np.asarray(positions)
        self._plan_sampling=sampling
        return self._goal_plan[0].copy(),copy.deepcopy(sampling[0])

    def _physical_state(self):
        return dict(qpos=self.data.qpos.tolist(),qvel=self.data.qvel.tolist(),
                    ctrl=self.data.ctrl.tolist(),time_s=float(self.data.time))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed,options=options)
        # Random pose initialization occurs only at an episode reset.
        self.data.qpos[self.kinematics.qadr]=self._start_q
        self.data.ctrl[self.actuator_ids]=self._start_q
        mujoco.mj_forward(self.model,self.data)
        self._command_position=self.data.site_xpos[self.kinematics.site_id].copy()
        self._joint_reference=self.data.ctrl[self.actuator_ids].copy()
        self._sequence_steps=0
        self._completed_goals=0
        self._sequence_goal_index=0
        self._leg_start_step=0
        self._leg_start_state=self._physical_state()
        self._leg_start_position=self._command_position.copy()
        obs=self._observation()
        self._last_observation={k:v.copy() for k,v in obs.items()}
        info=self._info(obs)
        info.update(start_mode=self.start_mode,start_sampling=copy.deepcopy(self._start_sampling),
                    start_position_m=obs['achieved_goal'].copy(),goal_plan_m=self._goal_plan.tolist(),
                    workspace_center_m=self.kinematics.home_position.tolist(),
                    goals_completed=0,goals_planned=self.goals_per_episode,goal_index=0,
                    goal_elapsed_steps=0,sequence_elapsed_steps=0,is_goal_success=False,goal_changed=False,
                    transition_desired_goal=self._goal.copy(),goal_result=None)
        return obs,info

    @staticmethod
    def _reward_goal(desired_goal,info):
        desired=np.asarray(desired_goal)
        if desired.shape[-1:]!=(3,) or not np.isfinite(desired).all():
            raise ValueError('Expected finite XYZ desired goals')
        if isinstance(info,dict):
            return info.get('transition_desired_goal',desired)
        if info is None:
            return desired
        records=np.asarray(info,dtype=object)
        if not all(isinstance(row,dict) for row in records.flat):
            return desired
        present=np.array(['transition_desired_goal' in row for row in records.flat]).reshape(records.shape+(1,))
        contexts=np.asarray([row.get('transition_desired_goal',[0.,0.,0.]) for row in records.flat])
        contexts=contexts.reshape(records.shape+(3,))
        return np.where(present,contexts,desired)

    def compute_reward(self,achieved_goal,desired_goal,info=None):
        return super().compute_reward(achieved_goal,self._reward_goal(desired_goal,info),info)

    def reward_components(self,achieved_goal,desired_goal,info):
        return super().reward_components(achieved_goal,self._reward_goal(desired_goal,info),info)

    def step(self,action):
        transition_goal=self._goal.copy()
        transition_index=self._sequence_goal_index
        obs,reward,terminated,truncated,info=super().step(action)
        self._sequence_steps+=1
        goal_steps=self._elapsed_steps
        reached=info['is_success']
        result=None
        if reached:
            self._completed_goals+=1
        if reached or terminated or truncated:
            result=dict(goal_index=transition_index,target_position_m=transition_goal.tolist(),
                        initial_position_m=self._leg_start_position.tolist(),
                        final_position_m=obs['achieved_goal'].tolist(),final_distance_m=info['distance_m'],
                        start_step=self._leg_start_step,end_step=self._sequence_steps,length=goal_steps,
                        success=bool(reached),safety_failure=info['is_safety_failure'],
                        reason=info['termination_reason'],detail=info.get('detail'),
                        start_state=copy.deepcopy(self._leg_start_state),end_state=self._physical_state(),
                        goal_sampling=copy.deepcopy(self._plan_sampling[transition_index]))
        changed=reached and self._completed_goals<self.goals_per_episode
        if changed:
            self._sequence_goal_index+=1
            self._goal=self._goal_plan[self._sequence_goal_index].copy()
            self._goal_sampling=copy.deepcopy(self._plan_sampling[self._sequence_goal_index])
            self._elapsed_steps=0
            self._episode_done=False
            terminated=False
            obs=self._observation()
            self._last_observation={k:v.copy() for k,v in obs.items()}
            self._leg_start_step=self._sequence_steps
            self._leg_start_state=self._physical_state()
            self._leg_start_position=obs['achieved_goal'].copy()
            info.update(termination_reason=None,is_success=False,
                        goal_sampling=copy.deepcopy(self._goal_sampling),
                        distance_m=float(np.linalg.norm(obs['achieved_goal']-obs['desired_goal'])))
        info.update(is_goal_success=bool(reached),goal_changed=bool(changed),goal_result=result,
                    transition_desired_goal=transition_goal,transition_goal_index=transition_index,
                    goal_index=self._sequence_goal_index,goals_completed=self._completed_goals,
                    goals_planned=self.goals_per_episode,goal_elapsed_steps=goal_steps,
                    sequence_elapsed_steps=self._sequence_steps,elapsed_steps=self._sequence_steps)
        return obs,reward,terminated,truncated,info
