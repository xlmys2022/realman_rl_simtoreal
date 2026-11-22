"""局部夹取与抬升的 MuJoCo 环境，提供四维末端动作、接触验收和 41 维观测。"""
import copy
import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from ...simulation.physics.joint_control import advance_synchronized_cycle, validate_target
from ...simulation.physics.grasp_contact import load_experiment, ContactRecorder
from ...simulation.devices.mujoco_gripper import Gripper, SIGNS, MAX_WIDTH, width_to_angle
from ...simulation.motion.kinematics import Kinematics, rotation_vector
from ...tasks.grasp.lift import GRASP_LIFT_CONFIG, REWARD, potential, shaped_reward


class GraspLiftEnv(gym.Env):
    """从近物体位姿开始执行接近、夹取、抬升与保持。

    kwargs 可覆盖局部任务配置，距离用 m、时长用 s。reset 根据种子初始化物体及
    参考点，step 执行世界系 XYZ 增量与绝对夹爪开口，返回接触和完成判据记录。
    保持计时以物理步推进，支撑净空与双侧接触来自实际仿真状态。
    """
    metadata = {"render_modes": ["ansi"], "render_fps": 20}

    def __init__(self, **kwargs):
        render_mode=kwargs.pop('render_mode',None)
        unknown=set(kwargs)-set(GRASP_LIFT_CONFIG)
        if unknown:raise ValueError('Unknown grasp configuration: '+str(sorted(unknown)))
        self.config=dict(GRASP_LIFT_CONFIG,**kwargs)
        c=self.config
        try:
            finite=np.asarray([v for k,v in c.items() if k!='record_physics'],dtype=float)
        except (TypeError,ValueError) as exc:
            raise ValueError('Grasp configuration must contain numeric scalars') from exc
        if finite.ndim!=1 or not np.isfinite(finite).all():raise ValueError('Grasp configuration must be finite scalars')
        if not (0<=c['object_radius']<=.005 and 0<=c['approach_distance']<=.03
                and .005<=c['lift_height']<=.04 and 0<c['height_tolerance']<c['lift_height']/2
                and .1<=c['hold_seconds']<=10 and 0<c['action_scale']<=.002
                and .03<=c['workspace_radius']<=.1 and 0<c['max_home_joint_delta']<=np.deg2rad(30)
                and .0005<=c['physics_timestep']<=.002
                and .021<=c['initial_width']<=MAX_WIDTH
                and 0<c['gripper_reference_speed']<=.5):
            raise ValueError('Invalid local grasp bounds, height, timing or action scale')
        if (isinstance(c['max_episode_steps'],bool) or not isinstance(c['max_episode_steps'],(int,np.integer))
                or c['max_episode_steps']<1 or not isinstance(c['record_physics'],bool)):
            raise ValueError('Expected positive integer max_episode_steps and boolean record_physics')
        if render_mode not in (None,'ansi'):raise ValueError('Only ansi rendering is supported by the environment')
        self.render_mode=render_mode
        self.model,self.data,self.joint_ids,self.actuator_ids,_,self.contact_policy=load_experiment(
            mass=c['mass'],friction=c['friction'])
        # Resolve loaded mimic/contact transients at 1 ms in this task. The
        # source scene, actuator gains and contact/constraint parameters stay
        # unchanged; every controller shares the same finer integration clock.
        self.model.opt.timestep=float(c['physics_timestep'])
        self.kinematics=Kinematics(self.model)
        self.physics_steps=round(c['control_period']/self.model.opt.timestep)
        if (not np.isclose(c['control_period'],.05,atol=1e-12,rtol=0)
                or not np.isclose(self.physics_steps*self.model.opt.timestep,.05,atol=1e-12,rtol=0)):
            raise ValueError('Grasp control_period must be .05 seconds and contain an integer number of physics steps')
        self.control_period=.05
        self.action_scale=float(c['action_scale'])
        self.max_episode_steps=int(c['max_episode_steps'])
        self.object_joint=self.model.joint('target_free').id
        self.object_qadr=int(self.model.jnt_qposadr[self.object_joint])
        self.object_vadr=int(self.model.jnt_dofadr[self.object_joint])
        self.object_body=self.model.body('target_object').id
        self.support_body=self.model.body('support').id
        self._support_position=self.model.body_pos[self.support_body].copy()
        self._object_pose=self.data.qpos[self.object_qadr:self.object_qadr+7].copy()
        self.support_rotation=self.data.xmat[self.support_body].reshape(3,3).copy()
        self.support_normal=self.support_rotation[:,2].copy()
        self._tcp_offset=self.kinematics.home_rotation.T@(self._object_pose[:3]-self.kinematics.home_position)
        self._probe=mujoco.MjData(self.model)
        self._hold_ticks=math.ceil(c['hold_seconds']/self.model.opt.timestep)
        self.action_space=spaces.Box(-1.,1.,shape=(4,),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(41,),dtype=np.float32)
        self._closed=False
        self._episode_done=True

    def _validate_q(self,q):
        return validate_target(self.model,self.joint_ids,self.kinematics.home_q,q,
                               max_home_delta=self.config['max_home_joint_delta'])

    def _ik(self,position,seed):
        return self.kinematics.solve_ik(position,seed,self.kinematics.home_rotation,
            max_iterations=600,damping=.001,seed_weight=1e-6,position_tolerance=1e-6)

    def reset(self, *, seed=None, options=None):
        """初始化场景和夹爪，返回初始观测及接近、抓取和抬升参考点。"""
        if options:raise ValueError('GraspLiftEnv has no reset options')
        if self._closed:raise RuntimeError('Environment is closed')
        super().reset(seed=seed)
        self._episode_done=True
        kin=self.kinematics
        for attempt in range(1,65):
            angle=self.np_random.uniform(0,2*np.pi)
            radius=self.config['object_radius']*np.sqrt(self.np_random.uniform())
            offset=self.support_rotation[:,:2]@(radius*np.array([np.cos(angle),np.sin(angle)]))
            grasp=kin.home_position+offset
            pre=grasp-self.support_normal*self.config['approach_distance']
            lifted=grasp+self.support_normal*self.config['lift_height']
            solutions=[self._ik(p,kin.home_q) for p in (pre,grasp,lifted)]
            if not all(r.success and r.q is not None for r in solutions):continue
            try:
                for result in solutions:self._validate_q(result.q)
            except ValueError:continue
            self.model.body_pos[self.support_body]=self._support_position+offset
            mujoco.mj_setConst(self.model,self._probe)
            mujoco.mj_resetDataKeyframe(self.model,self.data,self.model.key('home').id)
            self.data.qpos[self.object_qadr:self.object_qadr+7]=self._object_pose
            self.data.qpos[self.object_qadr:self.object_qadr+3]+=offset
            self.data.qpos[kin.qadr]=solutions[0].q
            self.data.ctrl[self.actuator_ids]=solutions[0].q
            for name,sign in SIGNS.items():
                self.data.qpos[self.model.joint(name).qposadr[0]]=sign*width_to_angle(self.config['initial_width'])
            self.data.ctrl[self.model.actuator('eg2_position').id]=width_to_angle(self.config['initial_width'])
            mujoco.mj_forward(self.model,self.data)
            try:self.contact_policy.check(self.data)
            except ValueError:continue
            break
        else:raise RuntimeError('No validated local grasp start after 64 attempts')
        self.gripper=Gripper(self.model,self.data,contact_policy=self.contact_policy,
                             reference_speed=self.config['gripper_reference_speed'])
        self.recorder=ContactRecorder(self.model,self.data,self.gripper,self.contact_policy,
                                      keep_history=self.config['record_physics'])
        self._initial_object=self.data.qpos[self.object_qadr:self.object_qadr+3].copy()
        self.object_goal=self._initial_object+self.support_normal*self.config['lift_height']
        self.grasp_reference=grasp.copy()
        self.pregrasp_reference=pre.copy()
        self.lift_reference=lifted.copy()
        self._command_position=kin.fk(solutions[0].q)[0]
        self._requested_width=self.config['initial_width']
        self._elapsed_steps=self._stable_ticks=self._max_stable_ticks=self._lost_ticks=0
        self._hold_anchor=None
        self._ever_lifted=False
        self._last_tick_good=False
        self._episode_done=False
        self._reset_attempts=attempt
        obs=self._observation()
        self._last_observation=obs.copy()
        info=self._info()
        self._last_info=copy.deepcopy(info)
        return obs,info

    def _geometry(self):
        m,d=self.model,self.data
        self._probe.qpos[:]=d.qpos
        mujoco.mj_forward(m,self._probe)
        position=d.qpos[self.object_qadr:self.object_qadr+3].copy()
        rotation=self._probe.xmat[self.object_body].reshape(3,3)
        normal=self.support_normal
        radius=.01*np.sum(np.abs(rotation.T@normal))
        clearance=float((position-self._probe.xpos[self.support_body])@normal-.003-radius)
        p=self._probe.site_xpos[self.kinematics.site_id]
        r=self._probe.site_xmat[self.kinematics.site_id].reshape(3,3)
        tcp=p+r@self._tcp_offset
        return position,tcp,clearance

    def _forces(self):
        row=self.recorder.last
        return np.zeros(3) if row is None else np.array([row['left_normal_n'],row['right_normal_n'],row['support_normal_n']])

    def _observe_tick(self,data):
        self.recorder(data)
        row=self.recorder.last
        bilateral=min(row['left_normal_n'],row['right_normal_n'])>.005
        height=row['support_clearance_m']
        self._ever_lifted |= bilateral and height>.003
        self._lost_ticks=self._lost_ticks+1 if self._ever_lifted and not bilateral else 0
        position=np.asarray(row['object_position_m'])
        goal_error=position-self.object_goal
        lateral=np.linalg.norm(goal_error-self.support_normal*(goal_error@self.support_normal))
        good=(bilateral and row['support_normal_n']<.001
              and height>=self.config['lift_height']-self.config['height_tolerance']
              and lateral<.003 and np.linalg.norm(row['object_qvel'][:3])<.002
              and np.linalg.norm(row['object_qvel'][3:])<.1)
        relative=np.asarray(row['object_relative_to_pads_m'])
        if good and self._hold_anchor is not None and np.linalg.norm(relative-self._hold_anchor)>=.002:
            good=False
        if good:
            if self._hold_anchor is None:self._hold_anchor=relative.copy()
            self._stable_ticks+=1
            self._max_stable_ticks=max(self._max_stable_ticks,self._stable_ticks)
        else:
            self._stable_ticks=0
            self._hold_anchor=None
        self._last_tick_good=bool(good)

    def _observation(self):
        position,tcp,clearance=self._geometry()
        kin=self.kinematics
        q=self.data.qpos[kin.qadr]
        jacp=np.empty((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self._probe,jacp,None,kin.site_id)
        v=self.data.qvel[self.object_vadr:self.object_vadr+6]
        obs=np.concatenate([(q-kin.home_q)/.5,self.data.qvel[kin.vadr]/.5,
            (position-tcp)/.05,(position-self._initial_object)/.05,(self.object_goal-position)/.05,
            v[:3]/.05,v[3:],
            [self.gripper.get_width()/MAX_WIDTH,self._requested_width/MAX_WIDTH,
             self.data.ctrl[self.gripper.actuator_id]/.82],self._forces()/np.array([.2,.2,.1]),
            [clearance/self.config['lift_height'],self._stable_ticks/self._hold_ticks],
            (jacp@self.data.qvel)/.05,(self._command_position-self._probe.site_xpos[kin.site_id])/.01])
        obs=obs.astype(np.float32)
        if obs.shape!=(41,) or not np.isfinite(obs).all():raise ValueError('Invalid grasp observation')
        return obs

    def _info(self,reason=None,detail=None,physics_steps=0):
        if not np.isfinite(self.data.time) or any(w.number for w in self.data.warning):
            raise ValueError('Invalid simulation time or MuJoCo numerical warning')
        position,tcp,clearance=self._geometry()
        forces=self._forces()
        error=float(np.linalg.norm(position-tcp))
        orientation=float(np.linalg.norm(rotation_vector(self.kinematics.home_rotation,
            self._probe.site_xmat[self.kinematics.site_id].reshape(3,3))))
        return dict(is_success=reason=='success',is_safety_failure=reason not in (None,'success','time_limit','dropped','object_lost'),
            termination_reason=reason,detail=detail,elapsed_steps=self._elapsed_steps,
            simulation_time_s=float(self.data.time),physics_steps=int(physics_steps),
            object_position_m=position.tolist(),object_goal_m=self.object_goal.tolist(),
            initial_object_position_m=self._initial_object.tolist(),grasp_position_m=tcp.tolist(),
            object_velocity_m_s=self.data.qvel[self.object_vadr:self.object_vadr+3].tolist(),
            object_angular_velocity_rad_s=self.data.qvel[self.object_vadr+3:self.object_vadr+6].tolist(),
            grasp_distance_m=error,support_clearance_m=clearance,
            world_z_lift_m=float(position[2]-self._initial_object[2]),
            support_normal_lift_m=float((position-self._initial_object)@self.support_normal),
            contact_forces_n=forces.tolist(),bilateral_contact=bool(min(forces[:2])>.005),
            contact_sample_time_s=None if self.recorder.last is None else self.recorder.last['time_s'],
            hold_time_s=self._stable_ticks*float(self.model.opt.timestep),
            max_hold_time_s=self._max_stable_ticks*float(self.model.opt.timestep),
            actual_width_m=self.gripper.get_width(),requested_width_m=self._requested_width,
            orientation_error_rad=orientation,command_position_m=self._command_position.tolist(),
            pregrasp_reference_m=self.pregrasp_reference.tolist(),grasp_reference_m=self.grasp_reference.tolist(),
            lift_reference_m=self.lift_reference.tolist(),support_normal=self.support_normal.tolist(),
            reset_attempts=self._reset_attempts,observation_valid=True)

    def _control_pose(self,target):
        kin=self.kinematics
        seed=self.data.ctrl[self.actuator_ids].copy()
        origin=kin.fk(seed)[0]
        point=target.copy()
        last_reason='No bounded Cartesian grasp step'
        for _ in range(12):
            result=self._ik(point,seed)
            if result.success and result.q is not None:
                try:
                    self._validate_q(result.q)
                    if np.max(np.abs(result.q-seed))<=np.deg2rad(.5)+1e-12:
                        return result.q,point
                except ValueError as exc:last_reason=str(exc)
            else:last_reason=result.reason
            point=origin+.5*(point-origin)
        raise ValueError('Grasp IK: '+str(last_reason))

    def step(self,action):
        """按四维归一化动作推进一个控制周期，输出 Gymnasium 五元组。"""
        if self._closed:raise RuntimeError('Environment is closed')
        if self._episode_done:raise gym.error.ResetNeeded('Call reset before stepping the grasp task')
        action=np.asarray(action,dtype=float)
        if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
            raise ValueError('Expected four finite actions in [-1,1]')
        previous=np.asarray(self._last_info['object_position_m'])
        tcp_before=np.asarray(self._last_info['grasp_position_m'])
        previous_distance=float(np.linalg.norm(previous-tcp_before))
        self._elapsed_steps+=1
        reason=detail=None
        before_time=float(self.data.time)
        target=self._command_position+action[:3]*self.action_scale
        try:
            self.gripper._check_state()
            if np.linalg.norm(target-self.kinematics.home_position)>self.config['workspace_radius']:
                reason='workspace_limit'
            else:
                q,point=self._control_pose(target)
                self._requested_width=float((action[3]+1)*MAX_WIDTH/2)
                self.gripper.set_width(self._requested_width)
                advance_synchronized_cycle(self.model,self.data,self.joint_ids,self.actuator_ids,q,self.gripper,
                    physics_steps=self.physics_steps,step_observer=self._observe_tick,
                    max_home_delta=self.config['max_home_joint_delta'])
                self._command_position=point.copy()
                r=self.data.site_xmat[self.kinematics.site_id].reshape(3,3)
                if np.linalg.norm(rotation_vector(self.kinematics.home_rotation,r))>np.deg2rad(5):
                    reason='orientation_limit'
        except (ValueError,RuntimeError,mujoco.FatalError) as exc:
            reason='ik_failure' if str(exc).startswith('Grasp IK:') else 'safety_failure'
            detail=str(exc)
        elapsed=float(self.data.time)-before_time
        physics_steps=round(elapsed/self.model.opt.timestep) if np.isfinite(elapsed) and elapsed>=0 else 0
        try:
            obs=self._observation()
            info=self._info(reason,detail,physics_steps)
            self._last_observation=obs.copy()
        except (ValueError,RuntimeError,mujoco.FatalError) as exc:
            obs=self._last_observation.copy()
            reason='numerical_failure'
            detail=str(exc)
            info=copy.deepcopy(self._last_info)
            info.update(observation_valid=False,elapsed_steps=self._elapsed_steps,physics_steps=physics_steps)
            if np.isfinite(self.data.time):info['simulation_time_s']=float(self.data.time)
        position=np.asarray(info['object_position_m'])
        if reason is None:
            if self._stable_ticks>=self._hold_ticks:reason='success'
            elif self._lost_ticks*self.model.opt.timestep>=.5:reason='dropped'
            elif (position-self._initial_object)@self.support_normal<-.02 or np.linalg.norm(position-self._initial_object)>.10:
                reason='object_lost'
        terminated=reason is not None
        truncated=not terminated and self._elapsed_steps>=self.max_episode_steps
        if truncated:reason='time_limit'
        self._episode_done=terminated or truncated
        info.update(is_success=reason=='success',termination_reason=reason,detail=detail,
            is_safety_failure=reason not in (None,'success','time_limit','dropped','object_lost'))
        bilateral=info['bilateral_contact']
        progress=(previous_distance-info['grasp_distance_m'])/.005
        lift_progress=float((position-previous)@self.support_normal)/self.config['lift_height']
        terms=dict(reach_progress=2*progress,distance_penalty=-.1*info['grasp_distance_m']/.05,
            step_cost=-.01,contact_bonus=.05*bilateral,lift_progress=10*lift_progress*bilateral,
            hold_bonus=.1*(self._last_tick_good and physics_steps==self.physics_steps and not info['is_safety_failure']),
            success_bonus=25.*info['is_success'],
            failure_penalty=-25.*info['is_safety_failure']-5.*(reason in ('dropped','object_lost')))
        reward=float(sum(terms.values()))
        info['reward_components']=terms
        if info['observation_valid']:self._last_info=copy.deepcopy(info)
        return obs,reward,terminated,truncated,info

    def render(self):
        if self.render_mode is None:return None
        i=self._info()
        return f"t={self.data.time:.2f}s opening={1000*i['actual_width_m']:.2f}mm clearance={1000*i['support_clearance_m']:.2f}mm hold={i['hold_time_s']:.2f}s"

    def close(self):
        self._closed=True


ENCODING=dict(name='grasp_relative_v1',input_dim=41,output_dim=41,
    method='asinh on position errors/displacements at 1mm; all other features unchanged',
    position_slice=[12,21],command_lag_slice=[38,41],running_normalization=False)


def encode(observation):
    """使用原固定 asinh 尺度编码误差通道，同时保留其他原始通道。"""
    value=np.asarray(observation,dtype=np.float32).copy()
    if value.shape[-1:]!=(41,) or not np.isfinite(value).all():
        raise ValueError('PPO grasp requires finite 41-dimensional observations')
    value[...,12:21]=np.arcsinh(value[...,12:21]*50.)
    value[...,38:41]=np.arcsinh(value[...,38:41]*10.)
    return value


class PPOGraspEnv(gym.Wrapper):
    """在局部抓取环境上应用原固定观测编码、势差奖励和可选的种子课程。"""
    def __init__(self,config,reward=None,randomize_height=False,randomize_approach=False):
        super().__init__(GraspLiftEnv(**config))
        self.task_config=self.env.config.copy()
        self.reward_config=copy.deepcopy(REWARD if reward is None else reward)
        self.randomize_height=bool(randomize_height)
        self.randomize_approach=bool(randomize_approach)
        if self.randomize_approach and self.task_config['approach_distance']<.005:
            raise ValueError('Random approach curriculum requires a maximum distance of at least 5mm')
        self._curriculum_rng=None
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(41,),dtype=np.float32)

    def reset(self,**kwargs):
        if self.randomize_height or self.randomize_approach:
            if kwargs.get('seed') is not None or self._curriculum_rng is None:
                seed=kwargs.get('seed')
                self._curriculum_rng=np.random.default_rng(None if seed is None else [int(seed),38117])
        if self.randomize_height:
            height=float(self._curriculum_rng.uniform(.005,self.task_config['lift_height']))
            self.env.config['lift_height']=height
            self.env.config['height_tolerance']=height*self.task_config['height_tolerance']/self.task_config['lift_height']
        if self.randomize_approach:
            self.env.config['approach_distance']=float(self._curriculum_rng.uniform(.005,self.task_config['approach_distance']))
        obs,info=self.env.reset(**kwargs)
        self.episode_config=self.env.config.copy()
        info['training_goal_height_m']=self.episode_config['lift_height']
        info['training_approach_distance_m']=self.episode_config['approach_distance']
        self.previous_phi=potential(info,self.episode_config,self.reward_config)
        return encode(obs),info

    def step(self,action):
        obs,base,terminated,truncated,info=self.env.step(action)
        reward,terms=shaped_reward(self.previous_phi,info,terminated,self.episode_config,self.reward_config)
        self.previous_phi=potential(info,self.episode_config,self.reward_config)
        info.update(environment_reward=base,environment_reward_components=info['reward_components'],
                    reward_components=terms,ppo_reward=reward,reward_mode=self.reward_config['name'],
                    training_goal_height_m=self.episode_config['lift_height'])
        info['training_approach_distance_m']=self.episode_config['approach_distance']
        return encode(obs),reward,terminated,truncated,info
