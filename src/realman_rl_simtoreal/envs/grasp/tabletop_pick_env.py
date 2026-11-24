"""关节空间桌面抓取环境，复用局部抬升任务的接触检查与物理保持判据。"""
import copy
import math

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from .grasp_lift_env import GraspLiftEnv
from ...paths import TABLETOP_PICK_SCENE
from ...simulation.physics.model import load_model,initialize,require
from ...simulation.motion.kinematics import Kinematics,rotation_vector
from ...simulation.physics.joint_control import advance_synchronized_cycle
from ...simulation.physics.grasp_contact import (
    ContactPolicy,ContactRecorder,PAD_PENETRATION,SUPPORT_PENETRATION,UNEXPECTED_PENETRATION)
from ...simulation.devices.mujoco_gripper import Gripper,SIGNS,MAX_WIDTH,width_to_angle
from ...tasks.grasp.tabletop_joint import TABLETOP_PICK_CONFIG, REWARD, potential, shaped_reward


class TabletopPickEnv(GraspLiftEnv):
    """以七维绝对目标执行完整桌面抓取。

    前六个动作是关节角（度），最后一个是归一化夹爪开口。kwargs 可调整起始姿态、
    物体区域与完成条件。reset 构造回合初始状态，step 对关节参考限速、预检后执行，
    并返回 56 维原始观测及诊断信息。
    """
    metadata={'render_modes':['ansi'],'render_fps':20}

    def __init__(self,**kwargs):
        render_mode=kwargs.pop('render_mode',None)
        unknown=set(kwargs)-set(TABLETOP_PICK_CONFIG)
        if unknown:raise ValueError('Unknown tabletop options: '+str(sorted(unknown)))
        self.config=copy.deepcopy(dict(TABLETOP_PICK_CONFIG,**kwargs));c=self.config
        self.initial_q=np.deg2rad(np.asarray(c['initial_q_deg'],dtype=float))
        require(self.initial_q.shape==(6,) and np.isfinite(self.initial_q).all(),'Six finite initial joint angles in degrees required')
        xy=np.asarray(c['object_xy'],dtype=float)
        require(xy.shape==(2,) and np.isfinite(xy).all(),'Finite object XY required')
        require(0<=c['object_radius']<=.1 and -.25<xy[0]<.65 and abs(xy[1])<.25,'Object outside the tabletop task area')
        require(.001<=c['mass']<=.1 and 0<=c['friction']<=2,'Invalid cube mass/friction')
        require(.021<=c['initial_width']<=MAX_WIDTH and 0<c['gripper_reference_speed']<=.5,'Invalid gripper configuration')
        require(np.isfinite(c['gripper_command_margin']) and 0<=c['gripper_command_margin']<=.005,'Invalid gripper command margin')
        require(.005<=c['lift_height']<=.1 and 0<c['height_tolerance']<c['lift_height']/2 and .1<=c['hold_seconds']<=10,'Invalid lift/hold configuration')
        self.goal_height=c['lift_height'] if c['goal_height'] is None else c['goal_height']
        require(np.isfinite(self.goal_height) and c['lift_height']<=self.goal_height<=.1,'Goal height must cover the success threshold')
        require(0<=c['start_fraction']<=1 and .002<=c['pregrasp_height']<=.15,'Invalid reset curriculum')
        require(c['approach_waypoint_height']==0 or .03<=c['approach_waypoint_height']<=.15,'Invalid approach waypoint height')
        require(.001<=c['approach_waypoint_tolerance']<=.02 and
                (c['approach_waypoint_height']==0 or c['approach_waypoint_height']-c['approach_waypoint_tolerance']>=.02),
                'Approach corridor must remain at least 20 mm above the cube center')
        require(c['control_period']==.05 and c['physics_timestep']==.001 and 0<c['joint_action_scale']<=np.deg2rad(.5),'Invalid protected control period or action scale')
        require(np.isfinite(c['reference_smoothing']) and 0<c['reference_smoothing']<=1,'Reference smoothing must be in (0,1]')
        require(isinstance(c['max_episode_steps'],int) and not isinstance(c['max_episode_steps'],bool) and c['max_episode_steps']>0,'Positive integer episode budget required')
        require(isinstance(c['record_physics'],bool) and isinstance(c['reject_unsafe_targets'],bool),'Boolean recording/shield flags required')
        require(render_mode in (None,'ansi'),'Only ansi environment rendering is supported')
        self.render_mode=render_mode
        self.model,self.joint_ids,self.actuator_ids=load_model(TABLETOP_PICK_SCENE,gravity_compensation=True)
        self.model.opt.timestep=.001
        body=self.model.body('target_object').id
        self.model.body_mass[body]=c['mass'];self.model.body_inertia[body]=c['mass']*.02**2/6
        for i in range(self.model.npair):
            name=self.model.pair(i).name
            if name.endswith('_target') and name!='support_target':self.model.pair_friction[i,:2]=c['friction']
        self.data=initialize(self.model)
        self.kinematics=Kinematics(self.model)
        self._validate_q(self.initial_q)
        self.contact_policy=ContactPolicy(self.model)
        self.physics_steps=50;self.control_period=.05
        self.max_episode_steps=c['max_episode_steps']
        self.object_joint=self.model.joint('target_free').id
        self.object_qadr=int(self.model.jnt_qposadr[self.object_joint]);self.object_vadr=int(self.model.jnt_dofadr[self.object_joint])
        self.object_body=body;self.support_body=self.model.body('support').id
        self.support_rotation=np.eye(3);self.support_normal=np.array([0.,0.,1.])
        self.table_height=float(self.model.body_pos[self.support_body,2]+self.model.geom_size[self.model.geom('support_top').id,2])
        self.tcp_site=self.model.site('grasp_site').id
        self._probe=mujoco.MjData(self.model);self._candidate=mujoco.MjData(self.model)
        self._tcp_offset=self.data.site_xmat[self.kinematics.site_id].reshape(3,3).T@(
            self.data.site_xpos[self.tcp_site]-self.data.site_xpos[self.kinematics.site_id])
        # The gripper's Z points down; the cube's parallel faces align with its jaws.
        self.grasp_rotation=np.diag([-1.,1.,-1.])
        self._hold_ticks=math.ceil(c['hold_seconds']/.001)
        limits=np.rad2deg(self.model.jnt_range[self.joint_ids])
        self.action_space=spaces.Box(np.r_[limits[:,0],-1.].astype(np.float32),
                                    np.r_[limits[:,1],1.].astype(np.float32),dtype=np.float32)
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(56,),dtype=np.float32)
        self._closed=False;self._episode_done=True

    def _validate_q(self,q):
        q=np.asarray(q,dtype=float)
        require(q.shape==(6,) and np.isfinite(q).all(),'Invalid joint reference')
        limits=self.model.jnt_range[self.joint_ids]
        require(np.all(q>limits[:,0]) and np.all(q<limits[:,1]),'Joint reference exceeds model limits')
        return q

    def _solve_goal(self,tcp,seed):
        return self.kinematics.solve_ik(np.asarray(tcp)-self.grasp_rotation@self._tcp_offset,seed,self.grasp_rotation,
            max_iterations=600,damping=.001,seed_weight=1e-6,position_tolerance=1e-6)

    def _goals(self,position):
        # Deterministic private calculations only; no physical warm-up or teacher actions.
        for seed in (np.deg2rad([0,15,84,0,81,0]),self.initial_q,np.zeros(6)):
            pre=self._solve_goal(position+np.array([0,0,self.config['pregrasp_height']]),seed)
            if not pre.success:continue
            grasp=self._solve_goal(position,pre.q)
            lifted=self._solve_goal(position+np.array([0,0,self.goal_height]),pre.q)
            if grasp.success and lifted.success:return pre.q,grasp.q,lifted.q
        raise RuntimeError('No tabletop grasp/lift IK goal at the sampled cube position')

    def reset(self,*,seed=None,options=None):
        """按种子设置桌面目标和课程起点，返回观测、初始角度与当前子目标。"""
        if self._closed:raise RuntimeError('Environment is closed')
        if options:raise ValueError('Tabletop reset options are recorded in the environment configuration')
        gym.Env.reset(self,seed=seed)
        c=self.config;kin=self.kinematics
        for attempt in range(64):
            angle=self.np_random.uniform(0,2*np.pi);radius=c['object_radius']*np.sqrt(self.np_random.uniform())
            xy=np.asarray(c['object_xy'])+radius*np.array([np.cos(angle),np.sin(angle)])
            position=np.r_[xy,self.table_height+.01]
            try:pre,grasp,lifted=self._goals(position)
            except RuntimeError:continue
            q=c['start_fraction']*self.initial_q+(1-c['start_fraction'])*pre
            self._validate_q(q)
            mujoco.mj_setConst(self.model,self._probe)
            mujoco.mj_resetDataKeyframe(self.model,self.data,self.model.key('home').id)
            self.data.qpos[kin.qadr]=q;self.data.ctrl[self.actuator_ids]=q
            self.data.qpos[self.object_qadr:self.object_qadr+7]=np.r_[position,1.,0.,0.,0.]
            for name,sign in SIGNS.items():
                self.data.qpos[self.model.joint(name).qposadr[0]]=sign*width_to_angle(c['initial_width'])
            self.data.ctrl[self.model.actuator('eg2_position').id]=width_to_angle(c['initial_width'])
            mujoco.mj_forward(self.model,self.data)
            try:self.contact_policy.check(self.data)
            except ValueError as exc:
                if c['start_fraction']==1:raise ValueError('Configured full-task initial pose collides: '+str(exc)) from exc
                continue
            break
        else:raise RuntimeError('No collision-free tabletop reset after 64 samples')
        self.grasp_q=grasp;self.lift_q=lifted;self.pregrasp_q=pre
        self._lift_goal_latched=False
        self._approach_complete=c['approach_waypoint_height']==0
        self.reach_q=self.grasp_q
        self.approach_reference=position+np.array([0,0,c['approach_waypoint_height']])
        if not self._approach_complete:
            solved=self._solve_goal(self.approach_reference,pre)
            require(solved.success,'No approach waypoint IK goal')
            self.reach_q=solved.q
        self.gripper=Gripper(self.model,self.data,contact_policy=self.contact_policy,reference_speed=c['gripper_reference_speed'])
        self.recorder=ContactRecorder(self.model,self.data,self.gripper,self.contact_policy,keep_history=c['record_physics'])
        self._initial_object=position.copy();self.object_goal=position+np.array([0,0,self.goal_height])
        self.grasp_reference=position.copy();self.pregrasp_reference=position+np.array([0,0,c['pregrasp_height']])
        self.lift_reference=self.object_goal.copy()
        p,r=kin.fk(q);self._command_position=p+r@self._tcp_offset
        self._requested_width=c['initial_width']
        self._elapsed_steps=self._stable_ticks=self._max_stable_ticks=self._lost_ticks=0
        self._hold_anchor=None;self._ever_lifted=False;self._last_tick_good=False
        self._episode_done=False;self._reset_attempts=attempt+1;self._action_rejected=False;self._rejection_detail=None
        self._reset_q=q.copy()
        self._update_reach_goal()
        obs=self._observation();info=self._info()
        self._last_observation=obs.copy();self._last_info=copy.deepcopy(info)
        return obs,info

    def _geometry(self):
        self._probe.qpos[:]=self.data.qpos;mujoco.mj_forward(self.model,self._probe)
        position=self.data.qpos[self.object_qadr:self.object_qadr+3].copy()
        rotation=self._probe.xmat[self.object_body].reshape(3,3)
        radius=.01*np.sum(np.abs(rotation.T@self.support_normal))
        clearance=float(position[2]-self.table_height-radius)
        return position,self._probe.site_xpos[self.tcp_site].copy(),clearance

    def _update_reach_goal(self):
        if self._approach_complete:return
        _,tcp,_=self._geometry()
        orientation=rotation_vector(self.grasp_rotation,self._probe.site_xmat[self.kinematics.site_id].reshape(3,3))
        error=tcp-self.approach_reference
        if (np.linalg.norm(error[:2])<.005 and abs(error[2])<self.config['approach_waypoint_tolerance']
                and np.linalg.norm(orientation)<.03):
            self._approach_complete=True
            self.reach_q=self.grasp_q

    def _observation(self):
        position,tcp,clearance=self._geometry();kin=self.kinematics
        q=self.data.qpos[kin.qadr];v=self.data.qvel[self.object_vadr:self.object_vadr+6]
        jac=np.empty((3,self.model.nv));mujoco.mj_jacSite(self.model,self._probe,jac,None,self.tcp_site)
        orientation=rotation_vector(self.grasp_rotation,self._probe.site_xmat[kin.site_id].reshape(3,3))
        value=np.concatenate([(q-self.initial_q)/np.pi,self.data.qvel[kin.vadr]/np.deg2rad(30),
            (position-tcp)/.05,(position-self._initial_object)/.05,(self.object_goal-position)/.05,
            v[:3]/.05,v[3:],
            [self.gripper.get_width()/MAX_WIDTH,self._requested_width/MAX_WIDTH,self.data.ctrl[self.gripper.actuator_id]/.82],
            self._forces()/np.array([.2,.2,.1]),[clearance/self.goal_height,self._stable_ticks/self._hold_ticks],
            (jac@self.data.qvel)/.05,(self._command_position-tcp)/.01,orientation/np.pi,
            (self.reach_q-q)/.1,(self.lift_q-q)/.1]).astype(np.float32)
        require(value.shape==(56,) and np.isfinite(value).all(),'Invalid tabletop observation')
        return value

    def _info(self,reason=None,detail=None,physics_steps=0):
        info=super()._info(reason,detail,physics_steps)
        q=self.data.qpos[self.kinematics.qadr]
        orientation=rotation_vector(self.grasp_rotation,self._probe.site_xmat[self.kinematics.site_id].reshape(3,3))
        info.update(task='tabletop_pick',initial_q_deg=np.rad2deg(self._reset_q).tolist(),
            configured_initial_q_deg=self.config['initial_q_deg'],initial_pose_name=self.config['initial_pose_name'],
            initial_pose_source=self.config['initial_pose_source'],start_fraction=self.config['start_fraction'],
            goal_height_m=self.goal_height,required_lift_height_m=self.config['lift_height'],
            table_height_m=self.table_height,table_normal=[0.,0.,1.],
            joint_position_rad=q.tolist(),command_joint_position_rad=self.data.ctrl[self.actuator_ids].tolist(),
            grasp_goal_joint_rad=self.grasp_q.tolist(),lift_goal_joint_rad=self.lift_q.tolist(),
            grasp_joint_error_rad=float(np.linalg.norm(q-self.grasp_q)),
            lift_joint_error_rad=float(np.linalg.norm(q-self.lift_q)),
            reach_goal_joint_rad=self.reach_q.tolist(),reach_joint_error_rad=float(np.linalg.norm(q-self.reach_q)),
            approach_complete=self._approach_complete,approach_reference_m=self.approach_reference.tolist(),
            approach_waypoint_tolerance_m=self.config['approach_waypoint_tolerance'],
            reach_goal_distance_m=(info['grasp_distance_m'] if self._approach_complete else
                float(np.linalg.norm(self.approach_reference-np.asarray(info['grasp_position_m'])))),
            orientation_error_rad=float(np.linalg.norm(orientation)),
            action_rejected=self._action_rejected,rejection_detail=self._rejection_detail)
        info['grasp_relative_lift_goal']=self._lift_goal_latched
        info['commanded_width_m']=self.gripper.target_width
        info['jaw_command_limited']=not np.isclose(self._requested_width,self.gripper.target_width,rtol=0,atol=1e-12)
        return info

    def _safe_reference(self,target):
        limits=self.model.jnt_range[self.joint_ids]
        if np.any(target<=limits[:,0]) or np.any(target>=limits[:,1]):return False,'joint_limit'
        if not self.config['reject_unsafe_targets']:return True,None
        probe=self._candidate;probe.qpos[:]=self.data.qpos;probe.qpos[self.kinematics.qadr]=target
        mujoco.mj_forward(self.model,probe)
        held=min(self._forces()[:2])>.005
        for contact in probe.contact:
            category=self.contact_policy.classify(*contact.geom)
            # A held free cube moves with the pads. A frozen-object FK probe
            # cannot predict that motion; the shared 1 ms physics checks still
            # enforce the original pad penetration and force/speed limits.
            if held and category.endswith('_pad_target'):continue
            limit=(PAD_PENETRATION if category.endswith('_pad_target') else
                   SUPPORT_PENETRATION if category=='target_support' else UNEXPECTED_PENETRATION)
            if not np.isfinite(contact.dist) or contact.dist < -limit:
                return False,category
        return True,None

    def step(self,action):
        """对七维动作执行碰撞预检和同步限速，推进仿真并更新完成状态。"""
        if self._closed or self._episode_done:raise RuntimeError('Reset an open environment before stepping')
        action=np.asarray(action,dtype=np.float32)
        if (action.shape!=(7,) or not np.isfinite(action).all()
                or np.any(action<self.action_space.low-1e-6) or np.any(action>self.action_space.high+1e-6)):
            raise ValueError('Six bounded joint positions in degrees and one normalized jaw action required')
        action=np.clip(action,self.action_space.low,self.action_space.high)
        self._elapsed_steps+=1;reason=detail=None;before=float(self.data.time)
        limits=self.model.jnt_range[self.joint_ids]
        requested=np.clip(np.deg2rad(action[:6].astype(float)),limits[:,0]+1e-5,limits[:,1]-1e-5)
        start=self.data.ctrl[self.actuator_ids].copy()
        delta=self.config['reference_smoothing']*(requested-start)
        # Share one interpolation fraction across all joints. Independent
        # per-axis clipping changes the path and can pitch the fingers into
        # the tabletop before the remaining joints reach their references.
        fraction=min(1.,self.config['joint_action_scale']/max(float(np.max(np.abs(delta))),1e-12))
        target=start+fraction*delta
        safe,self._rejection_detail=self._safe_reference(target);self._action_rejected=not safe
        if not safe:target=self.data.ctrl[self.actuator_ids].copy()
        try:
            self._requested_width=float((action[6]+1)*MAX_WIDTH/2)
            margin=self.config['gripper_command_margin']
            self.gripper.set_width(float(np.clip(self._requested_width,margin,MAX_WIDTH-margin)))
            advance_synchronized_cycle(self.model,self.data,self.joint_ids,self.actuator_ids,target,self.gripper,
                physics_steps=self.physics_steps,step_observer=self._observe_tick,max_home_delta=2*np.pi)
            p,r=self.kinematics.fk(self.data.ctrl[self.actuator_ids]);self._command_position=p+r@self._tcp_offset
        except (ValueError,RuntimeError,mujoco.FatalError) as exc:reason='safety_failure';detail=str(exc)
        ticks=round((float(self.data.time)-before)/.001)
        try:
            self._update_reach_goal()
            # A geometric observation target follows the actual grasp offset.
            # It does not move the arm. Object poses/forces are never modified.
            if not self._lift_goal_latched and min(self._forces()[:2])>.005:
                position,tcp,_=self._geometry()
                lift_tcp=self.object_goal+(tcp-position)+np.array([0,0,.002])
                solved=self._solve_goal(lift_tcp,self.grasp_q)
                if solved.success:
                    self.lift_q=solved.q;self.lift_reference=lift_tcp
                    self._lift_goal_latched=True
            obs=self._observation();info=self._info(reason,detail,ticks)
        except (ValueError,RuntimeError,mujoco.FatalError) as exc:
            obs=self._last_observation.copy();info=copy.deepcopy(self._last_info)
            reason='numerical_failure';detail=str(exc)
            info.update(observation_valid=False,elapsed_steps=self._elapsed_steps,physics_steps=ticks)
        if reason is None:
            if self._stable_ticks>=self._hold_ticks:reason='success'
            elif self._lost_ticks*.001>=.5:reason='dropped'
            elif info['object_position_m'][2]<self.table_height-.03:reason='object_lost'
        terminated=reason is not None;truncated=not terminated and self._elapsed_steps>=self.max_episode_steps
        if truncated:reason='time_limit'
        info.update(is_success=reason=='success',termination_reason=reason,detail=detail,
            is_safety_failure=reason not in (None,'success','time_limit','dropped','object_lost'))
        info['requested_joint_position_rad']=requested.tolist()
        info['commanded_joint_delta_rad']=(self.data.ctrl[self.actuator_ids]-start).tolist()
        self._episode_done=terminated or truncated
        self._last_observation=obs.copy();self._last_info=copy.deepcopy(info)
        # Algorithm-specific reward is supplied by PPOTabletopEnv.
        reward=float(25*info['is_success']-25*info['is_safety_failure']-.01)
        return obs,reward,terminated,truncated,info


ENCODING=dict(name='joint_goal_tabletop_v1',input_dim=56,output_dim=56,
    position_scale_m=.01,command_lag_scale_m=.001,joint_goal_scale_rad=.05,running_normalization=False)
ACTION_CONTRACT=dict(name='absolute_joint_positions_deg_v1',dimension=7,
    joint_units='degree',jaw_units='normalized_absolute_opening',rate_limited=True)


def encode(observation):
    """使用原固定 asinh 尺度编码误差通道，同时保留其他原始通道。"""
    value=np.asarray(observation,dtype=np.float32).copy()
    if value.shape[-1:]!=(56,) or not np.isfinite(value).all():raise ValueError('Finite 56-vector required')
    value[...,12:21]=np.arcsinh(value[...,12:21]*5)
    value[...,38:41]=np.arcsinh(value[...,38:41]*10)
    value[...,44:56]=np.arcsinh(value[...,44:56]*2)
    return value


class PPOTabletopEnv(gym.Wrapper):
    """在关节桌面环境上应用固定观测编码和势差奖励，保留动作执行前的目标供正则计算。"""
    def __init__(self,config,reward=None):
        super().__init__(TabletopPickEnv(**config))
        self.reward_config=copy.deepcopy(REWARD if reward is None else reward)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(56,),dtype=np.float32)

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);self.previous=potential(info,self.env.config,self.reward_config)
        self.action_info=copy.deepcopy(info)
        return encode(obs),info

    def step(self,action):
        self.env.recorder.phase='ppo'
        obs,base,terminated,truncated,info=self.env.step(action)
        reward,terms=shaped_reward(self.previous,info,terminated,self.env.config,self.reward_config,
                                  action=action,action_info=self.action_info)
        self.previous=potential(info,self.env.config,self.reward_config)
        self.action_info=copy.deepcopy(info)
        info.update(environment_reward=base,ppo_reward=reward,reward_components=terms)
        return encode(obs),reward,terminated,truncated,info
