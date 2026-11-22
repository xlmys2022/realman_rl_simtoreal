"""笛卡尔 PPO 模型创建、checkpoint 契约和共享评估薄适配。"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ..utils.artifacts import write_json as _write_json, append_json, file_hashes, archive_sources
from .common.artifacts import optimizer_steps, optimizer_state_hash
from ..tasks.grasp.tabletop_cartesian import CartesianTabletopConfig as Config
from ..evaluation import tabletop_cartesian as evaluation
from ..paths import PROJECT_ROOT as ROOT, TABLETOP_CARTESIAN_SCENE as DEFAULT_SCENE
from ..envs.grasp.tabletop_cartesian_env import (ACTION_CONTRACT,OBSERVATION_FIELDS,
                                         action_contract,observation_fields)
from ..tasks.grasp.tabletop_cartesian_reward import REWARD
from ..simulation.runtime.observation_presets import startup_contract
from ..simulation.physics.grasp_metrics import CONTACT_MEASUREMENT
from ..envs.grasp.tabletop_cartesian_vision_env import VISION_CONTRACT
from .cartesian_policy import CartesianPPOPolicy
from .cartesian_reach_policy import CartesianGoalFeatures, ReachGraspPPOPolicy
from .vision_policy import VisualCartesianGoalFeatures, VisualReachGraspPPOPolicy


def write_json(path,value):
    return _write_json(path,value,ensure_ascii=False,allow_nan=True)


def hashes(model):
    result={key:hashlib.sha256() for key in ('actor_mean','critic','all')}
    for name,tensor in model.policy.state_dict().items():
        blob=name.encode()+tensor.detach().cpu().contiguous().numpy().tobytes()
        result['all'].update(blob)
        if name.startswith(('mlp_extractor.policy_net.','action_net.')):result['actor_mean'].update(blob)
        if name.startswith(('mlp_extractor.value_net.','value_net.')):result['critic'].update(blob)
    return {k:v.hexdigest() for k,v in result.items()}


def provenance():
    files=list((ROOT/'src/realman_rl_simtoreal').rglob('*.py'))+[DEFAULT_SCENE]
    return file_hashes(sorted(files),root=ROOT)


def capture_sources(folder):
    """Archive this feature's sources and shared assets as training provenance."""
    files={ROOT/name for name in provenance()}
    files.update((ROOT/'assets').rglob('*'))
    files.update((ROOT/'scripts').rglob('*.py'))
    files.update(ROOT/name for name in ('pyproject.toml','requirements.txt','requirements-rl.txt','requirements-vision.txt'))
    archive_sources(Path(folder)/'source_snapshot.zip',sorted(files),root=ROOT)


def make_model(vector, *, seed, learning_rate, n_steps, batch_size, std, adaptive,
               exploration_profile, structured, vision, goal_features):
    """按真值或 RGB-D 观测契约创建既有的 PPO 策略与探索配置。"""
    exploration=dict(adaptive_exploration=adaptive,
        exploration_profile='distance' if vision else exploration_profile,
        exploration_distance=.04,motion_std_floor=.03,jaw_std_floor=.2,jaw_exploration_distance=.01,
        cage_xy_scale=.004,cage_z_scale=.005,grasp_offset_z=.003,jaw_far_std_floor=.1,
        jaw_held_std_floor=.1,grip_force_scale=.1,lift_exploration_gain=2.)
    policy_kwargs=dict(log_std_init=float(np.log(std)),**exploration)
    if structured:policy_kwargs.update(features_extractor_class=VisualCartesianGoalFeatures if vision else CartesianGoalFeatures,
                                      features_extractor_kwargs=goal_features)
    else:policy_kwargs['net_arch']=dict(pi=[128,128],vf=[128,128])
    policy_class=VisualReachGraspPPOPolicy if vision else ReachGraspPPOPolicy if structured else CartesianPPOPolicy
    model=PPO(policy_class,vector,
              learning_rate=learning_rate,n_steps=n_steps,batch_size=batch_size,
              n_epochs=10,gamma=REWARD['gamma'],gae_lambda=.95,clip_range=.2,ent_coef=.001,vf_coef=.5,
              max_grad_norm=.5,target_kl=.04,policy_kwargs=policy_kwargs,
              seed=seed,device='cpu',verbose=0)
    return model, exploration


def save(model,folder,config,complete=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);model.save(folder/'model.zip')
    write_json(folder/'config.json',config)
    write_json(folder/'state.json',dict(timesteps=int(model.num_timesteps),optimizer_steps=optimizer_steps(model),
        parameter_hashes=hashes(model),completed_update=bool(complete and model.num_timesteps>0)))


def load(path,*,allow_legacy_startup=False,allow_legacy_observation=False):
    """加载 CPU 笛卡尔 PPO 及保存的 JSON 配置。

    核对启动姿态、动作和真值或视觉观测契约。显式的 legacy 参数仅用于原有权重迁移路径。
    """
    path=Path(path);config=json.loads((path.parent/'config.json').read_text());torch.set_num_threads(1)
    legacy=config.get('schema',1)<2 or 'startup_contract' not in config
    if legacy:
        if not allow_legacy_startup:
            raise ValueError('Legacy near-object startup checkpoint: evaluate only a model trained from a fixed observation pose; '
                             'old weights may be supplied to --init-model for retraining')
        expected_action,expected_fields=ACTION_CONTRACT,OBSERVATION_FIELDS
    else:
        cfg=Config(**config['environment'])
        if config['startup_contract']!=startup_contract(cfg.observation_pose):
            raise ValueError('Checkpoint fixed observation pose contract does not match the configured preset')
        expected_action,expected_fields=action_contract(cfg),observation_fields(cfg)
    if config.get('action_contract')!=expected_action or config.get('observation_fields')!=[list(v) for v in expected_fields]:
        raise ValueError('Expected this project\'s Cartesian checkpoint; joint-space legacy models are incompatible')
    if config.get('contact_measurement')!=CONTACT_MEASUREMENT and not allow_legacy_observation:
        raise ValueError('Checkpoint uses the old contact-force observation; use its archived source for historical evaluation '
                         'or --init-model to retrain with inward grip forces')
    model=PPO.load(path,device='cpu')
    vision=not legacy and cfg.observation_source=='rgbd_hsv'
    if vision!=isinstance(model.policy,VisualReachGraspPPOPolicy):
        raise ValueError('Checkpoint policy and camera observation contract do not match; retrain an explicit transfer')
    if vision and config.get('vision_contract')!=VISION_CONTRACT:
        if not (allow_legacy_observation and config.get('vision_contract',{}).get('name') in
                ('wrist_rgbd_hsv_cube_v1','wrist_rgbd_hsv_cube_v2','wrist_rgbd_hsv_cube_v3')):
            raise ValueError('Unsupported visual sensor contract; transfer old visual weights through --init-model')
    if (model.observation_space.shape!=(sum(n for _,n in expected_fields),)
            or model.action_space.shape!=(expected_action['shape'],)):raise ValueError('Invalid policy dimensions')
    return model,config


def deterministic_action(model,observation):
    """直接读取现有动作分布的确定性输出，交给评估器执行原 Box 裁剪。"""
    tensor,_=model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        return model.policy.get_distribution(tensor).get_actions(deterministic=True).cpu().numpy().reshape(model.action_space.shape)


def evaluate(model,config,seeds,*,full=False,tick=None,environment=None,observation_pose=None,object_radius=None):
    """Keep the PPO API and JSON contract while sharing task evaluation."""
    def fingerprint():
        return (hashes(model),model.num_timesteps,model._n_updates,optimizer_steps(model),optimizer_state_hash(model))
    return evaluation.evaluate(lambda obs:deterministic_action(model,obs),config,seeds,
        action_shape=model.action_space.shape,full=full,tick=tick,environment=environment,
        observation_pose=observation_pose,object_radius=object_radius,
        state_fingerprint=fingerprint,sources=provenance,algorithm='PPO')


class Audit(BaseCallback):
    """记录每个笛卡尔 PPO 回合的执行摘要并检查采样和观测的有效性。"""
    def __init__(self,folder):super().__init__();self.folder=Path(folder);self.episodes=[]
    def _on_step(self):
        raw=np.asarray(self.locals['actions']);applied=np.asarray(self.locals['clipped_actions'])
        if raw.shape[-1]!=self.model.action_space.shape[0] or not np.isfinite(raw).all() or not np.array_equal(np.clip(raw,-1,1),applied):
            raise ValueError('Invalid PPO Cartesian actions')
        for info,done in zip(self.locals['infos'],self.locals['dones']):
            if not info['observation_valid']:raise ValueError('Invalid observation')
            if done:
                row=dict(step=int(self.model.num_timesteps),success=info['is_success'],reason=info['termination_reason'],
                         detail=info['detail'],length=info['episode']['l'],return_=info['episode']['r'],
                         clearance_m=info['clearance_m'],max_hold_seconds=info['max_hold_seconds'],
                         max_clearance_m=info['max_clearance_m'],max_bilateral_force_n=info['max_bilateral_force_n'],
                         min_tcp_object_distance_m=info['min_tcp_object_distance_m'])
                self.episodes.append(row)
                append_json(self.folder/'episodes.jsonl',row,allow_nan=True)
        return True
