"""Offline supervised Reach experiment; physics and expert stay unchanged.

BC has the SAC deterministic mean-path capacity, but no log-std head,
distribution, critic, entropy, TD targets or replay. It is not a SAC model.
"""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from ..envs.reach_env import ReachEnv
from .common import rule_action
from .sac_reach import ENV_CONFIG, ReachTransform, write_json, versions


FEATURE_KEYS = ('achieved_goal', 'desired_goal', 'observation')
FINAL_SEEDS = tuple(range(20000, 20100))
TRAINING_CONFIG = dict(learning_rate=.001, batch_size=256, epochs=300, seed=2026)


def empty_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output directory must be empty; never overwrite an experiment')
    path.mkdir(parents=True, exist_ok=True)
    return path


def features(encoded):
    """Same sorted Dict/Flatten order and float32 cast as SB3 CombinedExtractor."""
    result = np.concatenate([encoded[k] for k in FEATURE_KEYS], axis=-1).astype(np.float32)
    if result.shape[-1:] != (33,) or not np.isfinite(result).all():
        raise ValueError('Expected finite 33-dimensional processed Dict features')
    return result


def weight_digest(network):
    h = hashlib.sha256()
    for key, value in network.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class BCNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_pi = nn.Sequential(nn.Linear(33, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.mu = nn.Linear(256, 3)

    def forward(self, inputs):
        return torch.tanh(self.mu(self.latent_pi(inputs)))


class BCPolicy:
    def __init__(self, network, transform):
        self.network, self.transform = network.eval(), transform

    def predict(self, raw_observation):
        inputs = features(self.transform.encode(raw_observation))
        with torch.no_grad():
            result = self.network(torch.from_numpy(inputs)).cpu().numpy()
        if not np.isfinite(result).all() or np.any(np.abs(result) > 1):
            raise ValueError('Non-finite or illegal BC action')
        return result.astype(np.float32)


def save_policy(path, network, config):
    torch.save(dict(kind='reach_behavior_cloning', schema=1, feature_keys=list(FEATURE_KEYS),
                    network='33-256-ReLU-256-ReLU-3-tanh', state_dict=network.state_dict(),
                    config=config), path)


def load_policy(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved.get('kind') != 'reach_behavior_cloning' or saved.get('schema') != 1
            or saved.get('feature_keys') != list(FEATURE_KEYS)
            or saved.get('network') != '33-256-ReLU-256-ReLU-3-tanh'):
        raise ValueError('Not a supported BC model')
    with torch.random.fork_rng(devices=[]):
        network = BCNetwork()
    network.load_state_dict(saved['state_dict'], strict=True)
    if not all(torch.isfinite(p).all() for p in network.parameters()):
        raise ValueError('Non-finite BC parameters')
    return BCPolicy(network, ReachTransform(saved['config']['observation_transform'])), saved['config']


def stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return dict(count=0)
    return dict(count=int(values.size), mean=float(values.mean()), min=float(values.min()),
                median=float(np.median(values)), p95=float(np.percentile(values, 95)), max=float(values.max()))


def cosine(a, b):
    denominator = np.linalg.norm(a, axis=-1)*np.linalg.norm(b, axis=-1)
    mask = denominator > 1e-12
    return np.clip(np.sum(a[mask]*b[mask], axis=-1)/denominator[mask], -1., 1.)


def action_metrics(predictions, labels, raw):
    """Empty stopping strata explicitly have count=0 and null errors."""
    predictions, labels = np.asarray(predictions, dtype=float), np.asarray(labels, dtype=float)
    if predictions.shape != labels.shape or labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError('Expected matching Nx3 predictions and labels')
    if not np.isfinite(predictions).all() or not np.isfinite(labels).all():
        raise ValueError('Non-finite action metrics')
    error = predictions-labels
    label_norm, pred_norm = np.linalg.norm(labels, axis=1), np.linalg.norm(predictions, axis=1)
    def subset(mask):
        return dict(count=int(mask.sum()), mse=float(np.mean(error[mask]**2)) if mask.any() else None,
                    mae=float(np.mean(np.abs(error[mask]))) if mask.any() else None,
                    predicted_norm_mean=float(pred_norm[mask].mean()) if mask.any() else None)
    direction = cosine(predictions[label_norm > .01], labels[label_norm > .01])
    actual_error = raw['desired_goal']-raw['achieved_goal']
    command_error = raw['desired_goal']-raw['observation'][:, 18:21]
    actual_cos, command_cos = cosine(predictions, actual_error), cosine(predictions, command_error)
    return dict(mse=float(np.mean(error**2)), mae=float(np.mean(np.abs(error))),
                direction_cosine=stats(direction),
                toward_actual_fraction=float(np.mean(actual_cos > 0)) if len(actual_cos) else None,
                toward_command_fraction=float(np.mean(command_cos > 0)) if len(command_cos) else None,
                norm_mae=float(np.abs(pred_norm-label_norm).mean()),
                moving=subset(label_norm > .01), near_zero_stop=subset(label_norm <= .01),
                small_action=subset((label_norm > .01) & (label_norm < .1)),
                near_goal=subset(np.linalg.norm(actual_error, axis=1) <= .0015))


def dataset_digest(data):
    h = hashlib.sha256()
    for key in sorted(data):
        array = np.ascontiguousarray(data[key])
        h.update((key+str(array.dtype)+str(array.shape)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def collect_demonstrations(train_seeds, validation_seeds, *, environment=None):
    train_seeds, validation_seeds = list(train_seeds), list(validation_seeds)
    seeds = train_seeds+validation_seeds
    if (not train_seeds or not validation_seeds or any(s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds) or set(seeds) & set(FINAL_SEEDS)):
        raise ValueError('Nonempty disjoint nonnegative train/validation seeds, excluding final 20000..20099, required')
    config = dict(ENV_CONFIG if environment is None else environment)
    samples, episodes = [], []
    started = perf_counter()
    with ReachEnv(**config) as env:
        transform = ReachTransform.from_env(env)
        home = env.kinematics.home_position.copy()  # only used for distribution report
        for episode, seed in enumerate(seeds):
            obs, info = env.reset(seed=int(seed))
            group = 0 if episode < len(train_seeds) else 1
            goal = obs['desired_goal'].copy()
            begin = len(samples)
            for step in range(env.max_episode_steps):
                # Expert sees only the public Dict and the documented action scale.
                action = rule_action(obs, env.action_scale)
                if not env.action_space.contains(action):
                    raise ValueError('Expert returned illegal action')
                encoded = transform.encode(obs)
                record = {**{'raw_'+k:v.copy() for k,v in obs.items()},
                          **{'processed_'+k:v.copy() for k,v in encoded.items()},
                          'features':features(encoded), 'action':action.copy(),
                          'actual_position_m':obs['achieved_goal'].copy(),
                          'command_position_m':obs['observation'][18:21].copy(),
                          'target_position_m':goal.copy(), 'episode_index':episode,
                          'step_index':step, 'seed':seed, 'split':group}
                obs, reward, terminated, truncated, info = env.step(action)
                record.update(reward=reward, step_success=info['is_success'],
                              terminated=terminated, truncated=truncated,
                              simulation_time_s=info['simulation_time_s'])
                samples.append(record)
                if terminated or truncated:
                    break
            for record in samples[begin:]:
                record['episode_success'] = info['is_success']
            episodes.append(dict(seed=int(seed), split='train' if group == 0 else 'validation',
                                 episode_index=episode, length=len(samples)-begin,
                                 target_m=goal.tolist(), target_offset_m=(goal-home).tolist(),
                                 success=info['is_success'], reason=info['termination_reason'],
                                 final_distance_m=info['distance_m']))
    arrays = {k:np.asarray([r[k] for r in samples]) for k in samples[0]}
    norms = np.linalg.norm(arrays['action'], axis=1)
    offsets = np.array([e['target_offset_m'] for e in episodes])
    directions = offsets/np.linalg.norm(offsets, axis=1, keepdims=True)
    raw = {k:arrays['raw_'+k] for k in FEATURE_KEYS}
    distribution = dict(samples=len(samples), episodes=len(episodes),
        successes=sum(e['success'] for e in episodes), action_norm=stats(norms),
        moving_fraction=float(np.mean(norms > .01)), near_zero_stop_fraction=float(np.mean(norms <= .01)),
        action_component_min=arrays['action'].min(axis=0).tolist(), action_component_max=arrays['action'].max(axis=0).tolist(),
        action_direction_octants=dict(Counter(''.join('+' if x >= 0 else '-' for x in a) for a in arrays['action'][norms > .01])),
        target_radius_m=stats(np.linalg.norm(offsets, axis=1)), target_min_m=offsets.min(axis=0).tolist(),
        target_max_m=offsets.max(axis=0).tolist(), target_direction_mean=directions.mean(axis=0).tolist(),
        target_direction_octants=dict(Counter(''.join('+' if x >= 0 else '-' for x in d) for d in directions)),
        expert_direction=action_metrics(arrays['action'], arrays['action'], raw))
    manifest = dict(schema=1, environment=config, observation_transform=transform.config,
                    feature_keys=list(FEATURE_KEYS), train_seeds=train_seeds, validation_seeds=validation_seeds,
                    heldout_final_seeds=list(FINAL_SEEDS), split='whole episodes; 0=train, 1=validation',
                    indices='episode_index and step_index are zero-based; state/action precede env.step',
                    success_fields='step_success is returned by env.step; episode_success is final outcome',
                    data_digest=dataset_digest(arrays), distribution=distribution, episodes=episodes,
                    wall_s=perf_counter()-started)
    return arrays, manifest


def save_demonstrations(folder, train_seeds, validation_seeds):
    folder = empty_directory(folder)
    arrays, manifest = collect_demonstrations(train_seeds, validation_seeds)
    np.savez_compressed(folder/'demonstrations.npz', **arrays)
    manifest['archive_sha256'] = hashlib.sha256((folder/'demonstrations.npz').read_bytes()).hexdigest()
    write_json(folder/'manifest.json', manifest)
    return manifest


def load_demonstrations(path):
    path = Path(path).resolve()
    manifest = json.loads((path.parent/'manifest.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('Demonstration archive hash mismatch')
    with np.load(path, allow_pickle=False) as file:
        data = dict(file)
    if dataset_digest(data) != manifest['data_digest']:
        raise ValueError('Demonstration content hash mismatch')
    raw = {k:data['raw_'+k] for k in FEATURE_KEYS}
    encoded = ReachTransform(manifest['observation_transform']).encode(raw)
    for key in FEATURE_KEYS:
        if not np.array_equal(encoded[key], data['processed_'+key]):
            raise ValueError('Saved preprocessing differs from original 04-A transform')
    if not np.array_equal(features(encoded), data['features']):
        raise ValueError('Saved Dict flatten order mismatch')
    if (set(manifest['train_seeds']) & set(manifest['validation_seeds'])
            or (set(manifest['train_seeds']) | set(manifest['validation_seeds'])) & set(FINAL_SEEDS)):
        raise ValueError('Leaked target split')
    for split, key in [(0,'train_seeds'), (1,'validation_seeds')]:
        if set(data['seed'][data['split'] == split]) != set(manifest[key]):
            raise ValueError('Sample and manifest split differ')
    if not np.isfinite(data['features']).all() or not np.isfinite(data['action']).all() or np.any(abs(data['action']) > 1):
        raise ValueError('Illegal supervised data')
    return data, manifest


def train_bc(data_path, folder, *, epochs=300, batch_size=256, learning_rate=.001, seed=2026):
    if (not isinstance(epochs,int) or epochs < 1 or not isinstance(batch_size,int) or batch_size < 1
            or seed < 0 or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError('Positive epochs/batch/lr and nonnegative seed required')
    data, manifest = load_demonstrations(data_path)
    folder = empty_directory(folder)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    network = BCNetwork()
    initial = weight_digest(network)
    config = dict(kind='behavior_cloning_not_SAC', environment=manifest['environment'],
                  observation_transform=manifest['observation_transform'], feature_keys=list(FEATURE_KEYS),
                  training=dict(epochs=epochs,batch_size=batch_size,learning_rate=learning_rate,seed=seed,
                                optimizer='Adam',betas=[.9,.999],eps=1e-8,weight_decay=0.,
                                loss='mean squared normalized action error',device='cpu',torch_threads=1),
                  network='33-256-ReLU-256-ReLU-3-tanh', parameter_count=sum(p.numel() for p in network.parameters()),
                  difference_from_SAC='same deterministic mean path; no Gaussian log_std head, critic, entropy or RL updates',
                  train_seeds=manifest['train_seeds'],validation_seeds=manifest['validation_seeds'],
                  heldout_final_seeds=list(FINAL_SEEDS),dataset_path=str(Path(data_path).resolve()),
                  dataset_digest=manifest['data_digest'],checkpoint_selection='minimum validation MSE',
                  augmentation='none', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(folder/'config.json', config)
    write_json(folder/'dependencies.json', versions())
    save_policy(folder/'untrained_bc.pt',network,config)
    x,y=torch.from_numpy(data['features']),torch.from_numpy(data['action'])
    train_ids=torch.from_numpy(np.flatnonzero(data['split'] == 0))
    validation_ids=torch.from_numpy(np.flatnonzero(data['split'] == 1))
    optimizer=torch.optim.Adam(network.parameters(),lr=learning_rate)
    generator=torch.Generator().manual_seed(seed)
    history=[];best_loss=float('inf');best_state=None;best_epoch=0
    started=perf_counter()
    for epoch in range(1,epochs+1):
        network.train()
        order=train_ids[torch.randperm(len(train_ids),generator=generator)]
        for batch in order.split(batch_size):
            loss=nn.functional.mse_loss(network(x[batch]),y[batch])
            if not torch.isfinite(loss):
                raise ValueError('Non-finite BC loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                raise ValueError('Non-finite BC gradient')
            optimizer.step()
        network.eval()
        with torch.no_grad():
            train_loss=nn.functional.mse_loss(network(x[train_ids]),y[train_ids]).item()
            val_loss=nn.functional.mse_loss(network(x[validation_ids]),y[validation_ids]).item()
        history.append(dict(epoch=epoch,train_mse=train_loss,validation_mse=val_loss))
        if val_loss < best_loss:
            best_loss,best_epoch=val_loss,epoch
            best_state=copy.deepcopy(network.state_dict())
    save_policy(folder/'last_epoch_bc.pt',network,config)
    network.load_state_dict(best_state)
    policy=BCPolicy(network,ReachTransform(config['observation_transform']))
    metrics={}
    for name,ids in [('train',data['split'] == 0),('validation',data['split'] == 1)]:
        raw={k:data['raw_'+k][ids] for k in FEATURE_KEYS}
        metrics[name]=action_metrics(policy.predict(raw),data['action'][ids],raw)
    save_policy(folder/'best_bc.pt',network,config)
    summary=dict(best_epoch=best_epoch,best_validation_mse=best_loss,metrics=metrics,
                 training_wall_s=perf_counter()-started,initial_weight_digest=initial,
                 trained_weight_digest=weight_digest(network),parameters_updated=initial != weight_digest(network),
                 parameters_finite=all(torch.isfinite(p).all().item() for p in network.parameters()),
                 optimizer_updates=epochs*((len(train_ids)+batch_size-1)//batch_size))
    write_json(folder/'history.json',history)
    write_json(folder/'summary.json',summary)
    return summary


def evaluate_policy(name, seeds, *, policy=None, environment=None):
    """BC execution never calls expert; only the rule comparison uses rule_action."""
    if name not in ('rule','untrained','bc') or not len(seeds):
        raise ValueError('Expected rule/untrained/bc and nonempty seeds')
    if name != 'rule' and policy is None:
        raise ValueError('Learned/untrained policy required')
    config=dict(ENV_CONFIG if environment is None else environment)
    before=weight_digest(policy.network) if policy is not None else None
    episodes,records,step_times=[],[],[]
    started=perf_counter()
    with ReachEnv(**config) as env:
        if policy is not None and policy.transform.config != ReachTransform.from_env(env).config:
            raise ValueError('Saved transform does not match environment')
        for seed in seeds:
            obs,_=env.reset(seed=int(seed))
            goal=obs['desired_goal'].copy();initial=obs['achieved_goal'].copy()
            rewards=[];previous=None;reversed_steps=0;physical_steps=0
            begin=len(records)
            for step in range(env.max_episode_steps):
                actual=obs['achieved_goal'].copy();command=obs['observation'][18:21].copy()
                # No expert correction, blending or fallback in either neural branch.
                action=rule_action(obs,env.action_scale) if name == 'rule' else policy.predict(obs)
                if not env.action_space.contains(action):
                    raise ValueError('Illegal policy action')
                raw={k:v.copy() for k,v in obs.items()}
                tick=perf_counter()
                obs,reward,terminated,truncated,info=env.step(action)
                step_times.append(perf_counter()-tick)
                if not env.observation_space.contains(obs) or not all(np.isfinite(v).all() for v in obs.values()):
                    raise ValueError('Invalid evaluation observation')
                if reward != env.compute_reward(obs['achieved_goal'],obs['desired_goal'],info):
                    raise ValueError('Inconsistent evaluation reward')
                reversal=previous is not None and float(previous@action) < -1e-9
                reversed_steps+=int(reversal);previous=action.copy()
                records.append(dict(seed=int(seed),step=step+1,raw_observation=raw['observation'].tolist(),
                    actual_before_m=actual.tolist(),command_before_m=command.tolist(),goal_m=goal.tolist(),
                    actual_after_m=obs['achieved_goal'].tolist(),command_after_m=info['command_position_m'].tolist(),
                    action=action.tolist(),reversed_from_previous=bool(reversal),
                    reward=reward,distance_m=info['distance_m'],reason=info['termination_reason'],
                    physics_steps=info['physics_steps'],simulation_time_s=info['simulation_time_s']))
                rewards.append(reward);physical_steps+=info['physics_steps']
                if terminated or truncated:
                    break
            rows=records[begin:]
            actual_error=goal-np.array([r['actual_before_m'] for r in rows])
            command_error=goal-np.array([r['command_before_m'] for r in rows])
            actions=np.array([r['action'] for r in rows])
            actual_cos,command_cos=cosine(actions,actual_error),cosine(actions,command_error)
            command_path=sum(np.linalg.norm(np.subtract(r['command_after_m'],r['command_before_m'])) for r in rows)
            actual_path=sum(np.linalg.norm(np.subtract(r['actual_after_m'],r['actual_before_m'])) for r in rows)
            episodes.append(dict(seed=int(seed),goal_m=goal.tolist(),initial_position_m=initial.tolist(),
                final_position_m=obs['achieved_goal'].tolist(),success=info['is_success'],
                safety_failure=info['is_safety_failure'],reason=info['termination_reason'],
                terminated=bool(terminated),truncated=bool(truncated),length=len(rows),
                return_=float(sum(rewards)),discounted_return=float(np.dot(rewards,.99**np.arange(len(rewards)))),
                final_distance_m=info['distance_m'],physical_steps=physical_steps,
                actual_toward=int(np.sum(actual_cos > 0)),actual_directions=len(actual_cos),
                command_toward=int(np.sum(command_cos > 0)),command_directions=len(command_cos),
                reversed_steps=reversed_steps,reversal_pairs=max(0,len(rows)-1),
                command_path_m=command_path,actual_path_m=actual_path))
    if policy is not None and weight_digest(policy.network) != before:
        raise ValueError('Evaluation changed network parameters')
    successes=[e for e in episodes if e['success']]
    def fraction(count,total):
        denominator=sum(e[total] for e in episodes)
        return sum(e[count] for e in episodes)/denominator if denominator else None
    command_path=sum(e['command_path_m'] for e in episodes)
    summary=dict(episodes=len(episodes),success_rate=len(successes)/len(episodes),
        mean_success_steps=float(np.mean([e['length'] for e in successes])) if successes else None,
        mean_episode_length=float(np.mean([e['length'] for e in episodes])),
        mean_final_distance_m=float(np.mean([e['final_distance_m'] for e in episodes])),
        mean_return=float(np.mean([e['return_'] for e in episodes])),
        safety_failures=sum(e['safety_failure'] for e in episodes),
        termination_counts=dict(Counter(e['reason'] for e in episodes)),
        actual_toward_fraction=fraction('actual_toward','actual_directions'),
        command_toward_fraction=fraction('command_toward','command_directions'),
        reversed_fraction=fraction('reversed_steps','reversal_pairs'),
        command_path_m=command_path,actual_path_m=sum(e['actual_path_m'] for e in episodes),
        actual_to_command_path_ratio=sum(e['actual_path_m'] for e in episodes)/command_path if command_path else None)
    wall=perf_counter()-started
    return dict(policy=name,environment=config,summary=summary,episodes=episodes,records=records,
                parameters_unchanged=True,timing=dict(wall_s=wall,mean_step_s=float(np.mean(step_times)),
                                                     interaction_steps_per_s=len(step_times)/wall))
