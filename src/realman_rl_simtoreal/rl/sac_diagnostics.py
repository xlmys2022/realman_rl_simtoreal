"""Read-only physical-action and SB3 Bellman diagnostics; no optimizer/update."""
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from ..envs.reach_env import ReachEnv
from .common.baselines import rule_action
from .common.observation import checkpoint_transform, require_legacy, validate_model_observation
from .common.artifacts import write_json


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.policy.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if model.log_ent_coef is not None:
        digest.update(model.log_ent_coef.detach().cpu().numpy().tobytes())
    digest.update(str((model.num_timesteps, model._n_updates)).encode())
    return digest.hexdigest()


def load_bundle(path, *, replay=False):
    path = Path(path).resolve()
    config = json.loads((path.parent/'config.json').read_text())
    require_legacy(config['environment'], 'SAC diagnostics')
    transform = checkpoint_transform(config)
    torch.set_num_threads(1)
    model = SAC.load(path, device='cpu')
    validate_model_observation(model, transform)
    if replay:
        with gzip.open(path.parent/'replay_buffer.pkl.gz', 'rb') as file:
            model.load_replay_buffer(io.BytesIO(file.read()))
    return model, transform, config


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite diagnostic')
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


def direction(delta, error):
    size, distance = np.linalg.norm(delta), np.linalg.norm(error)
    if size < 1e-12 or distance < 1e-12:
        return dict(projection_m=None, angle_deg=None, cosine=None)
    projection = float(delta @ (error/distance))
    cosine = float(np.clip(projection/size, -1, 1))
    return dict(projection_m=projection, angle_deg=float(np.rad2deg(np.arccos(cosine))), cosine=cosine)


def trace_policy(name, seeds, *, model=None, transform=None, config=None):
    """All positions are SI/world. Rate-capped ctrl FK uses independent data."""
    require_legacy(config or {}, 'SAC command/reference trajectory diagnostics')
    if name not in ('sac', 'rule', 'idle', 'outward'):
        raise ValueError('Unknown diagnostic policy')
    before = fingerprint(model) if model is not None else None
    episodes, rows = [], []
    with ReachEnv(**config) as env:
        for seed in seeds:
            obs, _ = env.reset(seed=int(seed))
            episode = []
            for step in range(env.max_episode_steps):
                position = obs['achieved_goal'].copy()
                goal = obs['desired_goal'].copy()
                command = obs['observation'][18:21].copy()
                ctrl = obs['observation'][21:27].copy()
                if name == 'sac':
                    action, _ = model.predict(transform.encode(obs), deterministic=True)
                elif name == 'rule':
                    action = rule_action(obs, env.action_scale)
                elif name == 'idle':
                    action = np.zeros(3, dtype=np.float32)
                else:
                    action = np.ones(3, dtype=np.float32)
                if not env.action_space.contains(action):
                    raise ValueError('Illegal diagnostic action')
                requested_delta = action.astype(float)*env.action_scale
                obs, reward, terminated, truncated, info = env.step(action)
                actual_delta = obs['achieved_goal']-position
                new_command = info['command_position_m']
                cycle = info['control']
                rate_error = (None if cycle is None else
                              (np.asarray(cycle['requested_joint_target_rad'])-
                               cycle['commanded_joint_target_rad']).tolist())
                ctrl_position = env.kinematics.fk(obs['observation'][21:27])[0]
                row = dict(seed=int(seed), step=step+1, goal_m=goal.tolist(), actual_before_m=position.tolist(),
                           actual_after_m=obs['achieved_goal'].tolist(), command_before_m=command.tolist(),
                           command_after_m=new_command.tolist(), action=action.tolist(),
                           actual_error_before_m=(goal-position).tolist(), command_error_before_m=(goal-command).tolist(),
                           actual_direction=direction(requested_delta, goal-position),
                           command_direction=direction(requested_delta, goal-command),
                           requested_delta_m=requested_delta.tolist(), command_delta_m=(new_command-command).tolist(),
                           actual_delta_m=actual_delta.tolist(),
                           distance_improvement_m=float(np.linalg.norm(goal-position)-info['distance_m']),
                           ctrl_before_rad=ctrl.tolist(), ctrl_after_rad=obs['observation'][21:27].tolist(),
                           requested_joint_target_rad=None if cycle is None else cycle['requested_joint_target_rad'],
                           joint_rate_limited=False if cycle is None else cycle['command_rate_limited'],
                           joint_rate_error_rad=rate_error,
                           command_vs_ctrl_fk_error_m=float(np.linalg.norm(new_command-ctrl_position)),
                           physics_steps=info['physics_steps'], time_s=info['simulation_time_s'], reward=float(reward),
                           reason=info['termination_reason'], terminated=bool(terminated), truncated=bool(truncated),
                           rejected=bool(info.get('is_action_rejected', False)), rejection_reason=info.get('rejection_reason'),
                           is_safety_failure=info['is_safety_failure'])
                episode.append(row)
                rows.append(row)
                if terminated or truncated:
                    break
            for index, row in enumerate(episode):
                row['reversed_by_next'] = False
                row['pair_cancelled_length_m'] = 0.
                if index+1 < len(episode):
                    a, b = np.asarray(row['requested_delta_m']), np.asarray(episode[index+1]['requested_delta_m'])
                    row['reversed_by_next'] = bool(a@b < -1e-15)
                    row['pair_cancelled_length_m'] = float(max(0, (np.linalg.norm(a)+np.linalg.norm(b)-np.linalg.norm(a+b))/2))
            episodes.append(dict(seed=int(seed), goal_m=goal.tolist(), success=info['is_success'],
                                 reason=info['termination_reason'], length=len(episode),
                                 final_distance_m=info['distance_m'], return_=sum(r['reward'] for r in episode),
                                 discounted_return=float(sum(.99**i*r['reward'] for i,r in enumerate(episode))),
                                 rejected_actions=sum(r['rejected'] for r in episode),
                                 safety_failure=info['is_safety_failure']))
    if model is not None and fingerprint(model) != before:
        raise ValueError('Diagnostic changed model parameters/counters')
    return dict(policy=name, episodes=episodes, records=rows, summary=trace_summary(rows, episodes),
                weights_unchanged=True, projection_note='Action displacement projected on separate pre-step actual/command errors')


def trace_summary(rows, episodes):
    def facing(field):
        cos = np.array([r[field]['cosine'] for r in rows if r[field]['cosine'] is not None])
        return dict(valid_directions=len(cos), toward=int(np.sum(cos > 1e-6)), away=int(np.sum(cos < -1e-6)),
                    perpendicular=int(np.sum(abs(cos) <= 1e-6)), toward_fraction=float(np.mean(cos > 1e-6)) if len(cos) else None)
    command_path = sum(np.linalg.norm(r['command_delta_m']) for r in rows)
    actual_path = sum(np.linalg.norm(r['actual_delta_m']) for r in rows)
    return dict(episodes=len(episodes), steps=len(rows), success_rate=np.mean([e['success'] for e in episodes]).item(),
                mean_return=float(np.mean([e['return_'] for e in episodes])),
                mean_discounted_return=float(np.mean([e['discounted_return'] for e in episodes])),
                mean_episode_length=float(np.mean([e['length'] for e in episodes])),
                termination_counts=dict(Counter(e['reason'] for e in episodes)),
                mean_final_distance_m=float(np.mean([e['final_distance_m'] for e in episodes])),
                actual_facing=facing('actual_direction'), command_facing=facing('command_direction'),
                distance_improving_steps=sum(r['distance_improvement_m'] > 1e-8 for r in rows),
                distance_improving_fraction=float(np.mean([r['distance_improvement_m'] > 1e-8 for r in rows])),
                reversed_by_next_steps=sum(r['reversed_by_next'] for r in rows),
                reversed_by_next_fraction=float(np.mean([r['reversed_by_next'] for r in rows])),
                pair_cancelled_length_m=sum(r['pair_cancelled_length_m'] for r in rows),
                command_path_m=command_path, actual_path_m=actual_path,
                actual_to_command_path_ratio=actual_path/command_path if command_path else None,
                rate_limited_steps=sum(r['joint_rate_limited'] for r in rows),
                rate_limited_fraction=float(np.mean([r['joint_rate_limited'] for r in rows])),
                command_vs_ctrl_fk_error_m=distribution([r['command_vs_ctrl_fk_error_m'] for r in rows]),
                rejected_actions=sum(r['rejected'] for r in rows),
                rejected_by_reason={k:sum(r['rejection_reason']==k for r in rows) for k in ('workspace_limit','joint_command_limit')},
                safety_failures=sum(e['safety_failure'] for e in episodes), timeouts=sum(e['reason']=='time_limit' for e in episodes))


def td_diagnostic(model, *, seed=2026, batch_size=512, indices=None, rejected_indices=(),terminal_labels=None):
    """Same one-step soft target as installed SB3 SAC.train, never an update.

    Replay actions for current Q; one stochastic next action for target Q.
    Each row retains stored done, timeout and the true termination mask.
    These frozen, full-buffer statistics are NOT historical minibatch losses.
    """
    before = fingerprint(model)
    rb = model.replay_buffer
    if rb.n_envs != 1 or not rb.handle_timeout_termination:
        raise ValueError('This diagnostic requires single-env timeout-aware replay')
    inds = np.arange(rb.size()) if indices is None else np.asarray(indices, dtype=int)
    if not len(inds) or np.any(inds < 0) or np.any(inds >= rb.size()):
        raise ValueError('Invalid replay indices')
    arrays = {k:[] for k in ('q1','q2','target_q1_at_stored_action','target_q2_at_stored_action',
                             'next_target_q_min','next_log_prob','entropy_bonus','td_target','td_target_wrong_done',
                             'residual_q1','residual_q2')}
    alpha = float(model.log_ent_coef.detach().exp()) if model.log_ent_coef is not None else float(model.ent_coef_tensor)
    # fork_rng restores the caller's RNG. Direct indexed tensors consume no
    # NumPy RNG; single-env buffer stored data are already network-preprocessed.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(seed)
        for offset in range(0, len(inds), batch_size):
            index = inds[offset:offset+batch_size]
            obs = {k:torch.as_tensor(v[index,0],device=model.device) for k,v in rb.observations.items()}
            nxt = {k:torch.as_tensor(v[index,0],device=model.device) for k,v in rb.next_observations.items()}
            actions = torch.as_tensor(rb.actions[index,0],device=model.device)
            rewards = torch.as_tensor(rb.rewards[index,0,None],device=model.device)
            stored = torch.as_tensor(rb.dones[index,0,None],device=model.device)
            done = stored*(1-torch.as_tensor(rb.timeouts[index,0,None],device=model.device))
            next_actions, logp = model.actor.action_log_prob(nxt)
            next_q = torch.cat(model.critic_target(nxt,next_actions),dim=1).min(dim=1,keepdim=True).values
            entropy = -alpha*logp.reshape(-1,1)
            target = rewards+model.gamma*(1-done)*(next_q+entropy)
            wrong = rewards+model.gamma*(1-stored)*(next_q+entropy)
            q1,q2=model.critic(obs,actions)
            t1,t2=model.critic_target(obs,actions)
            for key,value in dict(q1=q1,q2=q2,target_q1_at_stored_action=t1,target_q2_at_stored_action=t2,
                                  next_target_q_min=next_q,next_log_prob=logp,entropy_bonus=entropy,
                                  td_target=target,td_target_wrong_done=wrong,
                                  residual_q1=target-q1,residual_q2=target-q2).items():
                arrays[key].append(value.cpu().numpy().ravel())
    arrays={k:np.concatenate(v) for k,v in arrays.items()}
    arrays.update(index=inds,reward=rb.rewards[inds,0].copy(),stored_done=rb.dones[inds,0].copy(),
                  timeout=rb.timeouts[inds,0].copy())
    arrays['effective_done']=arrays['stored_done']*(1-arrays['timeout'])
    arrays['is_action_rejected']=np.isin(inds,np.asarray(rejected_indices,dtype=int))
    # Labels only infer safe terminal from the known -201 baseline reward;
    # nonterminal rejected actions in B are reported separately below.
    labels=np.full(len(inds),'normal',dtype='<U20')
    labels[arrays['is_action_rejected']]='rejected'
    labels[(arrays['stored_done']==1)&(arrays['timeout']==1)]='timeout'
    labels[(arrays['effective_done']==1)&(arrays['reward'] < -200)]='safety_terminal'
    labels[(arrays['effective_done']==1)&(arrays['reward'] >= -200)]='success_terminal'
    if terminal_labels is not None:
        for row,index in enumerate(inds):
            if arrays['stored_done'][row]:
                category=terminal_labels.get(int(index))
                if category not in ('timeout','safety_terminal','success_terminal'):
                    raise ValueError('Explicit terminal label missing for stored transition')
                labels[row]=category
    arrays['category']=labels
    summaries={}
    for category in ('all','normal','rejected','timeout','safety_terminal','success_terminal'):
        selected=np.ones(len(inds),bool) if category=='all' else labels==category
        residual=np.concatenate([arrays['residual_q1'][selected],arrays['residual_q2'][selected]])
        summaries[category]=dict(count=int(selected.sum()),
           reward=distribution(arrays['reward'][selected]), q=distribution(np.r_[arrays['q1'][selected],arrays['q2'][selected]]),
           td_target=distribution(arrays['td_target'][selected]), residual=distribution(residual),
           residual_rmse=float(np.sqrt(np.mean(residual**2))) if len(residual) else None,
           critic_mse_loss=float(np.mean(residual**2)) if len(residual) else None,
           entropy_bonus=distribution(arrays['entropy_bonus'][selected]),
           timeout_bootstrap_difference=distribution((arrays['td_target']-arrays['td_target_wrong_done'])[selected]))
    mismatch=np.max(np.abs(rb.observations['desired_goal'][inds,0]-rb.next_observations['desired_goal'][inds,0]),axis=1)>1e-12
    summary=dict(seed=seed, transitions=len(inds),alpha=alpha,gamma=model.gamma,
                 stored_dones=int(arrays['stored_done'].sum()),timeouts=int(arrays['timeout'].sum()),
                 effective_terminations=int(arrays['effective_done'].sum()),
                 rejected_transitions=int(arrays['is_action_rejected'].sum()),
                 next_goal_mismatches=int(mismatch.sum()),
                 terminal_next_goal_mismatches=int(np.sum(mismatch & (arrays['stored_done']==1))),
                 categories=summaries, weights_unchanged=fingerprint(model)==before,
                 note='One stochastic next-action draw per stored transition; correct timeout bootstrap. '
                      'Current Q uses replay actions. Ordinary no-entropy MC returns are not TD labels.')
    if not summary['weights_unchanged']:
        raise ValueError('TD diagnostic changed model')
    return summary,arrays


def q_diagnostic(model):
    """Inspect ordinary SB3 networks/buffer; preserve the training sampling RNG."""
    if model.replay_buffer.size() == 0:
        return dict(buffer_size=0)
    state = np.random.get_state()
    try:
        batch = model.replay_buffer.sample(min(128, model.replay_buffer.size()))
    finally:
        np.random.set_state(state)
    with torch.no_grad():
        actions = model.actor(batch.observations, deterministic=True)
        values = torch.cat(model.critic(batch.observations, actions), dim=1).cpu().numpy()
        _, log_std, _ = model.actor.get_action_dist_params(batch.observations)
        std = torch.exp(log_std).cpu().numpy()
        finite = all(torch.isfinite(p).all().item() for p in model.policy.parameters())
        if model.log_ent_coef is not None:
            finite = finite and torch.isfinite(model.log_ent_coef).all().item()
    result = dict(buffer_size=int(model.replay_buffer.size()), q_min=float(values.min()), q_max=float(values.max()),
                  q_abs_max=float(np.abs(values).max()), q_mean=float(values.mean()), parameters_finite=bool(finite),
                  action_abs_mean=float(actions.abs().mean()), policy_std_mean=float(std.mean()))
    if not finite or not np.isfinite(values).all() or not np.isfinite(std).all():
        raise ValueError('Non-finite neural parameters/Q/std')
    if result['q_abs_max'] > 1e6:
        raise ValueError('Q magnitude exceeded the predeclared 1e6 divergence guard')
    return result
