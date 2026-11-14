#!/usr/bin/env python3
"""Frozen PPO goal sensitivity using only validated environment reset observations."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from realman_rl_simtoreal.envs.reach_env import ReachEnv
from realman_rl_simtoreal.tasks.reach import task_environment
from realman_rl_simtoreal.rl.ppo import load_ppo,parameter_hashes,optimizer_steps
from realman_rl_simtoreal.rl.common.observation import ReachTransform,features
from realman_rl_simtoreal.rl.common.artifacts import write_json


def probe(model_path,*,task='axis_pair_reach',distance_mm=5.):
    if task not in ('axis_pair_reach','axis_six_reach'):raise ValueError('Expected an axis curriculum task')
    if not np.isfinite(distance_mm) or distance_mm<=0:raise ValueError('Distance must be finite and positive')
    model,transform,saved=load_ppo(model_path)
    # Inside the ordinary observation these slices depend on the goal.
    # Both residuals (goal-ee and goal-command) agree at the home reset.
    goal_slice = {'legacy': slice(15, 18), 'fetch_like_v1': slice(0, 0),
                  'goal_relative_v1': slice(0, 3)}[transform.observation_mode]
    environment=task_environment(task,saved['environment'])
    # This explicit diagnostic distance never changes a training task/default.
    environment['target_offsets']=(np.array(environment['target_offsets'])*distance_mm/5).tolist()
    environment['target_radius']=distance_mm/1000
    before=parameter_hashes(model);counts=(model.num_timesteps,model._n_updates,optimizer_steps(model))
    rows=[];inputs=[];physical=[];other=[]
    with ReachEnv(**environment) as env:
        if transform.config!=ReachTransform.from_env(env).config:raise ValueError('Saved preprocessing is incompatible')
        for i in range(len(env.goal_candidates)):
            obs,info=env.reset(seed=87000,options={'goal_index':i})
            encoded=transform.encode(obs);inputs.append(features(encoded))
            physical.append(np.r_[env.data.qpos,env.data.qvel,env.data.ctrl,env.data.time])
            ordinary = obs['observation']
            non_goal = np.r_[ordinary[:goal_slice.start], ordinary[goal_slice.stop:]]
            other.append(np.r_[obs['achieved_goal'],non_goal])
            action,_=model.predict(encoded,deterministic=True)
            delta=obs['desired_goal']-obs['achieved_goal']
            denominator=np.linalg.norm(action)*np.linalg.norm(delta)
            cosine=float(action@delta/denominator) if denominator>1e-12 else None
            rows.append(dict(goal_index=i,goal_position_m=obs['desired_goal'].tolist(),
                offset_m=delta.tolist(),processed_relative_goal=encoded['observation'][goal_slice].tolist() or None,
                processed_features=inputs[-1].tolist(),action=action.tolist(),
                action_goal_cosine=cosine,toward_goal=None if cosine is None else cosine>0,
                goal_validation=env.candidate_validation[i]))
        for state in physical[1:]:np.testing.assert_array_equal(state,physical[0])
        for state in other[1:]:np.testing.assert_array_equal(state,other[0])
    if parameter_hashes(model)!=before or counts!=(model.num_timesteps,model._n_updates,optimizer_steps(model)):
        raise ValueError('Frozen sensitivity probe changed model parameters/counters')
    x=np.stack(inputs);actions=np.asarray([r['action'] for r in rows])
    changed=np.flatnonzero(np.any(x!=x[0],axis=0)).tolist()
    goal_features = dict(desired_goal=[3, 4, 5])
    if goal_slice.stop > goal_slice.start:
        goal_features['relative_goal'] = list(range(6 + goal_slice.start, 6 + goal_slice.stop))
    allowed = {i for indices in goal_features.values() for i in indices}
    if not set(changed)<=allowed:raise ValueError('Non-goal feature changed during same-state probe')
    responsive=not np.allclose(actions,actions[0],atol=1e-7,rtol=0)
    return dict(model=str(Path(model_path).resolve()),model_file_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        parameters_unchanged=True,model_parameter_hashes=before,task=task,diagnostic_distance_mm=distance_mm,
        environment=environment,physical_state_identical=True,changed_feature_indices=changed,
        observation_mode=transform.observation_mode,features_dim=transform.features_dim,
        expected_goal_features=goal_features,
        action_component_span=np.ptp(actions,axis=0).tolist(),action_signs=np.sign(actions).astype(int).tolist(),
        goal_responsive=responsive,initial_toward_count=sum(r['toward_goal'] is True for r in rows),
        classification='goal_insensitive_at_1e-7_action_tolerance' if not responsive else
                       'goal_sensitive_but_some_initial_directions_incorrect' if not all(r['toward_goal'] for r in rows) else
                       'goal_sensitive_initial_directions_toward_goal',
        note='Only validated reset + deterministic prediction; no physical step or closed-loop success claim.',rows=rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--task',choices=['axis_pair_reach','axis_six_reach'],default='axis_pair_reach')
    p.add_argument('--distance-mm',type=float,default=5.);p.add_argument('--json',type=Path,required=True)
    a=p.parse_args()
    try:
        if a.json.exists():raise ValueError('Output exists; choose a new --json')
        result=probe(a.model,task=a.task,distance_mm=a.distance_mm);write_json(a.json,result)
        print('index | world goal (m) | processed goal-ee | deterministic action | cosine')
        for r in result['rows']:
            print(r['goal_index'],r['goal_position_m'],r['processed_relative_goal'],r['action'],r['action_goal_cosine'])
        print(result['classification']);print('Saved',a.json);return 0
    except (ValueError,RuntimeError,OSError,ImportError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
