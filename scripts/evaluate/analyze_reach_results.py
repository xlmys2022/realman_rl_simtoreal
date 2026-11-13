#!/usr/bin/env python3
"""Read-only direction/radius and action-clipping audit of Reach evaluation JSON."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

import numpy as np


def vector(value, name):
    result=np.asarray(value,dtype=float)
    if result.shape!=(3,) or not np.isfinite(result).all():raise ValueError('Expected finite XYZ '+name)
    return result


def action_metrics(episodes):
    """Pool steps, not episode means. Never infer raw actions absent from old JSON."""
    applied=[];raw=[];positions=[];goals=[];first=[];first_errors=[]
    for e in episodes:
        a=np.asarray(e['actions'],dtype=float)
        if a.shape!=(e['length'],3) or not np.isfinite(a).all() or np.any(abs(a)>1):
            raise ValueError('Invalid applied actions/length')
        applied.extend(a);first.append(a[0]);first_errors.append(e['_offset'])
        if 'raw_actions' in e:
            u=np.asarray(e['raw_actions'],dtype=float)
            if u.shape!=a.shape or not np.isfinite(u).all() or not np.array_equal(np.clip(u,-1,1),a):
                raise ValueError('Raw actions do not reproduce applied Box clipping')
            raw.extend(u)
        if 'actual_positions_before_m' in e:
            p=np.asarray(e['actual_positions_before_m'],dtype=float)
            if p.shape!=a.shape or not np.isfinite(p).all():raise ValueError('Invalid actual position trajectory')
            positions.extend(p);goals.extend([e['target_position_m']]*len(p))
    a=np.asarray(applied).reshape(-1,3);f=np.asarray(first).reshape(-1,3)
    result=dict(steps=len(a),mean_applied_action=a.mean(axis=0).tolist() if len(a) else None,
        mean_first_action=f.mean(axis=0).tolist() if len(f) else None,
        first_action_plus_x_fraction=float(np.mean(f[:,0]>1e-8)) if len(f) else None,
        applied_positive_axis_fraction=(a>1e-8).mean(axis=0).tolist() if len(a) else None,
        applied_negative_axis_fraction=(a< -1e-8).mean(axis=0).tolist() if len(a) else None,
        applied_action_std=a.std(axis=0).tolist() if len(a) else None,
        near_zero_action_fraction=float(np.mean(np.linalg.norm(a,axis=1)<.01)) if len(a) else None)
    first_error=np.asarray(first_errors).reshape(-1,3)
    valid=(np.linalg.norm(f,axis=1)>1e-8)&(np.linalg.norm(first_error,axis=1)>1e-12)
    result['first_action_toward_initial_goal_fraction']=float(np.mean(np.sum(f[valid]*first_error[valid],axis=1)>0)) if valid.any() else None
    # Mixed old/new episodes are reported as unavailable, not silently pooled
    # against a denominator containing steps without their original records.
    result.update(raw_actions_complete=len(raw)==len(a) and len(a)>0,
                  actual_trajectory_complete=len(positions)==len(a) and len(a)>0)
    u=np.asarray(raw).reshape(-1,3)
    if result['raw_actions_complete']:
        clipped=np.abs(u)>1;un=np.linalg.norm(u,axis=1);an=np.linalg.norm(a,axis=1)
        valid=(un>1e-8)&(an>1e-8)
        angle=np.rad2deg(np.arccos(np.clip(np.sum(u[valid]*a[valid],axis=1)/(un[valid]*an[valid]),-1,1)))
        result.update(mean_raw_action=u.mean(axis=0).tolist(),box_clip_vector_fraction=float(clipped.any(axis=1).mean()),
            box_clip_component_fraction=float(clipped.mean()),raw_abs_max=float(abs(u).max()),
            clipping_direction_angle_samples=int(valid.sum()),
            clipping_direction_mean_deg=float(angle.mean()) if len(angle) else None,
            clipping_direction_p95_deg=float(np.percentile(angle,95)) if len(angle) else None,
            clipping_direction_max_deg=float(angle.max()) if len(angle) else None)
    if result['actual_trajectory_complete']:
        error=np.asarray(goals)-np.asarray(positions);valid=(np.linalg.norm(a,axis=1)>1e-8)&(np.linalg.norm(error,axis=1)>1e-12)
        dot=np.sum(a*error,axis=1)
        result.update(action_direction_samples=int(valid.sum()),
            action_toward_actual_goal_fraction=float(np.mean(dot[valid]>0)) if valid.any() else None)
        if result['raw_actions_complete']:
            valid=valid&(np.linalg.norm(u,axis=1)>1e-8)
            raw_dot=np.sum(u*error,axis=1)
            result['raw_action_toward_actual_goal_fraction']=float(np.mean(raw_dot[valid]>0)) if valid.any() else None
            result['toward_to_away_after_clipping_fraction']=float(np.mean((raw_dot[valid]>0)&(dot[valid]<=0))) if valid.any() else None
    return result


def group_summary(episodes,min_samples):
    n=len(episodes);success=[e for e in episodes if e['success']]
    return dict(samples=n,successes=len(success),success_rate=len(success)/n if n else None,
        small_sample=n<min_samples,mean_final_distance_mm=float(np.mean([e['final_distance_m'] for e in episodes])*1000) if n else None,
        mean_episode_length=float(np.mean([e['length'] for e in episodes])) if n else None,
        mean_success_steps=float(np.mean([e['length'] for e in success])) if success else None,
        termination_counts=dict(Counter(e['reason'] for e in episodes)),actions=action_metrics(episodes))


def analyze(document,*,min_samples=10):
    if min_samples<1:raise ValueError('min_samples must be positive')
    episodes=document.get('episodes')
    if not isinstance(episodes,list) or not episodes:raise ValueError('Expected nonempty per-episode evaluation JSON')
    rows=[];seeds=[]
    for item in episodes:
        e=dict(item)
        target=vector(e['target_position_m'],'target');initial=vector(e['initial_position_m'],'initial')
        final=vector(e['final_position_m'],'final');offset=target-initial;radius=float(np.linalg.norm(offset))
        if type(e['success']) is not bool or type(e['length']) is not int or e['length']<1:
            raise ValueError('Invalid success/length')
        if not np.isfinite(e['final_distance_m']) or not np.isclose(np.linalg.norm(target-final),e['final_distance_m'],rtol=0,atol=1e-9):
            raise ValueError('Final distance disagrees with physical positions')
        e['_offset']=offset.tolist();e['_radius_m']=radius
        e['_octant']=''.join('+' if x>0 else '-' for x in offset) if np.all(offset!=0) else 'on_coordinate_plane'
        e['_radius_bin']='3-4mm' if .003-1e-12<=radius<.004 else '4-5mm' if .004<=radius<=.005+1e-12 else 'outside_3-5mm'
        seeds.append(e['seed']);rows.append(e)
    if len(set(seeds))!=len(seeds) and not all('goal_index' in e for e in rows):
        raise ValueError('Repeated evaluation seeds; cannot count them as independent targets')
    directions={sign+axis:group_summary([e for e in rows if (e['_offset'][i]>0 if sign=='+' else e['_offset'][i]<0)],min_samples)
                for i,axis in enumerate('XYZ') for sign in ('+','-')}
    octants={''.join(signs):group_summary([e for e in rows if e['_octant']==''.join(signs)],min_samples)
             for signs in product('+-',repeat=3)}
    octants['on_coordinate_plane']=group_summary([e for e in rows if e['_octant']=='on_coordinate_plane'],min_samples)
    radii={key:group_summary([e for e in rows if e['_radius_bin']==key],min_samples)
           for key in ('3-4mm','4-5mm','outside_3-5mm')}
    assert sum(x['samples'] for x in octants.values())==len(rows)
    offsets=np.asarray([e['_offset'] for e in rows]);distances=np.asarray([e['_radius_m'] for e in rows])
    result=dict(schema=1,policy=document.get('policy'),deterministic=document.get('deterministic'),
        goal_scope=document.get('goal_scope'),seeds=seeds,min_group_samples=min_samples,
        reference='World XYZ offset from each episode initial ee position (reset home in ReachEnv).',
        notes=['Axis sign groups overlap; octants and radius bins partition episodes.',
               'small_sample is a descriptive count flag, not a confidence interval.',
               'Fixed-goal repetitions do not measure random-target generalization.',
               'Missing raw actions/actual trajectories in historical JSON remain unavailable; no reconstruction.',
               'All-action statistics pool steps; first-action statistics give one sample per episode.',
               'Box clipping angles compare raw/applied direction; PPO train/clip_fraction is a different metric.'],
        target_distribution=dict(unique_positions=len(set(tuple(e['target_position_m']) for e in rows)),
            offset_min_mm=(offsets.min(axis=0)*1000).tolist(),offset_max_mm=(offsets.max(axis=0)*1000).tolist(),
            distance_min_mm=float(distances.min()*1000),distance_max_mm=float(distances.max()*1000),
            distance_mean_mm=float(distances.mean()*1000)),
        overall=group_summary(rows,min_samples),directions=directions,octants=octants,radius_bins=radii,
        targets=[dict(seed=e['seed'],offset_mm=(np.array(e['_offset'])*1000).tolist(),radius_mm=e['_radius_m']*1000,
                      octant=e['_octant'],radius_bin=e['_radius_bin'],success=e['success'],reason=e['reason'],
                      final_distance_mm=e['final_distance_m']*1000,steps=e['length']) for e in rows])
    if any('goal_index' in e for e in rows):
        indices=[e['goal_index'] for e in rows]
        if len(set(indices))!=len(indices):raise ValueError('Repeated discrete candidates are not independent target evaluations')
        result['discrete_goals']={e['goal_label']:dict(group_summary([e],min_samples),
            goal_position_m=e['target_position_m'],first_action=e['actions'][0],motion=e.get('motion'),
            termination_reason=e['reason']) for e in rows}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--json',type=Path,required=True)
    p.add_argument('--min-samples',type=int,default=10);a=p.parse_args()
    try:
        if a.json.exists():raise ValueError('Analysis output exists; choose a new --json path')
        result=analyze(json.loads(a.evaluation.read_text()),min_samples=a.min_samples)
        result['source']=str(a.evaluation.resolve());a.json.parent.mkdir(parents=True,exist_ok=True)
        a.json.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        for kind in ('directions','octants','radius_bins'):
            for label,row in result[kind].items():
                print(kind,label,'n=',row['samples'],'success=',row['success_rate'],
                      'distance_mm=',row['mean_final_distance_mm'],'small_sample=',row['small_sample'])
        for label,row in result.get('discrete_goals',{}).items():
            print('distinct goal',label,'success=',row['successes'],'distance_mm=',row['mean_final_distance_mm'],
                  'steps=',row['mean_episode_length'],'first_action=',row['first_action'],
                  'reason=',row['termination_reason'],'motion=',row['motion'])
        print('Saved',a.json);return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
