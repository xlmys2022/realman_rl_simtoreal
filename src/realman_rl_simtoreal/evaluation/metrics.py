"""Pure result aggregation; preserve each historical JSON statistics contract."""
from collections import Counter
import numpy as np


REASONS = ('success', 'workspace_limit', 'joint_limit', 'ik_failure', 'orientation_limit',
           'collision', 'numerical_failure', 'time_limit', 'unreachable_target',
           'velocity_limit', 'drive_failure')


def summarize_episodes(episodes):
    counts = Counter(e['reason'] for e in episodes)
    successful = [e for e in episodes if e['success']]
    result = dict(episodes=len(episodes), success_rate=len(successful)/len(episodes) if episodes else None,
                mean_return=float(np.mean([e['return_'] for e in episodes])) if episodes else None,
                mean_discounted_return=float(np.mean([e['discounted_return'] for e in episodes])) if episodes else None,
                mean_episode_length=float(np.mean([e['length'] for e in episodes])) if episodes else None,
                mean_success_steps=float(np.mean([e['length'] for e in successful])) if successful else None,
                mean_final_distance_m=float(np.mean([e['final_distance_m'] for e in episodes])) if episodes else None,
                safety_failures=sum(e['safety_failure'] for e in episodes),
                rejected_actions=sum(e.get('rejected_actions',0) for e in episodes),
                termination_counts={k: counts[k] for k in dict.fromkeys((*REASONS, *counts))})
    if any('goals_planned' in e for e in episodes):
        planned=sum(e.get('goals_planned',1) for e in episodes)
        completed=sum(e.get('goals_completed',int(e['success'])) for e in episodes)
        attempted=sum(e.get('goals_attempted',1) for e in episodes)
        result.update(sequence_success_rate=result['success_rate'],goals_planned=planned,
                      goals_completed=completed,goals_attempted=attempted,
                      goal_success_rate=completed/attempted if attempted else None,
                      goal_completion_rate=completed/planned if planned else None)
    return result


def summarize_grasp_episodes(episodes):
    successful=[e for e in episodes if e['success']]
    return dict(episodes=len(episodes),successes=len(successful),success_rate=len(successful)/len(episodes) if episodes else None,
        termination_counts=dict(Counter(e['reason'] for e in episodes)),
        safety_failures=sum(e['is_safety_failure'] for e in episodes),
        mean_success_clearance_m=float(np.mean([e['final_support_clearance_m'] for e in successful])) if successful else None,
        mean_success_world_z_lift_m=float(np.mean([e['world_z_lift_m'] for e in successful])) if successful else None,
        mean_success_steps=float(np.mean([e['length'] for e in successful])) if successful else None)


def latency_summary(samples):
    values = np.asarray(samples, dtype=float)
    if not len(values):
        return dict(count=0, mean_s=None, median_s=None, p95_s=None, total_s=0.0)
    return dict(count=len(values), mean_s=float(np.mean(values)), median_s=float(np.median(values)),
                p95_s=float(np.percentile(values, 95)), total_s=float(np.sum(values)))


def goal_distribution(episodes):
    deltas = np.array([e['target_delta_m'] for e in episodes])
    octants = Counter(''.join('+' if x >= 0 else '-' for x in row) for row in deltas)
    return dict(min_xyz_m=deltas.min(axis=0).tolist(), max_xyz_m=deltas.max(axis=0).tolist(),
                mean_xyz_m=deltas.mean(axis=0).tolist(), octant_counts=dict(octants),
                min_radius_m=float(np.min(np.linalg.norm(deltas, axis=1))),
                max_radius_m=float(np.max(np.linalg.norm(deltas, axis=1))),
                total_sampling_attempts=sum(e['goal_sampling']['attempts'] for e in episodes),
                sampling_rejections=dict(sum((Counter(e['goal_sampling']['rejections']) for e in episodes), Counter())))


def summarize_baseline_episodes(episodes):
    successful = [e for e in episodes if e['success']]
    counts = Counter(e['reason'] for e in episodes)
    summary = dict(episodes=len(episodes), success_rate=len(successful)/len(episodes),
                   mean_return=float(np.mean([e['return_'] for e in episodes])),
                   mean_episode_length=float(np.mean([e['length'] for e in episodes])),
                   mean_success_steps=float(np.mean([e['length'] for e in successful])) if successful else None,
                   mean_final_distance_m=float(np.mean([e['final_distance_m'] for e in episodes])),
                   termination_counts=dict(counts), ik_failures=counts['ik_failure'],
                   command_boundary_failures=counts['workspace_limit'], joint_limit_failures=counts['joint_limit'],
                   collisions=counts['collision'], numerical_failures=counts['numerical_failure'],
                   timeouts=counts['time_limit'])
    return summary


def summarize_bc_episodes(episodes):
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
    return summary
