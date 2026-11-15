"""Write measured evaluation scalars to a training logger; never execute episodes."""
from ...evaluation.metrics import REASONS


def record_evaluation(logger, evaluation):
    for key,total in evaluation.get('reward_component_totals',{}).items():
        logger.record('evaluation/reward_components/'+key+'_mean_per_step',
                      total/sum(e['length'] for e in evaluation['episodes']))
    summary=evaluation['summary']
    for key in ('sequence_success_rate','goal_success_rate','goal_completion_rate','goals_completed','goals_planned'):
        if summary.get(key) is not None:
            logger.record('evaluation/'+key,summary[key])
    for tag,key in [('success_rate','success_rate'),('episode_length_mean','mean_episode_length'),
                    ('episode_reward_mean','mean_return'),('success_steps_mean','mean_success_steps'),
                    ('safety_failures','safety_failures')]:
        if summary[key] is not None:
            logger.record('evaluation/'+tag,summary[key])
    logger.record('evaluation/final_distance_mm',1000*summary['mean_final_distance_m'])
    logger.record('evaluation/timeouts',summary['termination_counts'].get('time_limit',0))
    logger.record('evaluation/action_rejection_fraction',summary['rejected_actions']/sum(e['length'] for e in evaluation['episodes']))
    if evaluation.get('bc_action_deviation') is not None:
        for key in ('mse','mae','max_abs'):
            logger.record('bc/evaluation_states_'+key,evaluation['bc_action_deviation'][key])
    for key,value in evaluation.get('motion',{}).items():
        if value is not None:logger.record('evaluation/motion/'+key,value)
    for label,row in evaluation.get('per_goal',{}).items():
        tag=label.replace('+','plus_').replace('-','minus_').lower()
        for key in ('success','episode_length','action_direction_fraction'):
            if row[key] is not None:logger.record(f'evaluation/goals/{tag}/{key}',float(row[key]))
        logger.record(f'evaluation/goals/{tag}/distance_mm',row['final_distance_m']*1000)
        if row['actual_path_m'] is not None:logger.record(f'evaluation/goals/{tag}/actual_path_mm',row['actual_path_m']*1000)
        for reason in REASONS:logger.record(f'evaluation/goals/{tag}/termination/'+reason,int(row['termination_reason']==reason))
        for axis,value in zip('xyz',row['first_action']):logger.record(f'evaluation/goals/{tag}/first_action_{axis}',value)
