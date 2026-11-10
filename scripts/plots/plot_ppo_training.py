#!/usr/bin/env python3
"""Export measured PPO evaluation/update metrics, never inferred success curves."""
import argparse
import json
import math
import os
from pathlib import Path


def load_curves(folder):
    folder=Path(folder)
    evaluations=json.loads((folder/'evaluations.json').read_text()) if (folder/'evaluations.json').exists() else []
    metrics=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines() if s] if (folder/'metrics.jsonl').exists() else []
    def series(rows,key):
        values={int(s):float(r[key]) for s,r in rows if r.get(key) is not None}
        if not all(math.isfinite(v) for v in values.values()):raise ValueError('Non-finite stored metric '+key)
        return [[s,v] for s,v in sorted(values.items())]
    result={key:series([(r['step'],r) for r in metrics],key) for key in
        ('policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction','explained_variance','policy_std_mean')}
    result.update(evaluation_success_rate=series([(r['step'],r['summary']) for r in evaluations],'success_rate'),
        evaluation_distance_mm=[[s,1000*v] for s,v in series([(r['step'],r['summary']) for r in evaluations],'mean_final_distance_m')],
        evaluation_episode_length=series([(r['step'],r['summary']) for r in evaluations],'mean_episode_length'))
    for key in ('vector_fraction','component_fraction'):
        result['action_box_clip_'+key]=series([(r['step'],r['rollout_box_clipping']) for r in metrics],key)
    return result


def export_plots(folder,output):
    curves=load_curves(folder);output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Plot output directory must be empty')
    output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(output.resolve()/'.matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs=[('evaluation_success','Independent PPO evaluation success','Success rate',[('evaluation_success_rate','PPO')]),
        ('evaluation_distance','Independent PPO final distance','Distance (mm)',[('evaluation_distance_mm','PPO')]),
        ('evaluation_episode_length','Independent PPO episode length','Environment steps',[('evaluation_episode_length','PPO')]),
        ('policy_gradient_loss','PPO policy gradient loss','Loss',[('policy_gradient_loss','Policy')]),
        ('value_loss','PPO value loss','Loss',[('value_loss','Value function')]),
        ('approx_kl','PPO approximate KL','Approximate KL',[('approx_kl','PPO update')]),
        ('clip_fraction','PPO likelihood-ratio clipping','Fraction',[('clip_fraction','Objective clip fraction')]),
        ('explained_variance','PPO explained variance (native rollout statistic)','Explained variance',[('explained_variance','Value function')]),
        ('entropy_loss','PPO entropy loss','Negative differential entropy',[('entropy_loss','Entropy loss')]),
        ('policy_std','PPO Gaussian action standard deviation','Raw action std',[('policy_std_mean','Policy std mean')]),
        ('action_box_clipping','Environment action Box clipping per rollout','Fraction',
         [('action_box_clip_vector_fraction','Vectors'),('action_box_clip_component_fraction','Components')])]
    generated,skipped=[],[]
    for stem,title,ylabel,items in specs:
        if not any(curves[key] for key,_ in items):
            skipped.append(dict(file=stem+'.png',reason='No measured data; no curve generated'));continue
        fig,axis=plt.subplots(figsize=(8,4.5),layout='constrained')
        for key,label in items:
            if curves[key]:
                x,y=zip(*curves[key]);axis.plot(x,y,marker='o',markersize=4,label=label)
        axis.set(title=title,xlabel='Actual training environment steps',ylabel=ylabel)
        axis.grid(alpha=.25);axis.legend(loc='best')
        if stem in ('evaluation_success','clip_fraction','action_box_clipping'):axis.set_ylim(-.02,1.02)
        fig.savefig(output/(stem+'.png'),dpi=160);plt.close(fig);generated.append(stem+'.png')
    result=dict(source=str(Path(folder).resolve()),curves=curves,generated=generated,skipped=skipped,
                note='Raw JSON measurements; evaluation success is never inferred from training reward.')
    (output/'plot_data.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--out',type=Path);a=p.parse_args()
    try:
        result=export_plots(a.run_dir,a.out or a.run_dir/'plots')
        print(dict(generated=result['generated'],skipped=result['skipped']))
        return 0 if result['generated'] else 1
    except (ValueError,OSError,ImportError) as exc:p.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
