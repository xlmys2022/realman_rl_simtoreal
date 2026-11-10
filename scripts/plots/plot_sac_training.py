#!/usr/bin/env python3
"""Export saved independent evaluation and training diagnostics to PNG, offline."""
import argparse
import json
import os
from pathlib import Path


def load_curves(folder):
    """JSON only, no environment, SAC model, smoothing or inferred success data."""
    folder=Path(folder)
    snapshots=json.loads((folder/'checkpoints.json').read_text()) if (folder/'checkpoints.json').exists() else []
    if (folder/'evaluations.json').exists():
        evaluations=json.loads((folder/'evaluations.json').read_text())
        evaluation_rows=[(r['step'],r['summary']) for r in evaluations]
    else:
        evaluation_rows=[(r['step'],r['evaluation']) for r in snapshots if r.get('evaluation') is not None]
    diagnostics=[]
    if (folder/'diagnostics.jsonl').exists():
        diagnostics=[json.loads(line) for line in (folder/'diagnostics.jsonl').read_text().splitlines() if line]
    diagnostics.extend(r['diagnostic'] for r in snapshots if r.get('diagnostic'))
    def series(rows,key):
        values={int(step):float(row[key]) for step,row in rows if row.get(key) is not None}
        import math
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Non-finite stored metric: '+key)
        return [[step,value] for step,value in sorted(values.items())]
    loss_rows=[(r['step'],r) for r in diagnostics]
    entropy=series(loss_rows,'entropy_coefficient') or series(loss_rows,'train/ent_coef')
    return dict(evaluation_success_rate=series(evaluation_rows,'success_rate'),
                evaluation_distance_mm=[[s,1000*v] for s,v in series(evaluation_rows,'mean_final_distance_m')],
                actor_loss=series(loss_rows,'train/actor_loss'),critic_loss=series(loss_rows,'train/critic_loss'),
                entropy_coefficient=entropy)


def export_plots(folder, output):
    curves=load_curves(folder)
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Plot output directory must be empty')
    output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(output.resolve()/'.matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specifications=[('evaluation_success.png','Independent evaluation success','Success rate',
                     [('evaluation_success_rate','SAC evaluation')]),
                    ('evaluation_distance.png','Independent evaluation final distance','Distance (mm)',
                     [('evaluation_distance_mm','SAC evaluation')]),
                    ('actor_critic_loss.png','Training loss (stored diagnostic updates)','Loss',
                     [('actor_loss','Actor'),('critic_loss','Critic')]),
                    ('entropy_coefficient.png','Entropy coefficient','Alpha',
                     [('entropy_coefficient','Entropy coefficient')])]
    generated,skipped=[],[]
    for filename,title,ylabel,items in specifications:
        if not any(curves[key] for key,_ in items):
            skipped.append(dict(file=filename,reason='No measured data; no synthetic curve generated'))
            continue
        available=[(key,label) for key,label in items if curves[key]]
        separate=filename=='actor_critic_loss.png'
        fig,axes=plt.subplots(len(available) if separate else 1,1,figsize=(8,6 if separate else 4.5),
                              squeeze=False,sharex=True,layout='constrained')
        for index,(key,label) in enumerate(available):
            axis=axes[index,0] if separate else axes[0,0]
            x,y=zip(*curves[key]);axis.plot(x,y,marker='o',markersize=4,label=label)
            axis.set(xlabel='Training environment steps',ylabel=label+' loss' if separate else ylabel)
            axis.grid(alpha=.25);axis.legend(loc='best')
            if filename=='evaluation_success.png':axis.set_ylim(-.02,1.02)
        fig.suptitle(title)
        fig.savefig(output/filename,dpi=160)
        plt.close(fig);generated.append(filename)
    result=dict(source=str(Path(folder).resolve()),generated=generated,skipped=skipped,
                curves=curves,note='Independent evaluation only; training reward is never treated as success.')
    (output/'plot_data.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,help='Default: run-dir/plots; must be empty')
    args=parser.parse_args()
    try:
        result=export_plots(args.run_dir,args.out or args.run_dir/'plots')
        print(json.dumps(dict(generated=result['generated'],skipped=result['skipped'])))
        return 0 if result['generated'] else 1
    except (ValueError,OSError,ImportError) as exc:
        parser.exit(1,str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
