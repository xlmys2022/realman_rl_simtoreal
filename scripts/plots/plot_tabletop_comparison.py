#!/usr/bin/env python3
"""绘制同一冻结 PPO 权重在两套桌面场景的成对评估。"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_report(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--corrected", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("选择新的图像路径")
    reports = [read_report(args.baseline), read_report(args.corrected)]
    alignments = [[(e['seed'], e['initial_object_position_m']) for e in report['episodes']] for report in reports]
    seeds_a = [seed for seed, _ in alignments[0]]
    seeds_b = [seed for seed, _ in alignments[1]]
    positions_a = [position for _, position in alignments[0]]
    positions_b = [position for _, position in alignments[1]]
    if (seeds_a != seeds_b or not np.allclose(positions_a, positions_b, atol=1e-8, rtol=0)
            or reports[0]['checkpoint_sha256'] != reports[1]['checkpoint_sha256']):
        parser.error("两组评估需要相同的权重、随机种子与桌面目标")
    fig, (stages, height) = plt.subplots(1, 2, figsize=(10, 4.1), layout="constrained")
    x = np.arange(3)
    for index, (report, color) in enumerate(zip(reports, ("#6d8191", "#167760"))):
        row = report['summary']
        series = [100 * row[name] for name in ('reach_success_rate', 'lift_success_rate', 'success_rate')]
        bars = stages.bar(x + (index - .5) * .34, series, .34, label=report['scene_profile'], color=color)
        stages.bar_label(bars, labels=[f"{value:g}%" for value in series], padding=2, fontsize=8)
        height.plot([e['seed'] for e in report['episodes']],
                    [1000 * e['metrics']['max_clearance_m'] for e in report['episodes']],
                    marker='o', label=report['scene_profile'], color=color)
    stages.set(title='Outcome by stage', ylabel='Successful episodes (%)', xticks=x,
               xticklabels=('Reach', 'Lift', 'Full grasp'), ylim=(0, 113))
    stages.legend(loc='upper right')
    threshold = reports[1]['environment']['lift_height'] - reports[1]['environment']['height_tolerance']
    height.axhline(1000 * threshold, color='#777777', ls='--', lw=1, label='Lift threshold')
    height.set(title='Peak cube clearance by seed', ylabel='Clearance above table (mm)', xlabel='Episode seed',
               xticks=[seed for seed, _ in alignments[0]])
    height.legend()
    fig.suptitle(f"Frozen visual PPO · {len(alignments[0])} matched targets")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == "__main__":
    main()
