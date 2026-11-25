"""实机安装场景和评估指标的附加 CLI 选项。"""

from pathlib import Path

from ...simulation.runtime.observation_presets import SCENE_PROFILES


def add_installation_arguments(setup, train, evaluate, render):
    """在原笛卡尔入口上注册显式场景、训练采样和指标选项。"""
    for command in (setup, train, evaluate, render):
        command.add_argument('--scene-profile', choices=SCENE_PROFILES,
                             help='选择原场景、实机安装名义场景或独立手眼候选')
        command.add_argument('--hand-eye-file', type=Path, help='手眼候选场景使用的本地标定文件')
    train.add_argument('--object-radius', type=float, help='桌面采样圆盘半径（m），范围 0–0.05')
    evaluate.add_argument('--metrics', action='store_true', help='记录分阶段、运动和视觉评估指标')
    evaluate.add_argument('--compact', action='store_true', help='保留每回合汇总及指标，省略逐步记录')
