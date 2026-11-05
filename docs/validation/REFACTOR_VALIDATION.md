# 02-D 工程结构重构验证

日期：2026-11-05。分支：`refactor/project-structure`。只做职责、导入、安装及文档位置迁移，没有提交、推送或创建其他工作区。

## 范围与文件

完整目录、迁移映射和依赖图见 [architecture.md](../architecture.md)。新增文件的职责如下：

- `pyproject.toml`：setuptools src 包发现、0.1.0 元数据，动态读取原有两份依赖文件。
- `src/realman_rl_simtoreal/__init__.py` 及三个子包的 `__init__.py`：轻量包标识，不初始化模型、SDK 或 Viewer。
- `paths.py`：从 editable 源码位置定位仓库资源及默认输出目录。
- `simulation/model.py`：原有加载、关节/执行器映射、结构检查、初始化、编译前重力补偿配置，以及独立数据中的纯重力诊断。
- `simulation/control.py`：原有关节目标检查、整段控制、阈值和总驱动力审计。
- `simulation/kinematics.py`：原有 FK、Jacobian、DLS IK，文件内容与旧实现一致。
- `simulation/ee_control.py`：原有末端运动检查、碰撞预检、执行及误差统计。
- `vendor/realman_fk.py`、`realman_ik.py`：唯一一套厂家适配实现，保留延迟加载与同一进程全局会话保护。
- `experiments/common.py`：共享 SE(3)、测试姿态、默认关节演示序列及纯文本结果格式；由 CLI 负责打印。
- `viewer.py`：仅供 CLI 调用的后端选择和惰性 Viewer 线程清理。
- `tests/test_package.py`：10项包、资源、无SDK、导入图、CLI及Viewer兼容测试。
- `docs/architecture.md`、本报告：当前调用约定和本次证据记录。

八个CLI文件、五个原测试文件更新导入；模型生成器只更新资源定位。删除旧 `scripts/kinematics.py`、`realman_fk.py`、`realman_ik.py`，没有转发副本。README更新安装、导入和链接，`.gitignore`增加构建/安装产物忽略。六份历史验证报告整体移动到本目录，原文不变；历史代码路径保留。

## 修改前基线

任何项目文件修改之前：

1. 完整运行原75项测试，**75/75通过，28.948秒**。
2. 保存17个模型资产文件的SHA256、原代码/测试和六份报告副本、环境包版本。
3. 从 `/tmp` 用目标环境的 `python -I` 和旧脚本绝对路径运行关节、末端、重力对照、全配置厂家FK及厂家IK对照，保存JSON、输出和退出码。

证据保存在 Git 忽略的 `outputs/refactor/baseline/`；它们是本次本地审计产物，不是项目运行依赖。

## Editable安装与依赖

目标解释器：`/home/mr-beaver/Documents/miniconda3/envs/realman_rl/bin/python`，Python 3.11.17。

先检查 `pip install --dry-run --no-build-isolation -e .` 的计划，仅增加本项目；再实际执行：

```bash
conda activate realman_rl
python -I -m pip --no-cache-dir install --no-build-isolation -e /home/mr-beaver/Documents/robot/realman_rl_simtoreal
python -I -m pip check
```

- 安装版本：`realman_rl_simtoreal==0.1.0`；`direct_url.json` 的 `dir_info.editable` 为 `true`。
- `pip check`：`No broken requirements found.`
- 原有包版本全部不变，仅新增本项目；没有升级或降级依赖。
- MuJoCo 3.17.0、NumPy 2.4.6、Robotic_Arm 1.1.7及算法库1.6.3-48513146沿用现有环境。
- 复用现有setuptools 84.0.0、wheel 0.48.0和pip 26.2.1。两份 requirements 文件没有修改；vendor依赖仅在 `extra == "vendor"` 时要求。
- 在项目目录及 `/tmp` 下用 `python -I` 导入，包来源均为仓库 `src/realman_rl_simtoreal`；不依赖当前目录、用户包或ROS的PYTHONPATH。

环境前后清单与安装审计见 `outputs/refactor/{baseline,after}/packages.json`、`after/install_audit.json`。本次支持保留源码仓库的editable使用方式；未验证脱离仓库的wheel资源安装。

## 回归结果

| 检查 | 修改前 | 修改后 |
|---|---:|---:|
| 原有自动测试 | 75/75 | 75/75，断言及精度阈值保留 |
| 新增包/入口测试 | — | 10/10 |
| 完整测试 | 75/75 | 85/85 |
| SDK不可用模拟 | — | 59项通过，26项可选厂家测试明确跳过 |
| 厂家II+SF IK主集 / 边界集 | 36/36、11/12 | 36/36、11/12 |
| 厂家II+ISFV IK主集 / 边界集 | 36/36、11/12 | 36/36、11/12 |
| MuJoCo IK主集 / 边界集 | 27/36、12/12 | 27/36、12/12 |
| 关节 / 末端 / 重力对照CLI退出码 | 0 / 0 / 0 | 0 / 0 / 0 |
| 厂家FK全配置CLI退出码 | 2（模型不一致） | 2（相同结论） |
| 厂家IK对照CLI退出码 | 0（审计通过） | 0（相同语义） |

重构后的所有75个原测试方法经导入限定名归一化后与基线一致。IK的0退出码仍表示求解结果、失败处理和重复性审计通过，不表示所有案例成功。J5负限位附近失败、MuJoCo远初值局部失败、失败原因、解分支和模型差异均继续复现。

完整测试命令：

```bash
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

`outputs/refactor/final_tests.log` 保存85项完整输出（最终一次30.713秒）；`package_tests.log`保存新增10项输出，`no_sdk_tests.log`保存SDK不可用时的完整测试结果。SDK验证采用可用性拦截以及独立进程的真实导入拦截，没有卸载SDK。被拦截时基础模型加载、重力补偿、FK/IK及控制仍能执行，厂家适配器构造明确抛出SDK不可用错误。

## 数值与资源一致性

关节控制、末端控制、PD/重力对照、全部厂家FK配置及厂家IK实验分别保存为 `baseline/` 和 `after/` 的JSON：

- **44,031个非耗时值逐值完全相等**，同时核对JSON键、类型和列表长度。不是用较松的近似比较代替一致性检查。
- 仅排除由 `perf_counter` 得到的993个计时字段值：`elapsed_s`、`native_elapsed_s`、`public_call_times_s`、`native_call_times_s`、`median_public_call_s`、`p95_public_call_s`、`median_native_call_s`；仍检查计时字段类型、长度及有限非负值。
- 仿真时间、全部物理值、状态、失败原因、关节解分支，以及重力JSON中的完整 `joint_sequence.log` 均参与精确比较。
- 例如默认PD末端实际绝对误差仍为 `0.0023915444025621405 m`，姿态误差仍为 `0.008718668804004764 rad`。未通过改变模型或补偿方式改善这些数字。
- 17个资产文件SHA256全部一致，包含原始URDF、兼容URDF、STL、MJCF、scene、SOURCE及相关元数据；重新生成动力学模型后仍逐字节一致。
- 六份移动后的历史报告与修改前副本逐字节一致。没有改写历史失败或实验数字。
- 核心提取函数/类的AST与基线一致，仅规范包内限定名。运动学与厂家FK整文件内容一致；关节结果格式化后的文本与两种控制器全部13段捕获日志一致。

审计摘要：`after/comparison_audit.json`、`core_audit.json`、`extraction_audit.json`。这些检查同时覆盖重力补偿仍在编译前配置，以及步进后、tick刷新缓存前调用力通道观察器的顺序。

## 包与CLI兼容测试

新增测试覆盖轻量初始化、隔离导入、资源定位、元数据/可选依赖、SDK缺失、无核心到脚本的依赖、无循环导入、无手工搜索路径修改、七个现有argparse入口的帮助及原错误退出码。

模型生成器原本没有argparse接口，未为它新增 `--help`；通过原有重新生成测试验证实际成功入口。静态检查及全部数值实验的成功入口仍有自动测试/实际运行覆盖。所有脚本均保留原绝对路径，从其他工作目录用 `python -I` 运行。

一次迁移中间运行暴露新 `simulation.model` 漏导入 `Path`，已补齐；随后原75项及新增测试通过。没有修改原断言、算法或阈值来处理这一导入错误。

## Viewer实际验证

在当前X11/XWayland桌面，以目标环境解释器、`-I`、脚本绝对路径及 `/tmp` 为工作目录运行：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/view_robot.py --duration 4
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_joints.py --viewer --duration 6
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/control_ee.py --viewer --duration 6
```

三者均找到实际MuJoCo窗口，截图检查可见完整机器人；三个进程均在定时退出后返回0。截图与日志位于 `outputs/refactor/viewer/`，`results.json`记录窗口、图片尺寸和退出码。

- 静态Viewer正常显示URDF零位，关闭后qpos不变，仿真时间为0。
- 关节Viewer可见动力学模型；末端Viewer可见机器人与目标标记。
- 关节及末端窗口在完整演示结束前定时停止，日志正确提示完整验收尚未完成。完整物理序列由无图形回归验证，不把这些短窗口运行算成完整控制验收。
- 新测试还覆盖显式后端优先、缺少显示服务的错误，以及正常/异常退出时只等待本次Viewer线程的行为。

**未验证：本次没有操作鼠标旋转/缩放、人工点击关闭按钮或交互式Ctrl+C。** 自动关闭及真实窗口渲染已验证；不据此宣称上述人工交互已经通过。

## 保留限制

editable安装依赖源码仓库和仓库内assets。run_segment仍是整段演示控制，home附近±5°及10mm等限制保持；没有变为Gymnasium单步接口。

临时ee_site仍不是标定TCP。厂家和MuJoCo的完整模型尚未统一，真实代际及6F-V传感器对应仍待确认。厂家Algo配置仍是进程全局、顺序使用，不支持同时运行不同配置。仿真重力补偿精度仍依赖当前模型参数，不能作为真实机械臂精度证明。

本次没有修改官方模型、SDK内部参数、IK算法或控制增益，没有连接真机，也没有进入Gymnasium或强化学习开发。
