# 03-C1 EG2-4C2导入与基础控制验证

日期：2026-11-07。分支：`feat/eg2-gripper`。使用现有realman_rl（Python3.11、MuJoCo3.17.0），未新增依赖、连接真机、调用运动接口、提交或推送。

## 来源及真实性结论

本次安装的是**官方明确命名的EG2-4C2模型资产**，不是另一型号。固定来源RealManRobot/rm_models提交 `776c1a7dd668ba67bfb03636015c59c8ef6e1b43` 的 `thirdparty/Two-finger Electric Gripper.urdf/`。URDF、CSV和package均对应EG2-4C2；7个STL完整，6个转动关节、5条mimic、质量及惯量均有定义。

保留13个原始文件（含许可证、包元数据及RMU01020安装图），SHA256/Git blob与固定树逐文件核对。具体资源调查、全部变更、几何公式和授权冲突见 [SOURCE.md](../../assets/grippers/eg2_4c2/SOURCE.md) 和upstream_manifest.json。根许可证Apache-2.0与package中的未注明变体BSD并存，没有自行覆盖。

但是模型**不等同于已确认的实物动力学**：质量仅94.2992g，产品参考231g；开口定义的最大值66.2173566mm，产品参考70mm。没有改尺寸/限位或猜测缺失质量去凑参数。安装板假定11g后新增模型载荷共105.2992g，而产品231g夹爪加同一假定板重时为242g，差异136.7008g。不能将后面的精细数值误差解读为真机精度。

## 新增与最小兼容修改

- `assets/grippers/eg2_4c2/upstream/`、manifest及SOURCE：完整来源、原始字节、限制及许可证。
- `eg2_4c2_mujoco.urdf`：只改mesh路径、加入明确的compiler配置，保留源文件。
- `eg2_4c2.xml`：独立6-hinge夹爪、一个servo、5条mimic，默认编译自身重力补偿。
- `rml63_eg2_4c2.xml`：原机械臂及全部相机附件，加实验板和夹爪。
- `scripts/build_gripper_model.py`：离线校验、转换及生成；不修改RML63资产。
- `simulation/gripper.py`：米制目标/实际宽度、张开/闭合、缓慢ctrl更新、物理推进与状态审计。
- `scripts/demo_gripper.py`：无界面验收、近景Viewer、完整/提前退出说明及JSON。
- `tests/test_gripper.py`：19项来源、结构、状态隔离、物理运动、接触及回归测试。

model.py移除“整个模型必须6个nq/nv/nu”的检查，改为六个命名机械臂hinge/actuator的映射；重力补偿遍历link_1后代，只补偿机器人，不补偿无关世界body。Kinematics仍只有六轴输入，但允许模型有额外坐标并正确切片Jacobian。ee_control的独立碰撞探针复制附件qpos。原control.py、执行器参数、ReachEnv及所有原测试均未改。

## 装配坐标

保留原ee_site=link_6原点，所有相机body/mesh及六维力/V相关link_6几何、质量、惯量和位姿均不变。原URDF未独立定义可标定的六维力坐标，不能从外观分离真实附件栈。

RMU01020图可核对直径57mm、总厚8mm；重量栏0.011在实验中按kg解释为11g，图中未注明重量单位，板重也未实称。图纸不能唯一确定与本机6F-V栈的安装方向、深度和已有板件是否重复。实际组合为**未经实物验证的实验同轴配置**，不是厂家装配标定：

| 变换 | 平移，m | 旋转 |
|---|---|---|
| ee_site → gripper_mount | [0,0,.008] | I |
| gripper_mount → grasp_site | [0,0,.147530235...] | I |
| ee_site → grasp_site | [0,0,.155530235...] | I |

安装板用固定圆柱代理及均匀圆柱惯量近似，未复刻孔/沉孔/凸台，原机械臂body惯量不融合、不改变。gripper_mount与原夹爪base重合。grasp_site为q=.41rad时两个末端接触带标记的名义中心，不是校准TCP。

四连杆使实际夹指中点的Z随开口改变，闭合约141.591mm、最大开口约146.585487mm（相对mount）。未来精确抓取需按开口标定或读取两个实际pad positions的中点；不能把固定grasp_site当作所有宽度下不变的真实接触中心。

## 运动、执行器与控制

q=4C2_Joint1，Joint2/5/6=q，Joint3/4=-q；轴、连接和限位来自URDF。正向关节0..0.82rad，反向-0.82..0rad。MuJoCo自动导入5个joint equality；不是对六个关节分别硬写位置。

开口由末端平直带上两个标记沿base X轴的投影定义，理想关系为：

`w(q)=2*[.0223*(1-cos(q))+.035591*sin(q)]`

输入单位m，单调反求主关节角。实际宽度在独立MjData上由当前全部实际qpos计算pad sites，因此能反映mimic残差；不会刷新实时力缓存。关节目标非法直接拒绝，70mm也拒绝，不裁剪成66.2mm假装支持产品全行程。

单个位置执行器eg2_position：kp=.05、kv=.002；各夹爪关节阻尼.0001，均是仿真整定值。主驱动总限幅±.2N·m，其他joint保留上游±1字段；20N夹持力没有直接当作某个关节的N·m。

命令按.5rad/s更新ctrl，低于源velocity=1。Mimic求解solref=.004/1、solimp=.999/.999/.001。较高增益试验曾产生明显柔性约束误差/越限，提高solver精度未改善；降低夹爪增益和阻尼后通过同样的越限/速度检查，没有虚构转子惯量或放宽检查。独立未补偿端点启动也曾触发保护，故独立模型默认编译补偿；组合演示显式load_model(..., gravity_compensation=True)。未补偿夹爪精度不是通过项。

所有开合经mj_step，只有初始化用官方home keyframe。set_width/open/close只选择目标；正常运动仅改变夹爪ctrl，六轴ctrl保持；原关节控制只改变六轴ctrl。两者推进共享物理时间，不能顺序调用后仍声称只推进一次。控制中检查有限值、关节/速度边界、激活的mimic、碰撞和总驱动力；六轴ForceMonitor仍在刷新缓存前审计。

## 碰撞设计与反例

保留7个视觉STL和7个独立碰撞mesh；视觉group1不参与接触，碰撞mask为1。MuJoCo凸包会填充凹槽，尚非精确指垫接触表面。只排除定位到的8个机构内部装配对，组合模型共9条exclude（含原base_link/link_1）；左右指垫、外物及非相邻机械臂仍参与碰撞。

裸URDF前向计算显示壳体/连杆凸包约8.7–13.6mm重叠，非树父子的闭环配合连杆约10.5mm。没有缩小STL或移动关节消除装配错误。过滤依据和各对清单保存在SOURCE。完整无物体轨迹最小接触距离约-5.0e-9m，仅数值级接触，无超过0.1mm的非预期穿透。

接触反例分别把固定小球放入左右指垫接触带，两个夹指均产生实际MuJoCo接触和正法向接触力，控制器拒绝继续穿透，时间不前进。折叠机械臂的base_link/link_2自碰撞反例仍被检出。不是只移动视觉网格，碰撞未被全局禁用；没有实现物体夹取流程。

## 结构与实际结果

| 模型 | nq / nv / nu | 关节与约束 |
|---|---|---|
| 原单独机械臂 | 6 / 6 / 6 | 六轴hinge，原场景不变 |
| 独立夹爪 | 6 / 6 / 1 | 六hinge，五mimic |
| arm+gripper | 12 / 12 / 7 | 原六轴＋六夹爪hinge，五mimic；base/板无freejoint |

组合重力补偿覆盖16个机器人body、12个joint，载荷补偿受关节执行器限幅，独立世界测试body不补偿。原六轴body质量/惯量/位姿、关节轴/限位、增益和力矩范围逐项相等。原机械臂单独场景仍运行，组合六轴FK/IK也使用六角接口。

无界面两轮开合，5530个物理步，总仿真11.06s：

| 指标 | 实测 |
|---|---|
| 最大目标开口 | 66.217356607mm |
| 闭合端点实际宽度 | 约0.00004522mm |
| 再张开端点误差 | 约0.000000619mm |
| 闭合到位时间 | 1.732s，到目标±0.2mm的首次时间 |
| 张开到位时间 | 1.736s，同一标准 |
| 开口峰值速度 | 约42.0001mm/s |
| 关节峰值速度 | 0.501335rad/s，低于源1rad/s |
| 主驱动峰值总力矩 | 0.00299126N·m，限幅±.2 |
| 最大mimic残差 | 0.000255171rad |
| 最大六轴home偏差 | 约0.00001848° |
| 六轴力矩饱和步数 | 0 |

各阶段末0.5s开口波动通过≤0.1mm验收；没有持续振荡、NaN/Inf或严重非预期碰撞。六轴峰值总驱动力矩约[.0000214,2.32095,4.86362,.093235,.720482,.000904]N·m，低于原[60,60,30,30,10,10]界限。补偿已路由到qfrc_actuator，未把qfrc_gravcomp重复相加。

这些微小端点误差依赖理想模型、计算参考点和同模型重力补偿；不等于硬件定位/夹持精度。产品1.3s闭合及20N夹持能力均未复现或标定。

## 测试与Viewer

修改前原134项全部通过（44.009s）；最终 **153/153通过（50.709s）**，原134项断言未修改。新增19项覆盖来源字节、模型/耦合、变换、宽度映射、稳定开合、实时qpos仅由mj_step修改、力缓存隔离、控制通道、带载保持、独立夹爪、无关body补偿隔离、外物接触、原碰撞、生成重现及其他工作目录CLI。

中途独立模型编译出现XML属性错误，原因是给body/joint配置补偿时误选到了同名的equality/joint节点；已把生成器限定到body下的joint，并同时编译验证两种输出模型。没有改变模型限位或修改测试断言绕过错误。

真实桌面Viewer已运行完整一轮“张开→闭合→再张开”，近景图可清晰看到两指变化及保留的相机，进程返回0，完整结果JSON已写入。关闭使用程序完成及上下文线程清理。鼠标操作、人工按钮关闭和交互式Ctrl+C本次未验证；提前退出分支保留明确未完成提示。

本地Git忽略证据位于outputs/gripper/：baseline_tests.log、all_tests.log、demo.json、demo.log、viewer_demo.json、viewer_result.json、viewer.log、三张viewer PNG、adapter.png、耦合参数诊断JSON及最终audit.json。所有源资源及旧机械臂哈希检查、依赖检查记录在审计中。

```bash
conda activate realman_rl
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/build_gripper_model.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/demo_gripper.py --cycles 2 --json /home/mr-beaver/Documents/robot/realman_rl_simtoreal/outputs/gripper/demo.json
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/demo_gripper.py --viewer --cycles 1
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

## 后续边界与特别回答

- **是否确切型号？** 是官方明确命名的EG2-4C2几何/运动学资产；不是已校准的231g/70mm实物模型，质量、行程和安装均有已记录差异。
- **是否真实物理夹指？** 是，6个有惯性的hinge body、5条约束及位置执行器由mj_step驱动，两指均能产生外物接触力。
- **grasp_site与ee_site？** 实验同向、+Z155.530235mm的固定名义参考；实际夹指中点随开口变化，尚不是TCP标定。
- **RM_API2能否明确映射？** 仅只读检查SDK文档：位置1..1000无量纲。未来可以统一米制接口，但开口与设备量的函数/方向/端点需真实标定；不能把模型66.2mm线性比例当作设备70mm标定。当前没有调用连接或运动接口。
- **后续可否加物体？** 可基于保留的物理接触能力继续实验；但需要改进凹槽碰撞表示、核实载荷、接触/摩擦和力控制，并在新的Pick任务中定义允许接触。当前开合安全检查会拒绝严重穿透，未实现抓取成功判据或策略。

ReachEnv代码、动作、奖励和默认6轴资源完全不变；本次没有物体抓取、RL训练、ROS2或真机通信。真实6F-V附件栈、标定TCP、实际传动/自锁、力与速度曲线及完整质量惯量仍未确认。
