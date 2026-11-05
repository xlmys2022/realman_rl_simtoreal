# 02-C2：RM_API2 逆运动学离线验证

日期：2026-11-05。沿用 Python 3.11.17 / realman_rl、MuJoCo 3.17.0、NumPy 2.4.6、Robotic_Arm 1.1.7、算法库 1.6.3-48513146。没有新增或改动依赖，`pip --no-cache-dir check` 通过。

## 结论与范围

官方 IK 可以离线运行，但本次结果不能概括为所有合法目标都可靠：

- II+SF、II+ISFV 各在 9 姿态 × 4 初值的主集上通过 **36/36** 次自模型 FK→IK→FK；边界集各为 **11/12**。
- 当前 MuJoCo DLS 主集为 **27/36**：home、near、other_elbow 初值各 9/9；far 初值 0/9。边界集为 **12/12**。已知 FK 可达目标的局部求解失败被保留，没有改求解器参数、增加迭代次数或自动换初值。
- home 目标在不同初值下得到两种不同肘部配置，均通过厂家 FK 验证。全解接口也可用；home 的 8 个原始分支有 2 个满足双方限位，另一姿态有 8 个合法周期等价解。
- **新的重要失败案例**：J5 接近负限位时，初值接近目标甚至等于目标，单解接口仍返回 1；同一目标的全解接口找到两组合法解，home 初值也成功。不能把返回码 1 当作目标不可达的证明。
- 厂家与 MuJoCo 完整模型仍不一致。交叉使用关节解时，另一模型末端仍出现约 23 mm（SF）或 11.5 mm（ISFV）偏差。本次没有改任何 URDF、MJCF、DH、底层控制器或现有 MuJoCo IK。

本程序只有离线运动学计算，不初始化通信、不创建机械臂连接、不调用运动控制接口、不向 MuJoCo 执行器发送厂家解，也不调用 `mj_step()`。原有动力学回归测试仍按原方式进行仿真积分。

## 调用方式与返回契约

参考：[官方 Algo 接口](https://develop.realman-robotics.com/robot/apipython/classes/algo/)、[逆解输入结构体](https://develop.realman-robotics.com/robot/apipython/struct/inverseKinematicsParams/)、[全解结构体](https://develop.realman-robotics.com/robot/apipython/struct/inverseKinematicsAllSolve/)。同时核对了本机安装包的 `rm_robot_interface.py` 和 `rm_ctypes_wrap.py`，SDK 发行来源与哈希沿用 `REALMAN_KINEMATICS_VALIDATION.md`，运行 JSON 也保存其版本、DH、限位、坐标系与文件哈希。

`RealManIK` 继承已有 `RealManFK`，每次配置只初始化一个 standalone Algo（handle id=0），IK 与 FK 使用同一厂家模型。顺序验证 **RM_MODEL_RM_63_II_E=3 + RM_MODEL_RM_SF_E=2 / RM_MODEL_RM_ISFV_E=5**；两者都是候选，均未确认对应真实旧版 6F-V。

```python
model, _, _ = control_joints.load_model()
kin = Kinematics(model)
vendor = RealManIK(kin.limits, model="II", sensor="SF")
p, R = vendor.fk(kin.home_q)
result = vendor.solve_ik(p, kin.home_q, R)
all_result = vendor.solve_all(p, kin.home_q, R)
```

公共接口输入位置 m、关节 rad、末端到基座的 3×3 主动旋转矩阵；参数顺序与现有 Kinematics.solve_ik 一致。旋转必须有限、正交且 det=+1；使用官方 `mju_mat2Quat` 编码为单位 **wxyz**，构造：

`rm_inverse_kinematics_params_t(q_in_deg, [x,y,z,w,x,y,z], flag=0)`

再调用 `rm_algo_inverse_kinematics(params)`。不是欧拉角相减；最终姿态残差为 `||Log(R_target R_actual.T)||`（rad），位置为欧氏范数（m）。关节输出 degree 转回 rad。

- `0`：SDK 声称成功，仍须检查六轴数量、有限值、双方限位及独立厂家 FK。只在全部通过时提供 `success=True, q=<六弧度>`。
- `1`：SDK 求解失败；`-1`：参考关节输入错误（全解文档还列出超限）；`-2`：目标四元数错误；未知非零码也明确拒绝。
- 本地非法输入、过期配置等拒绝返回 `sdk_code=None`。不伪造厂家错误码。所有失败 `q=None`，明确 reason；SDK 在失败时填充的 q_out 被丢弃，不推测其单位或赋予有效解语义。
- 返回字段含厂家返回码、位置/姿态残差、原始/最终双方限位标志、整数圈变换、完整调用耗时和 native 方法耗时。全解逐分支审计；被拒绝分支也是 `q=None`。
- 固定验收阈值为 **1e-5 m / 1e-4 rad**，与现有 MuJoCo 默认 IK 阈值相同，且高于所测 native float32 残差。不因边界失败放宽限位或误差要求。

算法遍历模式显式设为 True；不改 DH 或厂家限位。厂家库没有在此接口暴露内部迭代次数/最大迭代参数，不能把一次 native 调用说成一次数学迭代，也不声称已核实闭源实现为某一种纯解析算法。可确认它提供最多八个分支的六自由度全解与参考初值选解接口。

### 全局配置与坐标系

复用原 FK 的安装角、工作/工具坐标单位变换及活跃会话检查。新建另一 RealManFK/RealManIK 后，旧 IK 会话返回明确失败；旧 FK 的原有报错契约保持不变。SF→ISFV→SF 切换与重现性有测试。**不支持并发，不能混用外部直接修改 native Algo 全局状态的代码。** 该保护不声称能检测绕过适配器的所有外部 SDK 调用。

厂家目标在名义基座/默认末端（零工具）中；MuJoCo 目标在 world/ee_site 中。当前 base 固定且与 world 重合，ee_site 仍为 link_6 局部原点。主实验每套求解器使用自己的 FK 生成 p、R，并显式传入完整姿态，不暗用另一模型的绝对目标或仅位置 IK。

### 关节限位与整数圈

| 关节 | SDK ° | 当前 URDF/MuJoCo ° |
|---|---|---|
| J1、J2、J4、J5 | [-178,178] | [-177.960691,177.960691] |
| J3 | [-178,145] | [-177.960691,144.998429] |
| J6 | [-360,360] | [-359.817495,359.817495] |

读取厂家限位与现有模型限位，按两者交集检查初值及最终 q；不修改任何限位。另一个真实接口反例：厂家合法 J1=177.99°、初值177.9°，SDK 返回0且 FK 正确，但该分支超出当前模型的舍入限位，适配器返回失败/q=None，并保留“厂家合法、模型超限”标志。

全解原始角度表示依赖 q_in：同一姿态用 home 初值时可直接返回 -155°，用 far 初值则可能返回等价的205°。仅允许 `q'=q+2*pi*k`，从双方限位内选择最接近初值的合法表示，并重新 FK 验证。**不裁剪角度，不添加人为零偏，不用它消除 FK 模型差异。** ±180° 在 ±178° 限位内没有周期等价表示，必须拒绝。

## 实验集合与计时

主集使用与 FK 对照相同的 9 姿态。都是限位内运动学样本，不保证无碰撞或存在安全运动路径；没有运行这些姿态之间的物理运动。

| 姿态 | 来源关节角 J1..J6 ° |
|---|---|
| home | [  0. -20.  30.   0.  20.   0.] |
| J1_plus_3deg | [  3. -20.  30.   0.  20.   0.] |
| J2_plus_3deg | [  0. -17.  30.   0.  20.   0.] |
| J3_plus_3deg | [  0. -20.  33.   0.  20.   0.] |
| J4_plus_3deg | [  0. -20.  30.   3.  20.   0.] |
| J5_plus_3deg | [  0. -20.  30.   0.  23.   0.] |
| J6_plus_3deg | [  0. -20.  30.   0.  20.   3.] |
| combined | [  2. -22.  32.   1.  19.   2.] |
| diverse_fk_only | [ 25. -45.  55.  20.  40. -30.] |

四个共同初值（均合法，未用随机数）：

| 初值 | J1..J6 ° |
|---|---|
| home | [0,-20,30,0,20,0] |
| near | [2,-23,34,2,18,3] |
| other_elbow | [0,25,-55,0,-20,0] |
| far | [100,40,-70,60,80,100] |

每个主案例一次不计时预热 + 5 次同输入计时；各模型共36案例、180次计时。记录中位数/P95，所有重复的返回状态和成功关节解一致。边界、不可达、连续及失败诊断分别预热后测1次。计时不包含模型加载、JSON 输出和实验层额外 FK 复核。

厂家 public 调用包含输入转换、限位、返回检查、厂家 FK 复核；同时单独记录 SDK Python 方法（含 ctypes）的 native 时间。MuJoCo public 调用是现有 solve_ik，内部包含迭代 FK/Jacobian。两种公开接口的工作范围、实现语言和目标模型不同，**耗时仅为本机本数据集测量，不是同一机器人上的严格算法排名或实时性保证**。

## 内部 FK→IK→FK 对照

下表最大误差仅统计**成功且再次 FK 复核通过**的案例，失败另列，绝不以0代替失败残差。MuJoCo 仍用原 DLS 参数：150次上限、阻尼0.01、旋转权重0.2 m/rad、每次最大0.05 rad、seed_weight=1e-4、最多8档回溯；没有重试或调参。

| 求解器 | 成功/主集 | 最大位置残差 m | 最大姿态残差 rad | public 中位 ms | public P95 ms | native 中位 µs |
|---|---:|---:|---:|---:|---:|---:|
| MuJoCo DLS | 27/36 | 9.69383388e-06 | 1.89403493e-05 | 1.613746 | 5.219106 | — |
| RM_MODEL_RM_SF_E | 36/36 | 1.36774677e-07 | 3.19641506e-07 | 0.118178 | 0.127097 | 3.425 |
| RM_MODEL_RM_ISFV_E | 36/36 | 1.40618284e-07 | 3.20876513e-07 | 0.119295 | 0.128292 | 3.500 |

SDK 的本次残差比 DLS 小，不能据此认定 DLS 本质上达不到同样数值精度：DLS 达到既定10µm/100µrad容差即停止。所有数值都不是实物测量精度。

| 初值 | MuJoCo 成功 | SF 成功 | ISFV 成功 | MuJoCo public 中位 ms | SF public 中位 ms | ISFV public 中位 ms |
|---|---:|---:|---:|---:|---:|---:|
| home | 9/9 | 9/9 | 9/9 | 0.220119 | 0.118179 | 0.119300 |
| near | 9/9 | 9/9 | 9/9 | 0.320070 | 0.119197 | 0.118964 |
| other_elbow | 9/9 | 9/9 | 9/9 | 2.132863 | 0.117211 | 0.119992 |
| far | 0/9 | 9/9 | 9/9 | 4.930973 | 0.117692 | 0.119678 |

### 逐姿态主集结果（最大成功误差，跨4初值）

| 姿态 | MuJoCo 成功 | MuJoCo m / rad | SF 成功 | SF m / rad | ISFV 成功 | ISFV m / rad |
|---|---:|---|---:|---|---:|---|
| home | 3/4 | 3.66392e-06 / 3.97815e-06 | 4/4 | 1.20185e-07 / 3.19642e-07 | 4/4 | 1.1949e-07 / 2.35677e-07 |
| J1_plus_3deg | 3/4 | 5.03658e-06 / 1.29708e-05 | 4/4 | 8.94752e-08 / 5.80381e-08 | 4/4 | 1.11876e-07 / 2.30428e-07 |
| J2_plus_3deg | 3/4 | 9.37797e-06 / 1.89403e-05 | 4/4 | 1.49972e-08 / 6.11726e-08 | 4/4 | 1.19454e-07 / 6.3974e-08 |
| J3_plus_3deg | 3/4 | 9.69383e-06 / 5.54775e-06 | 4/4 | 3.36221e-08 / 3.30322e-08 | 4/4 | 1.19236e-07 / 2.5298e-07 |
| J4_plus_3deg | 3/4 | 6.20788e-06 / 5.39255e-06 | 4/4 | 1.30166e-07 / 1.47539e-07 | 4/4 | 1.19475e-07 / 3.20877e-07 |
| J5_plus_3deg | 3/4 | 5.41293e-06 / 1.18784e-05 | 4/4 | 5.22645e-08 / 1.1669e-07 | 4/4 | 1.40618e-07 / 2.29424e-07 |
| J6_plus_3deg | 3/4 | 3.56031e-06 / 1.12594e-05 | 4/4 | 6.71481e-08 / 6.20824e-08 | 4/4 | 8.9477e-08 / 2.3195e-07 |
| combined | 3/4 | 3.9142e-06 / 7.50234e-06 | 4/4 | 1.36775e-07 / 8.83791e-08 | 4/4 | 9.68578e-08 / 8.84094e-08 |
| diverse_fk_only | 3/4 | 1.06615e-06 / 1.05782e-05 | 4/4 | 9.65706e-08 / 1.54705e-07 | 4/4 | 9.09459e-08 / 2.3036e-07 |

### MuJoCo 远初值失败

这些目标由自身 FK 构造，已知存在合法解；失败反映当前局部迭代的初值敏感性，不是全局不可达的证明。失败返回 q=None。以下为求解器保存的内部末次残差，未暴露可执行的失败关节候选。

| 目标 | 迭代次数 | 末次位置残差 m | 末次姿态残差 rad | 原因 |
|---|---:|---:|---:|---|
| home | 65 | 0.0014244334 | 0.026231517 | Stalled: singular, joint-limited or locally unreachable target |
| J1_plus_3deg | 62 | 0.0048091042 | 0.053030468 | Stalled: singular, joint-limited or locally unreachable target |
| J2_plus_3deg | 64 | 0.00092028029 | 0.0055134956 | Stalled: singular, joint-limited or locally unreachable target |
| J3_plus_3deg | 59 | 0.00098765558 | 0.014538474 | Stalled: singular, joint-limited or locally unreachable target |
| J4_plus_3deg | 62 | 0.00096949068 | 0.030027828 | Stalled: singular, joint-limited or locally unreachable target |
| J5_plus_3deg | 67 | 0.00085324921 | 0.033142797 | Stalled: singular, joint-limited or locally unreachable target |
| J6_plus_3deg | 68 | 0.00082396266 | 0.031515913 | Stalled: singular, joint-limited or locally unreachable target |
| combined | 71 | 0.0039481676 | 0.055806337 | Stalled: singular, joint-limited or locally unreachable target |
| diverse_fk_only | 150 | 0.046225088 | 0.50672023 | Maximum iterations reached |

## 多解、初值与角度跳变

两个配置在 home 及 diverse 目标上，4个初值各选出了2种不同肘分支，不是仅差整数圈。以下示例为 SF，所有分支均以厂家 FK 验证同一个目标，不能按逐轴角度不等判错：

| 目标 | 分支 | J1..J6 ° |
|---|---:|---|
| home | 1 | [  0.000003 -19.999983  29.99996   -0.000004  20.000023   0.      ] |
| home | 2 | [  0.000003  21.280172 -49.337303  -0.000001  58.05714    0.      ] |
| diverse_fk_only | 1 | [ 25.000004 -44.999996  54.999996  19.999994  39.999992 -30.000008] |
| diverse_fk_only | 2 | [ 25.000004  22.496458 -74.337357  12.893146  99.847023 -12.178815] |

全解接口各返回8个分支：home中含±180° J1/J4等超限分支，过滤后2个；diverse中8个经合法周期表示及FK复核通过。home/far两种参考初值都单独记录全解。原始角度代表值会随初值变化，不能只检查一次原始输出便假定全解自动满足限位。

在这两组目标和四个初值的样本中，单解与已验证全解中最接近初值的 L2 距离一致到1e-4 rad量级。此为经验结果，不推断厂家闭源权重、所有姿态下的全局最优性或真实运动距离。

持续沿8个相邻小变化目标、以上一解为下一初值，SF最大单轴变化约3.000036°，ISFV约3.000025°，未发现大分支跳变。但独立 far 初值实验中单轴变化最大约119.847°；IK成功本身不限制关节移动幅度。这里都是角度集合的离线比较，不是速度/加速度或无碰撞轨迹验证。

## 边界与不可达目标

对每轴上下限，在双方交集内侧0.05°生成 FK 目标，初值再向内0.1°。不做接触检测或积分，不把这些极限姿态称为已验证安全姿态。

| 求解器 | 边界自模型成功 | 主动拒绝的明显不可达 [10,10,10] m |
|---|---:|---|
| MuJoCo | 12/12 | conservative reach bound；q=None，0次迭代 |
| RM_MODEL_RM_SF_E | 11/12 | 单解/全解 SDK code=1；q=None / 空候选 |
| RM_MODEL_RM_ISFV_E | 11/12 | 单解/全解 SDK code=1；q=None / 空候选 |

### J5 合法边界目标：单解失败而全解成功

两配置都在 `J5_lower_inside_0.05deg` 失败：来源 q=[0,-20,30,0,-177.91069117,0]°。距模型负限位内侧0.05°，距厂家负限位约0.08931°，双方均合法；目标来自各自厂家 FK。

- 初值 q5=-177.81069117°（其余同来源）：单解code=1。
- 初值正好等于来源q：仍code=1。取同一单位四元数q或-q，不改变失败。
- 初值 home：单解code=0，转到另一个肘分支，约 `[0,21.28019,-49.33734,0,-139.85355,0]°`，再次厂家FK通过。
- 原近邻初值调用全解：code=0，8个分支中2个满足限位与FK，其中一个接近原来源姿态，另一个为上述肘分支。
- 额外以单步模式False复查该近邻初值，两配置仍返回1；复查后恢复True。主实验固定True，未用模式切换覆盖失败结果。

这排除了目标全局不可达，并说明失败与单解初值/选支处理有关。可能涉及近腕部退化或限位处理，但**闭源库内部原因未确定**，不能宣称已定位具体实现缺陷。没有改变模式、DH、目标或验收阈值来抹掉失败；`boundary_failure_probes` 独立保存其他实验，原始计数仍是11/12。没有实现“单解失败便自动全解选一个”的控制策略。

此外，用mock验证非零返回码、NaN/Inf、错长度、code=0但FK误差过大、错误全解数量等防护。这些是测试注入反例，**不是声称SDK实际出现过这些错误成功输出**。本批真实成功输出没有发现超过规定残差的异常。

## 模型交叉诊断（不是公平 IK 精度比较）

仅对home、combined、diverse做双向诊断。使用已知 `T_world_base` 进行坐标变换（当前为单位阵）；**没有已标定法兰/ee_site的TCP变换，明确记为None**。没有拟合变换或加入人为偏移。

1. 厂家FK目标→经base/world变换给MuJoCo IK→分别用MuJoCo FK和厂家FK评估同一所得q。
2. MuJoCo FK目标→经world/base变换给厂家IK→分别用厂家FK和MuJoCo FK评估同一所得q。

这些数值目标属于不同名义末端参考点，不能拿其成功率评价谁更擅长求解同一物理目标。它只说明解与其模型绑定，且不能互换后假定另一末端仍正确。没有把任一结果送入执行器。

| 候选 | 来源姿态 | 输入方向 | 求解自身残差 m / rad | 另一模型残差 m / rad |
|---|---|---|---|---|
| RM_MODEL_RM_SF_E | home | vendor_pose_to_mujoco_ik | 5.2971379e-07 / 1.3838616e-06 | 0.023002374 / 0.0015837281 |
| RM_MODEL_RM_SF_E | home | mujoco_pose_to_vendor_ik | 1.1164311e-07 / 1.1364274e-07 | 0.023002295 / 0.0015836189 |
| RM_MODEL_RM_SF_E | combined | vendor_pose_to_mujoco_ik | 9.7670162e-07 / 1.6178149e-06 | 0.023001823 / 0.0015835116 |
| RM_MODEL_RM_SF_E | combined | mujoco_pose_to_vendor_ik | 2.4589595e-08 / 4.3474134e-08 | 0.023002004 / 0.0015835689 |
| RM_MODEL_RM_SF_E | diverse_fk_only | vendor_pose_to_mujoco_ik | 3.0561892e-07 / 2.8148303e-06 | 0.023050717 / 0.0015848221 |
| RM_MODEL_RM_SF_E | diverse_fk_only | mujoco_pose_to_vendor_ik | 8.0785632e-08 / 9.7853385e-08 | 0.023045804 / 0.0015842317 |
| RM_MODEL_RM_ISFV_E | home | vendor_pose_to_mujoco_ik | 1.8899561e-07 / 1.739712e-07 | 0.011502437 / 0.0015837994 |
| RM_MODEL_RM_ISFV_E | home | mujoco_pose_to_vendor_ik | 1.1175412e-07 / 1.2234791e-07 | 0.011502667 / 0.0015837837 |
| RM_MODEL_RM_ISFV_E | combined | vendor_pose_to_mujoco_ik | 4.353803e-07 / 4.2899823e-07 | 0.011502167 / 0.0015836352 |
| RM_MODEL_RM_ISFV_E | combined | mujoco_pose_to_vendor_ik | 2.0308193e-08 / 1.789737e-07 | 0.011502287 / 0.0015837556 |
| RM_MODEL_RM_ISFV_E | diverse_fk_only | vendor_pose_to_mujoco_ik | 5.0980305e-08 / 5.0304815e-07 | 0.011549668 / 0.0015843185 |
| RM_MODEL_RM_ISFV_E | diverse_fk_only | mujoco_pose_to_vendor_ik | 9.6780664e-08 / 2.1145891e-07 | 0.01154709 / 0.0015841702 |

差异量级与02-C1的末端长度/旋转近似证据一致；没有发现新的单位或坐标转换错误。新增发现主要是单解边界的初值相关失败、全解原始周期表示随q_in变化，以及对双方细微限位差异的显式拒绝。传感器映射、实物代际、机械零位、真实TCP和实机校准DH仍未确认。

## 自动化与文件职责

原有54项全部通过，新增21项全部通过：

```text
Ran 75 tests in 28.354s
OK
```

新增覆盖：矩阵/单位四元数近pi往返、非法输入与可选依赖缺失、整数圈不裁剪（包括整数输入）、自身FK往返、局部失败有界、多解与全解限位/FK、重复性计时、连续小变化、失败缓冲区丢弃、伪成功拒绝、当前模型限位门禁、J5边界失败保留、配置切换、交叉诊断标识、从其他目录运行CLI。数值审计通过不代表所有已知可达案例成功，更不代表两模型一致。

新测试还拦截通信初始化、连接、RoboticArm构造、运动API、mj_step、run_segment和SDK DH设置，确认完整对照仍可运行；正式资产字节和独立live数据qpos/qvel/ctrl不变。现有资源校验与重新生成回归继续通过。没有安装SDK的模拟检查中，4项通用测试通过、17项厂家IK测试明确跳过。

- 新增 `scripts/realman_ik.py`：复用FK配置、官方单解/全解、输入编码、返回码/限位/厂家FK复核、结构化结果；不实现厂家运动控制或自己的解析IK。
- 新增 `scripts/compare_ik.py`：自模型目标集、四初值、计时、多解/连续/边界/不可达和独立交叉诊断。不会改写失败结果或调用控制器。
- 新增 `tests/test_ik_comparison.py`：21项计算及防护测试；既有测试未改。
- 新增本文；修改 README 添加结果、运行入口、失败语义和可选SDK说明。
- `outputs/ik_comparison.json`：完整每姿态/初值目标、关节解、位置/旋转、限位、返回码、重复耗时、全解及失败诊断。另有 `outputs/ik_comparison.log`、`outputs/tests_02_c2.log`、`outputs/tests_ik_only.log`；均由现有Git规则忽略。

没有改 `kinematics.py`、`realman_fk.py`、`compare_fk.py`、`control_ee.py`、模型生成器、机器人资源、控制器、依赖锁定文件。没有新增GUI或重新验证Viewer。

## 复现与退出语义

```bash
conda activate realman_rl
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_ik.py
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_ik.py --sensor SF --repeats 5
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/compare_ik.py --sensor ISFV --repeats 5
python -I -m unittest discover -s /home/mr-beaver/Documents/robot/realman_rl_simtoreal/tests -v
```

默认同时顺序验证两配置；`--repeats` 为1–20，`--json PATH` 可指定输出。`-I` 隔离ROS PYTHONPATH与用户包，资源定位不依赖工作目录。若缺少可选SDK，按 requirements-vendor.txt 在realman_rl中安装即可，不能强制降级现有MuJoCo依赖。

退出码0：实验完成且重复性、数值、限位与失败处理审计通过；**不表示所有IK成功或厂家与MuJoCo模型一致**。退出码1：SDK/文件/运行错误；2：重复性或成功解复核等审计异常。`safety_audits_passed` 仅表示这些计算防护，不包括碰撞路径安全或真机安全。原有compare_fk的“2=模型不一致”退出语义保持原样。

## 后续条件与未解决事项

已有基础具备继续设计**纯MuJoCo仿真**Gymnasium环境原型的条件，可以继续依赖现有MuJoCo FK/IK与六轴控制；不需要等待真实TCP标定才能研究仿真接口。但厂家IK尚不能作为当前MuJoCo求解器的可互换替代，更不能据此认定sim-to-real几何、真机控制或安全条件已经满足。

厂家全解的显式筛选策略、近限位单解局部失败原因、完整工作空间与奇异姿态覆盖、路径碰撞检查、真实型号/TCP/零位/校准参数均未解决。IK成功只代表该模型中的终点位姿条件，不保证从初态到解的路径无碰撞或可执行。

本次没有实现Gymnasium、强化学习、厂家真机接口或新的轨迹规划；没有自动提交/推送GitHub。任务到此停止。
