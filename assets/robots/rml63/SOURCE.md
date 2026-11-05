# RML63 模型来源与兼容修改

## 固定来源

- 上游：[RealManRobot/rm_models](https://github.com/RealManRobot/rm_models)。
- 固定提交：`776c1a7dd668ba67bfb03636015c59c8ef6e1b43`（2026-10-30，`doc: Add new model files for RX75S`）。这不是 RML63 专用发布标签。
- 获取日期：2026-11-04（Asia/Shanghai）。
- 原始目录：[`RML63/urdf/RML63II/RML63II-6F-V`](https://github.com/RealManRobot/rm_models/tree/776c1a7dd668ba67bfb03636015c59c8ef6e1b43/RML63/urdf/RML63II/RML63II-6F-V)。
- URDF 的 robot 名称：`RML63II-6F-V`；`package.xml` 版本：`1.0.0`。
- 获取方法：按固定提交的 raw URL 下载 URDF、9 个被引用的 STL、`package.xml` 和仓库根 `LICENSE`，共 12 个文件、10,369,980 字节。逐文件计算 Git blob SHA-1，与本机干净的官方仓库副本在上述提交中的树对象比对通过。没有获取 CAD、其他型号 mesh、ROS 启动文件或夹爪。
- `upstream/` 全部保持上游原始字节；`SHA256SUMS` 保存这 12 个文件的 SHA-256。运行检查脚本时自动校验。

## 型号区别与匹配边界

本项目目标称为 **RML63-6F-V**，当前选用的实际资产为 **RML63II-6F-V**，经用户确认用于第一阶段导入和静态检查。尚未根据实物铭牌、序列号或厂家资料确认其代际对应关系，不能据此宣称与实物完全匹配。

固定提交包含以下完整 URDF 变体：

- II 型：`RML63II-B`、`RML63II-B-V`、`RML63II-6F`、`RML63II-6F-V`、`RML63II-6FB`、`RML63II-6FB-V`。
- III 型：`RML63III-B`、`RML63III-B-V`、`RML63III-6F`、`RML63III-6FB`、`RML63III-6FB-V`；没有同名 `RML63III-6F-V`。
- 从这些 URDF 可直接核实：II 型 `joint_1` 原点高度为 0.172 m，III 型为 0.1625 m；质量和惯量也存在版本差异，不能只改名称来切换代际。
- `-V` 文件包含相机支架和相机连杆；非 `-V` 文件只有底座与六个机械臂连杆。保留的相机是几何和惯性模型，并未创建 MuJoCo 图像传感器。
- 官方第三代参数说明区分 B 标准版、6F 分体式六维力版、6FB 一体式六维力版。对应 URDF 末端长度也不同：II-6F 为 0.1436 m，II-6FB 为 0.1323 m；带 V 后分别为 0.1666 m 和 0.1548 m。此处数值是 `joint_6` 的原点偏移，不直接当作 TCP 标定值。

参考：[官方第三代参数说明](https://develop.realman-robotics.com/en/robot/robotParameter/RML63OntologyParameters/)、[第四代参数说明](https://develop.realman-robotics.com/robot4th/robotParameter/RML63OntologyParameters/)。第四代页面的 RML63-6F-V D-H 数据列出 d1=162.5 mm、d6=155.1 mm，与当前 II 型 URDF 不一致；没有用网页参数覆盖 URDF。上述网页为获取当天参考资料，版本可能变化，固定提交才是本项目资产的可复现来源。

## 原始结构和参数

- 9 个 link，8 个 joint；`joint_1` 至 `joint_6` 是六个机械臂 revolute 关节。
- `camera_rojoint` 也是 revolute，范围 `[0, 1.05]` rad，但 effort 和 velocity 都是 0。这是上游相机支架附件定义，不将它解释成第七个机械臂驱动轴，也不擅自改成 fixed。
- `camera_joint` 为 fixed。MuJoCo 中固定连接不占运动 joint，因此兼容模型为 7 个 hinge，`nq=7, nv=7, nu=0`；保留固定 Body 后 `nbody=10`（包含 world）。
- 质量总和为 10.353902946 kg（包含固定底座和相机附件）。官方宣传中的标准版/六维力版自重不能直接作为此带附件 URDF 的验收值。
- 主轴局部轴均为 `[0, 0, 1]`；不同关节的局部坐标系不同，不表示其世界坐标轴相同。限位和 URDF 中的 RPY 近似值均原样保留，没有将 3.14 自动替换为 π。
- 原始 URDF 的关节 effort/velocity 不会自动创建 MuJoCo 执行器或运行时限速。六维力版本的几何也不等于已经实现力/力矩传感器。

## 兼容副本的完整修改范围

`rml63_mujoco.urdf` 仍是 URDF，不是 MJCF。与原始文件相比只有：

1. 将所有 `package://RML63II-6F-V/meshes/` 路径前缀改为相对于副本的 `upstream/meshes/`。
2. 增加注释及 MuJoCo compiler 扩展：`strippath=false`、`discardvisual=false`、`fusestatic=false`、`inertiafromgeom=false`、`balanceinertia=false`。保留视觉几何和固定连杆，明确使用原始惯量，不从 mesh 重新估算，不自动平衡惯量。
3. 对已复现的 `camera_link` 惯量异常执行用户批准的单项舍入修正，详见下节。

通过 XML 语义比对验证：撤销以上三项后，节点、属性与原始 URDF 一致。没有调整坐标变换、质量、其他惯量、关节类型、轴、限位或 mesh 数据；没有添加执行器、夹爪、地面或控制逻辑。

## 惯量异常：复现、修正及限制

在 MuJoCo 3.17.0 中，原始文件先报 mesh 路径错误；仅解析 mesh 路径后仍失败：

```text
Error: inertia must satisfy A + B >= C; use 'balanceinertia' to fix
Element name 'camera_link', id 9
```

原始相机固定连杆惯量矩阵的最小两项特征值之和比最大值小约 `3.000000073e-13 kg*m^2`。数值与导出精度相当，疑似舍入误差，但原因尚未获得厂家确认。

只在兼容副本修改 `camera_link` 的 `izz`：

- 原值：`4.9797e-08 kg*m^2`。
- 修正值：`4.979669e-08 kg*m^2`。
- 差值：`-3.1e-13 kg*m^2`，相对变化约 `-0.000622527%`。
- 原始 `ixx + iyy = 4.97967e-08`。修正值略小于该和；考虑原始非零交叉项重新计算特征值后，三角不等式余量约 `9.99999e-15 kg*m^2`，全部特征值为正。

此修正只为获得可加载的静态检查模型，**不是厂家测量值或动力学标定值**。原始文件未修改，检查脚本始终报告该原始异常。不得将此规则自动应用于其他异常，也未使用 `balanceinertia=true`、质量缩放或 mesh 惯量推断。

## 许可证记录

- 上游仓库根 `LICENSE` 声明 `Copyright 2024 realman-robotics` 和 Apache License 2.0；原文保存在 `upstream/LICENSE`。
- 同一模型的原始 `package.xml` 写有 `<license>BSD</license>`，未指定 BSD 变体，且含 TODO 作者/维护者字段。
- 两种声明存在不一致；本项目保留原文并记录，未自行选择其中一种覆盖另一种，也未给模型重新授权。

当前只验证文件完整性、URDF 导入、结构及数值可计算性。实物代际匹配、惯性参数真实性、碰撞精度、传感器、执行器和动态仿真均不属于已经验证的内容。

零位 `mj_forward()` 得到两个 `base_link` / `link_1` 接触，距离分别约为 `-0.002726524 m`、`-0.002726676 m`。这是当前 MuJoCo 碰撞表示下的穿透，不直接表示真机干涉。原始 STL、碰撞对和过滤设置均未更改；进入动力学阶段之前需要单独检查。

## 2026-11-05：动力学 MJCF 的派生修改

以上原始模型和第一阶段兼容 URDF 均保留，静态脚本行为不变。新增 `rml63_dynamics.xml` 和 `scene.xml`，可用 `scripts/build_dynamics_model.py` 重新生成机器人模型。完整本次验证见项目根目录 `DYNAMICS_VALIDATION.md`。

### 接触的定位与处理

1. 原始 STL 在零位的世界 Z 范围：`base_link` 约 `[0, 0.111300] m`；`link_1` 约 `[0.108000, 0.216720] m`。两者的 Z 包络在关节装配口重叠约 3.3 mm；MuJoCo 实际凸碰撞包络给出约 2.73 mm 穿透，接触位置约为世界 `(0.0395, -0.0122, 0.10936) m`。Z 范围重叠本身不证明原始三角形或实物干涉，不据此更改尺寸。
2. `base_link` 无关节，`body_weldid=0`，属于 world 固定组。MuJoCo 默认过滤同一刚体及相邻运动连杆，但 world 固定组存在例外，所以该对装配几何仍参与接触。其他相邻运动连杆通常已被默认过滤。[官方碰撞过滤说明](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection)
3. J1 在 `-90, -10, 0, 10, 90` 度进行仅前向计算的抽样，原始接触深度都约 2.726–2.730 mm，符合持续装配包络重叠的特征。该扫描用于诊断，未让控制器进行大幅运动。
4. 动力学模型只增加 `<exclude body1="base_link" body2="link_1"/>`，将这对相邻装配接触按常规相邻连杆处理。没有关闭全局碰撞、修改 STL、移动轴或缩放尺寸；非相邻连杆及外物与这两连杆的碰撞仍然保留。
5. 默认保持和控制序列无接触；反例中 `base_link/link_2` 折叠自碰撞仍被检测，放入连杆的独立球体也会被检测并令演示报错。当前仍沿用单个 STL 的凸碰撞包络，未宣称全工作空间碰撞精确。

### 底座、相机及转换精度

- 底座直接挂在 world 下且无 joint，位置和姿态刚性固定，不依赖地面摩擦或数值上很硬的约束。
- 删除派生 MJCF 中 `camera_rojoint`，在原始角度 0 rad 刚性连接 `camera_rolink` 和 `link_6`；`camera_link` 继续固定。保留两个相机 Body 的几何、位姿、质量及惯量，包括第一阶段已批准的微小惯量修正。没有额外的相机执行器或约束漂移。
- 删除该附件自由度后只有六个 hinge，`nq=nv=nu=6`；`nbody=10`，包括 world。`fusestatic=false` 保留连杆对应关系。
- 使用官方 `mj_saveLastXML()` 导出 MJCF，但其文本位数较短；生成器从原始编译模型重新写入 `.17g` 双精度的 Body 位姿、惯性坐标、质量、主惯量及关节参数。模型对比测试确认变换、惯量、几何世界位姿及六轴限位一致。没有新增尺寸或质量估计。

### 位置执行器与参数依据

- 每个 `joint_i` 对应一个 `position_i` 执行器，`gear=1`，输入为弧度，输出为 N·m。采用官方 `position` 的 `kp` 与 `kv`：`tau = kp*(q_target-q) - kv*qvel`；只做 PD，没有积分项或重力补偿。[官方位置执行器说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position)
- J1–J6 的 `kp`：`[500, 1000, 800, 200, 120, 80] N·m/rad`。
- `kv`：`[50, 90, 70, 15, 10, 5] N·m·s/rad`；额外 joint damping：`[1, 1, 0.5, 0.2, 0.1, 0.05] N·m·s/rad`。
- `ctrlrange` 保留各关节原始范围；执行器及关节合力矩限幅沿用 URDF effort：`[±60, ±60, ±30, ±30, ±10, ±10] N·m`。这些是上游模型字段，不是经真机验证的电机能力。
- 在测试姿态 `[0,-20,30,0,20,0]` 度，静态重力力矩约 `[0,2.444,-4.605,-0.093,-0.605,0.00091] N·m`。`kp` 使该姿态的预计重力偏差 `|tau_g/kp|` 小于 0.5°，同时避免通过无限增大增益来消除偏差。
- 以当前姿态质量矩阵对角项粗估 `2*sqrt(kp*M_ii)` 为 `[18.32,83.11,36.65,2.47,2.46,0.73]`。所选速度反馈偏保守，参考此量级并通过正负小角度、末段波动和时间步复测确认；耦合机械臂并非六个独立二阶系统，不能据此宣称全局临界阻尼。
- `gravity="0 0 -9.81"`，`timestep=0.002 s`，`integrator="implicitfast"`。该积分器适合含速度反馈的执行器；另以 0.001 s 复测。[官方速度反馈与积分器说明](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position)
- 没有添加未知电机转子惯量、齿轮参数或摩擦标定。PD 增益和 joint damping 是本项目的仿真调试参数，不是官方实测参数。本次控制脚本只接收六个关节目标，不处理末端位姿或逆运动学。

`scene.xml` 只包含上述机器人、灯光和地面。地面摩擦 `[0.7,0.005,0.0001]` 为通用场景设置；正常测试无地面接触，未作摩擦参数辨识。

## 任务 02-B：末端参考点

在 `scripts/build_dynamics_model.py` 中增加 `ee_site` 的生成逻辑，并重新生成 `rml63_dynamics.xml`。site 直接属于 `link_6`，局部 `pos="0 0 0"`、`quat="1 0 0 0"`；其位置及姿态分别由 `MjData.site_xpos` 和 `site_xmat` 给出，表示世界坐标及 site-to-world 旋转。

这是临时末端坐标系，不使用相机支架，也不是厂家标定的 TCP。绿色球形 site 只是可视化，无质量、惯量或碰撞；没有改变原有 Body、geom、执行器、重力、阻尼、限位或碰撞过滤。重新生成和回归测试均验证 site 保留、物理参数一致。

`control_ee.py` 的半透明红色目标球属于 Viewer `user_scn`，不加入物理模型。FK 与 IK 使用独立 `MjData`；正常运动仍由既有六个 position 执行器通过 `mj_step()` 实现。算法、朝向约束与实际 PD 误差的区别见项目根目录 `KINEMATICS_VALIDATION.md`。

## 任务 02-B+：加载时可选的重力补偿配置

基线 MJCF 及生成脚本不变。`control_joints.load_model(path, gravity_compensation=True)` 读取同一场景到官方 `MjSpec`，编译前只给 `link_1..link_6`、`camera_rolink`、`camera_link` 八个 Body 设置 `gravcomp=1`，并给六个主关节设置 `actuatorgravcomp=true`。底座固定在 world，不补偿；固定相机的重量仍计入上游运动关节。

`actuatorgravcomp` 把补偿计入 `qfrc_actuator`，与 PD 合成后由原 `actuatorfrcrange` 限幅，不能使用默认被动力通道绕过电机总力矩限值。标量执行器还保留各自原始限幅。此配置在编译前应用，避免仅改已编译 Body 数组而未更新重力补偿元数据。补偿模式的 `ngravcomp=8`、六轴路由标志和实际力通道均经过验证；缺省加载仍是 PD，重新加载关闭选项也恢复原模式。

这是确定性的加载配置，不是另一份机器人模型。没有改变任何质量、质心、惯量、几何、关节轴、重力大小、阻尼、执行器增益、时间步或碰撞规则，也不修改实际状态来保持姿态。物理参数一致性与基线重新生成由自动测试验证。完整证据见 `CONTROL_VALIDATION.md`。
