# RML63II-6FB 资产与实机参数

## 型号核实

控制器只读查询返回 `arm_model=3`（`RM_MODEL_RM_63_II_E`）、`force_type=3`（`RM_MODEL_RM_ISF_E`）、六轴、第三代控制器。SDK 高层显示 `RML_63`，其字符串转换把多个构型合并。实机报告的产品型号为 `RML63-6FB`。官方 6FB 本体没有 `-V` 相机附件。项目默认采用 `RML63II-6FB`。[厂家构型与 MDH](https://develop.realman-robotics.com/robot/robotParameter/RML63OntologyParameters/) · [SDK 枚举](https://develop.realman-robotics.com/robot/apic/type/) · [控制器回包](../../../outputs/validation/rml63ii_6fb/controller_readback.json)

官方 II 型 `d1` 为 172 mm，III 型为 162.5 mm。同为 II 型时，6FB、6FB-V、6F-V 的原始 J6 轴向长度分别为 132.3、154.8、166.6 mm。实机读回的 `d6` 为 **132.2 mm**。工作系 `Base`、工具系 `Arm_Tip` 均为单位变换，近零位回包对应基座系法兰。用实机 DH 计算时，两组近零位样本的位置差约 0.0051 mm 和 0.0093 mm。这个对照范围是两组近零位，工作空间内的真机绝对精度需要更多实测位姿。[读取 DH 的厂家接口](https://develop.realman-robotics.com/robot/apipython/classes/tipVelocityParameters/)

## 原始资源与生成

- 上游 [RealManRobot/rm_models](https://github.com/RealManRobot/rm_models) 固定提交 `776c1a7dd668ba67bfb03636015c59c8ef6e1b43`，路径为 `RML63/urdf/RML63II/RML63II-6FB`。
- `upstream/` 中的 URDF、七个 STL、`package.xml` 和许可证保持该提交的原始字节。读取官方 Git 对象后逐项核对 blob SHA-1，`SHA256SUMS` 保留相应 SHA-256。
- `rml63_mujoco.urdf` 使用本地 mesh 路径，保留上游质量、惯量、碰撞网格和力矩字段。关节原点与机械零位旋转取控制器读回的改进 DH。`lower/upper/velocity` 来自回包的度与度每秒，转换为弧度与弧度每秒。模型有七个连杆和六个转动关节。
- `rml63_dynamics.xml` 将兼容 URDF 转为带六个位置伺服的 MJCF。已有 `home` 关键帧用于 Reach。桌面完整任务仍实际从零角启动。设备的打包关节角未读到，未设为仿真关键帧。
- 动力学场景保留原有的 `base_link` / `link_1` 同轴装配排除。它只消除原始凸包在连接处的 2.73 mm 重叠。其余自碰撞、桌面、夹爪和物体碰撞按任务的原规则计算。

备份的旧 `RML63II-6F-V` 资产位于 [历史资产归档](../../../outputs/validation/rml63ii_6fb/previous_rml63ii_6f_v_assets.zip)，可由项目提交 `dd97ad28d8bf7824b1a49f98e1524ba71cee4b0c` 中的 `assets/` 重现。原训练产物和旧源代码的提交历史可用于复现对应实验。

在 `realman_rl` 环境中，从仓库根目录依次生成。

```bash
python -I scripts/models/build_dynamics_model.py
python -I scripts/models/build_gripper_model.py
python -I scripts/models/build_grasp_scene.py
python -I scripts/models/build_tabletop_scene.py
python -I scripts/models/build_tabletop_cartesian_scene.py
python -I scripts/models/inspect_model.py
```

## 坐标系及安装

- `world` 是 MuJoCo 世界系。裸臂的 `base_link` 与世界系重合。桌面场景的基座原点在世界系 `(0, 0, 0.75)` m，方向一致。
- `ee_site` 位于 `link_6` 原点，方向为法兰方向，对应控制器单位 `Arm_Tip`。仿真 FK/IK 的角度为 rad，返回世界系位置（m）和 site 到世界系的旋转。与控制器对照时使用基座系。
- `gripper_mount` 在法兰局部 `+Z` 8 mm，夹爪坐标方向与法兰一致。桌面 `grasp_site` 用 20 mm 开口下两指垫中点定义，笛卡尔任务通过该点做 FK/IK。法兰位姿和抓取 TCP 不可互换。[夹爪安装](../../grippers/eg2_4c2/SOURCE.md)
- `camera_link` 是视觉任务的独立刚性坐标系。其原点在法兰系 `(0.0845, 0.0325, -0.03552)` m，绕法兰 `+Z` 旋转 90°。该姿态沿用旧仿真光心的几何关系，场景只安装光学系，不引入未经测量的相机外壳和质量。`wrist_rgb` 在 `camera_link` 原点。RGB-D 模块换算 MuJoCo 光学方向后，将光学系三维点变换到基座系。

夹爪的实际安装变换、真实抓取 TCP 和相机手眼外参仍需实测。质量与惯量沿用 CAD，伺服参数用于仿真。

## 观察位与验证

`table_down` 的世界系抓取 TCP 目标为 `(0.10, 0, 1.10)` m，六轴重算为 `(0, -31.72666510, 91.96456974, 0, 119.76209534, 90)` 度。`forward` 也已基于 6FB 重算。两条路径实际从零位执行，分别用时 22.70 s 和 25.80 s，启动期间策略动作数为零。[观察位预设](../../../src/realman_rl_simtoreal/simulation/runtime/observation_presets.py)

`compare_fk.py --hardware-profile` 同时记录控制器读回的 MDH 对照和离线 SDK 名义对照。离线 SDK 1.1.7 的 `d6` 为 132.3 mm，因此名义法兰仍有 0.1 mm 长度差。`compare_ik.py --hardware-profile` 用各自 FK 生成九个目标位姿。MuJoCo DLS 从 home、邻近和另一肘形初值出发共通过 27/27，从远初值出发 0/9。名义 SDK 为 36/36，另有一个接近 J5 下限的原种子失败，切换为 home 初值后通过。不可达目标和结果限位审计通过。[FK 数据](../../../outputs/validation/rml63ii_6fb/fk.json) · [IK 数据](../../../outputs/validation/rml63ii_6fb/ik.json) · [安装与模型验证](../../../outputs/validation/rml63ii_6fb/verification.json)
