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
