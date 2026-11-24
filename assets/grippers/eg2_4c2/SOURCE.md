# EG2-4C2 来源、转换与实验装配

## 准确型号与原始资源

实际资产明确名为 **EG2-4C2**，不是把其他二指夹爪改名。来源：[RealManRobot/rm_models](https://github.com/RealManRobot/rm_models)，提交 `776c1a7dd668ba67bfb03636015c59c8ef6e1b43`，获取日期2026-11-07。原目录：`thirdparty/Two-finger Electric Gripper.urdf/`。URDF的robot名称、package名称及CAD导出CSV均为EG2-4C2，包版本1.0.0。

`upstream/`保留原始URDF、CSV、7个STL、package.xml、根LICENSE，以及因时目录的Readme和RMU01020安装板PDF，共13个文件。`upstream_manifest.json`记录原路径、固定提交raw URL、字节数、SHA256及Git blob SHA1；下载字节与固定提交树对象逐文件比对。没有下载其他型号、大型CAD或安装ROS。

同一thirdparty目录还调查了：通用静态Two-Finger Electric Gripper.STL、Two-Finger Electric Flexible Gripper.STL、Two-finger Electric Gripper.step；B型及视觉型二指转接板STEP；相机连接板；各厂家的末端适配目录。静态文件没有同等完整的关节语义，柔性夹爪也不能当作EG2-4C2；因此不用于替换本模型。因时目录有明确对应EG2-4C2的RMU01020安装板PDF/STEP，但没有本项目RML63II-6F-V完整附件栈的装配URDF。

许可证原文：仓库根LICENSE声明Apache-2.0；模型package.xml声明未指明变体的BSD，两者不一致，保留原文而不重新判定授权。原文件从不由生成器修改。

## 物理真实性边界

原始7个连杆质量合计 **0.0942992kg**。因时官方[产品选型手册V17](https://en.inspire-robots.com/wp-content/uploads/2024/04/Inspire-Robots-Products-Selection-Guide-V17.pdf)及任务给定参数为231g、双指总行程70mm、最大夹持力20N、全行程闭合约1.3s。现有URDF少于产品重量136.7008g，原因未确认，可能未包含完整驱动件。没有把缺失质量任意分摊到连杆，也没有缩放惯量去凑231g。

原URDF的0..0.82rad几何只对应约 **66.2173566mm** 的本项目开口定义，未达到70mm。没有扩大关节限位或修改STL。官方标称和仿真不符必须保留。原effort=1、velocity=1及SimpleTransmission reduction=1不能当作实物电机/传动标定。真实电机、减速器、摩擦、自锁、压力控制和实际装配均未确认。

## 可重复转换

在realman_rl editable环境运行：

```bash
python -I /home/mr-beaver/Documents/robot/realman_rl_simtoreal/scripts/build_gripper_model.py
```

生成 `eg2_4c2_mujoco.urdf`（仅本地mesh路径和明确的MuJoCo编译选项）、`eg2_4c2.xml`（独立夹爪）及 `rml63_eg2_4c2.xml`（实验组合场景）。读取和校验原始文件后用官方MuJoCo加载/导出；不启用balanceinertia，不从mesh重新估算惯性。导出后从编译的原模型恢复17位body位姿、惯性及关节数值，避免文本舍入。现有RML63资产全部原字节不变。

保留7个body、6个hinge及原STL。MuJoCo3.17.0自动把URDF的5条mimic转成joint equality，本生成器只命名和设置数值求解参数，不再重复添加一套耦合。

## 关节、开口与参考点

令q=4C2_Joint1。Joint2、Joint5、Joint6=q，Joint3、Joint4=-q；局部轴和连接原点均保留URDF。Joint1/2/5/6范围[0,.82]rad，Joint3/4范围[-.82,0]rad。6个坐标由5条柔性等式约束耦合为一个开合输入，不是6个独立驱动器。

开口定义在两个夹指末端平面上的标记之间，沿夹爪base的X方向投影。左Link5局部点[+.0177,.006,.027]m、右Link6局部点[-.0177,.006,.027]m位于STL末端平直接触带；这不是厂家TCP。理想mimic关系下：

`width(q) = 2 * [0.0223*(1-cos(q)) + 0.035591*sin(q)]`

长度来自原Joint5/6连接偏移和接触带坐标；0rad为闭合、0.82rad为最大模型开口。API用此单调关系反求角度；实际读数使用独立MjData中的两个实际pad site位置，包含mimic约束残差，绝不通过改写实时qpos伪造运动。

## 仿真安装关系

当前 `RML63II-6FB` 的 `ee_site` 位于 `link_6` 原点，方向对应法兰。腕部光学坐标系由桌面视觉场景单独配置。[本体与光学坐标系](../../robots/rml63/SOURCE.md)

RMU01020 图给出板外径 57 mm、总厚 8 mm、重量栏 0.011 及孔位。仿真按 11 g 计算，实物尚未称重。转接件与本机法兰的贴合深度、角向定位和螺孔匹配需要现场核查。

仿真采用同轴装配。在 `link_6` 下加固定的 8 mm 安装板代理，再将 EG2 底座沿法兰局部 +Z 放置 8 mm，旋转为单位阵。桌面场景的 `grasp_site` 使用 20 mm 开口下的夹垫中点。

- `T_ee_mount`：平移[0,0,.008]m，R=I；gripper_mount与4C2_baselink重合。
- `T_mount_grasp`：平移[0,0,.147530235...]m，R=I。grasp_site固定在名义q=.41rad时两末端标记的中心。
- `T_ee_grasp`：平移[0,0,.155530235...]m，R=I。

四连杆使实际夹指中心高度随开口变化：闭合时约.141591m、最大开口时约.146585487m（均相对mount），所以固定grasp_site只是名义抓取参考。未来若要精确TCP，应根据开口标定或使用 `Gripper.pad_positions()` 的实际中点；当前FK/IK和ReachEnv仍使用原ee_site，不把名义点伪称标定TCP。

安装板代理采用直径57mm、高8mm的圆柱，假定质量11g，均匀圆柱惯量近似；未复刻孔、沉孔和凸台，惯量未经厂家确认。它是新增固定body，不融合或修改原机械臂惯量。模型新增质量为105.2992g，与产品参考231g夹爪加同一假定板重的242g仍有明显差距。

## 执行器与动力学选择

保留原六轴执行器，新增一个 `eg2_position`，直接作用Joint1，gear=1，位置单位rad。模拟PD：kp=.05N·m/rad、kv=.002N·m·s/rad；六个夹爪joint各加.0001N·m·s/rad阻尼。参数是低增益仿真整定，不是厂家驱动器参数。

最初较强增益使轻小连杆的柔性mimic约束产生越限；提高solver精度没有改善。降低夹爪增益/阻尼后通过原位置/速度检查，不通过增加虚构转子惯量或放宽检查解决。mimic求解使用solref=[.004,1]、solimp=[.999,.999,.001]；保留数值约束残差，非理想刚性机械传动。

夹爪执行器和主joint的总驱动力限幅均为±.2N·m，比URDF的±1更保守。其他夹爪joint保留原±1N·m字段。**20N夹持力不是关节力矩，未直接设置成任何驱动力限幅**，当前模拟不代表20N夹持能力。

命令只按.5rad/s缓慢更新执行器参考，低于原velocity=1字段；全角度参考行程1.64s，加上动力学跟踪时间，不声称复现真机1.3s。正常运动全部经mj_step。

加载组合模型时可选原重力补偿。仅遍历link_1后代，包括新增夹爪及安装板；其他场景物体不补偿。所有机器人运动joint路由actuatorgravcomp，使原六轴总驱动力继续受原限幅，夹爪主驱动力也经总限幅；没有用被动力或外力注入旁路。关闭补偿仍能加载原始PD场景。

独立eg2_4c2.xml默认在编译前启用自身的重力补偿并路由至joint执行器限幅。较低增益未对未知传动重力负载作真实辨识；在开端点直接无补偿启动曾触发越限保护。因此独立开合使用已补偿配置，而不是关闭重力。组合XML保留PD基线，可通过load_model(..., gravity_compensation=True)显式开启；演示默认开启。未补偿夹爪的跟踪精度不在本次通过范围。

## 碰撞

视觉：7个原STL的group1、contype=conaffinity=0；接触：7个独立命名的mesh geom，保持contype=conaffinity=1。MuJoCo使用凸包，会填充STL凹槽，因此还不是精确指垫接触模型。摩擦[.7,.005,.0001]为一般仿真值。

直接转换在关节端点报告壳体/连杆约8.7–13.6mm的凸包重叠，及原树未明确闭环连接的连杆/夹指约10.5mm包络重叠。仅排除下列8对内部机构配合接触，未移动几何来消除它们：

- base与Link1、2、3、4：相邻铰接/壳体内运动。
- Link1/Link2、Link3/Link4：同侧平行四连杆配合包络。
- Link2/Link5、Link3/Link6：原URDF依靠mimic描述、未作为树父子的机构连接。

保留左右夹指之间、所有夹爪与外物、机械臂与外物及非相邻机械臂碰撞。没有全局关闭接触或删除碰撞几何。后续真实接触精度需要凸分解/接触标定，而非直接把本凸包当作实物表面。

## 后续接口边界

当前 `set_width(m)`、`get_width()`、`open()`、`close()` 仅在MuJoCo中工作。已只读检查安装SDK的rm_set_gripper_position文档：位置为1..1000、无量纲。可在未来建立米制开口适配层，但设备量与真实开口/行程的函数、方向、端点和标定尚未确认；不能把 `1+999*width/MAX_WIDTH` 当作已验证的真机线性映射。没有创建连接、调用运动接口或修改SDK。

本模型支持后续加入物体开展接触实验；当前只验证空载开合及固定障碍物接触反例，不代表已实现抓取、力控或Pick环境。验证结果见 `docs/validation/GRIPPER_VALIDATION.md`。
