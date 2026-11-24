"""加载并校验固定基座的 RML63 MuJoCo 模型，为执行器和运动学提供模型与关节索引。"""

from pathlib import Path

import mujoco
import numpy as np

from ...paths import DEFAULT_SCENE as SCENE

JOINTS = tuple(f"joint_{i}" for i in range(1, 7))

def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_model(path=SCENE, *, gravity_compensation=False):
    """从场景 XML 编译模型，返回 MjModel、六轴关节 ID 和位置执行器 ID。

    启用重力补偿时在编译前配置机械臂及刚性负载。场景缺少规定的关节、执行器、
    固定基座或力矩限制时抛出 ValueError。
    """
    path = str(Path(path).expanduser().resolve())
    if gravity_compensation:
        # 补偿参与编译出的执行器力矩约束，须在编译前设置关节与刚体标志。
        spec = mujoco.MjSpec.from_file(path)
        def on_arm(body):
            while body is not None:
                if body.name == "link_1":
                    return True
                body = body.parent
            return False
        # 重力补偿覆盖机械臂及其附件，桌面与自由物体仍按原重力积分。
        for body in spec.bodies:
            if on_arm(body):
                body.gravcomp = 1
        for joint in spec.joints:
            if on_arm(joint.parent):
                joint.actgravcomp = True
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(path)
    require(model.nq >= 6 and model.nv >= 6 and model.nu >= 6 and model.njnt >= 6,
            "Expected six named arm hinges and their actuators")
    joint_ids = np.array([model.joint(name).id for name in JOINTS])
    actuator_ids = np.array([model.actuator(f"position_{i}").id for i in range(1, 7)])
    for joint_id, actuator_id in zip(joint_ids, actuator_ids):
        require(model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE, "Expected a hinge joint")
        require(model.actuator_trntype[actuator_id] == mujoco.mjtTrn.mjTRN_JOINT
                and model.actuator_trnid[actuator_id, 0] == joint_id, "Wrong joint/actuator mapping")
        require(np.array_equal(model.actuator_gear[actuator_id], [1, 0, 0, 0, 0, 0]), "Expected direct-drive gear=1")
        require(model.jnt_limited[joint_id] and model.actuator_ctrllimited[actuator_id]
                and model.actuator_forcelimited[actuator_id], "Position and force limits are required")
        require(model.jnt_actfrclimited[joint_id], "Joint total-drive torque limit is required")
        require(np.allclose(model.jnt_range[joint_id], model.actuator_ctrlrange[actuator_id]), "Control/joint limits differ")
    base = model.body("base_link").id
    require(model.body_parentid[base] == 0 and model.body_jntnum[base] == 0
            and model.body_weldid[base] == 0, "Base must be rigidly fixed to world")
    # 纯本体没有相机。带腕部 RGB-D 的场景须把光学坐标系刚性接到法兰。
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_rgb")
    if camera >= 0:
        require(model.body_weldid[model.cam_bodyid[camera]] == model.body_weldid[model.body("link_6").id],
                "Wrist camera must be rigidly attached to link_6")
    require(np.allclose(model.opt.gravity, [0, 0, -9.81]), "Expected Earth gravity")
    require(not int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT), "Contact detection disabled")
    return model, joint_ids, actuator_ids


def initialize(model):
    """创建独立的 MjData，恢复 home 关键帧并刷新派生状态。"""
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def gravity_torque(model, q):
    """在独立静止状态计算重力广义力，q 按模型的 nq 坐标顺序排列。"""
    q = np.asarray(q, dtype=float)
    require(q.shape == (model.nq,) and np.isfinite(q).all(), "Expected finite model coordinates")
    probe = mujoco.MjData(model)
    probe.qpos[:] = q
    probe.qvel[:] = 0
    mujoco.mj_forward(model, probe)
    return probe.qfrc_bias.copy()

