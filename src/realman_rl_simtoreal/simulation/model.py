"""Load/validate the fixed-base six-axis model and create independent data."""

from pathlib import Path

import mujoco
import numpy as np

from ..paths import DEFAULT_SCENE as SCENE

JOINTS = tuple(f"joint_{i}" for i in range(1, 7))

def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_model(path=SCENE, *, gravity_compensation=False):
    path = str(Path(path).expanduser().resolve())
    if gravity_compensation:
        # Configure before compilation: merely editing body_gravcomp on an
        # already compiled baseline leaves its compiled gravity metadata disabled.
        spec = mujoco.MjSpec.from_file(path)
        for name in (*[f"link_{i}" for i in range(1, 7)], "camera_rolink", "camera_link"):
            spec.body(name).gravcomp = 1
        for name in JOINTS:
            spec.joint(name).actgravcomp = True
        model = spec.compile()
    else:
        model = mujoco.MjModel.from_xml_path(path)
    require((model.nq, model.nv, model.nu, model.njnt) == (6, 6, 6, 6), "Expected six arm hinges and six actuators")
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
    for name in ("camera_rolink", "camera_link"):
        require(model.body_weldid[model.body(name).id] == model.body_weldid[model.body("link_6").id],
                "Camera must be rigidly attached to link_6")
    require(np.allclose(model.opt.gravity, [0, 0, -9.81]), "Expected Earth gravity")
    require(not int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT), "Contact detection disabled")
    return model, joint_ids, actuator_ids


def initialize(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return data


def gravity_torque(model, q):
    """g(q): zero-velocity bias in private data; live moving bias is not pure gravity."""
    q = np.asarray(q, dtype=float)
    require(q.shape == (model.nq,) and np.isfinite(q).all(), "Expected finite model coordinates")
    probe = mujoco.MjData(model)
    probe.qpos[:] = q
    probe.qvel[:] = 0
    mujoco.mj_forward(model, probe)
    return probe.qfrc_bias.copy()

