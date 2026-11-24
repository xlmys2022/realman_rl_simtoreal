#!/usr/bin/env python3
"""从官方 6FB 资源与控制器参数生成兼容 URDF 和六轴 MJCF。"""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from realman_rl_simtoreal.hardware.realman_config import REALMAN_HARDWARE
from realman_rl_simtoreal.paths import DEFAULT_URDF, ROBOT_ASSETS as ASSETS, ROBOT_MODEL, UPSTREAM_URDF
from realman_rl_simtoreal.utils.geometry import modified_dh_transform

KP = (500, 1000, 800, 200, 120, 80)  # 仿真位置增益，N·m/rad。
KV = (50, 90, 70, 15, 10, 5)        # 仿真速度增益，N·m·s/rad。
DAMPING = (1, 1, 0.5, 0.2, 0.1, 0.05)


def numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def urdf_rpy(rotation):
    """把子连杆到父连杆的旋转矩阵转换为 URDF 固定轴 RPY（rad）。"""
    cos_pitch = np.hypot(rotation[0, 0], rotation[1, 0])
    pitch = np.arctan2(-rotation[2, 0], cos_pitch)
    if cos_pitch < 1e-12:
        # 俯仰 ±90° 时固定 yaw=0，剩余旋转归入 roll。
        return (np.arctan2(-rotation[1, 2], rotation[1, 1]), pitch, 0.)
    return (np.arctan2(rotation[2, 1], rotation[2, 2]),
            pitch, np.arctan2(rotation[1, 0], rotation[0, 0]))


def build_compatible_urdf():
    """保留厂家 CAD 的几何和惯量，写入实机关节零位变换与限位。"""
    root = ET.parse(UPSTREAM_URDF).getroot()
    if root.get("name") != ROBOT_MODEL:
        raise ValueError(f"需要官方 {ROBOT_MODEL} URDF")
    extension = ET.Element("mujoco")
    ET.SubElement(extension, "compiler", strippath="false", discardvisual="false",
                  fusestatic="false", inertiafromgeom="false", balanceinertia="false")
    root.insert(0, extension)
    root.insert(0, ET.Comment("由 build_dynamics_model.py 生成。厂家原始资源见 upstream/。"))
    for mesh in root.findall(".//mesh"):
        mesh.set("filename", f"upstream/meshes/{Path(mesh.get('filename')).name}")
    config = REALMAN_HARDWARE
    dh = config.controller_dh
    for index in range(config.arm_dof):
        joint = root.find(f"./joint[@name='joint_{index + 1}']")
        # 子系 Z 是转轴。机械零位偏置已进入 origin，FK 输入不再加一次。
        angle = np.deg2rad([dh.alpha_deg[index], dh.offset_deg[index]])
        zero = modified_dh_transform(dh.a_m[index], dh.d_m[index], *angle)
        origin = joint.find("origin")
        xyz = zero[:3, 3]
        origin.set("xyz", numbers(np.where(abs(xyz) < 1e-15, 0., xyz)))
        origin.set("rpy", numbers(urdf_rpy(zero[:3, :3])))
        limit = joint.find("limit")
        # 控制器回包的角度和速度为度及度每秒，URDF 存弧度及弧度每秒。
        for name, value in (("lower", config.joint_min_deg[index]),
                            ("upper", config.joint_max_deg[index]),
                            ("velocity", config.joint_max_speed_deg_s[index])):
            limit.set(name, numbers([np.deg2rad(value)]))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(DEFAULT_URDF, encoding="utf-8", xml_declaration=True)


def build():
    build_compatible_urdf()
    source = mujoco.MjModel.from_xml_path(str(DEFAULT_URDF))
    with tempfile.TemporaryDirectory(prefix="rml63-export-") as directory:
        path = Path(directory) / "export.xml"
        mujoco.mj_saveLastXML(str(path), source)
        root = ET.parse(path).getroot()
    root.set("model", f"{ROBOT_MODEL} joint position control")
    root.insert(0, ET.Comment("由 scripts/models/build_dynamics_model.py 生成，见 SOURCE.md。"))
    root.find("compiler").set("fusestatic", "false")
    root.find("compiler").set("meshdir", "upstream/meshes")
    for mesh in root.findall("asset/mesh"):
        mesh.set("file", Path(mesh.get("file")).name)
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")

    # 导出器会舍入小数，从已编译的 URDF 恢复位姿、质量和惯量精度。
    for body in root.findall(".//body"):
        original = source.body(body.get("name"))
        body.set("pos", numbers(original.pos))
        body.set("quat", numbers(original.quat))
        inertial = body.find("inertial")
        inertial.set("pos", numbers(original.ipos))
        inertial.set("quat", numbers(original.iquat))
        inertial.set("mass", numbers(original.mass))
        inertial.set("diaginertia", numbers(original.inertia))
        for index, geom in enumerate(body.findall("geom")):
            geom.set("name", body.get("name") + ("_visual" if index == 0 else "_collision"))
    # ee_site 是控制器 Arm_Tip 的法兰参考，抓取 TCP 在夹爪场景另设。
    ET.SubElement(root.find(".//body[@name='link_6']"), "site", name="ee_site",
                  pos="0 0 0", quat="1 0 0 0", type="sphere", size="0.008",
                  rgba="0.1 0.9 0.2 1", group="3")
    ET.SubElement(ET.SubElement(root, "contact"), "exclude", name="base_joint_assembly",
                  body1="base_link", body2="link_1")
    actuators = ET.SubElement(root, "actuator")
    for i, (kp, kv, damping) in enumerate(zip(KP, KV, DAMPING), 1):
        name = f"joint_{i}"
        joint = root.find(f".//joint[@name='{name}']")
        original = source.joint(name)
        joint.set("axis", numbers(original.axis))
        joint.set("pos", numbers(original.pos))
        joint.set("range", numbers(original.range))
        joint.set("damping", str(damping))
        force = joint.get("actuatorfrcrange")  # Imported official URDF effort bound.
        ET.SubElement(actuators, "position", name=f"position_{i}", joint=name,
                      kp=str(kp), kv=str(kv), ctrllimited="true", ctrlrange=numbers(original.range),
                      forcelimited="true", forcerange=force)
    # home 为 Reach 的仿真初始点，真实机械零位及打包装配姿态另行区分。
    home = numbers(np.deg2rad([0, -20, 30, 0, 20, 0]))
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home", qpos=home, ctrl=home)
    ET.indent(root, space="  ")
    output = ASSETS / "rml63_dynamics.xml"
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(output))
    if (compiled.nq, compiled.nv, compiled.nu) != (6, 6, 6):
        raise ValueError("Expected six coordinates, velocities and position actuators")
    print(f"Built {output}: nq={compiled.nq}, nv={compiled.nv}, nu={compiled.nu}")


if __name__ == "__main__":
    build()
