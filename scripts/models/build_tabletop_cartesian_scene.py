"""生成原桌面场景或独立的 RealMan 安装几何场景。"""
import argparse
import hashlib
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from realman_rl_simtoreal.paths import ROBOT_MODEL, TABLETOP_PICK_SCENE, TABLETOP_CARTESIAN_SCENE as DEFAULT_SCENE
from realman_rl_simtoreal.paths import TABLETOP_CARTESIAN_REALMAN_SCENE
from realman_rl_simtoreal.hardware.realman_config import REALMAN_TABLETOP
from realman_rl_simtoreal.simulation.devices.camera_geometry import camera_rotation


def apply_realman_installation(root):
    """只调整安装几何。保留原关节、接触保护及各附件惯性参数。"""
    hardware = REALMAN_TABLETOP
    support = root.find(".//body[@name='support']")
    height = float(support.get("pos").split()[2]) + float(support.find("geom").get("size").split()[2])
    root.find(".//body[@name='base_link']").set("pos", f"0 0 {height + hardware.base_above_table_m:g}")
    # 连接件代理含原 8 mm 安装板，完整夹爪子树随安装面后移。
    adapter = root.find(".//geom[@name='eg2_adapter_proxy']")
    half_length = hardware.adapter_length_m / 2
    adapter.set("pos", f"0 0 {half_length:.12g}")
    adapter.set("size", f"{hardware.adapter_radius_m:.12g} {half_length:.12g}")
    root.find(".//body[@name='4C2_baselink']").set("pos", f"0 0 {hardware.adapter_length_m:.12g}")
    root.find(".//site[@name='grasp_site']").set("pos", f"0 0 {hardware.gripper_mount_to_nominal_tcp_m:.12g}")
    cube = root.find(".//geom[@name='target_box']")
    cube.set("rgba", ".06 .72 .10 1")
    cube.set("size", " ".join([f"{hardware.cube_edge_m / 2:g}"] * 3))
    camera = root.find(".//camera[@name='wrist_rgb']")
    width, height = hardware.camera_width, hardware.camera_height
    camera.attrib.pop("fovy")
    camera.set("resolution", f"{width} {height}")
    camera.set("sensorsize", f"{width / 100000:g} {height / 100000:g}")
    camera.set("focalpixel", f"{hardware.camera_fx_px} {hardware.camera_fy_px}")
    # 用传感器中心到 OpenCV 左上像素中心的坐标约定构造 MuJoCo 偏心视锥。
    camera.set("principalpixel", f"{(width - 1) / 2 - hardware.camera_cx_px} "
               f"{(height - 1) / 2 - hardware.camera_cy_px}")


def build(*, realman=False):
    tree=ET.parse(TABLETOP_PICK_SCENE);root=tree.getroot()
    root.set('model',f'{ROBOT_MODEL} Cartesian PPO; independent kinematics; traditional startup')
    flange = root.find(".//body[@name='link_6']")
    # 6FB 本体没有 V 型相机附件。沿用仿真光心在法兰系的位置（米）与朝向。
    # 此刚性空坐标系只服务 RGB-D 渲染，不引入未测量的外壳或质量。
    body = ET.SubElement(flange, 'body', name='camera_link', pos='.0845 .0325 -.03552',
                         quat='0.7071067811865476 0 0 0.7071067811865476')
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,camera_rotation((0.,0.,1.),(1.,0.,0.)).ravel())
    ET.SubElement(body,'camera',name='wrist_rgb',pos='0 0 0',
                  quat=' '.join(map(str,quat)),fovy=str(90.))
    ET.SubElement(body,'site',name='camera_forward',pos='0 0 .05',size='.004',group='3',rgba='1 .7 0 1')
    ET.SubElement(root.find('worldbody'),'camera',name='top_view',pos='.2 0 3.2',xyaxes='1 0 0 0 1 0')
    scene = DEFAULT_SCENE
    if realman:
        scene = TABLETOP_CARTESIAN_REALMAN_SCENE
        root.set("model", f"{ROBOT_MODEL} measured tabletop installation, nominal camera parent")
        apply_realman_installation(root)
    ET.indent(root,space='  ');tree.write(scene,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(scene))
    return dict(scene=str(scene),nq=model.nq,nv=model.nv,nu=model.nu,
                sha256=hashlib.sha256(scene.read_bytes()).hexdigest())


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realman", action="store_true", help="生成独立的实机安装场景")
    print(build(realman=parser.parse_args().realman))
