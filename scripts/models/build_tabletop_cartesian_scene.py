"""Derive the Cartesian task scene; additions are visualization frames only."""
import hashlib
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from realman_rl_simtoreal.paths import TABLETOP_PICK_SCENE, TABLETOP_CARTESIAN_SCENE as DEFAULT_SCENE
from realman_rl_simtoreal.simulation.devices.camera_geometry import camera_rotation


def build():
    tree=ET.parse(TABLETOP_PICK_SCENE);root=tree.getroot()
    root.set('model','RealMan Cartesian PPO; independent kinematics; traditional startup')
    body=root.find(".//body[@name='camera_link']")
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,camera_rotation((0.,0.,1.),(1.,0.,0.)).ravel())
    ET.SubElement(body,'camera',name='wrist_rgb',pos='0 0 0',
                  quat=' '.join(map(str,quat)),fovy=str(90.))
    ET.SubElement(body,'site',name='camera_forward',pos='0 0 .05',size='.004',group='3',rgba='1 .7 0 1')
    ET.SubElement(root.find('worldbody'),'camera',name='top_view',pos='.2 0 3.2',xyaxes='1 0 0 0 1 0')
    ET.indent(root,space='  ');tree.write(DEFAULT_SCENE,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    return dict(scene=str(DEFAULT_SCENE),nq=model.nq,nv=model.nv,nu=model.nu,
                sha256=hashlib.sha256(DEFAULT_SCENE.read_bytes()).hexdigest())


if __name__=='__main__':print(build())
