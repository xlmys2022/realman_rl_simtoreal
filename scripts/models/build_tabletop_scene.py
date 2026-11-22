#!/usr/bin/env python3
"""Horizontal workbench with the fixed arm and a freely supported cube.

Derive only a new task asset. Upstream robot/gripper geometry, inertias,
actuators and contact protections remain those of the combined model.
The keyframe's zero arm angles are a model reference, not a packing pose;
the task must receive an explicitly identified initial arm configuration.
"""
import xml.etree.ElementTree as ET

import mujoco

from realman_rl_simtoreal.paths import GRASP_CONTACT_SCENE, TABLETOP_PICK_SCENE
from realman_rl_simtoreal.simulation.devices.mujoco_gripper import SIGNS,width_to_angle


def build():
    root=ET.parse(GRASP_CONTACT_SCENE).getroot()
    root.set('model','RML63II tabletop pick; free cube, fixed horizontal desk')
    world=root.find('worldbody')
    world.find("body[@name='base_link']").set('pos','0 0 0.75')
    support=world.find("body[@name='support']")
    support.set('pos','0.2 0 0.725');support.set('quat','1 0 0 0')
    top=support.find("geom[@name='support_top']")
    top.set('size','.6 .4 .025');top.set('friction','.8 .005 .0001')
    top.set('rgba','.48 .32 .18 1')
    for number,(x,y) in enumerate([(-.34,-.34),(-.34,.34),(.74,-.34),(.74,.34)]):
        ET.SubElement(world,'geom',name=f'table_leg_{number}',type='box',
                      pos=f'{x} {y} .35',size='.025 .025 .35',rgba='.22 .24 .26 1')
    obj=world.find("body[@name='target_object']")
    obj.set('pos','.4 0 .76');obj.set('quat','1 0 0 0')
    key=root.find("keyframe/key[@name='home']")
    positions=key.get('qpos').split();controls=key.get('ctrl').split()
    positions[:6]=['0']*6;controls[:6]=['0']*6
    positions[-7:]=['.4','0','.76','1','0','0','0']
    key.set('qpos',' '.join(positions));key.set('ctrl',' '.join(controls))
    # Calibrate the task marker to the actual pad midpoint at cube width,
    # using private FK only. The rigid mesh/adapter geometry is unchanged.
    source=mujoco.MjModel.from_xml_path(str(GRASP_CONTACT_SCENE))
    probe=mujoco.MjData(source)
    mujoco.mj_resetDataKeyframe(source,probe,source.key('home').id)
    for name,sign in SIGNS.items():
        probe.qpos[source.joint(name).qposadr[0]]=sign*width_to_angle(.02)
    mujoco.mj_kinematics(source,probe)
    pad=(probe.site_xpos[source.site('eg2_left_pad').id]+probe.site_xpos[source.site('eg2_right_pad').id])/2
    body=source.body('4C2_baselink').id
    local=probe.xmat[body].reshape(3,3).T@(pad-probe.xpos[body])
    root.find(".//site[@name='grasp_site']").set('pos',' '.join(format(float(x),'.17g') for x in local))
    root.find('option').set('timestep','.001')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    headlight=visual.find('headlight')
    if headlight is None:headlight=ET.SubElement(visual,'headlight')
    headlight.set('ambient','.35 .35 .35');headlight.set('diffuse','.65 .65 .65')
    ET.SubElement(world,'camera',name='table_overview',pos='1.5 -1.7 1.7',
                  xyaxes='.75 .66 0 -.2 .23 .95')
    ET.indent(root,space='  ')
    TABLETOP_PICK_SCENE.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(TABLETOP_PICK_SCENE,encoding='utf-8',xml_declaration=True)
    model=mujoco.MjModel.from_xml_path(str(TABLETOP_PICK_SCENE))
    print(f'Built {TABLETOP_PICK_SCENE}: nq={model.nq}, nv={model.nv}, nu={model.nu}; tabletop z=.75m')


if __name__=='__main__':build()
