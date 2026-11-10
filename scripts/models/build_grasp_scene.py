#!/usr/bin/env python3
"""Build an independent contact experiment; never rewrite the upstream/arm assets."""
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from realman_rl_simtoreal.paths import GRIPPER_ASSETS, GRIPPER_SCENE, GRASP_CONTACT_SCENE
from realman_rl_simtoreal.simulation.gripper import SIGNS, width_to_angle
from realman_rl_simtoreal.simulation.model import initialize

DESTINATION = GRASP_CONTACT_SCENE


def numbers(values):
    return ' '.join(format(float(x), '.17g') for x in values)


def pad_rectangles(side):
    """Ten original tooth crests + two conservative trough-floor rectangles.

    The STL tooth grooves are only 0.4mm deep: x=17.3..17.7mm.
    x=10mm is the BACK of the finger, not the trough. The two floor bands
    follow the original front footprint and lie inside the STL material;
    curved tooth flanks are under-approximated, never thickened outward.
    """
    link = '4C2_Link5' if side == 'left' else '4C2_Link6'
    sign = 1 if side == 'left' else -1
    raw = (GRIPPER_ASSETS/f'upstream/meshes/{link}.STL').read_bytes()
    dtype = np.dtype([('normal', '<f4', (3,)), ('v', '<f4', (3, 3)), ('attr', '<u2')])
    records = np.frombuffer(raw, dtype=dtype, offset=84)
    triangles = records['v'].astype(float)
    rectangles = []
    faces = triangles[np.max(np.abs(triangles[:, :, 0]-sign*.0177), axis=1) < 1e-7]
    for index in range(0, len(faces), 2):
        pair = faces[index:index+2]
        low, high = pair.reshape(-1, 3).min(0), pair.reshape(-1, 3).max(0)
        area = np.linalg.norm(np.cross(pair[:, 1]-pair[:, 0], pair[:, 2]-pair[:, 0]), axis=1).sum()/2
        if not np.isclose(area, (high[1]-low[1])*(high[2]-low[2]), atol=1e-10):
            raise ValueError('STL face is not a rectangular land')
        center, size = (low+high)/2, (high-low)/2
        center[0], size[0] = sign*.0177-sign*.00025, .00025
        rectangles.append((center, size))
    front = triangles[(sign*records['normal'][:, 0] > .1)
                      & ((sign*triangles[:, :, 0]).min(1) > .017)]
    # Separate the original lower/upper front footprints at the STL's step.
    for zlow, zhigh in [(-.005, .0105), (.0105, .0285)]:
        band = front[(front[:, :, 2].min(1) <= zlow+1e-7)
                     & (front[:, :, 2].max(1) >= zhigh-1e-7)]
        ymin, ymax = band[:, :, 1].min(), band[:, :, 1].max()
        floor = (sign*band[:, :, 0]).min()
        if not np.isclose(floor, .0173, atol=1e-7):
            raise ValueError('Unexpected tooth trough depth')
        rectangles.append((np.array([sign*(floor-.00025), (ymin+ymax)/2, (zlow+zhigh)/2]),
                           np.array([.00025, (ymax-ymin)/2, (zhigh-zlow)/2])))
    if len(rectangles) != 12:
        raise ValueError('Expected ten crests and two trough footprint bands per pad')
    return rectangles


def build():
    root = ET.parse(GRIPPER_SCENE).getroot()
    root.set('model', 'EG2-4C2 dynamic contact experiment; uncalibrated')
    for mesh in root.findall('asset/mesh'):
        mesh.set('file', os.path.relpath((GRIPPER_ASSETS/mesh.get('file')).resolve(), DESTINATION.parent))
    # Original finger hulls still collide with robot and support. Only the
    # target uses the measured inward pad layers instead of the convex hull.
    contact = root.find('contact')
    for side, link in [('left', '4C2_Link5'), ('right', '4C2_Link6')]:
        body = root.find(f".//body[@name='{link}']")
        hull = body.find(f"geom[@name='{link}_collision']")
        hull.set('contype', '4')
        hull.set('conaffinity', '5')
        for index, (center, size) in enumerate(pad_rectangles(side)):
            name = f'{side}_pad_{index}'
            ET.SubElement(body, 'geom', name=name, type='box', pos=numbers(center), size=numbers(size),
                          contype='8', conaffinity='2', mass='0', group='3', rgba='0 .8 .8 .35')
            ET.SubElement(contact, 'pair', name=f'{name}_target', geom1=name, geom2='target_box',
                          condim='3', friction='.7 .7 .005 .0001 .0001', solref='.006 1',
                          solimp='.95 .99 .001')
    # Calculate, without changing any running simulation, the pad midpoint at
    # 20mm opening. Object axes align with the experimental mount axes.
    model = mujoco.MjModel.from_xml_path(str(GRIPPER_SCENE))
    probe = initialize(model)
    for name, sign in SIGNS.items():
        probe.qpos[model.joint(name).qposadr[0]] = sign*width_to_angle(.02)
    mujoco.mj_forward(model, probe)
    center = (probe.site_xpos[model.site('eg2_left_pad').id]+probe.site_xpos[model.site('eg2_right_pad').id])/2
    rotation = probe.xmat[model.body('4C2_baselink').id].reshape(3, 3)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    world = root.find('worldbody')
    table = ET.SubElement(world, 'body', name='support', pos=numbers(center-rotation[:, 2]*.013), quat=numbers(quat))
    ET.SubElement(table, 'geom', name='support_top', type='box', size='.009 .02 .003',
                  friction='1 .005 .0001', rgba='.35 .35 .35 1')
    obj = ET.SubElement(world, 'body', name='target_object', pos=numbers(center), quat=numbers(quat))
    ET.SubElement(obj, 'freejoint', name='target_free')
    ET.SubElement(obj, 'inertial', pos='0 0 0', mass='.01', diaginertia=numbers([.01*.02**2/6]*3))
    ET.SubElement(obj, 'geom', name='target_box', type='box', size='.01 .01 .01',
                  contype='2', conaffinity='9', friction='.7 .005 .0001', rgba='.9 .35 .1 1')
    ET.SubElement(contact, 'pair', name='support_target', geom1='support_top', geom2='target_box',
                  condim='3', friction='1 1 .005 .0001 .0001', solref='.006 1', solimp='.999 .9999 .001')
    key = root.find('keyframe/key')
    key.set('qpos', key.get('qpos')+' '+numbers(np.r_[center, quat]))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space='  ')
    ET.ElementTree(root).write(DESTINATION, encoding='utf-8', xml_declaration=True)
    compiled = mujoco.MjModel.from_xml_path(str(DESTINATION))
    print(f'Built {DESTINATION}: nq={compiled.nq}, nv={compiled.nv}, nu={compiled.nu}; 24 pad layers')


if __name__ == '__main__':
    build()
