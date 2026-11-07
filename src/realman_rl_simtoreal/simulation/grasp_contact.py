"""Named contact policy and force records for the independent dynamic-cube scene.

This is an experiment, not a calibrated gripper force controller. Normal force
sums on each pad are reported separately from the vector force on the object.
"""
import mujoco
import numpy as np

from ..paths import GRASP_CONTACT_SCENE as SCENE
from .model import load_model, initialize, require
from .gripper import Gripper

PAD_PENETRATION = .0003  # 1.5% of the 20mm test cube, allowed contacts only
SUPPORT_PENETRATION = .0002
UNEXPECTED_PENETRATION = .0001  # unchanged empty-gripper threshold


class ContactPolicy:
    def __init__(self, model, *, hull=False):
        self.model = model
        self.target = model.geom('target_box').id
        self.support = model.geom('support_top').id
        self.pads = {}
        for index in range(model.ngeom):
            name = model.geom(index).name or ''
            for side in ('left', 'right'):
                if name.startswith(side+'_pad_'):
                    self.pads[index] = side
        if hull:
            self.pads[model.geom('4C2_Link5_collision').id] = 'left'
            self.pads[model.geom('4C2_Link6_collision').id] = 'right'
        self.object_dof = int(model.joint('target_free').dofadr[0])

    def classify(self, geom1, geom2):
        pair = {int(geom1), int(geom2)}
        if self.target in pair:
            other = geom2 if geom1 == self.target else geom1
            if other in self.pads:
                return self.pads[other]+'_pad_target'
            if other == self.support:
                return 'target_support'
        if self.support in pair:
            return 'robot_support'
        if self.target in pair:
            return 'target_other'
        return 'robot_internal_or_other'

    def check(self, data):
        v = data.qvel[self.object_dof:self.object_dof+6]
        require(np.isfinite(v).all() and np.linalg.norm(v[:3]) <= 1 and np.linalg.norm(v[3:]) <= 30,
                'Dynamic object velocity/numerical safety limit')
        for contact in data.contact:
            require(np.isfinite(contact.pos).all() and np.isfinite(contact.frame).all(),
                    'Non-finite contact geometry')
            category = self.classify(*contact.geom)
            limit = (PAD_PENETRATION if category.endswith('_pad_target') else
                     SUPPORT_PENETRATION if category == 'target_support' else UNEXPECTED_PENETRATION)
            require(np.isfinite(contact.dist) and contact.dist >= -limit,
                    f'{category}: penetration {contact.dist:.9g}m exceeds {limit:g}m; '
                    f'{tuple(self.model.geom(int(g)).name for g in contact.geom)}')


def contact_force_on_target(contact, force, target_geom):
    """contact.frame rows = world axes; force acts on geom[1], opposite on [0]."""
    require(target_geom in contact.geom, 'Contact does not involve target')
    world = np.asarray(contact.frame).reshape(3, 3).T @ np.asarray(force)[:3]
    return world if int(contact.geom[1]) == target_geom else -world


def load_experiment(*, mass=.01, friction=.7, hull=False):
    require(np.isfinite(mass) and .001 <= mass <= .1, 'Experiment mass outside 1..100g')
    require(np.isfinite(friction) and 0 <= friction <= 2, 'Invalid pad-target friction')
    model, jids, aids = load_model(SCENE, gravity_compensation=True)
    body = model.body('target_object').id
    model.body_mass[body] = mass
    model.body_inertia[body] = mass*.02**2/6
    # Explicit pairs avoid the geom friction mixing rule (max), so the low
    # friction case really changes this interface, not the support interface.
    for index in range(model.npair):
        if model.pair(index).name.endswith('_target') and model.pair(index).name != 'support_target':
            model.pair_friction[index, :2] = friction
    if hull:
        # Diagnostic only: replace the explicit pad pairs with original hull
        # pairs using MjSpec, to avoid pair contacts overriding geom masks.
        spec = mujoco.MjSpec.from_file(str(SCENE))
        for geom in spec.geoms:
            if geom.name.startswith(('left_pad_', 'right_pad_')):
                geom.contype = geom.conaffinity = 0
            if geom.name in ('4C2_Link5_collision', '4C2_Link6_collision'):
                geom.contype = geom.conaffinity = 1
        spec.geom('target_box').conaffinity = 1
        for pair in list(spec.pairs):
            if pair.name != 'support_target':
                spec.delete(pair)
        for side, link in [('left', '4C2_Link5'), ('right', '4C2_Link6')]:
            spec.add_pair(name=side+'_hull_target', geomname1=link+'_collision', geomname2='target_box',
                          condim=3, friction=[friction, friction, .005, .0001, .0001],
                          solref=[.006, 1], solimp=[.95, .99, .001, .5, 2])
        # Enable compensation before compile, with the same loader as baseline.
        # Saving to a string would lose mesh resolution; compile then load via
        # a temporary XML next to no asset is unnecessary: set subtree directly.
        for b in spec.bodies:
            parent = b
            while parent is not None and parent.name != 'link_1':
                parent = parent.parent
            if parent is not None:
                b.gravcomp = 1
        for joint in spec.joints:
            if joint.name != 'target_free':
                joint.actgravcomp = True
        model = spec.compile()
        model.body_mass[model.body('target_object').id] = mass
        model.body_inertia[model.body('target_object').id] = mass*.02**2/6
    # Refresh derived inertial constants after experiment-only mass changes.
    data = mujoco.MjData(model)
    mujoco.mj_setConst(model, data)
    data = initialize(model)
    policy = ContactPolicy(model, hull=hull)
    gripper = Gripper(model, data, contact_policy=policy)
    return model, data, jids, aids, gripper, policy


class ContactRecorder:
    """Call immediately after mj_step; retain every physical tick for audit."""
    def __init__(self, model, data, gripper, policy):
        self.model, self.data, self.gripper, self.policy = model, data, gripper, policy
        self.probe = mujoco.MjData(model)
        self.object = model.body('target_object').id
        self.support = model.body('support').id
        self.qadr = int(model.joint('target_free').qposadr[0])
        self.vadr = int(model.joint('target_free').dofadr[0])
        self.rows = []
        self.phase = 'initialize'
        self._seen_categories = set()

    def __call__(self, data):
        m, p = self.model, self.policy
        self.probe.qpos[:] = data.qpos
        mujoco.mj_forward(m, self.probe)  # never overwrite the live force cache
        g = self.gripper
        left, right = self.probe.site_xpos[g.left], self.probe.site_xpos[g.right]
        rotation = self.probe.xmat[g.base].reshape(3, 3)
        object_pos = data.qpos[self.qadr:self.qadr+3].copy()
        table_rotation = self.probe.xmat[self.support].reshape(3, 3)
        object_rotation = self.probe.xmat[self.object].reshape(3, 3)
        # Oriented box support distance, exact for the configured cube/plane.
        n = table_rotation[:, 2]
        radius = .01*np.sum(np.abs(object_rotation.T @ n))
        clearance = (object_pos-self.probe.xpos[self.support])@n-.003-radius
        row = dict(time_s=float(data.time), phase=self.phase,
                   object_position_m=object_pos.tolist(), object_quaternion_wxyz=data.qpos[self.qadr+3:self.qadr+7].tolist(),
                   object_qvel=data.qvel[self.vadr:self.vadr+6].tolist(),
                   object_relative_to_pads_m=(rotation.T@(object_pos-(left+right)/2)).tolist(),
                   support_clearance_m=float(clearance),
                   actual_width_m=float((right-left)@rotation[:, 0]), command_width_m=g.target_width,
                   actuator_reference_rad=float(data.ctrl[g.actuator_id]),
                   primary_servo_nm=float(data.actuator_force[g.actuator_id]),
                   primary_total_drive_nm=float(data.qfrc_actuator[m.jnt_dofadr[g.primary]]),
                   gripper_total_drive_nm=data.qfrc_actuator[g.vadr].tolist(),
                   arm_target_rad=data.ctrl[g.arm_monitor.aids].tolist(),
                   arm_actual_rad=data.qpos[m.jnt_qposadr[g.arm_monitor.jids]].tolist(),
                   max_penetration_m=max([0.]+[-float(c.dist) for c in data.contact]),
                   left_normal_n=0., right_normal_n=0., left_tangent_n=0., right_tangent_n=0.,
                   support_normal_n=0., contact_count=int(data.ncon), contacts=[], contacts_sampled=False)
        categories = {p.classify(*c.geom) for c in data.contact}
        # Scalars/forces at every 2ms tick; full contact vectors/points at 20Hz
        # and each category's first appearance. Avoid enormous redundant logs.
        detailed = len(self.rows) % 25 == 0 or bool(categories-self._seen_categories)
        self._seen_categories.update(categories)
        row['contacts_sampled'] = detailed
        for index, c in enumerate(data.contact):
            category = p.classify(*c.geom)
            force = np.zeros(6)
            mujoco.mj_contactForce(m, data, index, force)
            require(np.isfinite(force).all(), 'Non-finite contact force')
            world = contact_force_on_target(c, force, p.target) if p.target in c.geom else None
            if detailed:
                row['contacts'].append(dict(category=category, geoms=[m.geom(int(i)).name for i in c.geom],
                                           bodies=[m.body(int(m.geom_bodyid[i])).name for i in c.geom],
                                           distance_m=float(c.dist), position_world_m=c.pos.tolist(),
                                           force_contact_frame=force.tolist(),
                                           force_on_target_world_n=None if world is None else world.tolist()))
            if category in ('left_pad_target', 'right_pad_target'):
                side = category.split('_')[0]
                row[side+'_normal_n'] += float(force[0])
                row[side+'_tangent_n'] += float(np.linalg.norm(force[1:3]))
            elif category == 'target_support':
                row['support_normal_n'] += float(force[0])
        self.rows.append(row)


def groove_probe(*, hull=False):
    """Forward-only diagnostic, not the 20mm dynamic grasp experiment.

    A 0.1mm cube is initialized in a real Y tooth gap at left-pad local
    [17.5, 6, 18]mm. Its center is inside the old convex hull but outside STL
    solid surfaces. No physical trajectory or grip result is inferred.
    """
    m, d, _, _, _, policy = load_experiment(hull=hull)
    tiny = m.geom('target_box').id
    m.geom_size[tiny] = .00005
    finger = m.body('4C2_Link5').id
    rotation = d.xmat[finger].reshape(3, 3)
    qadr = int(m.joint('target_free').qposadr[0])
    d.qpos[qadr:qadr+3] = d.xpos[finger]+rotation@np.array([.0175, .006, .018])
    mujoco.mju_mat2Quat(d.qpos[qadr+3:qadr+7], rotation.ravel())
    mujoco.mj_forward(m, d)
    return [dict(category=policy.classify(*c.geom),
                 geoms=[m.geom(int(g)).name for g in c.geom], distance_m=float(c.dist),
                 position_world_m=c.pos.tolist(),
                 position_left_local_m=(rotation.T@(c.pos-d.xpos[finger])).tolist())
            for c in d.contact if tiny in c.geom]
