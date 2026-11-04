#!/usr/bin/env python3
"""Check the pinned RML63II-6F-V URDF and compiled MuJoCo model without a GUI."""

import argparse
import hashlib
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets" / "robots" / "rml63"
DEFAULT_MODEL = ASSET_DIR / "rml63_mujoco.urdf"
ARM_JOINTS = tuple(f"joint_{i}" for i in range(1, 7))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(text):
    values = np.array([float(value) for value in text.split()])
    require(values.shape == (3,) and np.isfinite(values).all(), f"Invalid vector: {text}")
    return values


def inertia_matrix(link):
    values = link.find("inertial/inertia").attrib
    return np.array([[float(values[f"i{a}{b}" if a <= b else f"i{b}{a}"])
                      for b in "xyz"] for a in "xyz"])


def check_upstream():
    entries = (ASSET_DIR / "SHA256SUMS").read_text().splitlines()
    for entry in entries:
        expected, name = entry.split(maxsplit=1)
        path = ASSET_DIR / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected,
                f"Upstream asset checksum mismatch: {path}")
    print(f"[PASS] Upstream SHA-256 checksums: {len(entries)} files")
    source = ET.parse(ASSET_DIR / "upstream/urdf/RML63II-6F-V.urdf").getroot()
    for link in source.findall("link"):
        eigenvalues = np.linalg.eigvalsh(inertia_matrix(link))
        margin = eigenvalues[0] + eigenvalues[1] - eigenvalues[2]
        if margin < 0:
            print(f"[WARN] Original {link.attrib['name']} inertia violates A+B>=C: "
                  f"margin={margin:.9e} kg*m^2; see SOURCE.md")
    print("[WARN] RML63II-6F-V hardware generation and physical parameters are not verified against the real arm.")


def check_urdf(path):
    root = ET.parse(path).getroot()
    require(root.tag == "robot" and root.get("name") == "RML63II-6F-V",
            "Expected the RML63II-6F-V URDF, not a different robot or MJCF")
    links = root.findall("link")
    joints = root.findall("joint")
    names = {link.attrib["name"] for link in links}
    expected_links = {"base_link", "camera_rolink", "camera_link"} | {
        f"link_{i}" for i in range(1, 7)}
    require(len(links) == len(names) and names == expected_links,
            f"Unexpected or duplicate URDF links: {sorted(names)}")
    by_name = {joint.attrib["name"]: joint for joint in joints}
    expected_joints = set(ARM_JOINTS) | {"camera_rojoint", "camera_joint"}
    require(len(joints) == len(by_name) and set(by_name) == expected_joints,
            f"Unexpected or duplicate URDF joints: {sorted(by_name)}")
    chain = [(name, "base_link" if i == 1 else f"link_{i-1}", f"link_{i}", "revolute")
             for i, name in enumerate(ARM_JOINTS, 1)]
    chain += [("camera_rojoint", "link_6", "camera_rolink", "revolute"),
              ("camera_joint", "camera_rolink", "camera_link", "fixed")]
    for name, parent, child, kind in chain:
        joint = by_name[name]
        require((joint.find("parent").get("link"), joint.find("child").get("link"),
                 joint.get("type")) == (parent, child, kind), f"Wrong topology/type: {name}")
    print(f"[PASS] URDF: {len(links)} links, {len(joints)} joints; six arm revolute joints + camera revolute + camera fixed")

    mesh_paths = set()
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        prefix = "package://RML63II-6F-V/"
        mesh_path = (path.parent.parent / filename[len(prefix):]
                     if filename.startswith(prefix) else path.parent / filename)
        mesh_path = mesh_path.resolve()
        require(mesh_path.is_file(), f"Missing mesh: {mesh_path}")
        mesh_paths.add(mesh_path)
    require(len(mesh_paths) == 9, f"Expected 9 unique meshes, found {len(mesh_paths)}")
    for mesh_path in sorted(mesh_paths):
        payload = mesh_path.read_bytes()
        require(len(payload) >= 84, f"Truncated binary STL: {mesh_path}")
        triangles = struct.unpack_from("<I", payload, 80)[0]
        require(triangles > 0 and len(payload) == 84 + 50 * triangles,
                f"Invalid binary STL size: {mesh_path}")
        print(f"  Mesh {mesh_path.name}: {len(payload)} bytes, {triangles} triangles")
    print("[PASS] All 9 referenced binary STL files are complete")

    total_mass = 0.0
    for link in links:
        name = link.attrib["name"]
        mass = float(link.find("inertial/mass").get("value"))
        com = vector(link.find("inertial/origin").get("xyz"))
        matrix = inertia_matrix(link)
        require(np.isfinite(mass) and mass > 0 and np.isfinite(matrix).all(),
                f"Non-positive mass or non-finite inertia: {name}")
        eigenvalues = np.linalg.eigvalsh(matrix)
        margin = eigenvalues[0] + eigenvalues[1] - eigenvalues[2]
        require(eigenvalues[0] > 0 and margin >= 0, f"Non-physical inertia: {name}; A+B-C={margin:.9e}")
        total_mass += mass
        print(f"  Link {name}: mass={mass:.9g} kg, COM={com.tolist()} m, "
              f"principal inertia={eigenvalues.tolist()} kg*m^2")
    print(f"[PASS] Selected URDF mass/inertia checks; total mass including fixed base={total_mass:.9f} kg")
    for name, _, _, kind in chain:
        joint = by_name[name]
        if kind == "fixed":
            print(f"  URDF {name}: fixed (no MuJoCo motion joint)")
            continue
        axis = vector(joint.find("axis").get("xyz"))
        require(np.linalg.norm(axis) > 0, f"Zero joint axis: {name}")
        limit = {key: float(value) for key, value in joint.find("limit").attrib.items()}
        require(all(np.isfinite(list(limit.values()))) and limit["lower"] < limit["upper"]
                and limit["lower"] <= 0 <= limit["upper"]
                and limit["effort"] >= 0 and limit["velocity"] >= 0, f"Invalid limits: {name}")
        print(f"  URDF {name}: axis={axis.tolist()} (joint frame), "
              f"effort={limit['effort']:g} N*m, velocity={limit['velocity']:g} rad/s")
    return root, by_name


def check_model(path, root, joints):
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    print(f"MuJoCo: nbody={model.nbody} (includes world), njnt={model.njnt}, "
          f"nq={model.nq}, nv={model.nv}, nu={model.nu}, nmesh={model.nmesh}, ngeom={model.ngeom}")
    require(model.nbody == 10, "Expected world plus all nine URDF bodies (fusestatic=false)")
    for body_id in range(model.nbody):
        name = model.body(body_id).name
        parent = "-" if body_id == 0 else model.body(int(model.body_parentid[body_id])).name
        print(f"  Body {body_id} {name}: parent={parent}, mass={model.body_mass[body_id]:.9g} kg, "
              f"COM={model.body_ipos[body_id].tolist()} m, "
              f"principal inertia={model.body_inertia[body_id].tolist()} kg*m^2")
    for link in root.findall("link"):
        body = model.body(link.get("name"))
        require(np.isclose(body.mass[0], float(link.find("inertial/mass").get("value"))),
                f"Compiled mass differs from URDF: {body.name}")
        require(np.allclose(body.ipos, vector(link.find("inertial/origin").get("xyz"))),
                f"Compiled COM differs from URDF: {body.name}")
        require(np.allclose(np.sort(body.inertia), np.linalg.eigvalsh(inertia_matrix(link)),
                            rtol=1e-5, atol=1e-15), f"Compiled inertia differs from URDF: {body.name}")
    expected = set(ARM_JOINTS) | {"camera_rojoint"}
    require({model.joint(i).name for i in range(model.njnt)} == expected,
            "Compiled motion joints do not match the six arm joints and camera joint")
    for joint_id in range(model.njnt):
        joint = model.joint(joint_id)
        source = joints[joint.name]
        kind = mujoco.mjtJoint(int(joint.type[0])).name
        require(joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE, f"Expected hinge: {joint.name}")
        body_id = int(joint.bodyid[0])
        require(model.body(body_id).name == source.find("child").get("link")
                and model.body(int(model.body_parentid[body_id])).name == source.find("parent").get("link"),
                f"Compiled joint topology differs: {joint.name}")
        axis = vector(source.find("axis").get("xyz"))
        require(np.allclose(joint.axis, axis / np.linalg.norm(axis)), f"Compiled axis differs: {joint.name}")
        limits = source.find("limit")
        expected_range = [float(limits.get("lower")), float(limits.get("upper"))]
        require(bool(joint.limited[0]) and np.allclose(joint.range, expected_range),
                f"Compiled limits differ: {joint.name}")
        role = "arm" if joint.name in ARM_JOINTS else "camera accessory"
        print(f"  Joint {joint_id} {joint.name}: {kind}, {role}, "
              f"axis={joint.axis.tolist()} (child body frame), "
              f"range={joint.range.tolist()} rad / {np.rad2deg(joint.range).round(6).tolist()} deg")
    require((model.nq, model.nv, model.nu) == (7, 7, 0), "Unexpected state/control dimensions")
    print("[PASS] Six arm hinges verified by name, type, topology, axis and limits; one additional camera hinge")
    print("[WARN] nu=0: URDF effort/velocity do not create actuators or enforce runtime velocity limits")
    mujoco.mj_forward(model, data)
    for name in ("qpos", "qvel", "qacc", "xpos", "xquat", "qfrc_bias"):
        require(np.isfinite(getattr(data, name)).all(), f"mj_forward produced non-finite {name}")
    mass_matrix = np.empty((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass_matrix)
    require(np.isfinite(mass_matrix).all() and np.linalg.eigvalsh(mass_matrix)[0] > 0,
            "Generalized mass matrix is not finite and positive definite")
    require(not any(warning.number for warning in data.warning), "MuJoCo reported a numerical warning")
    print(f"[PASS] mj_forward: finite state, poses, mass matrix and forces; time={data.time:g}, qpos={data.qpos.tolist()}")
    print(f"[INFO] Initial contacts={data.ncon}; contact geometry and dynamics are not validated for training")
    for contact in data.contact:
        body1 = model.body(int(model.geom_bodyid[contact.geom1])).name
        body2 = model.body(int(model.geom_bodyid[contact.geom2])).name
        print(f"  Contact {body1} / {body2}: distance={contact.dist:.9f} m")
    if any(contact.dist < 0 for contact in data.contact):
        print("[WARN] Initial collision penetration detected; geometry is unchanged and needs review before dynamic simulation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="RML63II-6F-V URDF path")
    args = parser.parse_args()
    global mujoco, np
    try:
        import mujoco
        import numpy as np

        path = args.model.expanduser().resolve()
        require(path.is_file(), f"Model file not found: {path}")
        print(f"Python: {sys.version.split()[0]} ({sys.executable})")
        print(f"MuJoCo: {mujoco.__version__}; NumPy: {np.__version__}")
        print(f"Model: {path}")
        check_upstream()
        root, joints = check_urdf(path)
        check_model(path, root, joints)
    except ImportError as exc:
        print(f"[FAIL] Missing dependency: {exc}. Install requirements.txt in realman_rl.", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, AttributeError, ET.ParseError, mujoco.FatalError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    print("[PASS] Model import and structural checks complete (warnings above remain applicable)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
