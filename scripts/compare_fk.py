#!/usr/bin/env python3
"""Offline RM_API2/MuJoCo FK measurements; exit 2 means model mismatch, not a test crash."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_joints as control
from kinematics import Kinematics, rotation_vector
from realman_fk import RealManFK, SDKUnavailable, MODELS, SENSORS, quaternion_matrix, euler_matrix
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# Numerical consistency criterion, not real-arm accuracy. Much larger than the
# measured float32 SDK conversion noise; smaller than the URDF rounding errors.
POSITION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE_RAD = np.deg2rad(0.001)


def transform(position, rotation):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rotation, position
    return result


def inverse(t):
    result = np.eye(4)
    result[:3, :3] = t[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ t[:3, 3]
    return result


def errors(a, b):
    dp = a[:3, 3] - b[:3, 3]
    angle = float(np.linalg.norm(rotation_vector(a[:3, :3], b[:3, :3])))
    distance = float(np.linalg.norm(dp))
    return dict(position_difference_m=dp.tolist(), position_error_m=distance,
                rotation_error_rad=angle, within_tolerance=bool(
                    distance <= POSITION_TOLERANCE_M and angle <= ROTATION_TOLERANCE_RAD))


def mdh_local(dh, index, angle):
    """Read-only diagnostic: Tx(a) Rx(alpha) Rz(q+offset) Tz(d)."""
    alpha, theta = np.deg2rad([dh["alpha"][index], dh["offset"][index]])
    rx = euler_matrix([alpha, 0, 0])
    rotation = rx @ euler_matrix([0, 0, theta + angle])
    return transform(np.array([dh["a"][index], 0, 0]) + rotation[:, 2]*dh["d"][index], rotation)


def mdh_fk(dh, q):
    """Audit the retrieved parameters/conventions against native FK, not a controller."""
    result = np.eye(4)
    for i, angle in enumerate(q):
        result = result @ mdh_local(dh, i, angle)
    return result


def sample_poses(home):
    poses = [("home", home.copy())]
    for i in range(6):
        q = home.copy()
        q[i] += np.deg2rad(3)
        poses.append((f"J{i+1}_plus_3deg", q))
    poses.extend([("combined", home + np.deg2rad([2, -2, 2, 1, -1, 2])),
                  ("diverse_fk_only", np.deg2rad([25, -45, 55, 20, 40, -30]))])
    return poses


def maxima(records):
    return dict(max_position_error_m=max(r["position_error_m"] for r in records),
                max_rotation_error_rad=max(r["rotation_error_rad"] for r in records),
                all_within_tolerance=all(r["within_tolerance"] for r in records))


def measure_configuration(kin, generation="II", sensor="SF"):
    """Compare frames without fitting or changing either robot's parameters."""
    model = kin.model
    data = control.initialize(model)  # Private reference data, never integrated.
    base = model.body("base_link").id
    world_from_base = transform(data.xpos[base], data.xmat[base].reshape(3, 3))
    base_from_world = inverse(world_from_base)
    sid = kin.site_id
    if (not np.array_equal(model.site_pos[sid], [0, 0, 0])
            or not np.array_equal(model.site_quat[sid], [1, 0, 0, 0])):
        raise ValueError("This comparison requires ee_site at link_6 origin, with identity orientation")
    link6 = model.body("link_6").id
    joint6 = model.joint("joint_6").id
    if (model.body_parentid[link6] != model.body("link_5").id
            or not np.array_equal(model.jnt_pos[joint6], [0, 0, 0])
            or not np.array_equal(model.jnt_axis[joint6], [0, 0, 1])):
        raise ValueError("Unexpected terminal chain; cannot define common wrist reference")
    r6_fixed = quaternion_matrix(model.body_quat[link6])
    vendor = RealManFK(generation, sensor)
    dh = vendor.info["dh"]
    if not np.array_equal(np.asarray(dh["a"])[4:], [0, 0]) or dh["d"][4] != 0:
        raise ValueError("Retrieved MDH does not have the expected coincident J4/J5 wrist centre")
    rows, mj_transforms, rm_transforms = [], [], []
    for name, q in sample_poses(kin.home_q):
        # fk validates finite values and the actual model's named-joint limits.
        pm, rm = kin.fk(q)
        tm = base_from_world @ transform(pm, rm)
        pv, rv = vendor.fk(q)
        tv = transform(pv, rv)
        pe, re = vendor.euler_fk(q)
        # link_5 origin is the wrist centre. Recover it using the actual fixed
        # link_6 transform, including URDF rounding, not an assumed axial length.
        r5 = tm[:3, :3] @ euler_matrix([0, 0, -q[5]]) @ r6_fixed.T
        wrist_m = tm[:3, 3] - r5 @ model.body_pos[link6]
        wrist_v = pv - rv[:, 2]*dh["d"][5]
        jp, jr = kin.jacobian(q)
        rows.append(dict(name=name, joint_deg=np.rad2deg(q).tolist(),
                         mujoco_world=transform(pm, rm).tolist(), mujoco_base=tm.tolist(),
                         vendor_base=tv.tolist(), absolute=errors(tm, tv),
                         wrist_mujoco_base_m=wrist_m.tolist(), wrist_vendor_base_m=wrist_v.tolist(),
                         common_wrist=errors(transform(wrist_m, tm[:3, :3]), transform(wrist_v, rv)),
                         sdk_mdh_audit=errors(mdh_fk(dh, q), tv),
                         sdk_euler_audit=errors(transform(pe, re), tv),
                         weighted_jacobian_min_singular_value=float(np.linalg.svd(
                             np.vstack([jp, 0.2*jr]), compute_uv=False)[-1])))
        mj_transforms.append(tm)
        rm_transforms.append(tv)
    mh, vh = mj_transforms[0], rm_transforms[0]
    # Home-derived transforms are diagnostics only. Validate on all OTHER poses;
    # never apply them to the primary comparison or claim calibrated frames.
    left_home, right_home = mh @ inverse(vh), inverse(vh) @ mh
    for row, tm, tv in zip(rows, mj_transforms, rm_transforms):
        relative_m, relative_v = inverse(mh) @ tm, inverse(vh) @ tv
        row.update(mujoco_relative_home=relative_m.tolist(), vendor_relative_home=relative_v.tolist(),
                   relative_home=errors(relative_m, relative_v),
                   base_displacement_difference_m=((tm[:3, 3]-mh[:3, 3])-(tv[:3, 3]-vh[:3, 3])).tolist(),
                   inferred_left_transform=(tm @ inverse(tv)).tolist(),
                   inferred_right_transform=(inverse(tv) @ tm).tolist(),
                   home_left_prediction=errors(tm, left_home @ tv),
                   home_right_prediction=errors(tm, tv @ right_home))
    joint_frames = []
    for i in range(6):
        bid = model.body(f"link_{i+1}").id
        local = transform(model.body_pos[bid], quaternion_matrix(model.body_quat[bid]))
        sdk_local = mdh_local(dh, i, 0)
        joint_frames.append(dict(joint=f"joint_{i+1}", mujoco_local_zero=local.tolist(),
                                 sdk_mdh_local_zero=sdk_local.tolist(), difference=errors(local, sdk_local)))
    summary = {key: maxima([r[key] for r in rows]) for key in
               ("absolute", "relative_home", "common_wrist", "sdk_mdh_audit", "sdk_euler_audit")}
    summary.update(home_left_validation=maxima([r["home_left_prediction"] for r in rows[1:]]),
                   home_right_validation=maxima([r["home_right_prediction"] for r in rows[1:]]),
                   numerically_consistent=summary["absolute"]["all_within_tolerance"],
                   same_robot_model_confirmed=False)
    return dict(sdk=vendor.info, world_from_mujoco_base=world_from_base.tolist(),
                base_alignment="Nominal URDF base and manufacturer MDH base; not a measured hardware extrinsic",
                tcp_correspondence_confirmed=False,
                mujoco_link6_fixed_translation_m=model.body_pos[link6].tolist(),
                joint_zero_frames=joint_frames, rows=rows, summary=summary,
                caveat="Sensor mapping and true TCP are unconfirmed. No fitted transform is used to declare equivalence.")


def run_comparison(generation="II", sensor="SF", path=control.SCENE):
    model, _, _ = control.load_model(path)
    kin = Kinematics(model)
    configurations = []
    for gen in MODELS if generation == "all" else [generation]:
        for force in SENSORS if sensor == "all" else [sensor]:
            configurations.append(measure_configuration(kin, gen, force))
    return dict(completed=True, motion_executed=False, simulation_steps=0,
                mujoco_version=mujoco.__version__, numpy_version=np.__version__,
                model_path=str(Path(path).resolve()),
                model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                position_tolerance_m=POSITION_TOLERANCE_M, rotation_tolerance_rad=ROTATION_TOLERANCE_RAD,
                configurations=configurations,
                all_numerically_consistent=all(c["summary"]["numerically_consistent"] for c in configurations),
                same_robot_model_confirmed=False)


def print_report(report):
    print("Offline FK only; 0 simulation steps. Sensor candidates are not confirmed hardware mappings.")
    print("p: metres, R: frame-to-base. Errors: SO(3) angle, not Euler subtraction.")
    print(f"Numerical criterion: {report['position_tolerance_m']*1000:g} mm / "
          f"{np.rad2deg(report['rotation_tolerance_rad']):g} deg; not real-arm accuracy.")
    for config in report["configurations"]:
        sdk = config["sdk"]
        print(f"\n{sdk['model']} ({sdk['model_value']}) / {sdk['sensor']} ({sdk['sensor_value']}); "
              f"SDK {sdk['sdk_version']}, algorithm {sdk['algorithm_version']}")
        print("Retrieved MDH:", sdk["dh"])
        print("pose                MuJoCo base xyz [m]                   SDK base xyz [m]                      dp_mm    dR_deg   relative_mm  relative_deg")
        for row in config["rows"]:
            mp, vp = np.array(row["mujoco_base"])[:3, 3], np.array(row["vendor_base"])[:3, 3]
            a, r = row["absolute"], row["relative_home"]
            print(f"{row['name']:19s} {np.array2string(mp, precision=8):39s} {np.array2string(vp, precision=8):39s} "
                  f"{a['position_error_m']*1000:9.6f} {np.rad2deg(a['rotation_error_rad']):9.6f} "
                  f"{r['position_error_m']*1000:12.6f} {np.rad2deg(r['rotation_error_rad']):12.6f}")
        for key in ("common_wrist", "home_left_validation", "home_right_validation", "sdk_mdh_audit", "sdk_euler_audit"):
            s = config["summary"][key]
            print(f"{key}: max {1000*s['max_position_error_m']:.6f} mm / "
                  f"{np.rad2deg(s['max_rotation_error_rad']):.6f} deg; within criterion={s['all_within_tolerance']}")
        status = "CONSISTENT_NUMERICALLY" if config["summary"]["numerically_consistent"] else "MISMATCH"
        print(f"[{status}] {config['caveat']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", choices=[*MODELS, "all"], default="II")
    parser.add_argument("--sensor", choices=[*SENSORS, "all"], default="SF", help="candidate, not confirmed 6F-V mapping")
    parser.add_argument("--model", type=Path, default=control.SCENE)
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/fk_comparison.json")
    args = parser.parse_args()
    try:
        report = run_comparison(args.generation, args.sensor, args.model)
        print_report(report)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"\nFull positions/rotation matrices/frame diagnostics: {args.json.resolve()}")
        return 0 if report["all_numerically_consistent"] else 2
    except (SDKUnavailable, OSError, ValueError, KeyError, RuntimeError, mujoco.FatalError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
