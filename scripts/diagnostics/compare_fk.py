#!/usr/bin/env python3
"""Offline RM_API2/MuJoCo FK measurements; exit 2 means model mismatch, not a test crash."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

from realman_rl_simtoreal.apps.visualization.viewer import configure_backend
configure_backend()
from realman_rl_simtoreal.simulation.physics import model as model_api
from realman_rl_simtoreal.paths import OUTPUTS
from realman_rl_simtoreal.hardware.realman_config import REALMAN_HARDWARE
from realman_rl_simtoreal.utils.geometry import transform, inverse, modified_dh_transform
from realman_rl_simtoreal.utils.demo_helpers import sample_poses
from realman_rl_simtoreal.simulation.motion.kinematics import Kinematics, rotation_vector
from realman_rl_simtoreal.vendor.realman_fk import (
    RealManFK,
    SDKUnavailable,
    MODELS,
    SENSORS,
    quaternion_matrix,
    euler_matrix,
)
import mujoco
import numpy as np

# 仅用于同一组数值模型的对照，实机姿态精度另外测量。
POSITION_TOLERANCE_M = 1e-5
ROTATION_TOLERANCE_RAD = np.deg2rad(0.001)


def errors(a, b):
    dp = a[:3, 3] - b[:3, 3]
    angle = float(np.linalg.norm(rotation_vector(a[:3, :3], b[:3, :3])))
    distance = float(np.linalg.norm(dp))
    return dict(position_difference_m=dp.tolist(), position_error_m=distance,
                rotation_error_rad=angle, within_tolerance=bool(
                    distance <= POSITION_TOLERANCE_M and angle <= ROTATION_TOLERANCE_RAD))


def mdh_local(dh, index, angle):
    """SDK 扭角与偏置用度，输入关节角 angle 已为弧度。"""
    alpha, offset = np.deg2rad([dh["alpha"][index], dh["offset"][index]])
    return modified_dh_transform(dh["a"][index], dh["d"][index], alpha, offset + angle)


def mdh_fk(dh, q):
    """Audit the retrieved parameters/conventions against native FK, not a controller."""
    result = np.eye(4)
    for i, angle in enumerate(q):
        result = result @ mdh_local(dh, i, angle)
    return result


def maxima(records):
    return dict(max_position_error_m=max(r["position_error_m"] for r in records),
                max_rotation_error_rad=max(r["rotation_error_rad"] for r in records),
                all_within_tolerance=all(r["within_tolerance"] for r in records))


def measure_configuration(kin, generation=REALMAN_HARDWARE.arm_generation, sensor=REALMAN_HARDWARE.sdk_force_type):
    """Compare frames without fitting or changing either robot's parameters."""
    model = kin.model
    data = model_api.initialize(model)  # Private reference data, never integrated.
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
                caveat="Nominal SDK dimensions are compared at the flange without a fitted transform.")


def run_comparison(generation=REALMAN_HARDWARE.arm_generation, sensor=REALMAN_HARDWARE.sdk_force_type,
                   path=model_api.SCENE):
    model, _, _ = model_api.load_model(path)
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


def compare_controller_dh(rows, profile):
    """与控制器读回的 MDH 对照。姿态使用基座到法兰，SDK 名义 DH 另行保留。"""
    measured = profile.controller_dh
    dh = dict(a=measured.a_m, d=measured.d_m, alpha=measured.alpha_deg, offset=measured.offset_deg)
    comparisons = []
    for row in rows:
        expected = mdh_fk(dh, np.deg2rad(row["joint_deg"]))
        comparisons.append(dict(name=row["name"], controller_base=expected.tolist(),
                                **errors(np.array(row["mujoco_base"]), expected)))
    return dict(summary=maxima(comparisons), rows=comparisons)


def print_report(report):
    print("Offline FK; 0 simulation steps. The SDK rows use nominal model dimensions.")
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
    parser.add_argument("--generation", choices=[*MODELS, "all"],
                        help=f"offline SDK generation (default {REALMAN_HARDWARE.arm_generation})")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--sensor", choices=[*SENSORS, "all"],
                          help=f"offline SDK sensor (default {REALMAN_HARDWARE.sdk_force_type})")
    selector.add_argument("--hardware-profile", action="store_true",
                          help="include a separate FK comparison against the controller's recorded DH")
    parser.add_argument("--model", type=Path, default=model_api.SCENE)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    try:
        profile = REALMAN_HARDWARE if args.hardware_profile else None
        if profile is not None:
            generation = args.generation or profile.arm_generation
            sensor = profile.sdk_force_type
            print(f"Recorded hardware {profile.model}, offline candidates {generation}/{sensor}")
        else:
            generation = args.generation or REALMAN_HARDWARE.arm_generation
            sensor = args.sensor or REALMAN_HARDWARE.sdk_force_type
        report = run_comparison(generation, sensor, args.model)
        if profile is not None:
            report["hardware_profile"] = asdict(profile)
            report["controller_dh_alignment"] = compare_controller_dh(report["configurations"][0]["rows"], profile)
        output_name = "fk_hardware_comparison.json" if profile else "fk_comparison.json"
        output = args.json or OUTPUTS / "simulation/kinematics" / output_name
        print_report(report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(f"\nFull positions/rotation matrices/frame diagnostics: {output.resolve()}")
        if profile is not None:
            alignment = report["controller_dh_alignment"]["summary"]
            print(f"Controller DH alignment: {alignment['max_position_error_m']*1000:.6f} mm, "
                  f"{alignment['max_rotation_error_rad']:.6g} rad")
            return 0 if alignment["all_within_tolerance"] else 2
        return 0 if report["all_numerically_consistent"] else 2
    except (SDKUnavailable, OSError, ValueError, KeyError, RuntimeError, mujoco.FatalError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
