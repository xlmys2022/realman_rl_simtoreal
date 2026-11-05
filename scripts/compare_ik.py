#!/usr/bin/env python3
"""Repeatable, offline self-model IK experiments and explicitly separate transfer diagnostics."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import control_joints as control
from compare_fk import sample_poses, transform, inverse
from kinematics import Kinematics, rotation_vector
from realman_fk import SDKUnavailable
from realman_ik import RealManIK, periodic_equivalent, POSITION_TOLERANCE_M, ORIENTATION_TOLERANCE_RAD
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def result_dict(result):
    values = asdict(result)
    if "q" in values:
        values["q"] = None if result.q is None else result.q.tolist()
    if "candidates" in values:
        values["candidates"] = [result_dict(r) for r in result.candidates]
    return values


def pose_errors(target_p, target_r, actual_p, actual_r):
    return dict(position_error_m=float(np.linalg.norm(np.asarray(target_p)-actual_p)),
                orientation_error_rad=float(np.linalg.norm(rotation_vector(np.asarray(target_r), actual_r))))


def seeds_for(home):
    return dict(home=home.copy(), near=home+np.deg2rad([2, -3, 4, 2, -2, 3]),
                other_elbow=np.deg2rad([0, 25, -55, 0, -20, 0]),
                far=np.deg2rad([100, 40, -70, 60, 80, 100]))


def measure(solver, fk, position, rotation, seed, repeats):
    # One untimed warm-up, then identical inputs. There are no random restarts.
    warm = solver(position, seed, rotation)
    times, native_times, stable = [], [], True
    result = warm
    for _ in range(repeats):
        start = perf_counter()
        result = solver(position, seed, rotation)
        times.append(perf_counter()-start)
        if hasattr(result, "native_elapsed_s"):
            native_times.append(result.native_elapsed_s)
        stable &= (result.success == warm.success and getattr(result, "sdk_code", None) == getattr(warm, "sdk_code", None))
        if result.success and warm.success:
            stable &= bool(np.allclose(result.q, warm.q, atol=1e-10, rtol=0))
    row = dict(seed_rad=np.asarray(seed).tolist(), target_position_m=np.asarray(position).tolist(),
               target_rotation=np.asarray(rotation).tolist(), result=result_dict(result),
               repeat_consistent=bool(stable), public_call_times_s=times,
               median_public_call_s=float(np.median(times)), native_call_times_s=native_times,
               independent_fk_audit=None, max_joint_change_rad=None, seed_distance_rad=None)
    if result.success:
        actual_p, actual_r = fk(result.q)
        audit = pose_errors(position, rotation, actual_p, actual_r)
        audit["passed"] = bool(audit["position_error_m"] <= POSITION_TOLERANCE_M
                               and audit["orientation_error_rad"] <= ORIENTATION_TOLERANCE_RAD)
        row.update(independent_fk_audit=audit, actual_position_m=actual_p.tolist(), actual_rotation=actual_r.tolist(),
                   max_joint_change_rad=float(np.max(np.abs(result.q-seed))),
                   seed_distance_rad=float(np.linalg.norm(result.q-seed)))
    elif result.q is not None:
        raise ValueError("Failed IK exposed joint angles instead of q=None")
    return row


def summarize(rows):
    successes = [r for r in rows if r["result"]["success"]]
    times = [t for r in rows for t in r["public_call_times_s"]]
    native = [t for r in rows for t in r["native_call_times_s"]]
    return dict(total=len(rows), successes=len(successes), success_rate=len(successes)/len(rows),
                max_position_error_m=max((r["independent_fk_audit"]["position_error_m"] for r in successes), default=None),
                max_orientation_error_rad=max((r["independent_fk_audit"]["orientation_error_rad"] for r in successes), default=None),
                median_public_call_s=float(np.median(times)), p95_public_call_s=float(np.percentile(times, 95)),
                median_native_call_s=float(np.median(native)) if native else None,
                repeat_consistent=all(r["repeat_consistent"] for r in rows),
                fk_audits_passed=all(r["independent_fk_audit"]["passed"] for r in successes))


def internal_experiment(solver, fk, poses, seeds, repeats):
    rows = []
    for name, q in poses:
        p, r = fk(q)  # Each solver gets ITS OWN FK target, never the other model's.
        for seed_name, seed in seeds.items():
            row = measure(solver, fk, p, r, seed, repeats)
            row.update(pose=name, seed_name=seed_name, source_q_rad=q.tolist())
            rows.append(row)
    return dict(rows=rows, summary=summarize(rows),
                by_seed={name: summarize([r for r in rows if r["seed_name"] == name]) for name in seeds})


def boundary_poses(home, limits):
    poses = []
    for i in range(6):
        for end, label in ((0, "lower"), (1, "upper")):
            q = home.copy()
            q[i] = limits[i, end] + np.deg2rad(.05)*(1 if end == 0 else -1)
            seed = q.copy()
            seed[i] += np.deg2rad(.1)*(1 if end == 0 else -1)
            poses.append((f"J{i+1}_{label}_inside_0.05deg", q, seed))
    return poses


def boundary_experiment(solver, fk, poses):
    rows = []
    for name, q, seed in poses:
        p, r = fk(q)
        row = measure(solver, fk, p, r, seed, 1)
        row.update(pose=name, source_q_rad=q.tolist())
        rows.append(row)
    return dict(rows=rows, summary=summarize(rows))


def unique_solutions(rows):
    distinct = []
    for row in rows:
        q = row["result"]["q"]
        if q is None:
            continue
        q = np.asarray(q)
        if not any(np.max(np.abs(np.arctan2(np.sin(q-d), np.cos(q-d)))) < 1e-3 for d in distinct):
            distinct.append(q)
    return [q.tolist() for q in distinct]


def multisolutions(vendor, poses, seeds, main_rows):
    groups = []
    for name, source in (poses[0], poses[-1]):
        p, r = vendor.fk(source)
        all_result = vendor.solve_all(p, seeds["home"], r)
        verified = [c.q for c in all_result.candidates if c.success]
        seed_checks = []
        for row in [r for r in main_rows if r["pose"] == name]:
            seed = np.array(row["seed_rad"])
            distances = [float(np.linalg.norm(periodic_equivalent(q, seed, vendor.limits)[0]-seed)) for q in verified]
            actual = row["seed_distance_rad"]
            nearest = min(distances) if distances else None
            seed_checks.append(dict(seed_name=row["seed_name"], selected_distance_rad=actual,
                                    nearest_enumerated_distance_rad=nearest,
                                    nearest_enumerated_l2=bool(actual is not None and nearest is not None
                                                              and abs(actual-nearest) < 1e-4)))
        groups.append(dict(pose=name, all_result=result_dict(all_result), seed_selection=seed_checks,
                           all_far_seed=result_dict(vendor.solve_all(p, seeds["far"], r)),
                           distinct_single_solutions=unique_solutions([r for r in main_rows if r["pose"] == name])))
    return groups


def continuity_experiment(vendor, poses, home):
    seed, rows = home.copy(), []
    for name, source in poses[:-1]:  # Small changes only; diverse pose is excluded.
        p, r = vendor.fk(source)
        row = measure(vendor.solve_ik, vendor.fk, p, r, seed, 1)
        row["pose"] = name
        rows.append(row)
        if not row["result"]["success"]:
            break
        seed = np.array(row["result"]["q"])
    return dict(rows=rows, summary=summarize(rows),
                max_joint_jump_rad=max((r["max_joint_change_rad"] for r in rows if r["result"]["success"]), default=None))


def investigate_boundary_failures(vendor, boundary_rows, home):
    """Independent probes, never retries that overwrite the original failure."""
    probes = []
    for row in boundary_rows:
        if row["result"]["success"]:
            continue
        p, r = np.array(row["target_position_m"]), np.array(row["target_rotation"])
        source = np.array(row["source_q_rad"])
        probes.append(dict(pose=row["pose"],
                           seed_at_target=measure(vendor.solve_ik, vendor.fk, p, r, source, 1),
                           seed_home=measure(vendor.solve_ik, vendor.fk, p, r, home, 1),
                           all_original_seed=result_dict(vendor.solve_all(p, np.array(row["seed_rad"]), r))))
    return probes


def cross_diagnostics(kin, vendor, poses):
    # Only the KNOWN world/base placement is transformed. No calibrated TCP
    # transform is available: these are deliberately different nominal endpoints.
    data = control.initialize(kin.model)
    bid = kin.model.body("base_link").id
    wb = transform(data.xpos[bid], data.xmat[bid].reshape(3, 3))
    bw = inverse(wb)
    rows = []
    for name, q in (poses[0], poses[-2], poses[-1]):
        for direction in ("vendor_pose_to_mujoco_ik", "mujoco_pose_to_vendor_ik"):
            if direction == "vendor_pose_to_mujoco_ik":
                pv, rv = vendor.fk(q)
                target = wb @ transform(pv, rv)
                result = kin.solve_ik(target[:3, 3], kin.home_q, target[:3, :3])
                other_target = transform(pv, rv)
                other_fk = vendor.fk
                own_fk = kin.fk
            else:
                pm, rm = kin.fk(q)
                target = bw @ transform(pm, rm)
                result = vendor.solve_ik(target[:3, 3], kin.home_q, target[:3, :3])
                other_target = transform(pm, rm)
                other_fk = kin.fk
                own_fk = vendor.fk
            row = dict(pose=name, direction=direction, result=result_dict(result),
                       requested_numeric_pose=target.tolist(), own_model_error=None, other_model_error=None)
            if result.success:
                row["own_model_error"] = pose_errors(target[:3, 3], target[:3, :3], *own_fk(result.q))
                row["other_model_error"] = pose_errors(other_target[:3, 3], other_target[:3, :3], *other_fk(result.q))
            rows.append(row)
    return dict(world_from_mujoco_base=wb.tolist(), tcp_transform=None,
                caveat="Nominal base alignment only; factory flange and ee_site differ. Model-transfer diagnostic, NOT a fair IK accuracy comparison.",
                rows=rows)


def run_comparison(repeats=5, sensors=("SF", "ISFV"), path=control.SCENE):
    if not isinstance(repeats, int) or not 1 <= repeats <= 20:
        raise ValueError("repeats must be an integer in [1,20]")
    if not sensors or len(set(sensors)) != len(sensors) or any(s not in ("SF", "ISFV") for s in sensors):
        raise ValueError("Select SF and/or ISFV candidates, sequentially")
    model, _, _ = control.load_model(path)
    kin = Kinematics(model)
    poses, seeds = sample_poses(kin.home_q), seeds_for(kin.home_q)
    mj = internal_experiment(kin.solve_ik, kin.fk, poses, seeds, repeats)
    boundaries = boundary_poses(kin.home_q, kin.limits)
    mj_boundaries = boundary_experiment(kin.solve_ik, kin.fk, boundaries)
    unreachable_p = np.array([10., 10., 10.])
    mj_unreachable = measure(kin.solve_ik, kin.fk, unreachable_p, kin.home_rotation, kin.home_q, 1)
    configs = []
    for sensor in sensors:
        vendor = RealManIK(kin.limits, "II", sensor)
        experiment = internal_experiment(vendor.solve_ik, vendor.fk, poses, seeds, repeats)
        boundary = boundary_experiment(vendor.solve_ik, vendor.fk, boundaries)
        _, home_r = vendor.fk(kin.home_q)
        unreachable = measure(vendor.solve_ik, vendor.fk, unreachable_p, home_r, kin.home_q, 1)
        configs.append(dict(sdk=vendor.info, internal=experiment,
                            boundaries=boundary,
                            boundary_failure_probes=investigate_boundary_failures(vendor, boundary["rows"], kin.home_q),
                            unreachable=unreachable,
                            unreachable_all=result_dict(vendor.solve_all(unreachable_p, kin.home_q, home_r)),
                            multisolutions=multisolutions(vendor, poses, seeds, experiment["rows"]),
                            continuity=continuity_experiment(vendor, poses, kin.home_q),
                            cross=cross_diagnostics(kin, vendor, poses)))
    summaries = [mj["summary"], mj_boundaries["summary"]]
    for c in configs:
        summaries.extend([c["internal"]["summary"], c["boundaries"]["summary"], c["continuity"]["summary"]])
    safe = all(s["repeat_consistent"] and s["fk_audits_passed"] for s in summaries)
    safe &= not mj_unreachable["result"]["success"] and all(
        not c["unreachable"]["result"]["success"] and not c["unreachable_all"]["success"] for c in configs)
    safe &= all(a["success"] and all(
        r["success"] or r["reason"] == "No periodic equivalent satisfies BOTH manufacturer/model joint limits"
        for r in a["candidates"]) for c in configs for g in c["multisolutions"]
                for a in (g["all_result"], g["all_far_seed"]))
    anomalies = []
    for c in configs:
        rows = c["internal"]["rows"] + c["boundaries"]["rows"] + c["continuity"]["rows"] + c["cross"]["rows"]
        for row in rows:
            r = row["result"]
            if r.get("sdk_code") == 0 and not r["success"] and r["reason"] != "No periodic equivalent satisfies BOTH manufacturer/model joint limits":
                anomalies.append(dict(sensor=c["sdk"]["sensor"], pose=row["pose"], reason=r["reason"]))
    safe &= not anomalies
    return dict(completed=True, motion_executed=False, simulation_steps=0,
                same_robot_model_confirmed=False, safety_audits_passed=bool(safe), integrity_anomalies=anomalies, repeats=repeats,
                timing_scope="Warmed public solve calls; vendor includes conversion/range/FK audits. Native SDK method time also recorded. Extra experiment FK audit is outside timing.",
                position_tolerance_m=POSITION_TOLERANCE_M, orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
                mujoco_version=mujoco.__version__, numpy_version=np.__version__, model_path=str(Path(path).resolve()),
                mujoco_internal=mj, mujoco_boundaries=mj_boundaries, mujoco_unreachable=mj_unreachable,
                configurations=configs)


def print_report(report):
    print("Offline only; each solver uses its OWN FK targets. No controllers or simulation steps.")
    print("Same-robot model remains UNCONFIRMED; SF/ISFV are candidates.")
    items = [("MuJoCo DLS", report["mujoco_internal"])]
    items.extend((c["sdk"]["sensor"], c["internal"]) for c in report["configurations"])
    print("solver                 verified/total max_pos_m      max_rot_rad    median_public_ms  p95_public_ms")
    for label, experiment in items:
        s = experiment["summary"]
        print(f"{label:23s} {s['successes']:2d}/{s['total']:<2d} {s['max_position_error_m']!s:14s} "
              f"{s['max_orientation_error_rad']!s:14s} {1000*s['median_public_call_s']:.6f} {1000*s['p95_public_call_s']:.6f}")
        for seed, stats in experiment["by_seed"].items():
            print(f"  seed={seed}: {stats['successes']}/{stats['total']}")
    for c in report["configurations"]:
        print(f"\n{c['sdk']['sensor']}: unreachable SDK code={c['unreachable']['result']['sdk_code']}, "
              f"q={c['unreachable']['result']['q']}; limits={c['sdk']['vendor_limits_rad']}")
        for group in c["multisolutions"]:
            a = group["all_result"]
            print(f"  {group['pose']}: native branches={a['reported_count']}, "
                  f"verified legal branches={sum(r['success'] for r in a['candidates'])}, "
                  f"single-solve distinct branches={len(group['distinct_single_solutions'])}")
        print(f"  boundary self-FK targets: {c['boundaries']['summary']['successes']}/12")
        for probe in c["boundary_failure_probes"]:
            print(f"  BOUNDARY FAILURE {probe['pose']}: same-seed code={probe['seed_at_target']['result']['sdk_code']}; "
                  f"home-seed code={probe['seed_home']['result']['sdk_code']}; "
                  f"all-solve accepted={sum(r['success'] for r in probe['all_original_seed']['candidates'])}")
        for row in c["cross"]["rows"]:
            print(f"  TRANSFER DIAGNOSTIC {row['pose']} {row['direction']}: success={row['result']['success']}; "
                  f"other-model error={row['other_model_error']}")
    print(f"MuJoCo unreachable: {report['mujoco_unreachable']['result']['reason']}")
    print(f"Completed={report['completed']}; safety audits={report['safety_audits_passed']}; model equivalence=False")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensor", choices=("SF", "ISFV", "both"), default="both")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model", type=Path, default=control.SCENE)
    parser.add_argument("--json", type=Path, default=ROOT / "outputs/ik_comparison.json")
    args = parser.parse_args()
    try:
        report = run_comparison(args.repeats, ("SF", "ISFV") if args.sensor == "both" else (args.sensor,), args.model)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print_report(report)
        print(f"Full per-pose/seed/error/timing results: {args.json.resolve()}")
        return 0 if report["safety_audits_passed"] else 2
    except (SDKUnavailable, OSError, ValueError, KeyError, RuntimeError, mujoco.FatalError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
