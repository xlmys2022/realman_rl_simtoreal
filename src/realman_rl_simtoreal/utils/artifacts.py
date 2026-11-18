"""Shared artifact I/O; callers retain their own serialization and provenance contracts."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import zipfile
from ..paths import PROJECT_ROOT, GRASP_CONTACT_SCENE


def experiment_metadata():
    """Fingerprint the actual combined robot, gripper and contact task."""
    assets={GRASP_CONTACT_SCENE}
    for folder in ('assets/robots/rml63','assets/grippers/eg2_4c2'):
        assets.update(p for p in (PROJECT_ROOT/folder).rglob('*') if p.is_file())
    sources=[PROJECT_ROOT/'src/realman_rl_simtoreal'/p for p in (
        'envs/grasp_lift_env.py','tasks/grasp_lift.py','evaluation/grasp_lift.py','utils/artifacts.py','evaluation/metrics.py',
        'simulation/grasp_controller.py','simulation/gripper.py','simulation/grasp_contact.py',
        'simulation/control.py','simulation/kinematics.py','simulation/model.py')]
    return dict(assets=file_hashes(sorted(assets),root=PROJECT_ROOT),
                sources=file_hashes(sorted(sources),root=PROJECT_ROOT),
                runtime=dependency_versions(('mujoco','numpy','gymnasium')),
                action_semantics=['world_dx','world_dy','world_dz','absolute_gripper_opening'],
                observation_dim=41,action_dim=4)


def write_json(path, value, *, ensure_ascii=True, allow_nan=False):
    """Write an indented JSON document; compatibility callers specify their flags."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=ensure_ascii, allow_nan=allow_nan)+'\n')


def append_json(path, value, *, ensure_ascii=True, allow_nan=False):
    """Append one JSONL record without implicitly creating a missing experiment."""
    with Path(path).open('a') as file:
        file.write(json.dumps(value, ensure_ascii=ensure_ascii, allow_nan=allow_nan)+'\n')


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def file_hashes(paths, *, root):
    """Keep caller-selected files, order and root-relative keys unchanged."""
    return {str(Path(p).relative_to(root)): file_sha256(p) for p in paths}


def dependency_versions(names):
    return {name: metadata.version(name) for name in names}


def runtime_versions(**runtime):
    """Record installed packages plus caller-supplied framework runtime fields."""
    return dict(python=sys.version, executable=sys.executable, **runtime,
                packages={d.metadata['Name']: d.version for d in metadata.distributions()})


def archive_sources(destination, paths, *, root):
    """Archive a caller-defined ordered source set; never select extra sources."""
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            path = Path(path)
            if path.is_file():
                archive.write(path, path.relative_to(root))


def empty_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output directory must be empty; never overwrite an experiment')
    path.mkdir(parents=True, exist_ok=True)
    return path
