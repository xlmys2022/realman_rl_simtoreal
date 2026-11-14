"""Shared JSON artifacts and unchanged dependency/resource fingerprints."""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import torch

from ...paths import PROJECT_ROOT


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append_json(path, value):
    with Path(path).open('a') as file:
        file.write(json.dumps(value, allow_nan=False)+'\n')


def versions():
    import sys
    return dict(python=sys.version, executable=sys.executable, torch_cuda=torch.version.cuda,
                packages={d.metadata['Name']: d.version for d in metadata.distributions()})


def asset_hashes():
    files = [p for p in (PROJECT_ROOT/'assets/robots/rml63').rglob('*') if p.is_file()]
    files += [PROJECT_ROOT/'src/realman_rl_simtoreal'/name for name in
              ('envs/reach_env.py', 'envs/sequence_reach_env.py', 'envs/factory.py',
               'simulation/kinematics.py', 'simulation/control.py', 'simulation/model.py')]
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def empty_directory(path):
    path = Path(path).resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output directory must be empty; never overwrite an experiment')
    path.mkdir(parents=True, exist_ok=True)
    return path
