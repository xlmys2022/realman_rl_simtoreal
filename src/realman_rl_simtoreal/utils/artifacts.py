"""提供 JSON、JSONL、摘要和源码归档，具体实验决定字段和来源范围。"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import zipfile

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
