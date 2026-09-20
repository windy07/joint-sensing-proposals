"""Portable post-training tools. The registered training source is read-only."""
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def write(path, value):
    """Unique temporary name and bounded replacement retry on Windows readers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode_numeric(value):
        if hasattr(value, 'tolist'):
            return value.tolist()
        raise TypeError('NOT_JSON_SERIALIZABLE:' + type(value).__name__)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                      default=encode_numeric).encode()
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
        for attempt in range(8):
            try:
                os.replace(name, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(.025 * 2**attempt)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def adapter_digest():
    return digest({p.name: sha(p) for p in sorted(ROOT.glob('*.py'))})


def load_science(project):
    project = Path(project).resolve()
    science = project / 'ACTIVE/MAINLINE_RESEARCH/spatial_scheduling'
    if not (science / 'spatial_lab/scheduling.py').is_file():
        raise ValueError('NOT_A_STAGE6_PROJECT')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    loaded = sys.modules.get('spatial_lab')
    if loaded and Path(loaded.__file__).resolve().parent != science / 'spatial_lab':
        raise ValueError('DIFFERENT_PROJECT_ALREADY_IMPORTED')
    sys.path.insert(0, str(science))
    return science
