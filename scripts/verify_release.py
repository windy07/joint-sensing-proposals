"""Check original source bytes and, optionally, the selected weights and environment."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b''): h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models-root', type=Path)
    ap.add_argument('--environment', action='store_true')
    args = ap.parse_args()
    files = json.loads((ROOT/'provenance/original_files.json').read_text(encoding='utf-8'))['files']
    for entry in files:
        if sha(ROOT/entry['path']) != entry['sha256']:
            raise ValueError('Changed original file: ' + entry['path'])
    result = {'original_files_verified':len(files), 'checkpoints_verified':0}
    if args.models_root:
        models = json.loads((ROOT/'checkpoints/manifest.json').read_text(encoding='utf-8'))['models']
        for entry in models:
            p = args.models_root/entry['arm']/str(entry['model_seed'])/'best.pt'
            if sha(p) != entry['sha256']:
                raise ValueError('Checkpoint mismatch: ' + str(p))
        result['checkpoints_verified'] = len(models)
    if args.environment:
        versions = {'torch':'2.10.0', 'numpy':'2.3.5', 'scipy':'1.17.0', 'threadpoolctl':'3.6.0'}
        actual = {k:importlib.metadata.version(k).split('+')[0] for k in versions}
        if actual != versions:
            raise ValueError('Install the pinned dependencies: ' + str(actual))
        sys.path.insert(0, str(ROOT/'project/ACTIVE/MAINLINE_RESEARCH/spatial_scheduling'))
        from spatial_lab.binding import source_digest
        plan=json.loads((ROOT/'project/STAGE6_SERVER/plan_12800.json').read_text(encoding='utf-8'))
        if source_digest() != plan['source_digest']:
            raise ValueError('Scientific source does not match the experiment plan')
        result['environment'] = actual
    print(json.dumps(result))


if __name__ == '__main__':
    main()
