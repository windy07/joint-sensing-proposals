"""Check retained artifacts and their frozen bindings before statistical reporting."""
import argparse
import csv
import math
from pathlib import Path
from common import read, write, sha, load_science
from frozen_contract import validate_frozen, validate_checkpoint
from runner import validate_design


def verify(root, design_path, output):
    root, output = Path(root), Path(output)
    load_science(root/'project')
    design = read(design_path)
    validate_design(design, read(root/'project/STAGE6_SERVER/plan_12800.json'))
    contract = validate_frozen(design)
    jobs = {j['id']: j for j in design['jobs']}
    receipts = {p.stem: p for p in (output/'receipts').glob('*.json')}
    if set(receipts) != set(jobs):
        raise ValueError('RESULT_SET_DOES_NOT_MATCH_FROZEN_MATRIX')
    artifacts = []
    for ident, job in jobs.items():
        row = read(receipts[ident])
        if (row.get('status') != 'COMPLETE' or row.get('job') != job
                or row.get('design_digest') != design['design_digest']
                or row.get('contract_digest') != design['contract_digest']
                or row.get('actual_optimizer_updates') != 0
                or row.get('actual_world_advances') != job['configuration']['cycles']):
            raise ValueError('INVALID_COMPLETED_RECEIPT:' + ident)
        if job['arm'] is not None:
            validate_checkpoint(contract, job['arm'], job['model_seed'], row.get('checkpoint_sha256'))
        elif row.get('checkpoint_sha256') is not None:
            raise ValueError('ANALYTIC_BASELINE_HAS_CHECKPOINT:' + ident)
        if not math.isfinite(row['summary']['mean_gospa_m']):
            raise ValueError('NONFINITE_PRIMARY_METRIC:' + ident)
        case = output/'cases'/(ident+'.json.gz')
        if not case.is_file() or sha(case) != row['case_sha256']:
            raise ValueError('RETAINED_CASE_CHANGED:' + ident)
        entry = {'job': ident, 'case_sha256': row['case_sha256'],
                 'receipt_sha256': sha(receipts[ident]), 'tables': {}}
        for name, expected in (('cycles', job['configuration']['cycles']),
                               ('actions', row['tables']['action_rows'])):
            path = output/'tables'/ident/(name+'.csv')
            with path.open(encoding='utf-8-sig', newline='') as stream:
                count = sum(1 for _ in csv.DictReader(stream))
            if count != expected:
                raise ValueError('EXPORTED_TABLE_COUNT_CHANGED:' + ident + ':' + name)
            entry['tables'][name] = {'rows': count, 'sha256': sha(path)}
        artifacts.append(entry)
    result = {'status': 'COMPLETE', 'episodes': len(jobs),
              'worlds': len({j['world_seed'] for j in jobs.values()}),
              'contract_digest': design['contract_digest'], 'artifacts': artifacts,
              'actual_optimizer_updates': 0, 'actual_world_advances': 0}
    write(output/'RESULT_INTEGRITY.json', result)
    return {k: v for k, v in result.items() if k != 'artifacts'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('root', 'design', 'output'):
        ap.add_argument('--'+name, required=True)
    args = ap.parse_args()
    print(verify(args.root, args.design, args.output))
