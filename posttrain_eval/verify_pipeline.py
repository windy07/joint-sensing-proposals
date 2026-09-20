"""One full, 64-cycle development smoke run with a retained trained checkpoint.

Copies or hard-links supplied model artifacts into a private test layout. It never
trains, changes supplied artifacts, or executes the confirmation partition.
"""
from pathlib import Path
import argparse
import csv
import os
import shutil
import sys
import time
from common import read, write, digest, load_science, sha
from design import make_design
from runner import run_job


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--model-directory', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('FULL_EPISODE_VERIFICATION_RUNS_ON_LINUX')
    started = time.perf_counter()
    project, out = Path(args.project), Path(args.output)
    original_path = project / 'STAGE6_SERVER/plan_12800.json'
    original = read(original_path)
    design = make_design(original)
    design['design_digest'] = digest(design)
    design_path = out / 'development_design.json'
    write(design_path, design)
    model_root = out / '_local_models'
    dest = model_root / 'integrated_agent/633431'
    dest.mkdir(parents=True, exist_ok=True)
    for name in ('best.pt', 'epoch_010.pt', 'TRAIN_RECEIPT.json'):
        source, target = Path(args.model_directory) / name, dest / name
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        if sha(source) != sha(target):
            raise ValueError('LOCAL_MODEL_FIXTURE_CHANGED')
    chosen = next(j for j in design['jobs'] if j['split'] == 'development' and
                  j['arm'] == 'integrated_agent' and j['model_seed'] == 633431 and j['variant'] == 'n0f1')
    confirm = next(j for j in design['jobs'] if j['split'] == 'confirmation')
    result_root = out / 'smoke_results'
    preview = run_job(project, original_path, design_path, chosen['id'], model_root, result_root)
    assert preview['status'] == 'PREVIEW' and preview['actual_world_advances'] == 0
    try:
        run_job(project, original_path, design_path, confirm['id'], model_root, result_root, execute=True)
    except ValueError as exc:
        assert 'CONFIRMATION_NOT_FROZEN' in str(exc)
    else:
        raise AssertionError('CONFIRMATION_STARTED')
    before = time.perf_counter()
    receipt = run_job(project, original_path, design_path, chosen['id'], model_root, result_root, execute=True)
    elapsed = time.perf_counter() - before
    reused = run_job(project, original_path, design_path, chosen['id'], model_root, result_root, execute=True)
    assert reused['reused'] and reused['actual_world_advances_this_invocation'] == 0
    result = read(result_root / 'cases' / (chosen['id'] + '.json.gz'))
    assert len(result['trace']) == 64 and result['world_advances'] == 64
    assert len(result['memory_lifecycle']) == 64
    assert all(c['commit']['meter']['numeric_materializations'] == 0 for c in result['trace'])
    assert all(c['commit']['meter']['score_calls'] <= 12 for c in result['trace'])
    assert all(c['commit']['meter']['construction'] <= chosen['configuration']['construction_cap'] for c in result['trace'])
    assert all(not c['commit']['failures'] for c in result['trace'])
    with (result_root / 'tables' / chosen['id'] / 'cycles.csv').open(encoding='utf-8-sig', newline='') as stream:
        cycles = list(csv.DictReader(stream))
    with (result_root / 'tables' / chosen['id'] / 'actions.csv').open(encoding='utf-8-sig', newline='') as stream:
        actions = list(csv.DictReader(stream))
    assert len(cycles) == 64 and sorted({int(r['epoch']) for r in actions}) == list(range(64))
    from spatial_lab.binding import source_digest
    report = {'status': 'PASSED', 'kind': 'LOCAL_PIPELINE_SMOKE_NOT_PERFORMANCE_COMPARISON',
              'job_id': chosen['id'], 'world_seed': chosen['world_seed'], 'variant': chosen['variant'],
              'model_seed': chosen['model_seed'], 'actual_world_advances': 0 if receipt.get('reused') else 64,
              'completed_case_world_advances': 64, 'actual_optimizer_updates': 0,
              'source_digest': source_digest(), 'confirmation_not_executed': True,
              'resume_reused_identical_completed_result': True, 'cycle_rows': len(cycles), 'action_rows': len(actions),
              'feedback_cycles': sum(c['commit']['feedback_object'] is not None for c in result['trace']),
              'single_memory_update_each_cycle': True, 'numeric_parameters_unchanged': True,
              'score_failures': 0, 'episode_invocation_seconds': elapsed, 'total_seconds': time.perf_counter() - started}
    write(out / 'PIPELINE_SMOKE.json', report)
    print(__import__('json').dumps(report))


if __name__ == '__main__':
    main()
