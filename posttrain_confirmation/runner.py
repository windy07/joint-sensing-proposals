"""One frozen confirmation episode using the validated scientific evaluator."""
from contextlib import contextmanager
from pathlib import Path
import functools
import os
import sys
import time
import traceback
from common import read, write, sha, digest, adapter_digest, load_science
from design import audit_model
from frozen_contract import validate_frozen, validate_checkpoint


def validate_design(design, original):
    if design.get('design_digest') != digest({k: v for k, v in design.items() if k != 'design_digest'}):
        raise ValueError('DESIGN_CONTENT_CHANGED')
    if design['base_plan_digest'] != digest(original):
        raise ValueError('BASE_PLAN_CHANGED')
    if design['adapter_digest'] != adapter_digest():
        raise ValueError('ADAPTER_CHANGED_REBUILD_DEVELOPMENT_DESIGN')
    from spatial_lab.binding import source_digest
    if design['base_source_digest'] != source_digest():
        raise ValueError('SCIENTIFIC_SOURCE_CHANGED')
    validate_frozen(design)


@contextmanager
def intervention(job):
    """Process-local replacement, restored even when a case fails."""
    from spatial_lab import run
    original = run.run_policy
    if job['arm'] is not None:
        from policy import run_policy
        run.run_policy = functools.partial(run_policy, numeric_parameters=job['variant'][1] == '1',
                                          feedback_enabled=job['variant'][3] == '1')
    try:
        yield
    finally:
        run.run_policy = original


def run_job(project, original_path, design_path, job_id, models_root, output, *, device='cpu', execute=False):
    load_science(project)
    original, design = read(original_path), read(design_path)
    validate_design(design, original)
    jobs = [j for j in design['jobs'] if j['id'] == job_id]
    if len(jobs) != 1:
        raise ValueError('UNKNOWN_JOB')
    job = jobs[0]
    if job['split'] != 'confirmation':
        raise ValueError('CONFIRMATION_JOBS_ONLY')
    if not execute:
        return {'status': 'PREVIEW', 'job': job, 'actual_optimizer_updates': 0, 'actual_world_advances': 0}
    if sys.platform != 'linux':
        raise RuntimeError('EVALUATION_RUNS_ON_LINUX')
    import torch
    from spatial_lab import work_budget
    from spatial_lab.run import run_case
    from spatial_lab.spatial_graph import Priority
    from trace_export import export_trace
    torch.set_num_threads(1)
    if device.startswith('cuda'):
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        cap = float(os.environ.get('RADAR_EVAL_GPU_ALLOCATOR_GIB', '1.0'))
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(cap * 2**30 / total, 0)
    output = Path(output)
    receipt_path = output / 'receipts' / (job_id + '.json')
    case_path = output / 'cases' / (job_id + '.json.gz')
    lock = output / 'locks' / (job_id + '.lock')
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    lock.write_text(str(os.getpid()), encoding='ascii')
    receipt = {'job': job, 'design_digest': design['design_digest'], 'status': 'RUNNING',
               'contract_digest': design['contract_digest'],
               'device': device, 'pid': os.getpid(), 'started_ns': time.time_ns(), 'actual_optimizer_updates': 0}
    receipt['physical_gpu_uuid'] = os.environ.get('CUDA_VISIBLE_DEVICES') if device.startswith('cuda') else None
    try:
        audited = None
        if job['arm'] is not None:
            audited = audit_model(Path(models_root) / job['arm'] / str(job['model_seed']),
                                  job['arm'], job['model_seed'], original)
        expected_checkpoint = audited['checkpoint_sha256'] if audited else None
        if audited:
            validate_checkpoint(validate_frozen(design), job['arm'], job['model_seed'], expected_checkpoint)
        if receipt_path.exists():
            old = read(receipt_path)
            if old['status'] == 'COMPLETE':
                if (old['design_digest'] != design['design_digest'] or old['job'] != job
                    or old['checkpoint_sha256'] != expected_checkpoint or not case_path.is_file()
                    or old['case_sha256'] != sha(case_path)):
                    raise ValueError('COMPLETED_RESULT_BINDING_CHANGED')
                return {**old, 'reused': True, 'actual_world_advances_this_invocation': 0}
        receipt['checkpoint_sha256'] = expected_checkpoint
        write(receipt_path, receipt)
        learned = Priority(audited['checkpoint'], device=device, evaluation_split='confirmation') if audited else None
        if job['arm'] is None:
            method = job['variant']
        else:
            method = 'numeric_network' if job['variant'][1] == '1' else 'fixed_network'
        # Restart failed episodes from cycle zero. No partially committed cycle is reused.
        token = work_budget.enable_server(design['design_digest'])
        try:
            with work_budget.prohibit_optimizers(), intervention(job):
                result = run_case(job['world_seed'], job['configuration'], method, learned)
        finally:
            work_budget.disable_server(token)
        if len(result['trace']) != job['configuration']['cycles'] or result['world_advances'] != len(result['trace']):
            raise ValueError('INCOMPLETE_EPISODE')
        result['posttrain_evaluation'] = {'job': job, 'design_digest': design['design_digest'],
                                        'adapter_digest': design['adapter_digest'],
                                        'contract_digest': design['contract_digest'], 'device': device}
        write(case_path, result)
        exported = export_trace(result, output / 'tables' / job_id, job)
        receipt.update(status='COMPLETE', summary=result['summary'], case_sha256=sha(case_path),
                       actual_world_advances=result['world_advances'], tables=exported,
                       gpu_peak_allocated_gib=torch.cuda.max_memory_allocated(0)/2**30 if device.startswith('cuda') else None,
                       gpu_peak_reserved_gib=torch.cuda.max_memory_reserved(0)/2**30 if device.startswith('cuda') else None,
                       finished_ns=time.time_ns())
        write(receipt_path, receipt)
        return receipt
    except Exception as exc:
        failure = {**receipt, 'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc()}
        # Keep an existing successful receipt intact if its reuse check failed.
        dest = output / 'failures' / (job_id + '_' + str(time.time_ns()) + '.json')
        write(dest, failure)
        if not receipt_path.exists() or read(receipt_path).get('status') != 'COMPLETE':
            write(receipt_path, failure)
        raise
    finally:
        lock.unlink()
