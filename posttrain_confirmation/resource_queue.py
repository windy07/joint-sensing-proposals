"""Resource-tuned queue around the unchanged, bound development evaluator."""
from pathlib import Path
from collections import Counter
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
ADAPTER = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTER))
from common import read, write, sha, load_science
from runner import validate_design
from resource_policy import read_memory, additions

GIB = 2**30


def alive(pid):
    try:
        return bool(pid and Path(f'/proc/{pid}/cmdline').read_bytes())
    except (FileNotFoundError, ProcessLookupError):
        return False


def resources(previous=None):
    root = Path('/sys/fs/cgroup')
    q, period = (root / 'cpu.max').read_text().split()
    quota = len(os.sched_getaffinity(0)) if q == 'max' else int(q) / int(period)
    stat = dict(line.split() for line in (root / 'cpu.stat').read_text().splitlines())
    now = time.monotonic()
    cpu = {'t': now, 'usage': int(stat['usage_usec'])}
    used = ((cpu['usage'] - previous['usage']) / 1e6 / (now - previous['t'])) if previous else 0.
    maximum = (root / 'memory.max').read_text().strip()
    if maximum == 'max':
        maximum = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                           if x.startswith('MemTotal:'))) * 1024
    else:
        maximum = int(maximum)
    current = int((root / 'memory.current').read_text())
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    gpus = []
    for line in text.strip().splitlines():
        i, uuid, total, free, util = [v.strip() for v in line.split(',')]
        gpus.append({'index': int(i), 'uuid': uuid, 'total_gib': float(total)/1024,
                     'free_gib': float(free)/1024, 'utilization': float(util)})
    procs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                     '--format=csv,noheader,nounits'], text=True, timeout=15)
    for gpu in gpus:
        gpu['compute_pids'] = [int(line.split(',')[1]) for line in procs.splitlines()
                               if line.split(',')[0].strip() == gpu['uuid']]
    return {'cpu_quota': min(quota, len(os.sched_getaffinity(0))), 'cpu_used': used,
            **read_memory(), 'gpus': gpus}, cpu


def training(root):
    occupied, ready, unfinished = set(), set(), []
    run = Path(root) / 'run'
    for path in (run / 'LINUX_EPOCH10_RUN/receipts').glob('*.json'):
        row = read(path)
        if row.get('gpu') is not None and alive(row.get('pid')):
            occupied.add(row['gpu'])
    for path in (run / 'models').glob('*/*/TRAIN_RECEIPT.json'):
        row = read(path)
        ident = (path.parent.parent.name, int(path.parent.name))
        files = all((path.parent / name).is_file() for name in ('best.pt', 'epoch_010.pt'))
        if row.get('status') == 'COMPLETE' and row.get('optimizer_updates') == 32000 and files:
            ready.add(ident)
        else:
            unfinished.append({'arm': ident[0], 'seed': ident[1], 'updates': row.get('optimizer_updates'),
                               'status': row.get('status')})
    return occupied, ready, unfinished


def release_count(gpus, occupied, own_pids, active_gpus):
    available = [g for g in gpus if g['index'] not in occupied and
                 all(p in own_pids or not alive(p) for p in g['compute_pids'])]
    count = sum(g['free_gib'] >= 28. or g['index'] in active_gpus for g in available)
    return count, available


def main():
    if sys.platform != 'linux':
        raise RuntimeError('LINUX_QUEUE_REQUIRED')
    import fcntl
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--design', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--max-workers', type=int, default=64)
    ap.add_argument('--gpu-workers', type=int, default=24)
    ap.add_argument('--gpu-allocator-gib', type=float, default=1.0)
    ap.add_argument('--required-free-gpus', type=int, default=2)
    ap.add_argument('--poll-seconds', type=float, default=3.)
    ap.add_argument('--jobs-file', help='Explicit development-only resource diagnostic subset')
    ap.add_argument('--pilot-evidence', help='Completed evaluation receipts for the same adapter and GPUs')
    args = ap.parse_args()
    if args.required_free_gpus < 1 or not 1 <= args.max_workers <= 70:
        raise ValueError('REQUIRE_RELEASED_GPUS_AND_AT_MOST_SEVENTY_WORKERS')
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    project = root / 'project'
    original_path = project / 'STAGE6_SERVER/plan_12800.json'
    load_science(project)
    design, original = read(args.design), read(original_path)
    validate_design(design, original)
    jobs = [j for j in design['jobs'] if j['split'] == 'confirmation']
    if args.jobs_file:
        raise ValueError('CONFIRMATION_REQUIRES_FULL_FROZEN_MATRIX')
    if args.pilot_evidence and out == Path(args.pilot_evidence).resolve():
        raise ValueError('RESOURCE_DIAGNOSTIC_MUST_PRESERVE_ORIGINAL_RESULTS')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'coordinator.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(out / 'EXECUTION.json', {'design_digest': design['design_digest'], 'arguments': vars(args),
          'pid': os.getpid(), 'mode': 'confirmation', 'learned_backend': 'CUDA float32',
          'nonlearning_backend': 'CPU float64 physical calculations', 'training_updates': 0,
          'gpu_rule': 'no live training worker or foreign compute process on an admitted GPU',
          'world_count': len({j['world_seed'] for j in jobs}), 'episode_limit': len(jobs),
          'resource_queue_sha256': sha(__file__),
          'resource_policy_sha256': sha(Path(__file__).with_name('resource_policy.py')),
          'purpose': 'frozen confirmation', 'contract_digest': design['contract_digest']})
    complete, failed = set(), set()
    expected = {j['id']: j for j in jobs}
    for path in (out / 'receipts').glob('*.json'):
        row = read(path)
        if row['design_digest'] != design['design_digest']:
            raise ValueError('QUEUE_RESULT_DESIGN_CHANGED')
        if row.get('job') != expected.get(row.get('job', {}).get('id')):
            raise ValueError('QUEUE_RESULT_JOB_CHANGED')
        if row['status'] == 'COMPLETE':
            case = out / 'cases' / (row['job']['id'] + '.json.gz')
            if not case.is_file() or sha(case) != row['case_sha256']:
                raise ValueError('QUEUE_COMPLETED_CASE_CHANGED')
            complete.add(row['job']['id'])
        elif row['status'] == 'RUNNING' and alive(row.get('pid')):
            raise ValueError('PREVIOUS_EVALUATION_WORKER_STILL_RUNNING')
        else:
            failed.add(row['job']['id'])
    active, gpu_successes = {}, Counter()
    if args.pilot_evidence:
        evidence = Path(args.pilot_evidence)
        initial_hw, _ = resources()
        uuid_index = {g['uuid']: g['index'] for g in initial_hw['gpus']}
        parent_design = read(root/'posttrain_eval/development_design.json')
        if parent_design['design_digest'] != design['parent_development_design_digest']:
            raise ValueError('GPU_PILOT_PARENT_DESIGN_CHANGED')
        for shared in ('policy.py', 'trace_export.py', 'paired_statistics.py'):
            if sha(ADAPTER/shared) != sha(root/'posttrain_eval'/shared):
                raise ValueError('GPU_PILOT_SHARED_BEHAVIOR_CHANGED:' + shared)
        design_jobs = {j['id']: j for j in parent_design['jobs'] if j['split'] == 'development'}
        reused = []
        for path in sorted((evidence/'receipts').glob('*.json')):
            row = read(path)
            gpu = uuid_index.get(row.get('physical_gpu_uuid'))
            if gpu is None or gpu_successes[gpu] >= 2 or row.get('status') != 'COMPLETE':
                continue
            job = row.get('job', {})
            case = evidence/'cases'/(job.get('id', '')+'.json.gz')
            if (row.get('design_digest') != parent_design['design_digest'] or job != design_jobs.get(job.get('id'))
                or row.get('actual_optimizer_updates') != 0
                or row.get('actual_world_advances') != job['configuration']['cycles']
                or not case.is_file() or sha(case) != row.get('case_sha256')):
                raise ValueError('INVALID_GPU_PILOT_EVIDENCE')
            if (row.get('gpu_peak_allocated_gib') or 0) >= args.gpu_allocator_gib:
                raise ValueError('PILOT_EXCEEDS_GPU_ALLOCATOR_CAP')
            gpu_successes[gpu] += 1
            reused.append({'job': job['id'], 'gpu': gpu, 'source': str(path)})
            if len(reused) == 2*len(uuid_index):
                break
        write(out/'REUSED_GPU_PILOTS.json', reused)
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous, last_ramp, ceiling = None, time.monotonic(), min(16, args.max_workers)
    initial_events = read_memory()['memory_events']
    while True:
        for ident, entry in list(active.items()):
            if entry['process'].poll() is None:
                continue
            for stream in entry['streams']:
                stream.close()
            path = out / 'receipts' / (ident + '.json')
            row = read(path) if path.exists() else {}
            if entry['process'].returncode == 0 and row.get('status') == 'COMPLETE':
                complete.add(ident)
                if entry['gpu'] is not None:
                    gpu_successes[entry['gpu']] += 1
            else:
                failed.add(ident)
                write(out / 'queue_failures' / (ident + '.json'),
                      {'job': entry['job'], 'returncode': entry['process'].returncode,
                       'worker_receipt': row, 'log_directory': str(out / 'logs')})
            del active[ident]
        hw, previous = resources(previous)
        if any(hw['memory_events'].get(k, 0) > initial_events.get(k, 0) for k in ('oom', 'oom_kill')):
            stopping[0] = True
            write(out/'MEMORY_STOP.json', {'before': initial_events, 'after': hw['memory_events']})
        occupied, ready, unfinished = training(root)
        own_pids = {x['process'].pid for x in active.values()}
        released_count, available_gpus = release_count(hw['gpus'], occupied, own_pids,
                                                       {x['gpu'] for x in active.values()})
        release_gate = released_count >= args.required_free_gpus
        if time.monotonic()-last_ramp >= 15 and not failed:
            ceiling = min(args.max_workers, ceiling+16)
            last_ramp = time.monotonic()
        capacity = additions(hw, len(active), min(args.max_workers, ceiling))
        capacity = min(capacity, 8)
        if (out / 'STOP').exists():
            stopping[0] = True
        if release_gate and not failed and not stopping[0]:
            # Interleave worlds/methods from the deterministic design; no metric-based prioritization.
            for job in jobs:
                if capacity <= 0:
                    break
                ident = job['id']
                if ident in complete or ident in active:
                    continue
                device, gpu = 'cpu', None
                if job['arm'] is not None:
                    if (job['arm'], job['model_seed']) not in ready:
                        continue
                    options = []
                    for candidate in available_gpus:
                        count = sum(x['gpu'] == candidate['index'] for x in active.values())
                        # First two full episodes calibrate each released GPU before a wider ramp.
                        cap = args.gpu_workers if gpu_successes[candidate['index']] >= 2 else 2
                        if count < cap and candidate['free_gib'] > args.gpu_allocator_gib + 2.:
                            options.append((count, -candidate['free_gib'], candidate))
                    if not options:
                        continue
                    gpu = min(options, key=lambda x: x[:2])[2]
                    device = 'cuda:0'
                    gpu['free_gib'] -= args.gpu_allocator_gib + .6
                logdir = out / 'logs'
                logdir.mkdir(exist_ok=True)
                streams = [(logdir / (ident + '.' + name + '.txt')).open('ab') for name in ('stdout', 'stderr')]
                env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
                       'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONIOENCODING': 'utf-8',
                       'CUDA_VISIBLE_DEVICES': gpu['uuid'] if gpu else '',
                       'RADAR_EVAL_GPU_ALLOCATOR_GIB': str(args.gpu_allocator_gib)}
                command = [sys.executable, '-B', '-u', str(ADAPTER/'posttrain.py'),
                    'run-job', '--project', str(project), '--server-plan', str(original_path),
                    '--design', str(Path(args.design).resolve()), '--job', ident,
                    '--models-root', str(root / 'run/models'), '--output', str(out), '--device', device, '--execute']
                proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=streams[0], stderr=streams[1],
                                        env=env, start_new_session=True)
                active[ident] = {'process': proc, 'streams': streams, 'job': job,
                                 'gpu': gpu['index'] if gpu else None}
                capacity -= 1
        body = {'status': 'FAILED_DRAINING' if failed and active else 'FAILED' if failed else
                         'STOPPING' if stopping[0] and active else 'STOPPED' if stopping[0] else
                         'COMPLETE' if len(complete) == len(jobs) else
                         'WAITING_FOR_RELEASED_GPUS' if not release_gate and not active else 'RUNNING',
                'updated_ns': time.time_ns(), 'pid': os.getpid(), 'complete': len(complete), 'failed': len(failed),
                'running': len(active), 'total': len(jobs), 'pending': len(jobs)-len(complete)-len(failed)-len(active),
                'active': [{'job': x['job']['id'], 'arm': x['job']['arm'], 'seed': x['job']['model_seed'],
                            'gpu': x['gpu'], 'pid': x['process'].pid} for x in active.values()],
                'resources': hw, 'training_gpus': sorted(occupied), 'models_ready': len(ready),
                'training_unfinished': unfinished, 'admission_ceiling': ceiling,
                'released_gpus': released_count, 'required_released_gpus': args.required_free_gpus,
                'waiting_for_released_gpu': not release_gate, 'optimizer_updates': 0}
        write(out / 'QUEUE_STATUS.json', body)
        with (out/'RESOURCE_HISTORY.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({k: body[k] for k in ('updated_ns', 'complete', 'running', 'pending',
                               'failed', 'resources', 'admission_ceiling')})+'\n')
        print(json.dumps({k:v for k,v in body.items() if k not in ('active','resources','training_unfinished')}), flush=True)
        if not active and (failed or stopping[0] or len(complete) == len(jobs)):
            return 2 if failed else 0
        time.sleep(max(3., args.poll_seconds))


if __name__ == '__main__':
    sys.exit(main())
