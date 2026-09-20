"""Adaptive Linux development queue alongside the existing training coordinator.

Non-learning references use CPUs. Learned policies use GPUs after training on
that GPU has exited. A failed episode stops new submissions, preserving outputs.
"""
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
from common import read, write, sha, load_science
from runner import validate_design

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
    return {'cpu_quota': quota, 'cpu_used': used, 'memory_current_gib': current/GIB,
            'memory_free_gib': (maximum-current)/GIB, 'gpus': gpus}, cpu


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


def additions(hw, active_count, max_workers, *, memory_reserve=12., worker_memory=2., cpu_reserve=5., cpu_per_job=1.):
    memory_slots = math.floor(max(0., hw['memory_free_gib']-memory_reserve) / worker_memory)
    cpu_slots = math.floor(max(0., hw['cpu_quota']-hw['cpu_used']-cpu_reserve) / cpu_per_job)
    return max(0, min(max_workers-active_count, memory_slots, cpu_slots))


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
    ap.add_argument('--max-workers', type=int, default=50)
    ap.add_argument('--gpu-workers', type=int, default=20)
    ap.add_argument('--gpu-allocator-gib', type=float, default=1.0)
    ap.add_argument('--required-free-gpus', type=int, default=2)
    ap.add_argument('--poll-seconds', type=float, default=10.)
    args = ap.parse_args()
    if args.required_free_gpus < 1 or not 1 <= args.max_workers <= 50:
        raise ValueError('REQUIRE_AT_LEAST_ONE_RELEASED_GPU_AND_AT_MOST_FIFTY_WORKERS')
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    project = root / 'project'
    original_path = project / 'STAGE6_SERVER/plan_12800.json'
    load_science(project)
    design, original = read(args.design), read(original_path)
    validate_design(design, original)
    jobs = [j for j in design['jobs'] if j['split'] == 'development']
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'coordinator.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(out / 'EXECUTION.json', {'design_digest': design['design_digest'], 'arguments': vars(args),
          'pid': os.getpid(), 'mode': 'development', 'learned_backend': 'CUDA float32',
          'nonlearning_backend': 'CPU float64 physical calculations', 'training_updates': 0,
          'gpu_rule': 'no live training worker or foreign compute process on an admitted GPU',
          'world_count': len({j['world_seed'] for j in jobs}), 'episode_limit': len(jobs)})
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
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    previous, last_ramp, ceiling = None, time.monotonic(), 4
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
        occupied, ready, unfinished = training(root)
        own_pids = {x['process'].pid for x in active.values()}
        released_count, available_gpus = release_count(hw['gpus'], occupied, own_pids,
                                                       {x['gpu'] for x in active.values()})
        release_gate = released_count >= args.required_free_gpus
        if time.monotonic()-last_ramp >= 30 and not failed:
            ceiling = min(args.max_workers, ceiling+4)
            last_ramp = time.monotonic()
        capacity = additions(hw, len(active), min(args.max_workers, ceiling))
        capacity = min(capacity, 4)  # admission samples refresh before another burst
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
                command = [sys.executable, '-B', '-u', str(Path(__file__).with_name('posttrain.py')),
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
        print(json.dumps({k:v for k,v in body.items() if k not in ('active','resources','training_unfinished')}), flush=True)
        if not active and (failed or stopping[0] or len(complete) == len(jobs)):
            return 2 if failed else 0
        time.sleep(max(3., args.poll_seconds))


if __name__ == '__main__':
    sys.exit(main())
