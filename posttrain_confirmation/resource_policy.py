"""Container-aware admission policy; no changes to scientific jobs or models."""
import math
from pathlib import Path

GIB = 2**30


def memory_budget(current, maximum, high, stat, host_available):
    """Credit 80% of clean, unmapped file cache, respecting both cgroup limits.

    Shared memory, dirty/writeback pages, and mapped pages are excluded. The
    discount and a separate admission reserve absorb reclamation latency.
    Host MemAvailable is an additional ceiling, never a container limit.
    """
    limit = min(maximum, high) if high is not None else maximum
    excluded = sum(stat.get(k, 0) for k in ('shmem', 'file_dirty', 'file_writeback', 'file_mapped'))
    clean = max(0, stat.get('file', 0) - excluded)
    credit = int(clean * .8)
    available = max(0, min(limit, host_available, limit - current + credit))
    return {'memory_current_gib': current/GIB, 'memory_limit_gib': maximum/GIB,
            'memory_high_gib': high/GIB if high is not None else None,
            'memory_raw_headroom_gib': max(0, limit-current)/GIB,
            'memory_available_gib': available/GIB, 'memory_cache_credit_gib': credit/GIB,
            'memory_file_cache_gib': stat.get('file', 0)/GIB,
            'memory_anonymous_gib': stat.get('anon', 0)/GIB,
            'memory_dirty_gib': stat.get('file_dirty', 0)/GIB}


def read_memory(cgroup=Path('/sys/fs/cgroup'), meminfo=Path('/proc/meminfo')):
    host = {line.split(':', 1)[0]: int(line.split()[1])*1024
            for line in meminfo.read_text().splitlines() if len(line.split()) >= 3}
    def limit(name):
        value = (cgroup/name).read_text().strip()
        return None if value == 'max' else int(value)
    stat = {k: int(v) for k, v in (line.split() for line in (cgroup/'memory.stat').read_text().splitlines())}
    budget = memory_budget(int((cgroup/'memory.current').read_text()),
                           limit('memory.max') or host['MemTotal'], limit('memory.high'),
                           stat, host.get('MemAvailable', host['MemFree']))
    pressure = {}
    for line in (cgroup/'memory.pressure').read_text().splitlines():
        kind, *fields = line.split()
        pressure[kind] = {k: float(v) for k, v in (f.split('=') for f in fields)}
    budget['memory_pressure'] = pressure
    budget['memory_events'] = {k: int(v) for k, v in
                               (line.split() for line in (cgroup/'memory.events').read_text().splitlines())}
    return budget


def additions(hw, active_count, max_workers, *, memory_reserve=16., worker_memory=2.5,
              cpu_reserve=7., cpu_per_job=1.):
    pressure = hw.get('memory_pressure', {})
    if pressure.get('full', {}).get('avg10', 0) > 1. or pressure.get('some', {}).get('avg10', 0) > 5.:
        return 0
    memory_slots = math.floor(max(0., hw['memory_available_gib']-memory_reserve) / worker_memory)
    cpu_slots = math.floor(max(0., hw['cpu_quota']-hw['cpu_used']-cpu_reserve) / cpu_per_job)
    total_cpu_slots = max(0, math.floor(hw['cpu_quota']-cpu_reserve)-active_count)
    return max(0, min(max_workers-active_count, memory_slots, cpu_slots, total_cpu_slots))
