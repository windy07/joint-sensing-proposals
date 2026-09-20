"""Fast dependency/configuration checks; no world, weights or optimization."""
from __future__ import annotations
import importlib.metadata,sys,platform,os
from pathlib import Path
from .core import ContractError
from .binding import dependency_contract,source_manifest,import_origins
from .actions import device

SUPPORTED_PUBLIC_VERSIONS={'numpy':'2.3.5','scipy':'1.17.0','torch':'2.10.0','threadpoolctl':'3.6.0'}

def preflight(config=None,device_name='cpu',require_device=False):
    actual=dependency_contract()
    if actual!=SUPPORTED_PUBLIC_VERSIONS:
        raise ContractError('DEPENDENCY_PREFLIGHT: expected '+str(SUPPORTED_PUBLIC_VERSIONS)+' got '+str(actual))
    if sys.version_info <(3,11):raise ContractError('PYTHON_311_OR_NEWER_REQUIRED')
    from threadpoolctl import threadpool_info,threadpool_limits
    import torch
    torch.set_num_threads(1)
    global _LIVE_THREAD_LIMITER
    _LIVE_THREAD_LIMITER=threadpool_limits(limits=1)
    if any(p.get("num_threads",1)>1 for p in threadpool_info()):
        raise ContractError("SINGLE_THREAD_RUNTIME_LIMIT_FAILED")
    if require_device and device_name.startswith('cuda') and not torch.cuda.is_available():
        raise ContractError('REQUESTED_CUDA_DEVICE_UNAVAILABLE')
    if config is not None:
        if config.get('action_schema')!='ain.spatial.executable.pulse_scan.v4':raise ContractError('ACTION_SCHEMA_PREFLIGHT')
        if not 1<=config.get('forecast_horizon',1)<=3:raise ContractError('PUBLIC_FORECAST_HORIZON_LIMIT')
        if type(config['cycles']) is not int or config['cycles']<=0 or not 0<config['dt_s']<=10:raise ContractError('TIME_CONFIGURATION')
        if not 1<=config['max_tracks']<=6 or not 1<=config['max_simultaneous']<=4:raise ContractError('FINITE_DIMENSION_LIMIT')
        if not 0<=config['root_score_budget']<=64 or not 1<=config['return_capacity']<=8:raise ContractError('DECISION_BUDGET')
        if config['construction_cap']<config['candidate_pool_cap']+12:raise ContractError('CONSTRUCTION_CAP_TOO_SMALL_FOR_FEEDBACK')
        for p in ('pfa_cell','packet_loss_probability','emission_probability'):
            if not 0<=config[p]<=1 or p=='pfa_cell' and config[p] in (0,1):raise ContractError('PROBABILITY_DOMAIN:'+p)
        for n in range(4):
            d=device(config,n)
            for k,v in d.items():
                if isinstance(v,(int,float)) and v<=0:raise ContractError('DEVICE_PARAMETER:'+k)
            if d['aperture_efficiency']>1 or d['noise_factor']<1 or d['loss']<1:raise ContractError('PASSIVE_DEVICE_CONSTANT')
        if config.get('birth_process')=='capped_geometric' and not 0<config.get('birth_probability_per_cycle',0)<=1:raise ContractError('BIRTH_HAZARD')
    return {'python':sys.version,'platform':platform.platform(),'public_dependencies':actual,
            'installed_builds':{n:importlib.metadata.version(n) for n in actual},
            'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            'threadpools':threadpool_info(),'requested_device':device_name,'cuda_available':torch.cuda.is_available(),
            'source_manifest':source_manifest(),'actual_imports':import_origins(),
            'optimizer_updates':0,'world_advances':0,'status':'PREFLIGHT_PASSED'}
