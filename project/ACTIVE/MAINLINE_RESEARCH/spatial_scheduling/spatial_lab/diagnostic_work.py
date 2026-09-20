"""Explicit small CPU diagnostics, with a separate cumulative attempt ledger.

The Stage4 ledger is historical and is never reset or reused here. The contract
is supplied by the authorized caller. No context is enabled by importing this
module, and diagnostic checkpoints cannot enter confirmatory evaluation.
"""
from contextlib import contextmanager
from pathlib import Path
import os,time,traceback
from .core import ContractError,read_json,write_json,digest
from .binding import source_digest
from .work_budget import _SCOPE

@contextmanager
def _locked(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.lock').open('a+b') as stream:
        if os.name=='nt':
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
            msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try:
            data=read_json(path) if path.exists() else None
            yield data
        finally:
            if os.name=='nt':stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

@contextmanager
def diagnostic_authority(contract_path,output,job,*,execute=False):
    if not execute:raise ContractError('DIAGNOSTIC_REQUIRES_EXPLICIT_EXECUTE')
    if _SCOPE.get() is not None:raise ContractError('NESTED_WORK_AUTHORITY')
    c=read_json(contract_path);budget=c['resource_budget']
    if c.get('mode')!='development-diagnostic':raise ContractError('DIAGNOSTIC_CONTRACT_MODE')
    if (not 0<=budget['world_advance_attempts']<=128 or
        not 0<=budget['optimizer_attempts']<=96 or budget['device']!='cpu' or budget['threads']!=1):
        raise ContractError('SMALL_DIAGNOSTIC_BUDGET')
    cases=c['allowed_cases']
    for item in cases:
        cfg=item['configuration']
        # A short prefix can retain the production horizon/resource pacing.
        # The independent transition cap is charged before World mutates state.
        prefix=item.get('max_world_advances',cfg['cycles'])
        max_horizon=64 if 'max_world_advances' in item else 8
        max_scores=12 if 'max_world_advances' in item else 6
        max_prefix=16 if 'max_world_advances' in item else 8
        if (type(item['seed']) is not int or type(prefix) is not int or not 0<prefix<=max_prefix or
            not prefix<=cfg['cycles']<=max_horizon or cfg['forecast_horizon']>2 or cfg['root_score_budget']>max_scores):
            raise ContractError('SMALL_DIAGNOSTIC_CASE_DOMAIN')
    ledger=Path(output)/'LOCAL_COMPUTE_LEDGER.json';identity=digest(c)
    with _locked(ledger) as data:
        if data is None:data={'schema':'ain.local.diagnostic.compute.v1','contract_digest':identity,
            'world_attempts':0,'optimizer_attempts':0,'jobs':{},'events':[]}
        if data['contract_digest']!=identity:raise ContractError('DIAGNOSTIC_CONTRACT_CHANGED')
        if job in data['jobs']:raise ContractError('DIAGNOSTIC_JOB_ALREADY_ATTEMPTED')
        data['jobs'][job]={'status':'RUNNING','source_digest':source_digest(),'world_attempts':0,'optimizer_attempts':0}
        write_json(ledger,data)
    token=_SCOPE.set({'kind':'diagnostic','contract':c,'ledger':ledger,'job':job})
    error=None
    try:yield
    except BaseException as exc:
        error={'error':repr(exc),'traceback':traceback.format_exc()};raise
    finally:
        _SCOPE.reset(token)
        with _locked(ledger) as data:
            data['jobs'][job].update(status='FAILED' if error else 'COMPLETE',failure=error)
            write_json(ledger,data)

def charge(kind,**fields):
    scope=_SCOPE.get()
    if not scope or scope['kind']!='diagnostic':raise ContractError('DIAGNOSTIC_SCOPE_REQUIRED')
    c=scope['contract'];budget=c['resource_budget']
    if kind=='world':
        matches=[x for x in c['allowed_cases'] if x['seed']==fields['seed'] and digest(x['configuration'])==fields['configuration_digest']]
        if len(matches)!=1 or not 0<=fields['epoch']<matches[0].get('max_world_advances',matches[0]['configuration']['cycles']):
            raise ContractError('UNREGISTERED_DIAGNOSTIC_WORLD')
        counter,limit='world_attempts',budget['world_advance_attempts']
    elif kind=='optimizer':
        if fields['arm'] not in c['backbones'] or fields['seed'] not in c['model_seeds']:
            raise ContractError('UNREGISTERED_DIAGNOSTIC_MODEL')
        counter,limit='optimizer_attempts',budget['optimizer_attempts']
    else:raise ContractError('UNKNOWN_DIAGNOSTIC_CHARGE')
    with _locked(scope['ledger']) as data:
        if data[counter]>=limit:raise ContractError('DIAGNOSTIC_CUMULATIVE_LIMIT:'+counter)
        data[counter]+=1;data['jobs'][scope['job']][counter]+=1
        data['events'].append({'kind':kind,'job':scope['job'],'time_ns':time.time_ns(),**fields})
        write_json(scope['ledger'],data)
