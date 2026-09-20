"""Fail-closed local work authority: default zero worlds and zero optimizers.

Local functional worlds use one persistent, locked ledger at the release root.
An attempted advance is charged BEFORE any state is changed, including failures.
The two registered fixtures can each be attempted only once in this workspace.
Server execution is a separate explicit CLI authority, never inferred on import.
"""
from __future__ import annotations
from contextvars import ContextVar
from contextlib import contextmanager
from pathlib import Path
import os,time,traceback
from .core import ContractError,read_json,write_json,digest

_RELEASE=Path(__file__).resolve().parents[4]
_SCOPE=ContextVar('stage4_work_authority',default=None)
LEDGER=_RELEASE/'STAGE4_EVIDENCE/COMPUTE_LEDGER.json'

@contextmanager
def locked_ledger():
    LEDGER.parent.mkdir(parents=True,exist_ok=True);lock=LEDGER.with_suffix('.lock')
    with lock.open('a+b') as stream:
        if os.name=='nt':
            import msvcrt
            stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try:
            d=read_json(LEDGER) if LEDGER.exists() else {'schema':'ain.stage4.local_work.v1','optimizer_updates':0,
                  'world_advance_attempts':0,'world_limit':12,'fixtures':{},'backward_attempts':{},'events':[],
                  'historical_stage3_world_advances':23,'historical_worlds_reexecuted':0}
            yield d
            write_json(LEDGER,d)
        finally:
            if os.name=='nt':stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)

def log_event(kind,**details):
    with locked_ledger() as d:d['events'].append({'kind':kind,'time_ns':time.time_ns(),**details})

@contextmanager
def functional_authority(name,seed,config):
    from .binding import source_manifest
    binding=read_json(_RELEASE/'STAGE4_BINDINGS/FINAL_BINDING.json')
    if binding['source_manifest']!=source_manifest():raise ContractError('FUNCTIONAL_REQUIRES_FROZEN_FINAL_SOURCE')
    registry=read_json(_RELEASE/'STAGE4_BINDINGS/FUNCTIONAL_PREREGISTRATION.json')
    spec=registry['fixtures'].get(name)
    if not spec or spec['seed']!=seed or spec['configuration']!=config:raise ContractError('UNREGISTERED_FUNCTIONAL_WORLD')
    with locked_ledger() as d:
        if name in d['fixtures']:raise ContractError('FIXTURE_ALREADY_ATTEMPTED_NO_RESEED_OR_RETRY')
        if len(d['fixtures'])>=2 or d['world_advance_attempts']+config['cycles']>12:raise ContractError('CUMULATIVE_FUNCTIONAL_WORLD_LIMIT')
        d['fixtures'][name]={'seed':seed,'config_digest':digest(config),'source_digest':source_manifest()['digest'],
                            'status':'RUNNING','attempts':0,'limit':config['cycles'],'next_epoch':0}
    token=_SCOPE.set({'kind':'functional','name':name,'seed':seed,'configuration':config})
    error=None
    try:yield
    except BaseException as exc:error={'error':repr(exc),'traceback':traceback.format_exc()};raise
    finally:
        _SCOPE.reset(token)
        with locked_ledger() as d:d['fixtures'][name].update(status='FAILED' if error else 'COMPLETE',failure=error)

def charge_world(seed,epoch,config):
    scope=_SCOPE.get()
    if scope is None:raise ContractError('ZERO_WORLD_DEFAULT_EXPLICIT_REGISTERED_FIXTURE_OR_SERVER_REQUIRED')
    if scope['kind']=='server':return
    if scope['kind']=='diagnostic':
        from .diagnostic_work import charge
        charge('world',seed=seed,epoch=epoch,configuration_digest=digest(config))
        return
    if seed!=scope['seed'] or config!=scope['configuration']:raise ContractError('WORLD_AUTHORITY_IDENTITY')
    with locked_ledger() as d:
        r=d['fixtures'][scope['name']]
        if d['world_advance_attempts']>=12 or r['attempts']>=r['limit'] or epoch!=r['next_epoch']:
            raise ContractError('CUMULATIVE_WORLD_LIMIT_OR_NONCONSECUTIVE_ADVANCE')
        d['world_advance_attempts']+=1;r['attempts']+=1;r['next_epoch']+=1
        d['events'].append({'kind':'world_advance_attempt','fixture':scope['name'],'seed':seed,'epoch':epoch,'time_ns':time.time_ns()})

def charge_backward(arm):
    with locked_ledger() as d:
        if d['backward_attempts'].get(arm,0)>=1:raise ContractError('ONE_LOCAL_BACKWARD_PER_BACKBONE')
        d['backward_attempts'][arm]=1;d['events'].append({'kind':'backward_attempt','arm':arm,'time_ns':time.time_ns()})

def enable_server(plan_digest):
    if os.environ.get('RADAR_STAGE4_LOCAL_VERIFY')=='1' or os.environ.get('RADAR_LOCAL_BUDGET_FILE'):
        raise ContractError('SERVER_EXECUTION_FORBIDDEN_IN_LOCAL_CHECK_ENVIRONMENT')
    return _SCOPE.set({'kind':'server','plan_digest':plan_digest})

def disable_server(token):_SCOPE.reset(token)
def require_optimizer_authority():
    if (_SCOPE.get() or {}).get('kind') not in ('server','diagnostic'):raise ContractError('LOCAL_OPTIMIZER_UPDATES_FORBIDDEN')

def charge_optimizer(arm,seed,update):
    require_optimizer_authority()
    if _SCOPE.get()['kind']=='diagnostic':
        from .diagnostic_work import charge
        charge('optimizer',arm=arm,seed=seed,update=update)

def training_purpose():
    return 'development_diagnostic' if (_SCOPE.get() or {}).get('kind')=='diagnostic' else 'registered_server_training'

@contextmanager
def prohibit_optimizers():
    """Runtime tripwire for bounded verification, also catches foreign callers."""
    import torch.optim
    original=torch.optim.Optimizer.__init__
    def forbidden(*args,**kwargs):raise ContractError('LOCAL_OPTIMIZER_CONSTRUCTION_FORBIDDEN')
    torch.optim.Optimizer.__init__=forbidden
    old=os.environ.get('RADAR_STAGE4_LOCAL_VERIFY');os.environ['RADAR_STAGE4_LOCAL_VERIFY']='1'
    try:yield
    finally:
        torch.optim.Optimizer.__init__=original
        if old is None:os.environ.pop('RADAR_STAGE4_LOCAL_VERIFY',None)
        else:os.environ['RADAR_STAGE4_LOCAL_VERIFY']=old
