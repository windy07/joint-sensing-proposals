"""Units, immutable bindings, deterministic exogenous streams and bounded work."""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from pathlib import Path
from fractions import Fraction
import hashlib,json,math,os,time,uuid
import numpy as np
import scipy.linalg as _scipy_linalg  # load both BLAS runtimes before enforcing limits
from threadpoolctl import threadpool_limits
# Environment variables may be too late when an embedding process preloads NumPy.
# This live limit covers unit scripts and the executable mainline alike.
_THREAD_LIMITER = threadpool_limits(limits=1)

class ContractError(ValueError):pass
class BudgetExhausted(RuntimeError):pass

def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canon(x)).hexdigest()
def rng(seed,*parts):
    n=int.from_bytes(hashlib.sha256(canon([int(seed),*parts])).digest()[:8],'little')
    return np.random.default_rng(n)
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def atomic_replace(tmp,path,timeout_s=10.):
    """Retry only transient Windows reader/rename conflicts; keep failed temp bytes."""
    deadline=time.monotonic()+timeout_s;delay=.01
    while True:
        try:
            os.replace(tmp,path)
            return
        except OSError as exc:
            if os.name!='nt' or getattr(exc,'winerror',None) not in (5,32,33) or time.monotonic()>=deadline:
                raise
            time.sleep(min(delay,max(0.,deadline-time.monotonic())))
            delay=min(.25,delay*2)

def atomic_save(path,writer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with tmp.open('wb') as stream:
        writer(stream);stream.flush();os.fsync(stream.fileno())
    atomic_replace(tmp,path)

def write_json(path,obj):
    payload=json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8')
    atomic_save(path,lambda stream:stream.write(payload))
def _legacy_read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def wrap(a):return (np.asarray(a)+np.pi)%(2*np.pi)-np.pi

def transition(dt,q):
    F=np.eye(4);F[0,2]=dt;F[1,3]=dt
    Q=q*np.array([[dt**3/3,0,dt**2/2,0],[0,dt**3/3,0,dt**2/2],[dt**2/2,0,dt,0],[0,dt**2/2,0,dt]])
    return F,Q

@dataclass(frozen=True,order=True)
class Task:
    node:int
    kind:str
    item:int=-1
    role:str=''
    mode:str='full'
    start_s:float=0.
    power_W:float=0.
    dwell_s:float=0.
    bearing_rad:float=0.
    scan:tuple=()
    reference_node:int=-1
    reference_id:str=''
    calibration:str=''
    carrier_Hz:float=3.e9
    bandwidth_Hz:float=1.e7
    pulse_width_s:float=2.e-6
    pri_s:float=1.e-4
    beam_rad:float=0.
    waveform:str='LFM'
    initial_bearing_rad:float=0.
    requested_start_s:float|None=None
    def __post_init__(self):
        object.__setattr__(self,'scan',tuple(tuple(float(x) for x in b) for b in self.scan))
    def to_dict(self):return asdict(self)
    def semantic_id(self):
        # Persistent role identity, NOT a scoring-cache identity. Owner track IDs
        # are monotone. Numeric alternatives intentionally share this memory.
        return (f'action/node={self.node}/role={self.role}/mode={self.mode}/ref={self.reference_node}'
                f'/cal={self.calibration}/task={self.kind}/object={self.item}')
    def execution_id(self):return digest(self.to_dict())

@dataclass
class Meter:
    B:int=16
    construction_cap:int=256
    score_calls:int=0
    score_completed:int=0
    score_failures:int=0
    construction:int=0
    forecast_updates:int=0
    cholesky:int=0
    cache_hits:int=0
    priority_comparisons:int=0
    structural_pair_inspections:int=0
    feedback_seconds:float|None=None
    proposal_seconds:float=0.
    reference_seconds:float=0.
    risk_kernel_calls:int|None=0
    network_forwards:int=0
    numeric_optimization_calls:int=0
    numeric_materializations:int=0
    regenerations:int=0
    network_seconds:float|None=None
    optimization_seconds:float|None=None
    feasibility_rejections:int=0
    # Deliberately not named native forks: this component does not call the DFG.
    native_graph_forks:int=0
    def charge(self):
        if self.score_calls>=self.B:raise BudgetExhausted('VERIFICATION_BUDGET')
        self.score_calls+=1
    def construct(self):
        if self.construction>=self.construction_cap:raise BudgetExhausted('CONSTRUCTION_BUDGET')
        self.construction+=1
    def report(self):return asdict(self)

def psd(P,name='covariance'):
    if not np.isfinite(P).all() or not np.allclose(P,P.T,rtol=1e-8,atol=1e-8):raise ContractError(name+':NONFINITE_OR_ASYMMETRIC')
    cholesky(P)

def gaussian_update(m,P,H,R,innovation,meter=None):
    S=H@P@H.T+R; C=cholesky(S)
    if meter is not None:meter.cholesky+=1
    PH=P@H.T;K=solve(C.T,solve(C,PH.T)).T
    z=solve(C,innovation)
    loglik=-.5*(len(z)*math.log(2*math.pi)+2*np.log(np.diag(C)).sum()+z@z)
    I=np.eye(len(m))-K@H
    Q=I@P@I.T+K@R@K.T
    return m+K@innovation,(Q+Q.T)/2,float(loglik),float(z@z)

# Explicit NumPy factor/solve accounting for the decision path. Torch backbone
# linear algebra is counted/timed as neural inference, not guessed as Cholesky.
from contextvars import ContextVar
from contextlib import contextmanager
_LINEAR_WORK=ContextVar('decision_linear_work',default=None)
def _counted_linear(name,*args,**kwargs):
    work=_LINEAR_WORK.get()
    if work is not None:work[name]=work.get(name,0)+1
    return getattr(np.linalg,name)(*args,**kwargs)
def solve(*a,**k):return _counted_linear('solve',*a,**k)
def cholesky(*a,**k):return _counted_linear('cholesky',*a,**k)
def eigvalsh(*a,**k):return _counted_linear('eigvalsh',*a,**k)
def inverse(*a,**k):return _counted_linear('inv',*a,**k)
@contextmanager
def decision_linear_work():
    work={k:0 for k in ('solve','cholesky','eigvalsh','inv')};token=_LINEAR_WORK.set(work)
    try:yield work
    finally:_LINEAR_WORK.reset(token)

# Large immutable cases and restart snapshots are compressed; small receipts
# retain ordinary JSON and the same Windows sharing-violation retry logic.
_write_plain_json=write_json
def write_json(path,obj):
    path=Path(path)
    if path.suffix=='.gz':
        import gzip
        payload=json.dumps(clean(obj),ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
        atomic_save(path,lambda f:f.write(gzip.compress(payload,compresslevel=1,mtime=0)))
    else:_write_plain_json(path,obj)
def read_json(path):
    path=Path(path)
    if path.suffix=='.gz':
        import gzip
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return _legacy_read_json(path)
