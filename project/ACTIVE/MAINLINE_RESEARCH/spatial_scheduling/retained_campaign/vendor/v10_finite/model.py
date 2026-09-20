"""Immutable finite R0 law, legal schedules, and independently callable Bayes scorer.

A provenance group is ONE Bernoulli measurement. Different groups are independent
conditional on all 16 declared latent rows. Duplicate ports copy the same bit.
No realized latent row or measurement is generated or exposed to a proposer.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from fractions import Fraction as F
from itertools import product, combinations
import hashlib
import json
import math
import random
import time
from typing import Callable
import numpy as np

SCHEMA = 'ain.r0.association-priority.v10.1'
EXECUTION_WORK = {'generated_contexts':0,'paid_score_attempts':0,'exact_score_calls':0}

class Rejection(ValueError): pass
class Exhausted(RuntimeError): pass

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(obj): return hashlib.sha256(canonical(obj)).hexdigest()

def strict_json(raw):
    def pairs(xs):
        out = {}
        for k,v in xs:
            if k in out: raise Rejection('DUPLICATE_JSON_KEY')
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: (_ for _ in ()).throw(Rejection('NONFINITE_JSON')))

def rat(x):
    if not isinstance(x,str) or len(x)>80: raise Rejection('RATIONAL_TOKEN_REQUIRED')
    try: v=F(x)
    except (ValueError, ZeroDivisionError) as e: raise Rejection('RATIONAL_TOKEN') from e
    if max(v.numerator.bit_length(),v.denominator.bit_length())>128: raise Rejection('PRECISION_CAP')
    return v

def rs(x): return str(F(x))

def integer(x,lo,hi):
    if type(x) is not int or not lo<=x<=hi: raise Rejection('INTEGER_CAP')
    return x

@dataclass(frozen=True)
class Channel:
    id: str
    provenance: str
    q: tuple[F,...]
    energy: F
    available: bool
    acquire: F
    arrive: F

@dataclass(frozen=True)
class Context:
    context_id: str
    context_digest: str
    law_digest: str
    group_id: str
    regime: str
    seed: int
    x: tuple[F,...]
    prior: tuple[F,...]
    association: tuple[int,...]
    source: tuple[int,...]
    decisions: tuple[F,...]
    channels: tuple[Channel,...]
    length: F
    start: F
    terminal: F
    resource: F
    max_size: int
    lambda_tail: F
    rho: F
    lambda_energy: F
    threshold: F
    raw: bytes

    @property
    def risk_scale(self): return float(self.length*self.length)

    def legal(self, action):
        if not isinstance(action,tuple) or len(action)>self.max_size or len(set(action))!=len(action):
            raise Rejection('SCHEDULE_SHAPE')
        if any(type(i) is not int or not 0<=i<len(self.channels) for i in action):
            raise Rejection('SCHEDULE_CHANNEL')
        if tuple(sorted(action))!=action: raise Rejection('NONCANONICAL_SCHEDULE')
        cs=[self.channels[i] for i in action]
        if any(not c.available or c.acquire<self.start or c.arrive>self.terminal for c in cs):
            raise Rejection('UNAVAILABLE_OR_NONCAUSAL_SCHEDULE')
        if sum((c.energy for c in cs),F(0))>self.resource: raise Rejection('RESOURCE_EXHAUSTED')
        return action

    def energy(self, action): return sum((self.channels[i].energy for i in action),F(0))
    def groups(self, action):
        result=[]; seen=set()
        for i in action:
            c=self.channels[i]
            if c.provenance not in seen: seen.add(c.provenance); result.append(i)
        return tuple(result)
    def available(self):
        out=[]
        for i in range(len(self.channels)):
            try: self.legal((i,));out.append(i)
            except Rejection: pass
        return tuple(out)


def seal(body):
    b=dict(body); b.pop('digest',None)
    return {**b,'digest':digest(b)}


def load_context(obj):
    if isinstance(obj,(bytes,str)): obj=strict_json(obj)
    if not isinstance(obj,dict): raise Rejection('CONTEXT_OBJECT')
    b=dict(obj);supplied=b.pop('digest',None)
    if supplied!=digest(b): raise Rejection('STALE_CONTEXT_DIGEST')
    if b.get('schema')!=SCHEMA or b.get('origin')!='DECLARED_SYNTHETIC_DEVELOPMENT': raise Rejection('MODEL_FAMILY')
    if b.get('native_mapping') is not None: raise Rejection('NATIVE_MAPPING_UNSUPPORTED')
    if b.get('units')!={'position':'m','time':'s','energy':'J','loss':'m2','lambda_energy':'m2/J','observation':'binary'}:
        raise Rejection('UNITS')
    if b.get('joint_contract')!='INDEPENDENT_GROUP_DRAWS_GIVEN_X_ASSOCIATION_SOURCE;DUPLICATE_PORTS_COPY':
        raise Rejection('JOINT_LAW_UNAVAILABLE')
    states=b.get('states'); cs=b.get('channels')
    if not isinstance(states,list) or len(states)!=16 or not isinstance(cs,list) or not 1<=len(cs)<=32:
        raise Rejection('STATE_OR_CHANNEL_CAP')
    expected=[(j,a,s) for j in range(4) for a in range(2) for s in range(2)]
    if [(v.get('target_index'),v.get('association'),v.get('source')) for v in states]!=expected:
        raise Rejection('STATE_ORDER')
    p=tuple(rat(v['p']) for v in states);x=tuple(rat(v['x_m']) for v in states)
    if any(v<0 for v in p) or sum(p)!=1: raise Rejection('PRIOR_MASS')
    if any(len(set(x[4*j:4*j+4]))!=1 for j in range(4)): raise Rejection('TARGET_COORDINATES')
    ds=tuple(rat(v) for v in b['decisions_m'])
    if not ds or len(ds)>8 or tuple(sorted(set(ds)))!=ds: raise Rejection('DECISION_CLASS')
    for v in x:
        if v not in ds: raise Rejection('DECISION_CLASS_TARGET_SUPPORT')
    start=rat(b['decision_s']);end=rat(b['terminal_s']);L=rat(b['length_m']);cap=rat(b['resource_J'])
    if end<=start or L<=0 or cap<0: raise Rejection('TIME_SCALE_OR_RESOURCE')
    lt=rat(b['lambda_tail_m2']);rho=rat(b['rho_m']);le=rat(b['lambda_energy_m2_per_J']);tau=rat(b['threshold_m2'])
    if min(lt,rho,le)<0: raise Rejection('LOSS_PARAMETERS')
    channels=[];ident=set();groups={}
    for c in cs:
        if not isinstance(c,dict) or not isinstance(c.get('id'),str) or c['id'] in ident: raise Rejection('CHANNEL_ID')
        ident.add(c['id']);g=c.get('provenance')
        if not isinstance(g,str) or not g or len(g)>80: raise Rejection('PROVENANCE')
        q=tuple(rat(t) for t in c.get('p_plus_given_z',[]))
        if len(q)!=16 or any(t<0 or t>1 for t in q): raise Rejection('LIKELIHOOD_ROWS')
        en=rat(c['energy_J']);ac=rat(c['acquire_s']);ar=rat(c['arrival_s'])
        if en<0 or ar<ac or type(c['available']) is not bool: raise Rejection('CHANNEL_TIME_RESOURCE')
        if g in groups and groups[g]!=(q,ac): raise Rejection('INCONSISTENT_DUPLICATE_PROVENANCE')
        groups[g]=(q,ac)
        channels.append(Channel(c['id'],g,q,en,c['available'],ac,ar))
    # Identity-independent law signature: arbitrary node/group names are omitted;
    # equal q groups remain distinct draws unless grouped by provenance.
    group_records=[]
    for g in groups:
        ports=[{'energy':rs(c.energy),'available':c.available,'acquire':rs(c.acquire),'arrival':rs(c.arrive)}
               for c in channels if c.provenance==g]
        group_records.append({'q':[rs(t) for t in groups[g][0]],'ports':sorted(ports,key=canonical)})
    law=digest({'states':states,'groups':sorted(group_records,key=canonical),'decisions':b['decisions_m'],
                'endpoint':b['terminal_s'],'resource':b['resource_J'],'max_size':b['max_size'],
                'loss':[b['lambda_tail_m2'],b['rho_m'],b['lambda_energy_m2_per_J']], 'tau':b['threshold_m2']})
    return Context(str(b['context_id']), supplied,law,str(b['group_id']),str(b['regime']),integer(b['seed'],0,2**63-1),
        x,p,tuple(v['association'] for v in states),tuple(v['source'] for v in states),ds,tuple(channels),
        L,start,end,cap,integer(b['max_size'],1,4),lt,rho,le,tau,canonical(obj))


def generate(seed:int,n:int=8,max_size:int=4,regime:str|None=None):
    """Generate model parameters, NOT latent realizations or sensing outcomes."""
    EXECUTION_WORK['generated_contexts']+=1
    integer(n,4,32);rng=random.Random(seed)
    if regime is None: regime=rng.choices(['cheap','ambiguous','empty','restricted'],[3,5,1,1])[0]
    if regime not in ('cheap','ambiguous','empty','restricted'): raise Rejection('REGIME')
    L=rng.choice([F(1,2),F(1),F(2)]);center=rng.choice([F(-1),F(0),F(1)])
    positions=[center+L*v for v in [-3,-1,1,3]]
    weights=[rng.randint(1,5) for _ in range(4)];px=[F(v,sum(weights)) for v in weights]
    pa=[rng.choice([F(1,4),F(1,2),F(3,4)]) for _ in range(4)]
    ps=[rng.choice([F(2,5),F(1,2),F(3,5)]) for _ in range(2)]
    states=[]
    for j,a,s in product(range(4),range(2),range(2)):
        pp=px[j]*(pa[j] if a else 1-pa[j])*(ps[a] if s else 1-ps[a])
        states.append({'target_index':j,'x_m':rs(positions[j]),'association':a,'source':s,'p':rs(pp)})
    channels=[];unique=max(4,n-n//5)
    for g in range(unique):
        e=rng.choice([F(1,100),F(3,100),F(2,25),F(3,20),F(1,4)])
        if regime=='cheap': kind='direct';bit=g%3;perm=list(range(4));sm=0
        else:
            kind=(['direct','alias','association','source'][g] if g<4 else
                  rng.choice(['direct','alias','alias','association','source','xor']))
            bit=rng.randrange(3);perm=list(range(4));rng.shuffle(perm);sm=rng.randrange(2)
            if g==0: e=F(1,4)  # a lawful noisy direct port, not a promised winner
        vals=[]
        for st in states:
            j,a,s=st['target_index'],st['association'],st['source']
            jj=perm[j] if a and kind in ('alias','xor') else j
            targetbit=(int(jj>=2),jj%2,int(jj>=1))[bit]
            signbit=(a if kind=='association' else s if kind=='source' else
                     targetbit ^ ((a if kind=='xor' else 0)) ^ (s if kind in ('alias','xor') and sm else 0))
            vals.append(rs(1-e if signbit else e))
        ac=F(1,10);ar=rng.choice([F(1,5),F(2,5),F(3,5)])
        channels.append({'id':f'c{g:02d}','provenance':f'g{g:02d}','p_plus_given_z':vals,
            'energy_J':rs(rng.choice([F(1,2),F(3,4),F(1)])), 'available':rng.random()>.08,
            'acquire_s':rs(ac),'arrival_s':rs(ar)})
    for i in range(unique,n):
        base=dict(channels[rng.randrange(unique)]);base['id']=f'c{i:02d}'
        base['energy_J']=rs(rng.choice([F(1,2),F(3,4),F(1)]));channels.append(base)
    rng.shuffle(channels) # node ordering carries no latent recipe role
    cap=rng.choice([F(3,2),F(2),F(5,2),F(3)])
    if regime=='restricted':
        for i,c in enumerate(channels):
            c['available']=i==0;c['arrival_s']='6/5' if i==0 else c['arrival_s']
    tau=(-F(1,100) if regime=='empty' else F(13,20))*L*L
    obj=seal({'schema':SCHEMA,'origin':'DECLARED_SYNTHETIC_DEVELOPMENT','native_mapping':None,
        'context_id':f'R0-{seed}', 'group_id':f'latent-context-{seed}', 'seed':seed,'regime':regime,
        'units':{'position':'m','time':'s','energy':'J','loss':'m2','lambda_energy':'m2/J','observation':'binary'},
        'joint_contract':'INDEPENDENT_GROUP_DRAWS_GIVEN_X_ASSOCIATION_SOURCE;DUPLICATE_PORTS_COPY',
        'states':states,'decisions_m':[rs(v) for v in sorted(set(positions))], 'channels':channels,
        'length_m':rs(L),'decision_s':'0','terminal_s':'1','resource_J':rs(cap),'max_size':max_size,
        'lambda_tail_m2':rs(L*L/2),'rho_m':rs(2*L),'lambda_energy_m2_per_J':rs(L*L/40),'threshold_m2':rs(tau)})
    return load_context(obj)

@dataclass
class Meter:
    score_cap:int
    construction_cap:int
    score_attempts:int=0
    score_completions:int=0
    construction_attempts:int=0
    legal_rejections:int=0
    duplicate_proposals:int=0
    outcome_rows:int=0
    weighted_likelihood_entries:int=0
    loss_products:int=0
    preprocessing_entries:int=0
    priority_candidates:int=0
    affinity_products:int=0
    covariance_solves:int=0
    selection_comparisons:int=0
    reference_exception:bool=False
    score_seconds:float=0.
    priority_seconds:float=0.
    prepare_seconds:float=0.

    def construct(self,ctx,a):
        if self.construction_attempts>=self.construction_cap: raise Exhausted('CONSTRUCTION_BUDGET')
        self.construction_attempts+=1
        try: return ctx.legal(tuple(sorted(a)))
        except Rejection: self.legal_rejections+=1;return None
    def report(self):
        d=asdict(self);d['risk_kernel_calls']=None if self.reference_exception else self.score_attempts
        d['native_graph_forks']=0;d['native_graph_solves']=0
        return d

@dataclass(frozen=True)
class Score:
    context_digest:str
    action:tuple[int,...]
    risk_m2:float
    squared_error_m2:float
    tail_probability:float
    energy_J:float
    outcome_count:int
    def report(self):
        return {**asdict(self),'action':list(self.action),'formal_certificate':False,
                'assurance':'NUMERICAL_FINITE_SUM_IN_DECLARED_R0_LAW','physical_validation':False}


def finite_score(ctx:Context,action:tuple[int,...],meter:Meter):
    """No proposer scores, priorities, labels, or audit rows are inputs."""
    groups=ctx.groups(action);z=len(ctx.prior)
    y=np.asarray(list(product([0,1],repeat=len(groups))),dtype=np.int8).reshape(-1,len(groups)) if groups else np.zeros((1,0),np.int8)
    joint=np.ones((len(y),z),dtype=np.float64)
    for t,i in enumerate(groups):
        q=np.asarray([float(v) for v in ctx.channels[i].q]);joint*=np.where(y[:,t,None]==1,q,1-q)
    w=joint*np.asarray([float(v) for v in ctx.prior])
    xx=np.asarray([float(v) for v in ctx.x]);dd=np.asarray([float(v) for v in ctx.decisions])
    sq=(xx[:,None]-dd[None,:])**2
    tail=(np.abs(xx[:,None]-dd[None,:])>float(ctx.rho)).astype(float)
    loss=sq+float(ctx.lambda_tail)*tail
    values=w@loss; chosen=values.argmin(axis=1)
    rr=float(values[np.arange(len(y)),chosen].sum())
    mse=float((w*sq[:,chosen].T).sum()); tp=float((w*tail[:,chosen].T).sum())
    meter.outcome_rows+=len(y);meter.weighted_likelihood_entries+=len(y)*z*max(1,len(groups))
    meter.loss_products+=len(y)*z*len(dd)
    energy=float(ctx.energy(action))
    return Score(ctx.context_digest,action,rr+float(ctx.lambda_energy)*energy,mse,tp,energy,len(y))


def check_score(ctx,a,row):
    if not isinstance(row,Score) or row.context_digest!=ctx.context_digest or row.action!=a: raise Rejection('REFERENCE_BINDING')
    v=[row.risk_m2,row.squared_error_m2,row.tail_probability,row.energy_J]
    if not all(type(t) is float and math.isfinite(t) for t in v): raise Rejection('REFERENCE_NONFINITE')
    if min(v)<-1e-12 or row.tail_probability>1+1e-12 or row.energy_J!=float(ctx.energy(a)): raise Rejection('REFERENCE_DOMAIN')
    expected=row.squared_error_m2+float(ctx.lambda_tail)*row.tail_probability+float(ctx.lambda_energy)*row.energy_J
    if abs(row.risk_m2-expected)>1e-10*max(1,ctx.risk_scale): raise Rejection('REFERENCE_OBJECTIVE')
    return row


def paid_score(ctx,action,meter,reference:Callable=finite_score):
    ctx.legal(action)
    if meter.score_attempts>=meter.score_cap: raise Exhausted('SCORE_BUDGET')
    meter.score_attempts+=1;EXECUTION_WORK['paid_score_attempts']+=1;t=time.perf_counter()
    try: result=check_score(ctx,action,reference(ctx,action,meter))
    except Exception:
        meter.reference_exception=True
        raise
    finally: meter.score_seconds+=time.perf_counter()-t
    meter.score_completions+=1;return result


def exact_score(ctx,action):
    """Independent rational loop, only for tiny contract cases; not a policy oracle."""
    EXECUTION_WORK['exact_score_calls']+=1
    ctx.legal(action);groups=ctx.groups(action);risk=F(0);mse=F(0);tail=F(0)
    for yy in product([0,1],repeat=len(groups)):
        ww=[]
        for z,p in enumerate(ctx.prior):
            v=p
            for i,y in zip(groups,yy):
                q=ctx.channels[i].q[z];v*=q if y else 1-q
            ww.append(v)
        candidates=[]
        for d in ctx.decisions:
            se=sum((p*(x-d)**2 for p,x in zip(ww,ctx.x)),F(0))
            tp=sum((p for p,x in zip(ww,ctx.x) if abs(x-d)>ctx.rho),F(0))
            candidates.append((se+ctx.lambda_tail*tp,d,se,tp))
        value,_,se,tp=min(candidates)
        risk+=value;mse+=se;tail+=tp
    return {'risk_m2':risk+ctx.lambda_energy*ctx.energy(action),'squared_error_m2':mse,'tail_probability':tail}


def enumerate_legal(ctx,meter):
    if len(ctx.channels)>8: raise Rejection('FULL_ENUMERATION_CAP')
    out=[]
    for k in range(ctx.max_size+1):
        for a in combinations(range(len(ctx.channels)),k):
            aa=meter.construct(ctx,a)
            if aa is not None: out.append(aa)
    return out


def exact_mechanism(association_probability=F(1,2)):
    """One decisive control, never used as a training/evaluation law."""
    states=[]
    for j,a,s in product(range(4),range(2),range(2)):
        states.append({'target_index':j,'x_m':'-1' if j<2 else '1','association':a,'source':s,
                       'p':rs(F(1,8)*(association_probability if a else 1-association_probability))})
    qs=[[],[],[],[]]
    for st in states:
        sign=st['target_index']>=2;a=st['association']
        qs[0].append(rs(F(3,4) if sign else F(1,4)))
        qs[1].append(rs(int(sign^bool(a))))
        qs[2].append(rs(a));qs[3].append(qs[1][-1])
    b=strict_json(generate(987123,4,2,'cheap').raw)
    b.update(context_id='EXACT-MECHANISM',group_id='EXACT-MECHANISM',states=states,decisions_m=['-1','1'],
             length_m='1',resource_J='4',lambda_tail_m2='0',rho_m='1',lambda_energy_m2_per_J='0',threshold_m2='1/2')
    b['channels']=[{'id':name,'provenance':g,'p_plus_given_z':q,'energy_J':'1','available':True,
                    'acquire_s':'1/10','arrival_s':'1/5'} for name,g,q in zip(['C','U','V','U-copy'],['c','u','v','u'],qs)]
    return load_context(seal(b))
