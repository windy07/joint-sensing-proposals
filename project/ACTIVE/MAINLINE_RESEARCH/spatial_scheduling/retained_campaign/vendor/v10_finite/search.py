"""Bounded proposals and post-commit, separately paid common-reference audit."""
from __future__ import annotations
from itertools import combinations
from dataclasses import asdict
import math
import random
import time
import numpy as np
from .model import Context,Meter,Score,Rejection,Exhausted,paid_score,finite_score,enumerate_legal,digest

METHODS=('random','analytic_decoder','beam_block','learned')
FEATURE_DIM=64
AUX_DIM=8

class Features:
    """All inputs are pre-decision declared law/metadata. No scored objective rows."""
    def __init__(self,ctx:Context,meter:Meter,affinity=True):
        t=time.perf_counter();self.ctx=ctx;self.meter=meter
        self.p=np.array([float(x) for x in ctx.prior]);self.x=np.array([float(x) for x in ctx.x])
        self.q=np.array([[float(x) for x in c.q] for c in ctx.channels])
        self.scale=ctx.risk_scale;mean=self.p@self.x
        ey=2*self.q-1;my=ey@self.p;vary=1-my*my
        self.cov=ey@(self.p*(self.x-mean));self.vary=vary;self.varx=float(self.p@(self.x-mean)**2)
        n=len(ctx.channels);cap=max(float(ctx.resource),1e-12);duration=float(ctx.terminal-ctx.start)
        self.group=[c.provenance for c in ctx.channels]
        ppA=sum(float(p) for p,a in zip(ctx.prior,ctx.association) if a)
        ppS=sum(float(p) for p,s in zip(ctx.prior,ctx.source) if s)
        entropy=lambda p: -sum(v*math.log(max(v,1e-300)) for v in (p,1-p) if v>0)
        rows=[]
        for i,c in enumerate(ctx.channels):
            meta=[float(c.available),float(c.energy)/cap,float(c.acquire-ctx.start)/duration,
                  float(c.arrive-ctx.start)/duration,self.group.count(c.provenance)/n,
                  my[i],vary[i],self.cov[i]/float(ctx.length),float(self.p@(self.q[i]*(1-self.q[i]))),
                  entropy(ppA),entropy(ppS),float(ctx.lambda_tail)/self.scale,float(ctx.rho/ctx.length),
                  float(ctx.lambda_energy)/self.scale,float(ctx.resource)/4,n/32]
            rows.append(np.r_[self.q[i],self.p,(self.x-mean)/float(ctx.length),meta])
        self.nodes=np.asarray(rows,dtype=np.float32)
        if self.nodes.shape!=(n,FEATURE_DIM): raise Rejection('FEATURE_DIMENSION')
        self.nodes.setflags(write=False)
        self.aff=None
        if affinity:
            self.aff=np.sqrt(self.q[:,:,None]*self.q[:,None,:])+np.sqrt((1-self.q[:,:,None])*(1-self.q[:,None,:]))
            dist=(self.x[:,None]-self.x[None,:])**2+float(ctx.lambda_tail)*(np.abs(self.x[:,None]-self.x[None,:])>float(ctx.rho))
            self.pair_weight=self.p[:,None]*self.p[None,:]*dist
            meter.preprocessing_entries+=self.aff.size+self.pair_weight.size
        meter.preprocessing_entries+=self.nodes.size+self.q.size+len(self.p)
        meter.prepare_seconds+=time.perf_counter()-t

    def singleton_priority(self,a):
        self.meter.priority_candidates+=1
        i=a[0];gain=self.cov[i]**2/self.vary[i] if self.vary[i]>1e-14 else 0.
        return self.varx-gain+float(self.ctx.lambda_energy*self.ctx.channels[i].energy)

    def rank(self,actions):
        if self.aff is None: raise Rejection('AFFINITY_NOT_PREPARED')
        t=time.perf_counter();out=[]
        for a in actions:
            v=np.ones_like(self.pair_weight)
            for i in self.ctx.groups(a): v*=self.aff[i];self.meter.affinity_products+=v.size
            out.append(float((self.pair_weight*v).sum())+float(self.ctx.lambda_energy*self.ctx.energy(a)))
        self.meter.priority_candidates+=len(actions);self.meter.priority_seconds+=time.perf_counter()-t
        return out

    def masks_aux(self,actions):
        n=len(self.ctx.channels);masks=np.zeros((len(actions),n),np.float32);aux=[]
        duration=float(self.ctx.terminal-self.ctx.start);cap=max(float(self.ctx.resource),1e-12)
        for r,a in enumerate(actions):
            gg=[self.group[i] for i in a]
            for i in a: masks[r,i]=1/gg.count(self.group[i]) # one unit per observation group
            tt=[float(self.ctx.channels[i].arrive-self.ctx.start)/duration for i in a]
            aux.append([len(a)/4,len(set(gg))/4,(len(a)-len(set(gg)))/4,
                        float(self.ctx.energy(a))/cap,min(tt,default=0),max(tt,default=0),
                        float(not a),self.varx/self.scale])
        return masks,np.asarray(aux,dtype=np.float32)

    def covariance_proxy(self,a):
        """Diagnostic only; not finite-decision Bayes risk or tail assurance."""
        if not a:return self.varx
        self.meter.covariance_solves+=1
        ey=2*self.q[list(a)]-1;my=ey@self.p
        cov=(ey*self.p)@ey.T-my[:,None]*my[None,:]
        for j,i in enumerate(a):
            for k,l in enumerate(a):
                if self.group[i]==self.group[l]: cov[j,k]=1-my[j]*my[k]
        return float(self.varx-self.cov[list(a)]@np.linalg.pinv(cov,hermitian=True)@self.cov[list(a)])


def unique_construct(ctx,meter,items,seen):
    out=[]
    for a in items:
        # Keep the already constructed prefix: construction exhaustion must not
        # discard legal candidates or forbid spending remaining score budget.
        if meter.construction_attempts>=meter.construction_cap:break
        aa=meter.construct(ctx,a)
        if aa is None:continue
        if aa in seen:meter.duplicate_proposals+=1;continue
        seen.add(aa);out.append(aa)
    return out


def ranked(actions,priority,meter):
    if not actions:return []
    values=priority(actions)
    if len(values)!=len(actions) or not all(math.isfinite(float(v)) for v in values): raise Rejection('PRIORITY_VALUES')
    meter.selection_comparisons+=len(actions)*max(1,math.ceil(math.log2(max(2,len(actions)))))
    return [a for _,a in sorted(zip(values,actions),key=lambda va:(float(va[0]),va[1]))]


def decoder_pool(ctx,meter,priority,width):
    """Same construction decoder for learned and no-learning analytic priority."""
    av=ctx.available();pool=[];seen=set()
    try:
        for k in (1,2):
            if k<=ctx.max_size:pool+=unique_construct(ctx,meter,combinations(av,k),seen)
        frontier=ranked([a for a in pool if len(a)==min(2,ctx.max_size)],priority,meter)[:width]
        for k in range(3,ctx.max_size+1):
            new=[]
            for a in frontier:
                new+=unique_construct(ctx,meter,(tuple(sorted(a+(i,))) for i in av if i not in a),seen)
            pool+=new;frontier=ranked(new,priority,meter)[:width]
    except Exhausted:
        pass
    return ranked(pool,priority,meter)


def random_pool(ctx,meter,rng,maximum):
    out=[];seen=set();av=ctx.available()
    if not av:return out
    upper=sum(math.comb(len(av),k) for k in range(1,min(ctx.max_size,len(av))+1))
    while len(out)<min(maximum,upper) and meter.construction_attempts<meter.construction_cap:
        k=rng.randint(1,min(ctx.max_size,len(av)));a=tuple(sorted(rng.sample(av,k)))
        aa=meter.construct(ctx,a)
        if aa is None: continue
        if aa in seen:meter.duplicate_proposals+=1;continue
        seen.add(aa);out.append(aa)
    return out


def classify(ctx,row):
    if row is None:return None
    tol=1e-10*max(1,ctx.risk_scale);v=row.risk_m2-float(ctx.threshold)
    return True if v < -tol else False if v>tol else None


def run_policy(ctx:Context,settings:dict,method:str,learned_priority=None,reference=finite_score):
    if method not in METHODS:raise Rejection('METHOD')
    K=settings['K'];B=settings['B'];C=settings['construction_cap'];width=settings['beam_width']
    if type(K) is not int or not 1<=K<=8 or type(B) is not int or not 0<=B<=128:raise Rejection('POLICY_CAP')
    mt=Meter(B,C);start=time.perf_counter();archive={};errors=[];trace=[];stop='BUDGET_OR_POOL_COMPLETE'
    f=Features(ctx,mt,affinity=method!='random');av=ctx.available()
    anchor=min(((i,) for i in av),key=lambda a:(f.singleton_priority(a),a)) if av else None
    def evaluate(a):
        if a in archive:return
        row=paid_score(ctx,a,mt,reference);archive[a]=row
        trace.append({'call':mt.score_attempts,'action':list(a),'risk_m2':row.risk_m2})
    def attempt(a):
        aa=mt.construct(ctx,a)
        if aa is not None:evaluate(aa)
    try:
        attempt(()) # fallback consumes one score and one K slot
        if anchor is not None:attempt(anchor)
        if method=='random':
            pool=random_pool(ctx,mt,random.Random(ctx.seed ^ settings['random_seed']),max(B*3,16))
        elif method in ('analytic_decoder','learned'):
            if method=='learned':
                if learned_priority is None:raise Rejection('MISSING_LEARNED_POLICY')
                priority=learned_priority.bind(f,mt)
            else:priority=f.rank
            pool=decoder_pool(ctx,mt,priority,width)
        else:
            seen=set();base=[]
            for k in (1,2):
                if k<=ctx.max_size:base+=unique_construct(ctx,mt,combinations(av,k),seen)
            base=ranked(base,f.rank,mt)
            first_target=max(mt.score_attempts,(B+1)//2)
            for a in base:
                if mt.score_attempts>=first_target:break
                evaluate(a)
            # Paid-risk beam can cross non-improving one-step intermediates by
            # adding two channels at once; it never calls the scorer for ranking.
            parents=sorted((a for a in archive if a),key=lambda a:(archive[a].risk_m2,a))[:width]
            new=[]
            for a in parents:
                remaining=[i for i in av if i not in a]
                for k in (1,2):
                    if len(a)+k<=ctx.max_size:
                        new+=unique_construct(ctx,mt,(tuple(sorted(a+b)) for b in combinations(remaining,k)),seen)
            pool=ranked(base+new,f.rank,mt)
        for a in pool:
            if mt.score_attempts>=B:break
            evaluate(a)
    except Exhausted as exc:stop=str(exc)
    except Exception as exc:
        stop='UNKNOWN_AFTER_EXCEPTION';errors.append({'type':type(exc).__name__,'message':str(exc)})
    if stop=='BUDGET_OR_POOL_COMPLETE' and mt.construction_attempts>=mt.construction_cap:
        stop='CONSTRUCTION_CAPPED_PREFIX_RETAINED'
    # Protected output: fallback and analytic singleton, then best verified extras.
    out=[]
    if () in archive:out.append(())
    if anchor is not None and anchor in archive and len(out)<K:out.append(anchor)
    mt.selection_comparisons+=len(archive)*max(1,math.ceil(math.log2(max(2,len(archive)))))
    for a in sorted(archive,key=lambda a:(archive[a].risk_m2,a)):
        if len(out)>=K:break
        if a not in out:out.append(a)
    exists=any(classify(ctx,r) is True for r in archive.values())
    hits=any(classify(ctx,archive[a]) is True for a in out)
    miss=False if hits else (True if exists and all(classify(ctx,archive[a]) is False for a in out) else None)
    body={'schema':'ain.v10.policy-commit.v1','method':method,'context_digest':ctx.context_digest,
          'law_digest':ctx.law_digest,'settings':settings,'archive':[r.report() for r in archive.values()],
          'output':[list(a) for a in out],'protected_singleton':None if anchor is None else list(anchor),
          'trace':trace,'stop':stop,'errors':errors,'status':'ABSTAIN' if not out else 'UNKNOWN' if errors else 'VERIFIED_OUTPUT_UNDER_BUDGET',
          'Q_nonempty_before_audit':True if exists else None,'conditional_omission_before_audit':miss,
          'meter':mt.report(),'elapsed_seconds':time.perf_counter()-start,
          'claim':'BEST_KNOWN_UNDER_BUDGET;NOT_GLOBAL','formal_certificate':False}
    return {**body,'commit_digest':digest(body)}


def validate_commit(ctx,run):
    if not isinstance(run,dict):raise Rejection('COMMIT_TYPE')
    b=dict(run);d=b.pop('commit_digest',None)
    if digest(b)!=d or run['context_digest']!=ctx.context_digest or run['law_digest']!=ctx.law_digest:raise Rejection('COMMIT_BINDING')
    archive=set()
    for r in run['archive']:
        a=tuple(r['action']);ctx.legal(a)
        if a in archive or r['context_digest']!=ctx.context_digest:raise Rejection('ARCHIVE_BINDING')
        if any(not isinstance(r.get(k),(float,int)) or not math.isfinite(r[k]) for k in ('risk_m2','squared_error_m2','tail_probability','energy_J')):raise Rejection('NONFINITE_COMMITTED_ROW')
        archive.add(a)
    output=[tuple(a) for a in run['output']]
    if len(output)>run['settings']['K'] or len(set(output))!=len(output) or any(a not in archive for a in output):raise Rejection('OUTPUT_BINDING')
    if output and () not in output:raise Rejection('MISSING_FALLBACK')
    return archive


def freeze_reference(ctx,runs,audit_settings):
    """Called ONLY after commits. No audit rows or proposer scores used to select it."""
    names=[r['method'] for r in runs]
    if len(names)!=len(set(names)) or not names:raise Rejection('METHOD_COMMITMENTS')
    mt=Meter(0,audit_settings['construction_cap']);union=set()
    for r in runs:union.update(validate_commit(ctx,r))
    full=len(ctx.channels)<=audit_settings['full_enumeration_channels']
    if full:
        reference=enumerate_legal(ctx,mt)
    else:
        f=Features(ctx,mt);pool=decoder_pool(ctx,mt,f.rank,audit_settings['beam_width'])
        reference=list(union|set(pool[:audit_settings['constructive_candidates']])|{()})
    if len(reference)>audit_settings['B']:raise Rejection('AUDIT_BUDGET_CANNOT_COVER_FROZEN_REFERENCE')
    body={'context_digest':ctx.context_digest,'method_commits':{r['method']:r['commit_digest'] for r in runs},
          'actions':[list(a) for a in sorted(set(reference))],'full_domain':full,
          'construction':mt.report(),'frozen_after_commits_before_scores':True}
    return {**body,'digest':digest(body)}


def report_output(ctx,out,rows,reference,full,verified_rows=None):
    out=[tuple(a) for a in out];refs=[tuple(a) for a in reference]
    known=[a for a in refs if a in rows and classify(ctx,rows[a]) is True]
    all_ref_unqualified=all(a in rows and classify(ctx,rows[a]) is False for a in refs)
    all_out_unqualified=all(a in rows and classify(ctx,rows[a]) is False for a in out)
    hit=any(a in rows and classify(ctx,rows[a]) is True for a in out)
    exists=True if known else False if full and all_ref_unqualified else None
    # An incomplete reference cannot establish global emptiness. Independent
    # earlier verification can nevertheless establish nonemptiness (C13).
    if exists is None and verified_rows and any(classify(ctx,r) is True for r in verified_rows):exists=True
    coverage=True if hit else False if exists is True and all_out_unqualified else None
    missed=False if hit or exists is False else True if coverage is False else None
    complete_ref=all(a in rows for a in refs)
    regret=None;best=None
    if out and all(a in rows for a in out) and complete_ref:
        best=min(out,key=lambda a:(rows[a].risk_m2,a));regret=rows[best].risk_m2-min(rows[a].risk_m2 for a in refs)
    return {'Q_nonempty':exists,'Q_empty':None if exists is None else not exists,
            'reference_Q_nonempty':True if known else False if all_ref_unqualified else None,
            'conditional_coverage':coverage,'unconditional_missed_witness':missed,
            'finite_reference_regret_m2':regret,'normalized_regret':None if regret is None else regret/ctx.risk_scale,
            'output_risk_m2':None if best is None else rows[best].risk_m2,
            'tail_probability':None if best is None else rows[best].tail_probability,
            'reference_complete':complete_ref,'global_domain_enumerated':full,
            'reference_limit':'FULL_DECLARED_GRAMMAR' if full else 'FROZEN_COMMON_UNION_NOT_GLOBAL',
            'assurance':'NUMERICAL_FINITE_SUM_WITH_TOLERANCE;NOT_OUTWARD_CERTIFIED'}


def audit(ctx,runs,frozen,audit_settings):
    for r in runs:validate_commit(ctx,r)
    b=dict(frozen);d=b.pop('digest',None)
    if d!=digest(b) or frozen['context_digest']!=ctx.context_digest or frozen['method_commits']!={r['method']:r['commit_digest'] for r in runs}:
        raise Rejection('FROZEN_REFERENCE_BINDING')
    refs=[tuple(a) for a in frozen['actions']]
    if len(set(refs))!=len(refs) or any(a not in refs for r in runs for a in validate_commit(ctx,r)):raise Rejection('INCOMPLETE_COMMON_UNION')
    if frozen['full_domain']:
        check=enumerate_legal(ctx,Meter(0,10000))
        # count this explicit domain-validation work, not an uncharged oracle
        validation_attempts=sum(math.comb(len(ctx.channels),k) for k in range(ctx.max_size+1))
        if set(check)!=set(refs):raise Rejection('FALSE_FULL_DOMAIN')
    else:validation_attempts=len(refs)
    mt=Meter(audit_settings['B'],audit_settings['construction_cap']);mt.construction_attempts=validation_attempts
    rows={};errors=[];start=time.perf_counter()
    for a in refs:
        try:rows[a]=paid_score(ctx,a,mt)
        except Exception as exc:errors.append({'action':list(a),'type':type(exc).__name__,'message':str(exc)})
    reports={}
    for run in runs:
        agreement=True
        for row in run['archive']:
            a=tuple(row['action'])
            if a not in rows or abs(rows[a].risk_m2-float(row['risk_m2']))>1e-9*max(1,ctx.risk_scale):agreement=False
        reports[run['method']]={**report_output(ctx,run['output'],rows,refs,frozen['full_domain']),
            'score_agreement':agreement,'policy_status':run['status'],'policy_meter':run['meter'],
            'policy_seconds':run['elapsed_seconds']}
    return {'status':'COMPLETE' if len(rows)==len(refs) and not errors else 'PARTIAL',
            'context_digest':ctx.context_digest,'frozen_digest':d,'errors':errors,
            'rows':[r.report() for r in rows.values()],'reports':reports,'meter':mt.report(),
            'elapsed_seconds':time.perf_counter()-start}
