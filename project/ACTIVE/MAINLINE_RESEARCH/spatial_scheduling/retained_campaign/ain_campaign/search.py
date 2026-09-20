"""Protected bounded construction, paid verification, commit-before-reference."""
from __future__ import annotations
from itertools import combinations,product
import math,time,random
import numpy as np
from v10_finite import model as v10
from v10_finite.search import Features as V10Features
from .common import *
from .contexts import Context
from .risk import Scorer
from .stable_math import mixture_weights

class AnalyticPriority:
 def __init__(self,ctx,meter):
  self.ctx=ctx;self.d=ctx.doc();self.m=meter;t=time.perf_counter()
  if ctx.track=='finite':
   self.vm=v10.Meter(0,0);self.feat=V10Features(v10.load_context(self.d['finite_law']),self.vm)
   meter.descriptor_entries+=self.vm.preprocessing_entries
  else:
   self.w,_=mixture_weights(self.d);self.mu=np.asarray(self.d['means']);self.P=np.asarray(self.d['covariances'][0]);center=self.w@self.mu;dm=self.mu-center;self.C=self.P+np.einsum('i,ij,ik->jk',self.w,dm,dm);self.var=self.C[0,0]
   meter.descriptor_entries+=self.C.size+self.mu.size
  meter.prepare_seconds+=time.perf_counter()-t
 def rank(self,actions):
  t=time.perf_counter();ans=[]
  if self.ctx.track=='finite':
   before=self.vm.affinity_products;ans=self.feat.rank(actions);self.m.affinity_products+=self.vm.affinity_products-before
  else:
   for a in actions:
    if not a:val=float(self.var)
    else:
     cs=[self.d['channels'][i] for i in self.ctx.groups(a)];H=np.asarray([c['H'] for c in cs]);sd=np.asarray([c['noise_sd_m'] for c in cs]);l=np.asarray([c['common_loading'] for c in cs]);R=np.diag(sd*sd)+np.outer(l,l);cy=H@self.C[:,0];S=H@self.C@H.T+R;val=float(self.var-cy@np.linalg.solve(S,cy));self.m.linear_solves+=1
    ans.append(val+self.d['lambda_energy']*sum(self.d['channels'][i]['energy_J'] for i in a))
  self.m.priority_candidates+=len(actions);self.m.priority_seconds+=time.perf_counter()-t;return ans
 def moment_proxy(self,a):
  if self.ctx.track=='finite':
   val=float(self.feat.covariance_proxy(a));self.m.pseudoinverses+=int(bool(a));return val+self.d['lambda_energy']*sum(self.d['channels'][i]['energy_J'] for i in a)
  return self.rank([a])[0]

def add_candidate(ctx,a,meter,pool,seen):
 """Charge attempted construction, preserving an already built legal prefix."""
 try:meter.construct()
 except BudgetExhausted:return False
 a=tuple(sorted(a))
 try:ctx.legal(a)
 except ContractError:meter.rejected+=1;return True
 if a not in seen:pool.append(a);seen.add(a)
 return True

def construct_pool(ctx,meter,priority,limit=128,width=4,random_seed=0):
 pool=[];seen=set();avail=ctx.available()
 for a in [(),*((i,) for i in avail)]:
  if not add_candidate(ctx,a,meter,pool,seen):return pool
 frontier=[a for a in pool if len(a)==1]
 for depth in range(2,ctx.doc()['max_size']+1):
  if not frontier:break
  ps=priority.rank(frontier);frontier=[a for _,a in sorted(zip(ps,frontier),key=lambda x:(x[0],x[1]))[:width]]
  new=[];new_seen=set()
  for base in frontier:
   for i in avail:
    if i in base:continue
    before=len(pool)
    if not add_candidate(ctx,base+(i,),meter,pool,seen):return pool[:limit]
    if len(pool)>before:new.append(pool[-1])
    if len(pool)>=limit:return pool
  frontier=new
 # Declared independently seeded exploratory proposals, not outcome-conditioned.
 rr=random.Random(random_seed)
 for _ in range(min(32,max(0,limit-len(pool)))):
  if not avail:break
  k=rr.randint(1,min(ctx.doc()['max_size'],len(avail)));a=tuple(rr.sample(avail,k))
  if not add_candidate(ctx,a,meter,pool,seen):break
  if len(pool)>=limit:break
 return pool

def protected_candidate(ctx,meter,analytic):
 options=[];seen=set();avail=ctx.available()
 for a in [(i,) for i in avail]:
  if not add_candidate(ctx,a,meter,options,seen):break
 for a in combinations(avail,2):
  if len(ctx.groups(a))<2:continue
  if not add_candidate(ctx,a,meter,options,seen):break
 if not options:return ()
 values=analytic.rank(options);return min(zip(values,options),key=lambda x:(x[0],x[1]))[1]

def run_policy(ctx,method,K,B,construct_cap,learned=None,protected=True,random_seed=0,shared=True,mode='production',pool_limit=128,nodes=64):
 if not 1<=K<=8 or not 0<=B<=32:raise ContractError('ONLINE_BUDGET')
 start=time.perf_counter();meter=Meter(B,construct_cap);scorer=Scorer(ctx,meter,mode,shared,nodes);analytic=AnalyticPriority(ctx,meter);rows=[];error=None;reserve=();candidate_failures=[]
 if learned is not None:learned.prepare(ctx,meter)
 priority=learned if learned is not None else analytic
 try:
  if protected and K>=2:reserve=protected_candidate(ctx,meter,analytic)
  # Every method gets the same lawful protective reserve and common construction cap.
  if method=='random':
   pool=[()];seen={()};avail=ctx.available();rr=random.Random(random_seed)
   while len(pool)<pool_limit and meter.construction<meter.construct_cap and avail:
    k=rr.randint(1,min(ctx.doc()['max_size'],len(avail)))
    if not add_candidate(ctx,tuple(rr.sample(avail,k)),meter,pool,seen):break
  else:pool=construct_pool(ctx,meter,priority,pool_limit,4,random_seed)
  # A complete reference is NOT called during construction.
  ordered=[()]+([reserve] if reserve else [])
  rest=[a for a in pool if a not in ordered]
  if method=='random':random.Random(random_seed+1).shuffle(rest)
  else:
   ranks=priority.rank(rest);rest=[a for _,a in sorted(zip(ranks,rest),key=lambda x:(x[0],x[1]))]
  queue=ordered+rest;seen_eval=set()
  while queue and meter.scores<B:
   a=queue.pop(0)
   if a in seen_eval:continue
   seen_eval.add(a)
   try:rows.append(scorer.evaluate(a))
   except Exception as exc:
    candidate_failures.append({'action':list(a),'error':type(exc).__name__+':'+str(exc),'paid_call':meter.scores})
    continue
   if method=='beam_block' and len(rows)>=2:
    # Paid incumbent determines the next local edits. No hidden scoring.
    best=tuple(min(rows,key=lambda r:r['risk_m2'])['action']);new=[];se=set(seen_eval)|set(queue)
    for i in ctx.available():
     for proposed in (tuple(j for j in best if j!=i),tuple(sorted(set(best)|{i}))):
      if meter.construction>=meter.construct_cap:break
      add_candidate(ctx,proposed,meter,new,se)
    if new:
     rv=analytic.rank(new);queue=[a for _,a in sorted(zip(rv,new),key=lambda x:(x[0],x[1]))]+queue
 except Exception as exc:
  error=type(exc).__name__+':'+str(exc)
 ranked=sorted(rows,key=lambda r:(r['risk_m2'],tuple(r['action'])))
 output=[()]
 if protected and reserve and K>=2 and any(tuple(r['action'])==reserve for r in rows):output.append(reserve)
 for r in ranked:
  a=tuple(r['action'])
  if a not in output and len(output)<K:output.append(a)
 verified={tuple(r['action']):r for r in rows};usable=[a for a in output if a in verified]
 selected=min(usable,key=lambda a:(verified[a]['risk_m2'],a)) if usable else ()
 tau=ctx.doc()['threshold_m2'];witness=any(r['risk_m2']<=tau for r in rows)
 body={'schema':'ain.v11.commit.v1','context_digest':ctx.digest,'method':method,'K':K,'B':B,'construct_cap':construct_cap,'protected':protected,'reserved':list(reserve),'output':[list(a) for a in output],'selected':list(selected),'archive':rows,'known_nonempty_before_audit':bool(witness),'status':('PARTIAL_VERIFIED' if usable else 'UNKNOWN') if (error or candidate_failures) else ('ABSTAINED' if not usable else 'COMPLETE'),'error':error,'candidate_failures':candidate_failures,'partial_verification':bool(candidate_failures or error),'meter':meter.report(),'policy_seconds':time.perf_counter()-start,'assurance':'NUMERICAL_DECLARED_LAW','formal_certificate':False}
 return seal(body)

def freeze_reference(ctx,commits,cap=256,construct_cap=2048,exhaustive_domain_cap=128):
 """All output/archival bytes commit before this independent expansion."""
 for c in commits:
  unseal(c)
  if c['context_digest']!=ctx.digest:raise ContractError('CROSS_CONTEXT_REFERENCE')
 actions=set()
 for c in commits:
  actions.update(tuple(a) for a in c['output']);actions.update(tuple(r['action']) for r in c['archive'])
 m=Meter(0,construct_cap);analytic=AnalyticPriority(ctx,m);independent=construct_pool(ctx,m,analytic,min(32,cap),4,seed_for('V11_REFERENCE',ctx.digest))
 actions.update(independent);complete=False;avail=ctx.available()
 size=sum(math.comb(len(avail),j) for j in range(min(len(avail),ctx.doc()['max_size'])+1))
 if size<=min(cap,exhaustive_domain_cap):
  for j in range(ctx.doc()['max_size']+1):
   for a in combinations(avail,j):
    try:ctx.legal(a);actions.add(a)
    except ContractError:pass
  complete=True
 if len(actions)>cap:raise ContractError('AUDIT_CAP_DOES_NOT_COVER_COMMITTED_UNION')
 return seal({'schema':'ain.v11.reference.v1','context_digest':ctx.digest,'commit_digests':[c['digest'] for c in commits],'actions':[list(a) for a in sorted(actions)],'complete_grammar':complete,'construction_meter':m.report()})

def audit(ctx,reference,commits,cap=256,nodes=64):
 unseal(reference)
 if reference['context_digest']!=ctx.digest or reference['commit_digests']!=[c['digest'] for c in commits]:raise ContractError('AUDIT_BINDING')
 m=Meter(cap,0);s=Scorer(ctx,m,nodes=nodes);rows=[];failures=[]
 for a in reference['actions']:
  try:rows.append(s.evaluate(tuple(a)))
  except Exception as e:failures.append({'action':a,'error':repr(e)})
 lookup={tuple(r['action']):r for r in rows};minimum=min((r['risk_m2'] for r in rows),default=None);tau=ctx.doc()['threshold_m2'];qualified={tuple(r['action']) for r in rows if r['numeric_error_estimate_m2']<=1e-5*ctx.doc()['length_m']**2 and r['risk_m2']+r['numeric_error_estimate_m2']<=tau}
 numeric_ambiguous=any(r['numeric_error_estimate_m2']>1e-5*ctx.doc()['length_m']**2 or abs(r['risk_m2']-tau)<=r['numeric_error_estimate_m2'] for r in rows)
 qstate='KNOWN_NONEMPTY_NUMERICAL' if qualified else ('KNOWN_EMPTY_NUMERICAL' if reference['complete_grammar'] and not failures and not numeric_ambiguous else 'UNKNOWN')
 reports=[]
 for c in commits:
  unseal(c);output=[tuple(a) for a in c['output']];sel=tuple(c['selected']);row=lookup.get(sel);covered=bool(set(output)&qualified) if qstate=='KNOWN_NONEMPTY_NUMERICAL' else None
  if qstate=='KNOWN_NONEMPTY_NUMERICAL' and not covered:
   if any(a not in lookup or lookup[a]['risk_m2']-lookup[a]['numeric_error_estimate_m2']<=tau or lookup[a]['numeric_error_estimate_m2']>1e-5*ctx.doc()['length_m']**2 for a in output):covered=None
  reports.append({'method':c['method'],'commit_digest':c['digest'],'execution_status':c['status'],'Q_state':qstate,'coverage':covered,'unconditional_miss':(not covered) if covered is not None else (False if qstate=='KNOWN_EMPTY_NUMERICAL' else None),'risk_m2':None if row is None else row['risk_m2'],'tail_probability':None if row is None else row['tail_probability'],'finite_reference_regret_m2':None if row is None or minimum is None else row['risk_m2']-minimum,'numerical_ambiguity':numeric_ambiguous,'abstained':c['status'] not in ('COMPLETE','PARTIAL_VERIFIED'),'assurance':'NUMERICAL_NOT_FORMAL'})
 return seal({'schema':'ain.v11.audit.v1','context_digest':ctx.digest,'reference_digest':reference['digest'],'reference_complete':reference['complete_grammar'],'status':'COMPLETE' if not failures else 'PARTIAL','reports':reports,'rows':rows,'failures':failures,'meter':m.report()})
