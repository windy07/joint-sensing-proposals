"""Immutable causal contexts; finite-law branch reuses adopted V10 parser/law."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction as F
import json,math
import numpy as np
from v10_finite import model as v10
from .common import *
from .stable_math import mixture_weights

@dataclass(frozen=True)
class Context:
 raw:bytes
 def __post_init__(self):
  d=json.loads(self.raw);unseal(d)
  if d.get('schema')!='ain.v11.context.v1' or d.get('track') not in ('finite','dynamic'):raise ContractError('CONTEXT_SCHEMA')
  if d.get('truth_access') is not False:raise ContractError('TRUTH_ACCESS')
  if d['units']!={'position':'m','velocity':'m/s','time':'s','energy':'J','risk':'m2'}:raise ContractError('UNITS')
  if any(k in d for k in ('truth','realized_mode','audit','reference_scores','future_observations')):raise ContractError('CAUSAL_LEAK')
  require_finite([d['length_m'],d['epoch'],d['resource_J'],d['threshold_m2']],'CONTEXT')
  if d['length_m']<=0 or d['resource_J']<0 or not 0<=d['epoch']<=32:raise ContractError('CONTEXT_RANGE')
  ch=d['channels'];ids=[c['id'] for c in ch]
  if len(set(ids))!=len(ch) or not 1<=len(ch)<=32:raise ContractError('CHANNEL_IDS')
  if d['track']=='finite':v10.load_context(d['finite_law'])
  else:
   w=np.asarray(d['weights']);mu=np.asarray(d['means']);cov=np.asarray(d['covariances'])
   if len(w) not in (1,2) or mu.shape!=(len(w),3) or cov.shape!=(len(w),3,3):raise ContractError('COMPLETE_MIXTURE')
   mixture_weights(d)
   require_finite(w,'WEIGHTS');require_finite(mu,'MEANS');require_finite(cov,'COVARIANCES')
   for p in cov:np.linalg.cholesky(p)
   if not np.allclose(cov,cov[0],rtol=1e-9,atol=1e-10*d['length_m']**2):raise ContractError('UNSUPPORTED_UNEQUAL_COVARIANCE')
   if d['posterior_authority']!='DynamicFactorGraph' or d['coordinate_units']!=['m','m/s','m']:raise ContractError('POSTERIOR_SCOPE')
   # Source/likelihood roots are binding metadata, never evidence of authenticity.
   roots={}
   for c in ch:
    require_finite([*c['H'],c['noise_sd_m'],c['common_loading'],c['acquire_s'],c['arrival_s'],c['energy_J']],'CHANNEL')
    if c['noise_sd_m']<=0:raise ContractError('NONPOSITIVE_PRIVATE_NOISE')
    signature=(c['H'],c['noise_sd_m'],c['common_loading'],c['acquire_s'])
    if c['root'] in roots and roots[c['root']]!=signature:raise ContractError('DUPLICATED_PROVENANCE_LAW_CONFLICT')
    roots[c['root']]=signature
    if c['arrival_s']<c['acquire_s'] or c['energy_J']<0 or len(c['H'])!=3:raise ContractError('CHANNEL_SCHEMA')
 def doc(self):return json.loads(self.raw)
 @property
 def id(self):return self.doc()['context_id']
 @property
 def digest(self):return self.doc()['digest']
 @property
 def track(self):return self.doc()['track']
 def legal(self,a):
  d=self.doc()
  if not isinstance(a,tuple) or tuple(sorted(a))!=a or len(set(a))!=len(a) or len(a)>d['max_size'] or any(type(i) is not int or not 0<=i<len(d['channels']) for i in a):raise ContractError('SCHEDULE_SHAPE')
  cs=[d['channels'][i] for i in a]
  if any(not c['available'] or c['acquire_s']<d['decision_s'] or c['arrival_s']>d['terminal_s'] for c in cs):raise ContractError('UNAVAILABLE_OR_LATE')
  if sum(F(str(c['energy_J'])) for c in cs)>F(str(d['resource_J'])):raise ContractError('ENERGY_EXHAUSTED')
  return a
 def available(self):
  out=[]
  for i in range(len(self.doc()['channels'])):
   try:self.legal((i,));out.append(i)
   except ContractError:pass
  return tuple(out)
 def groups(self,a):
  d=self.doc();seen=set();ans=[]
  for i in a:
   root=d['channels'][i]['root']
   if root not in seen:seen.add(root);ans.append(i)
  return tuple(ans)

def from_doc(d):return Context(canonical(seal(d)))
def finite_context(seed,n,family='ID',index=1):
 """Change likelihood/availability/resources, never use any evaluated outcome."""
 base=v10.generate(int(seed),n,4,'cheap' if index%5==0 else 'ambiguous')
 b=json.loads(base.raw);b.pop('digest');rr=rng('V11_FINITE_FAMILY',seed,family)
 if family=='spectrum':
  e=F(int(rr.choice([20,30,40])),100)
  for c in b['channels']:c['p_plus_given_z']=[str(e+(1-2*e)*F(q)) for q in c['p_plus_given_z']]
 elif family=='association':
  # Keep target marginal; change conditional association probabilities outside train range.
  for j in range(4):
   mass=sum(F(x['p']) for x in b['states'] if x['target_index']==j)
   pa=F(1 if j%2 else 19,20)
   for st in b['states']:
    if st['target_index']==j:st['p']=str(mass*(pa if st['association'] else 1-pa)/2)
 elif family=='provenance':
  for k in range(n//2,n):
   src=b['channels'][k%(max(1,n//3))]
   for key in ('provenance','p_plus_given_z','acquire_s'):b['channels'][k][key]=src[key]
 elif family=='availability':
  for c in b['channels']:
   c['available']=bool(rr.random()>.5)
   if rr.random()<.2:c['arrival_s']='6/5'
 elif family=='topology':
  b['max_size']=2;b['resource_J']='5/4'
 if family in ('topology','availability') and index%8==0:b['resource_J']='1/100';b['threshold_m2']='-1/100'
 law=v10.load_context(v10.seal(b));ch=[]
 for i,c in enumerate(law.channels):
  ch.append({'id':c.id,'root':c.provenance,'available':c.available,'energy_J':float(c.energy),'acquire_s':float(c.acquire),'arrival_s':float(c.arrive),'q':[float(t) for t in c.q], 'carrier_Hz':9e9 if family=='spectrum' else 3e9,'bandwidth_Hz':5e6 if family=='spectrum' else 20e6})
 return from_doc({'schema':'ain.v11.context.v1','track':'finite','origin':'DECLARED_R0_COMPLETE_FINITE','truth_access':False,'context_id':f'V11-F-{seed}','world_id':f'V11-F-{seed}','world_seed':int(seed),'family':family,'epoch':0,'decision_s':0.,'terminal_s':1.,'units':{'position':'m','velocity':'m/s','time':'s','energy':'J','risk':'m2'},'length_m':float(law.length),'resource_J':float(law.resource),'max_size':law.max_size,'threshold_m2':float(law.threshold),'lambda_energy':float(law.lambda_energy),'rho_m':float(law.rho),'lambda_tail_m2':float(law.lambda_tail),'channels':ch,'finite_law':json.loads(law.raw),'arrived_history':[]})
