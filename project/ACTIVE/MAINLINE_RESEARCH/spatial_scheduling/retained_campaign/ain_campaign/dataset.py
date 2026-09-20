"""World-sharded, model-independent labels with separate train/dev namespaces."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from itertools import product
import os,time,traceback,math
import numpy as np
from .common import *
from .contexts import Context,finite_context
from .native import DynamicWorld
from .search import AnalyticPriority,construct_pool,protected_candidate
from .risk import Scorer

def world_spec(config,split,index,track=None,family=None):
 track=track or ('finite' if index%2==0 else 'dynamic')
 family=family or 'ID'
 ns=config['seed_namespace']+':'+split+':'+track+':'+family
 seed=seed_for(ns,index);counts=config['channels'][track]
 return {'track':track,'family':family,'index':index,'seed':seed,'channels':counts[index%len(counts)],'split':split,'unit_id':f'{split}-{track}-{family}-{index:06d}'}

def label_context(ctx,config):
 budget=config['labels']['candidates_per_context'];m=Meter(budget,config['labels']['construction_cap']);priority=AnalyticPriority(ctx,m);protect=protected_candidate(ctx,m,priority)
 pool=construct_pool(ctx,m,priority,max(budget,32),4,seed_for('V11_LABEL_POOL',ctx.digest));pool=list(dict.fromkeys([(),*((protect,) if protect else ()),*pool]))[:budget]
 scorer=Scorer(ctx,m,config['numerics']['mode'],nodes=config['numerics']['nodes']);rows=[]
 for r in scorer.evaluate_many(pool,config['labels']['chunk']):
  a=tuple(r['action']);r['moment_proxy_m2']=priority.moment_proxy(a);rows.append(r)
 aux={'indices':[],'association':[],'novelty':[],'protected_action':list(protect)};d=ctx.doc();available=set(ctx.available())
 for i,c in enumerate(d['channels']):
  if i not in available:continue
  aux['indices'].append(i);aux['novelty'].append(1./sum(x['root']==c['root'] for x in d['channels']))
  if ctx.track=='finite':
   old=scorer.old;p=scorer.p;q=scorer.q[i];av=np.asarray(old.association)
   # A paid extra label calculation, not a proposal input.
   ambiguity=0.
   for yy in (0,1):
    joint=p*(q if yy else 1-q);mass=float(joint.sum())
    if mass>0:pa=float(joint@av)/mass;ambiguity+=mass*4*pa*(1-pa)
   m.outcome_rows+=2;aux['association'].append(float(ambiguity))
  else:
   # Source-visible mode separation remaining under this action, using only a
   # paid singleton score if present. No hidden realized origin is used.
   r=next((r for r in rows if tuple(r['action'])==(i,)),None)
   if r is None:aux['indices'].pop();aux['novelty'].pop();continue
   aux['association'].append(float(min(1.,4*r.get('remaining_association_variance',0.))))
 return {'context':ctx.doc(),'candidates':[list(a) for a in pool],'scores':rows,'aux':aux,'meter':m.report(),'label_semantics':'complete_task_risk_stable_float64_v2','label_hash':digest(rows)}

def build_world(config,spec):
 t=time.perf_counter();samples=[];trajectory=[];native_ops=None
 if spec['track']=='finite':samples.append(label_context(finite_context(spec['seed'],spec['channels'],spec['family'],spec['index']),config))
 else:
  world=DynamicWorld(spec['seed'],spec['channels'],spec['family'],config['horizon'],spec['index'])
  for _ in range(config['horizon']):
   ctx=world.begin_cycle();rec=label_context(ctx,config);samples.append(rec)
   # Fixed complete-label teacher creates subsequent causal training histories;
   # not used for final trajectories and shared by every representation.
   best=min(rec['scores'],key=lambda r:(r['risk_m2'],r['action']));trajectory.append(world.apply(ctx,tuple(best['action'])))
  native_ops=world.ops.used
 return {'schema':'ain.v11.label_world.v1','status':'COMPLETE','spec':spec,'samples':samples,'teacher_trajectory':trajectory,'native_operations':native_ops,'seconds':time.perf_counter()-t}

def _one_label_job(config,spec,path,binding):
 path=Path(path);rp=path.with_suffix('.receipt.json')
 if completed(rp,binding):return read_json(rp)
 start=time.perf_counter()
 try:
  v=build_world(config,spec);atomic_json(path,v)
  return receipt(rp,binding,[path],unit=spec,seconds=time.perf_counter()-start,score_attempts=sum(x['meter']['scores'] for x in v['samples']))
 except Exception as exc:
  atomic_json(path.with_suffix('.failure.json'),{'spec':spec,'error':repr(exc),'traceback':traceback.format_exc(),'status':'FAILED'})
  receipt(rp,binding,[],status='FAILED',error=repr(exc),unit=spec)
  raise

def generate_labels(config,out,workers=None):
 out=Path(out);binding=digest({'config':config,'source':source_digest(),'stage':'labels'});jobs=[]
 for split,n in [('train',config['size']['train']),('development',config['size']['development'])]:
  for i in range(n):
   sp=world_spec(config,split,i);path=out/'data'/split/(sp['unit_id']+'.json');jobs.append((config,sp,str(path),digest([binding,sp])))
 workers=min(workers or config['resources']['label_workers'],16);start=time.perf_counter();failures=[]
 if workers==1:
  for args in jobs:
   try:_one_label_job(*args)
   except Exception as e:failures.append({'unit':args[1],'error':repr(e)})
 else:
  with ProcessPoolExecutor(max_workers=workers) as pool:
   futures={pool.submit(_one_label_job,*args):args[1] for args in jobs}
   for fut in as_completed(futures):
    try:fut.result()
    except Exception as e:failures.append({'unit':futures[fut],'error':repr(e)})
 rows=[]
 for _,spec,path,b in jobs:
  p=Path(path);rp=p.with_suffix('.receipt.json')
  if completed(rp,b):rows.append({'spec':spec,'path':p.relative_to(out).as_posix(),'sha256':filehash(p),'receipt':rp.relative_to(out).as_posix()})
 manifest={'schema':'ain.v11.dataset.v1','binding':binding,'status':'COMPLETE' if len(rows)==len(jobs) else 'PARTIAL','expected_worlds':len(jobs),'completed_worlds':len(rows),'units':rows,'failures':failures,'seconds':time.perf_counter()-start,'score_attempts':sum(read_json(out/x['receipt']).get('score_attempts',0) for x in rows)}
 atomic_json(out/'DATASET_MANIFEST.json',manifest)
 if failures:raise ContractError('LABEL_STAGE_PARTIAL')
 return manifest

def data_manifest(out):
 out=Path(out);m=read_json(out/'DATASET_MANIFEST.json')
 if m['status']!='COMPLETE' or not m['units']:raise ContractError('DATASET_INCOMPLETE')
 for r in m['units']:
  p=out/r['path']
  if not p.is_file() or filehash(p)!=r['sha256']:raise ContractError('DATASET_HASH')
 return m
