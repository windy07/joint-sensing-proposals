"""Committed policy trajectories, then separately frozen context-specific references."""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import time,traceback,math
from .common import *
from .contexts import Context,finite_context
from .native import DynamicWorld
from .dataset import world_spec
from .models import LearnedPriority
from .train import load_model
from .search import run_policy,freeze_reference,audit
from .risk import Scorer


def training_jobs(config):
 jobs=[{'arm':a,'seed':s} for a in ARMS for s in config['model_seeds']]
 jobs += [{'arm':a,'seed':s} for a in config['ablations'].get('retrain_arms',['no_assoc_supervision']) for s in config['model_seeds'][:config['ablations']['retrain_seeds']]]
 return jobs

def freeze_plan(config,out):
 out=Path(out);selected=[];data=read_json(out/'DATASET_MANIFEST.json')
 for j in training_jobs(config):
  p=out/'training'/f"{j['arm']}_{j['seed']}";r=read_json(p/'TRAIN_RECEIPT.json')
  expected=digest({'config':config,'dataset':data['binding'],'source':source_digest(),'arm':j['arm'],'seed':j['seed']})
  if not completed(p/'TRAIN_RECEIPT.json',expected):raise ContractError('TRAINING_NOT_COMPLETE')
  c=read_json(p/'SELECTED.json');f=out/c['path']
  if filehash(f)!=c['sha256']:raise ContractError('SELECTED_CHECKPOINT_HASH')
  selected.append({**j,**c})
 data=read_json(out/'DATASET_MANIFEST.json')
 plan=seal({'schema':'ain.v11.frozen_plan.v1','source_digest':source_digest(),'contract_sha256':filehash(ROOT/'EXPERIMENT_CONTRACT.json'),'config_digest':digest(config),'dataset_digest':filehash(out/'DATASET_MANIFEST.json'),'checkpoints':selected,'final_seed_namespace':config['seed_namespace']+':final','families':config['families'],'worlds_per_family_per_track':config['size']['final_per_family'],'tracks':['finite','dynamic'],'primary':'paired dynamic normalized capped terminal position loss plus mean energy penalty','threshold_selection':'no data-dependent qualification threshold tuning','selection':'minimum fixed development-pool mean normalized task risk; earlier step breaks ties','reference':'commit all methods first, then context-specific union plus fixed constructive pool','sensitivity':config['statistics']})
 path=out/'FROZEN_PLAN.json'
 if path.exists():
  if read_json(path)!=plan:raise ContractError('FROZEN_PLAN_CONFLICT')
 else:
  if (out/'final').exists() and any((out/'final').rglob('*.json')):raise ContractError('FINAL_ALREADY_OPEN_BEFORE_FREEZE')
  atomic_json(path,plan)
 return plan

def get_plan(config,out):
 out=Path(out);p=read_json(out/'FROZEN_PLAN.json');unseal(p)
 if p.get('contract_sha256')!=filehash(ROOT/'EXPERIMENT_CONTRACT.json'):raise ContractError('FROZEN_CONTRACT_CHANGED')
 if p['config_digest']!=digest(config) or p['source_digest']!=source_digest():raise ContractError('FROZEN_PLAN_SOURCE_CONFIG')
 for c in p['checkpoints']:
  if not (out/c['path']).exists() or filehash(out/c['path'])!=c['sha256']:raise ContractError('FROZEN_CHECKPOINT_CHANGED')
 return p

def evaluation_jobs(config,plan):
 jobs=[]
 for c in plan['checkpoints']:
  jobs.append({'method':c['arm'],'seed':c['seed'],'checkpoint':c,'protected':c['arm']!='original_agent','tag':f"{c['arm']}_{c['seed']}"})
  if c['arm']=='integrated_agent' and c['seed'] in config['model_seeds'][:config['ablations']['toggle_seeds']]:
   jobs.append({'method':'integrated_no_protection','seed':c['seed'],'checkpoint':c,'protected':False,'tag':f"integrated_no_protection_{c['seed']}"})
 for c in plan['checkpoints']:
  if config['ablations'].get('original_protection_toggle',False) and c['arm']=='original_agent' and c['seed'] in config['model_seeds'][:config['ablations']['toggle_seeds']]:
   jobs.append({'method':'original_with_protection','seed':c['seed'],'checkpoint':c,'protected':True,'tag':f"original_with_protection_{c['seed']}"})
 for c in CONTROLS:jobs.append({'method':c,'seed':config['control_seed'],'checkpoint':None,'protected':True,'tag':c})
 return jobs

def settings(config,index):
 o=config['online'];answer=[{'tag':'main','K':o['K'],'B':o['B']}]
 if index<config['diagnostics']['curve_worlds_per_family']:
  answer += [{'tag':f'B{b}','K':4,'B':b} for b in (4,8,32)]
  answer += [{'tag':f'K{k}','K':k,'B':16} for k in (2,8)]
 return answer

def evaluate_one(config,spec,setting,job,model=None):
 start=time.perf_counter();records=[];failure=None;world=None
 policy=LearnedPriority(model,spec['unit_id']+':'+setting['tag']) if model is not None else None
 try:
  world=DynamicWorld(spec['seed'],spec['channels'],spec['family'],config['horizon'],spec['index']) if spec['track']=='dynamic' else None
  for cycle in range(config['horizon'] if world is not None else 1):
   ctx=world.begin_cycle() if world is not None else finite_context(spec['seed'],spec['channels'],spec['family'],spec['index'])
   commit=run_policy(ctx,job['method'],setting['K'],setting['B'],config['online']['construction_cap'],learned=policy,protected=job['protected'],random_seed=seed_for('V11_RANDOM_POLICY',spec['seed'],cycle,job['seed']),mode=config['numerics']['mode'],pool_limit=config['online']['pool_limit'],nodes=config['numerics']['nodes'])
   if commit['status']=='UNKNOWN':
    records.append({'cycle':cycle,'context':ctx.doc(),'commit':commit,'actual':None})
    raise ContractError('POLICY_COMPUTATION_FAILED:'+str(commit.get('error')))
   actual=world.apply(ctx,tuple(commit['selected'])) if world is not None else None
   records.append({'cycle':cycle,'context':ctx.doc(),'commit':commit,'actual':actual})
 except Exception as exc:
  failure={'error':repr(exc),'traceback':traceback.format_exc(),'after_cycles':len(records),'attempted_cycle':locals().get('cycle'),'last_context':None if 'ctx' not in locals() else ctx.doc(),'last_commit':None if 'commit' not in locals() else commit}
 # Numerical failures are scientific outcomes, not omitted worlds. Infrastructure
 # hashes/IO failures remain exceptions at the outer job boundary.
 L=records[0]['context']['length_m'] if records else (world.length if world else 1.)
 if world is not None and failure is None and len(records)==config['horizon']:
  actuals=[r['actual'] for r in records];e2=[a['squared_error_m2']/L**2 for a in actuals];energies=[a['energy_J'] for a in actuals]
  endpoint=min(e2[-1],25.)+.01*sum(energies)/config['horizon']
  trajectory=sum(min(v,25.) for v in e2)/len(e2)+.01*sum(energies)/len(e2)
  large=bool(actuals[-1]['large_error']);raw=e2[-1]
 elif spec['track']=='finite' and records and failure is None:
  c=records[0]['commit'];r=next((r for r in c['archive'] if r['action']==c['selected']),None)
  endpoint=None if r is None else r['risk_m2']/L**2;trajectory=None;large=None;raw=None
 else:endpoint=25.24 if spec['track']=='dynamic' else None;trajectory=25.24 if spec['track']=='dynamic' else None;large=True if spec['track']=='dynamic' else None;raw=None
 return {'schema':'ain.v11.final_world.v1','execution_status':'FAILED' if failure else 'COMPLETE','spec':spec,'setting':setting,'method':job['method'],'model_seed':job['seed'],'protected':job['protected'],'records':records,'failure':failure,'primary_normalized_loss':endpoint,'trajectory_normalized_loss':trajectory,'raw_terminal_normalized_squared_error':raw,'large_error':large,'length_m':L,'native_operations':None if world is None else world.ops.used,'seconds':time.perf_counter()-start,'is_confirmation':False}

def evaluate_job(config,out,tag,device='cpu'):
 out=Path(out);plan=get_plan(config,out);jobs=evaluation_jobs(config,plan);job=next(j for j in jobs if j['tag']==tag);binding=digest([plan['digest'],job]);base=out/'final'/tag;base.mkdir(parents=True,exist_ok=True)
 done=base/'EVAL_RECEIPT.json'
 if completed(done,binding):return read_json(done)
 model=None
 if job['checkpoint']:
  cp=job['checkpoint'];model,_,_=load_model(out/cp['path'],config,cp['arm'],cp['seed'],device,cp['sha256'])
 start=time.perf_counter();rows=[];failed=0
 for track in ('finite','dynamic'):
  for family in config['families']:
   for index in range(config['size']['final_per_family']):
    spec=world_spec(config,'final',index,track,family)
    for setting in settings(config,index):
     path=base/track/family/f"{index:05d}_{setting['tag']}.json";rp=path.with_suffix('.receipt.json');b=digest([binding,spec,setting])
     if not completed(rp,b):
      result=evaluate_one(config,spec,setting,job,model);result['plan_digest']=plan['digest'];result['checkpoint_sha256']=None if not job['checkpoint'] else job['checkpoint']['sha256'];atomic_json(path,result)
      receipt(rp,b,[path],scientific_execution_status=result['execution_status'])
     content=read_json(path);failed+=int(content['execution_status']=='FAILED');rows.append({'path':path.relative_to(out).as_posix(),'sha256':filehash(path),'spec':spec,'setting':setting,'scientific_status':content['execution_status']})
 atomic_json(base/'INDEX.json',rows)
 return receipt(done,binding,[base/'INDEX.json'],completed_worlds=len(rows),scientific_failed_worlds=failed,seconds=time.perf_counter()-start)

def commit_evaluation(config,out):
 out=Path(out);p=get_plan(config,out);rows=[]
 for j in evaluation_jobs(config,p):
  base=out/'final'/j['tag'];b=digest([p['digest'],j]);rp=base/'EVAL_RECEIPT.json'
  if not completed(rp,b):raise ContractError('FINAL_JOB_INCOMPLETE:'+j['tag'])
  indexed=read_json(base/'INDEX.json');expected={(t,f,i,s['tag']) for t in ('finite','dynamic') for f in config['families'] for i in range(config['size']['final_per_family']) for s in settings(config,i)}
  actual={(r['spec']['track'],r['spec']['family'],r['spec']['index'],r['setting']['tag']) for r in indexed}
  if actual!=expected or len(actual)!=len(indexed):raise ContractError('INCOMPLETE_OR_DUPLICATE_FINAL_INDEX')
  for r in indexed:
   if r['spec']!=world_spec(config,'final',r['spec']['index'],r['spec']['track'],r['spec']['family']):raise ContractError('FINAL_SPEC_CHANGED')
   if not (out/r['path']).is_file() or filehash(out/r['path'])!=r['sha256']:raise ContractError('FINAL_SHARD_HASH')
   rows.append({**r,'tag':j['tag']})
 manifest=seal({'plan_digest':p['digest'],'source_digest':source_digest(),'status':'COMMITTED','rows':rows})
 dest=out/'EVALUATION_COMMIT.json'
 if dest.exists() and read_json(dest)!=manifest:raise ContractError('COMMITTED_EVALUATION_CHANGED')
 atomic_json(dest,manifest);return manifest

def _audit_unit(args):
 config,out,unit,rows,manifest_digest=args;out=Path(out);path=out/'audit'/unit['track']/unit['family']/f"{unit['index']:05d}.json";rp=path.with_suffix('.receipt.json');binding=digest([manifest_digest,unit])
 if completed(rp,binding):return str(path)
 groups=defaultdict(list);worlds=[]
 for row in rows:
  f=out/row['path']
  if filehash(f)!=row['sha256']:raise ContractError('COMMITTED_WORLD_CHANGED')
  v=read_json(f);worlds.append({'tag':row['tag'],'setting':v['setting'],'path':row['path'],'primary_normalized_loss':v['primary_normalized_loss'],'trajectory_normalized_loss':v['trajectory_normalized_loss'],'large_error':v['large_error'],'execution_status':v['execution_status'],'native_operations':v['native_operations'],'seconds':v['seconds']})
  for rec in v['records']:groups[rec['context']['digest']].append((row['tag'],v['setting'],rec))
 audits=[];shared_diagnostics=[]
 for key,items in groups.items():
  ctx=Context(canonical(items[0][2]['context']));commits=[x[2]['commit'] for x in items]
  ref=freeze_reference(ctx,commits,config['audit']['cap'],config['audit']['construction_cap'],config['audit'].get('exhaustive_domain_cap',128));a=audit(ctx,ref,commits,config['audit']['cap'],config['numerics']['audit_nodes'])
  audits.append({'context_digest':key,'length_m':ctx.doc()['length_m'],'members':[{'tag':x[0],'setting':x[1]['tag'],'cycle':x[2]['cycle']} for x in items],'reference':ref,'audit':a})
  # Fixed committed candidates, independently re-evaluated after commitment.
  if unit['index']<config['diagnostics']['shared_evaluation_worlds']:
   for shared in (True,False):
    mm=Meter(len(ref['actions']),0);ss=Scorer(ctx,mm,shared=shared,nodes=config['numerics']['audit_nodes']);vals=[];diag_errors=[]
    for xx in ref['actions']:
     try:vals.append(ss.evaluate(tuple(xx)))
     except Exception as exc:diag_errors.append({'action':xx,'error':repr(exc)})
    shared_diagnostics.append({'context_digest':key,'shared':shared,'failures':diag_errors,'meter':mm.report(),'score_digest':digest(vals),'maximum_absolute_difference_from_audit':max((abs(x['risk_m2']-y['risk_m2']) for x,y in zip(vals,a['rows'])),default=0.),'agreement':len(vals)==len(a['rows']) and all(abs(x['risk_m2']-y['risk_m2'])<=config['numerics']['comparison_atol_m2']+config['numerics']['comparison_rtol']*abs(y['risk_m2']) for x,y in zip(vals,a['rows']))})
 value={'schema':'ain.v11.audited_unit.v1','unit':unit,'committed_manifest':manifest_digest,'worlds':worlds,'contexts':audits,'shared_evaluation':shared_diagnostics,'status':'COMPLETE' if all(x['audit']['status']=='COMPLETE' for x in audits) else 'NUMERICAL_PARTIAL'}
 atomic_json(path,value);receipt(rp,binding,[path],scientific_status=value['status']);return str(path)

def audit_all(config,out,workers=None):
 from concurrent.futures import ProcessPoolExecutor
 out=Path(out);plan=get_plan(config,out);m=read_json(out/'EVALUATION_COMMIT.json');unseal(m);groups=defaultdict(list)
 if m['plan_digest']!=plan['digest']:raise ContractError('AUDIT_PLAN_BINDING')
 for r in m['rows']:
  s=r['spec'];groups[(s['track'],s['family'],s['index'])].append(r)
 args=[(config,str(out),{'track':k[0],'family':k[1],'index':k[2]},rs,m['digest']) for k,rs in groups.items()]
 workers=min(workers or config['resources']['label_workers'],16)
 if workers==1:paths=[_audit_unit(a) for a in args]
 else:
  with ProcessPoolExecutor(max_workers=workers) as pool:paths=list(pool.map(_audit_unit,args,chunksize=1))
 result={'schema':'ain.v11.audit_manifest.v1','status':'COMPLETE','committed_manifest':m['digest'],'units':[{'path':Path(p).relative_to(out).as_posix(),'sha256':filehash(p)} for p in paths],'expected_units':len(args)}
 atomic_json(out/'AUDIT_MANIFEST.json',result);return result
