"""Finite Windows/Linux campaign: calibration, shards, jobs, freeze, final, audit."""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import sys,os,time,traceback,copy,threading
from .common import *
from .planning import preflight,calibrate,validate_config
from .dataset import generate_labels
from .train import train_job
from .evaluate import training_jobs,freeze_plan,evaluation_jobs,evaluate_job,commit_evaluation,audit_all
from .analysis import analyze,export_results
from .processes import run_child


def smoke(config,out,device='cpu'):
 """Same scientific stage functions as the campaign, one tiny local process."""
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 if not config.get('smoke'):raise ContractError('SMOKE_REQUIRES_BOUNDED_CONFIG')
 start=time.perf_counter();source=source_digest();state={'status':'RUNNING','source_digest':source,'config_digest':digest(config)};atomic_json(out/'CAMPAIGN_STATE.json',state)
 try:
  preflight(config,out,False);effective=calibrate(config,out);generate_labels(effective,out,1)
  for j in training_jobs(effective):train_job(effective,out,j['arm'],j['seed'],device)
  plan=freeze_plan(effective,out)
  for j in evaluation_jobs(effective,plan):evaluate_job(effective,out,j['tag'],device)
  commit_evaluation(effective,out);audit_all(effective,out,1);summary=analyze(effective,out)
  state.update(status='COMPLETE',scientific_status=summary['status'],seconds=time.perf_counter()-start)
 except Exception as exc:
  state.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start);atomic_json(out/'CAMPAIGN_STATE.json',state);raise
 atomic_json(out/'CAMPAIGN_STATE.json',state);return state

def run_campaign(config_path,out,resume=False,require_cuda=True,tier=None):
 raw=validate_config(read_json(config_path));out=Path(out);out.mkdir(parents=True,exist_ok=True);source=source_digest();contract=filehash(ROOT/'EXPERIMENT_CONTRACT.json');seed_manifest=filehash(ROOT/'HISTORICAL_SEED_EXCLUSIONS.json');binding=digest([raw,source,contract,seed_manifest]);sp=out/'CAMPAIGN_STATE.json';now=time.time()
 if sp.exists():
  state=read_json(sp)
  if state['binding']!=binding:raise ContractError('INCOMPATIBLE_CAMPAIGN_RESUME')
  if not resume:raise ContractError('EXISTING_CAMPAIGN_REQUIRES_RESUME')
  if state.get('status')=='COMPLETE':
   # Still check immutable plan and its checkpoint bytes, not the status alone.
   from .evaluate import get_plan
   get_plan(read_json(out/'EFFECTIVE_CONFIG.json'),out)
   am=read_json(out/'AUDIT_MANIFEST.json')
   for x in am['units']:
    if filehash(out/x['path'])!=x['sha256']:raise ContractError('RESUME_AUDIT_HASH')
   for x in read_json(out/'EVALUATION_COMMIT.json')['rows']:
    if filehash(out/x['path'])!=x['sha256']:raise ContractError('RESUME_FINAL_HASH')
   return state
 else:state={'schema':SCHEMA,'binding':binding,'status':'RUNNING','started_unix':now,'deadline_unix':now+raw['resources']['campaign_hours']*3600,'source_digest':source,'config_digest':digest(raw),'contract_sha256':contract,'historical_seed_manifest_sha256':seed_manifest,'stage_receipts':[]}
 atomic_json(out/'REQUESTED_CONFIG.json',raw);state['status']='RUNNING';atomic_json(sp,state)
 lock=threading.Lock();cancel=threading.Event()
 def remaining():return state['deadline_unix']-time.time()
 def one(stage,args,env=None):
  cmd=[sys.executable,'-B',str(ROOT/'run.py'),stage,'--config',str(out/'REQUESTED_CONFIG.json' if stage in ('preflight','calibrate') else out/'EFFECTIVE_CONFIG.json'),'--output',str(out),*args]
  name=stage+'_'+str(len(list((out/'logs').glob('*.receipt.json'))) if (out/'logs').exists() else 0)+'_'+digest(args)[:8]
  if cancel.is_set():raise ContractError('CAMPAIGN_CANCELLED')
  r=run_child(cmd,out/'logs'/(name+'.log'),remaining(),env,str(ROOT),cancel)
  with lock:
   state['stage_receipts'].append({'stage':stage,'args':args,'receipt':str(Path(r['log']).with_suffix('.receipt.json').relative_to(out)),'status':r['status']});atomic_json(sp,state)
  if r['exit_status']!=0:cancel.set();raise ContractError('STAGE_'+r['status']+':'+stage)
  return r
 try:
  one('preflight',['--require-cuda'] if require_cuda else [])
  one('calibrate',['--tier',tier] if tier else [])
  config=read_json(out/'EFFECTIVE_CONFIG.json');one('labels',[])
  devices=config['resources']['gpu_devices'] if require_cuda else []
  # Deterministic queue assignment; at most one independent job per GPU.
  def run_queue(q,device):
   for stage,args in q:
    env={'CUDA_VISIBLE_DEVICES':str(device)} if device is not None else {'CUDA_VISIBLE_DEVICES':''}
    one(stage,args+['--device','cuda:0' if device is not None else 'cpu'],env)
  jobs=training_jobs(config);n=max(1,len(devices));queues=[[] for _ in range(n)]
  for i,j in enumerate(jobs):queues[i%n].append(('train',['--arm',j['arm'],'--seed',str(j['seed'])]))
  with ThreadPoolExecutor(max_workers=n) as pool:
   futures=[pool.submit(run_queue,q,devices[i] if devices else None) for i,q in enumerate(queues)]
   try:
    for f in as_completed(futures):f.result()
   except BaseException:
    cancel.set();raise
  one('freeze',[]);plan=read_json(out/'FROZEN_PLAN.json');queues=[[] for _ in range(n)]
  for i,j in enumerate(evaluation_jobs(config,plan)):queues[i%n].append(('evaluate',['--tag',j['tag']]))
  with ThreadPoolExecutor(max_workers=n) as pool:
   futures=[pool.submit(run_queue,q,devices[i] if devices else None) for i,q in enumerate(queues)]
   try:
    for f in as_completed(futures):f.result()
   except BaseException:
    cancel.set();raise
  one('commit',[]);one('audit',[]);one('analyze',[])
  summary=read_json(out/'analysis'/'SUMMARY.json');state.update(status='COMPLETE',scientific_status=summary['status'],completed_unix=time.time(),elapsed_seconds=time.time()-state['started_unix'])
 except BaseException as exc:
  state.update(status='TIMED_OUT_PARTIAL' if remaining()<=0 else 'FAILED_PARTIAL',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.time()-state['started_unix']);atomic_json(sp,state);raise
 atomic_json(sp,state);return state
