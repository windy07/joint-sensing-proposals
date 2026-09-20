"""Hardware-only workload choice, source preflight, immutable campaign bindings."""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
import os,sys,platform,time,copy,math,shutil
import numpy as np
import torch,scipy
from .common import *
from .models import make_model,config_for,objective,schedule_scores
from .graph_adapter import graph_context
from .contexts import finite_context
from .native import DynamicWorld
from .risk import Scorer
from .dataset import label_context
from .search import run_policy


def validate_config(c):
 if c.get('schema')!='ain.v11.campaign.v1' or not set(c['families'])<=set(FAMILIES):raise ContractError('CONFIG_SCHEMA')
 if len(set(c['model_seeds']))!=len(c['model_seeds']) or not 1<=len(c['model_seeds'])<=5:raise ContractError('MODEL_SEEDS')
 if not 1<=c['horizon']<=8 or not 1<=c['online']['K']<=8 or not 0<=c['online']['B']<=32:raise ContractError('CONFIG_BOUNDS')
 if not 1<=c['resources']['label_workers']<=16 or c['audit']['cap']>4096:raise ContractError('RESOURCE_BOUNDS')
 if any(c['size'][k]<=0 for k in ('train','development','final_per_family','updates')):raise ContractError('EMPTY_CAMPAIGN')
 return c

def preflight(config,out,require_cuda=False):
 from .seed_guard import validate_seed_plan
 seeds=validate_seed_plan(config)
 import matplotlib
 out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();torch.set_num_threads(config['resources']['torch_threads'])
 bindings=read_json(ROOT/'VENDOR_SOURCE_BINDINGS.json');bad=[]
 entries=bindings.get('files',bindings) if isinstance(bindings,dict) else bindings
 if isinstance(entries,dict):entries=list(entries.values())
 for e in entries:
  path=ROOT/e.get('destination',e.get('path',''))
  if not path.is_file() or filehash(path)!=e['sha256']:bad.append(str(path))
 if bad:raise ContractError('VENDOR_SOURCE_CHANGED:'+str(bad[:3]))
 counts={};classes={}
 for arm in ARMS:
  model=make_model(arm,config,config['model_seeds'][0]);counts[arm]=sum(p.numel() for p in model.parameters());classes[arm]=type(model).__name__;del model
 gpu=[]
 for i in range(torch.cuda.device_count()):
  p=torch.cuda.get_device_properties(i)
  with torch.cuda.device(i):bf16=torch.cuda.is_bf16_supported()
  gpu.append({'index':i,'name':p.name,'total_memory_bytes':p.total_memory,'compute_capability':[p.major,p.minor],'bf16_supported':bf16})
 if require_cuda and len(gpu)<len(config['resources']['gpu_devices']):raise ContractError('INSUFFICIENT_GPUS_FOR_DECLARED_CAMPAIGN')
 if require_cuda:
  for i in config['resources']['gpu_devices'] or [0]:
   a=torch.ones((4,4),device=f'cuda:{i}');(a@a).sum().item();torch.cuda.synchronize(i)
 ratio={a:counts[a]/counts['integrated_agent'] for a in ARMS}
 if not config.get('smoke') and max(abs(x-1) for x in ratio.values())>config['model']['parameter_count_tolerance']:raise ContractError('PARAMETER_BUDGET_MISMATCH:'+str(ratio))
 r={'schema':SCHEMA,'status':'COMPLETE','python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'gpus':gpu,'cpu_count_interface':os.cpu_count(),'free_disk_bytes':shutil.disk_usage(out).free,'model_parameters':counts,'backbone_classes':classes,'relative_parameter_count':ratio,'source_digest':source_digest(),'contract_sha256':filehash(ROOT/'EXPERIMENT_CONTRACT.json'),'configuration_digest':digest(config),'verified_vendor_files':len(entries),'seconds':time.perf_counter()-start,'reported_hardware_not_independently_assumed':True,'seed_plan':seeds}
 stable={k:r[k] for k in ('python','executable','platform','numpy','scipy','matplotlib','torch','torch_cuda_build','gpus')}
 ep=out/'ENVIRONMENT_BINDING.json'
 if ep.exists() and read_json(ep)!=stable:raise ContractError('ENVIRONMENT_CHANGED_ON_RESUME')
 atomic_json(ep,stable);r['environment_digest']=digest(stable)
 atomic_json(out/'PREFLIGHT.json',r);return r

def calibrate(config,out,force_tier=None):
 out=Path(out);rp=out/'RESOURCE_PLAN.json';binding=digest([config,source_digest(),filehash(ROOT/'EXPERIMENT_CONTRACT.json'),filehash(ROOT/'HISTORICAL_SEED_EXCLUSIONS.json'),'HARDWARE_ONLY'])
 if rp.exists():
  r=read_json(rp)
  if r['binding']!=binding:raise ContractError('RESOURCE_PLAN_BINDING')
  return read_json(out/'EFFECTIVE_CONFIG.json')
 if (out/'DATASET_MANIFEST.json').exists():raise ContractError('CALIBRATION_AFTER_LABELS')
 if config.get('smoke'):
  chosen=copy.deepcopy(config);r={'binding':binding,'status':'COMPLETE','tier':'SMOKE','not_performed':'hardware throughput calibration omitted for tiny wiring preset','optimizer_updates':0,'source_digest':source_digest()}
 else:
  torch.set_num_threads(config['resources']['torch_threads']);device='cuda:0' if torch.cuda.is_available() else 'cpu';t0=time.perf_counter();ctx=finite_context(seed_for(config['seed_namespace'],'calibration'),max(config['channels']['finite']),'ID',0)
  lc=copy.deepcopy(config);lc['labels']['candidates_per_context']=8;rec=label_context(ctx,lc);label_s=time.perf_counter()-t0
  world=DynamicWorld(seed_for(config['seed_namespace'],'calibration-dynamic'),max(config['channels']['dynamic']),'ID',1,1);t=time.perf_counter();dc=world.begin_cycle();sm=Meter(8,0);sc=Scorer(dc,sm,nodes=config['numerics']['nodes']);acts=[(),*((i,) for i in dc.available())][:8];rv=[sc.evaluate(a) for a in acts];native_label_s=time.perf_counter()-t
  model=make_model('integrated_agent',config,config['model_seeds'][0],device);optimizer=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate']);g=graph_context(ctx,Meter(0,0),device)
  t=time.perf_counter()
  with torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=config['training']['mixed_precision'] and device.startswith('cuda')):
   loss,_,_,_=objective(model,g,[tuple(a) for a in rec['candidates']],rec['scores'],rec['aux'],ctx,'integrated_agent')
  loss.backward();optimizer.step()
  if device.startswith('cuda'):torch.cuda.synchronize()
  update_s=time.perf_counter()-t;del optimizer,model
  t=time.perf_counter();cr=run_policy(dc,'analytic_constructive',4,16,config['online']['construction_cap'],nodes=config['numerics']['nodes']);policy_s=time.perf_counter()-t
  # Prices are measured locally once, not tuned to policy quality. Worst-arm and
  # CPU/GPU queue imbalance covered by explicit conservative safety factor.
  gpus=max(1,len(config['resources']['gpu_devices']) if torch.cuda.is_available() else 1);workers=config['resources']['label_workers'];factor=config['resources']['calibration_safety_factor']
  def estimate(size):
   contexts=(size['train']+size['development'])*(1+config['horizon'])/2
   labels=contexts*max(label_s,native_label_s)*config['labels']['candidates_per_context']/8/workers
   job_count=4*len(config['model_seeds'])+len(config['ablations'].get('retrain_arms',['no_assoc_supervision']))*config['ablations']['retrain_seeds']
   training=job_count*size['updates']*(1+config['horizon'])/2*update_s/gpus
   development=job_count*4*size['development']*(1+config['horizon'])/2*(update_s/3)/gpus
   eval_job_count=job_count+3+config['ablations']['toggle_seeds']*(1+int(config['ablations'].get('original_protection_toggle',False)))
   decisions=eval_job_count*len(config['families'])*(size['final_per_family']+5*config['diagnostics']['curve_worlds_per_family'])*(1+config['horizon'])
   evaluation=decisions*(policy_s+update_s/2)/gpus
   audit=decisions*max(native_label_s,label_s)*6/workers
   return factor*(labels+training+development+evaluation+audit)
  estimates={'default_s':estimate(config['size']),'reduced_s':estimate(config['reduced_size'])};tier=force_tier or ('default' if estimates['default_s']<=config['resources']['campaign_hours']*3600*.9 else 'reduced');chosen=copy.deepcopy(config)
  if tier=='reduced':chosen['size']=config['reduced_size'];chosen['diagnostics']['curve_worlds_per_family']=min(16,chosen['diagnostics']['curve_worlds_per_family'])
  r={'binding':binding,'status':'COMPLETE','tier':tier,'measured_label_8_finite_s':label_s,'measured_label_8_dynamic_s':native_label_s,'measured_one_update_s':update_s,'measured_analytic_policy_s':policy_s,'estimated_seconds':estimates,'optimizer_updates':1,'calibration_contexts':2,'calibration_score_attempts':rec['meter']['scores']+sm.scores+cr['meter']['scores'],'source_digest':source_digest(),'estimates_not_performance_evidence':True,'even_reduced_may_hit_cap':estimates['reduced_s']>config['resources']['campaign_hours']*3600}
  chosen['resource_tier']=tier
 atomic_json(out/'EFFECTIVE_CONFIG.json',chosen);atomic_json(rp,r);return chosen
