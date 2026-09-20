"""World-ordered training and development-only selection of retained backbones."""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from dataclasses import asdict
import math,time,random,json
import numpy as np
import torch
from iclr2027_graph_agent.checkpoint import CheckpointMetadata,save_atomic_checkpoint,load_verified_checkpoint
from .common import *
from .contexts import Context
from .graph_adapter import graph_context,FEATURE_BINDING
from .models import make_model,objective,config_for,schedule_scores
from .dataset import data_manifest

class WorldCache:
 def __init__(self,out,maxsize=48):self.out=Path(out);self.cache=OrderedDict();self.max=maxsize
 def get(self,row):
  key=row['path']
  if key not in self.cache:
   if len(self.cache)>=self.max:self.cache.popitem(last=False)
   self.cache[key]=read_json(self.out/key)
  else:self.cache.move_to_end(key)
  return self.cache[key]

def load_model(path,config,arm,seed,device='cpu',expected_hash=None):
 model=make_model(arm,config,seed,device)
 payload,h=load_verified_checkpoint(Path(path),expected={'feature_schema_sha256':FEATURE_BINDING,'model_config_sha256':digest(asdict(config_for(arm,config))),'source_tree_sha256':source_digest()},map_location=device,expected_file_sha256=expected_hash)
 if payload['extra'].get('arm')!=arm:raise ContractError('CHECKPOINT_ARM')
 model.load_state_dict(payload['model_state']);return model,payload,h

def evaluate_development(model,config,out,rows,cache,arm):
 model.eval();device=next(model.parameters()).device;losses=[];worlds=[];meter=Meter(0,0)
 with torch.no_grad():
  for row in rows:
   world=cache.get(row);values=[];memory=None;last_nodes=None
   for rec in world['samples']:
    ctx=Context(canonical(rec['context']));g=graph_context(ctx,meter,str(device))
    if memory is not None:memory=align_memory(memory,last_nodes,g.node_ids,model.cfg.memory_dim,g.node_features)
    acts=[tuple(a) for a in rec['candidates']];scores,o=schedule_scores(model,g,acts,memory);memory=o.next_memory;last_nodes=g.node_ids
    order=torch.argsort(scores,descending=True).cpu().tolist();rr=rec['scores']
    reserved=[]
    if () in acts:reserved.append(acts.index(()))
    if arm!='original_agent' and config['online']['K']>=2:
     pa=tuple(rec['aux'].get('protected_action',()))
     if pa in acts and acts.index(pa) not in reserved:reserved.append(acts.index(pa))
    selected=(reserved+[i for i in order if i not in reserved])[:config['online']['B']]
    if config['online']['K']==2 and len(reserved)==2:selected=reserved
    # Known labels in one fixed development pool; counted as selection work,
    # never called a budget-matched online rollout.
    val=min(rr[j]['risk_m2'] for j in selected)/ctx.doc()['length_m']**2 if selected else 25.
    values.append(val)
   worlds.append(float(np.mean(values)))
 return float(np.mean(worlds)),{'worlds':len(worlds),'context_forward_calls':sum(len(cache.get(r)['samples']) for r in rows),'normalized_world_values':worlds,'selection_reference':'fixed_development_label_pool_with_matched_reserve_v2','meter':meter.report()}

def align_memory(memory,old_ids,new_ids,dim,reference):
 index={k:i for i,k in enumerate(old_ids)}
 return torch.stack([memory[index[k]] if k in index else reference.new_zeros(dim) for k in new_ids])

def train_job(config,out,arm,seed,device='cpu'):
 out=Path(out);job=out/'training'/f'{arm}_{seed}';job.mkdir(parents=True,exist_ok=True);manifest=data_manifest(out)
 expected_dataset=digest({'config':config,'source':source_digest(),'stage':'labels'})
 if manifest['binding']!=expected_dataset:raise ContractError('TRAINING_DATASET_CONFIG_SOURCE')
 binding=digest({'config':config,'dataset':manifest['binding'],'source':source_digest(),'arm':arm,'seed':seed});done=job/'TRAIN_RECEIPT.json'
 if completed(done,binding):return read_json(done)
 model=make_model(arm,config,seed,device);model.train();torch.set_num_threads(config['resources']['torch_threads']);torch.use_deterministic_algorithms(True)
 opt=torch.optim.AdamW(model.parameters(),lr=config['training']['learning_rate'],weight_decay=1e-4)
 max_steps=config['size']['updates'];warm=min(200,max_steps//10)
 def factor(step):
  if step<warm:return (step+1)/max(1,warm)
  return .5*(1+math.cos(math.pi*(step-warm)/max(1,max_steps-warm)))
 sched=torch.optim.lr_scheduler.LambdaLR(opt,factor);start_step=0;source=source_digest();cache=WorldCache(out)
 train_rows=[r for r in manifest['units'] if r['spec']['split']=='train'];dev_rows=[r for r in manifest['units'] if r['spec']['split']=='development']
 last=job/'last.pt';checkpoint_rows=[];gradmax=0.;auxgrad=0.;steps_history=[];start=time.perf_counter()
 if last.exists():
  payload,_=load_verified_checkpoint(last,expected={'training_config_sha256':binding,'source_tree_sha256':source,'feature_schema_sha256':FEATURE_BINDING},map_location=device)
  model.load_state_dict(payload['model_state']);opt.load_state_dict(payload['optimizer_state']);sched.load_state_dict(payload['scheduler_state']);start_step=payload['metadata']['global_step'];checkpoint_rows=payload['extra'].get('checkpoints',[])
  torch.random.set_rng_state(payload['torch_random_state'].cpu())
  # Checkpoint map_location moves all tensors to the training device, but CUDA
  # generators require their serialized ByteTensor state on the CPU.
  if torch.cuda.is_available() and payload.get('cuda_random_state'):torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_random_state']])
 initial=next(model.parameters()).detach().cpu().clone()
 checkpoints=set(max(1,int(max_steps*x)) for x in (.25,.5,.75,1.))
 # Same world order across arms at each paired seed. Whole-world cycles remain ordered.
 orders={}
 def row_at(step):
  epoch=step//len(train_rows)
  if epoch not in orders:
   ii=list(range(len(train_rows)));random.Random(seed_for('V11_TRAIN_ORDER',seed,epoch)).shuffle(ii);orders[epoch]=ii
  return train_rows[orders[epoch][step%len(train_rows)]]
 def save(path,step,devmetric=float('inf')):
  safe_metric=devmetric if math.isfinite(devmetric) else 1e30
  meta=CheckpointMetadata('iclr2027.checkpoint.v11',step//max(1,len(train_rows)),step,digest(asdict(config_for(arm,config))),FEATURE_BINDING,manifest['binding'],binding,source,filehash(out/'DATASET_MANIFEST.json'),filehash(out/'DATASET_MANIFEST.json'),manifest['binding'],manifest['binding'],{'normalized_risk':safe_metric})
  return save_atomic_checkpoint(path,model=model,optimizer=opt,scheduler=sched,metadata=meta,extra={'arm':arm,'seed':seed,'checkpoints':checkpoint_rows,'recorded_steps':step})
 for step in range(start_step,max_steps):
  model.train();opt.zero_grad(set_to_none=True);world=cache.get(row_at(step));memory=None;last_ids=None;previous=None;terms=[]
  for rec in world['samples']:
   ctx=Context(canonical(rec['context']));meter=Meter(0,0);g=graph_context(ctx,meter,device)
   if memory is not None:memory=align_memory(memory,last_ids,g.node_ids,model.cfg.memory_dim,g.node_features)
   acts=[tuple(a) for a in rec['candidates']]
   mixed=config['training'].get('mixed_precision',False) and str(device).startswith('cuda')
   with torch.autocast(device_type='cuda' if str(device).startswith('cuda') else 'cpu',dtype=torch.bfloat16,enabled=mixed):
    loss,logs,prev,mem=objective(model,g,acts,rec['scores'],rec['aux'],ctx,arm,previous,memory)
    scaled=loss/len(world['samples'])
   if not torch.isfinite(scaled):raise ContractError('NONFINITE_TRAINING_LOSS')
   scaled.backward();terms.append(logs);previous=(prev[0].detach(),prev[1]);memory=mem.detach();last_ids=g.node_ids
  gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));gradmax=max(gradmax,gn)
  for name,p in model.named_parameters():
   if ('association_risk' in name or '.assoc.' in name or 'provenance_novelty' in name or '.prov.' in name) and p.grad is not None:auxgrad=max(auxgrad,float(p.grad.norm()))
  opt.step();sched.step();s=step+1
  event={'step':s,'world':world['spec']['unit_id'],'gradient_norm':gn,'terms':terms,'seconds':time.perf_counter()-start};steps_history.append(event)
  with (job/'training.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(event)+'\n')
  if s in checkpoints:
   metric,dev=evaluate_development(model,config,out,dev_rows,cache,arm);cp=job/f'step_{s:06d}.pt';h=save(cp,s,metric)
   checkpoint_rows.append({'step':s,'path':cp.relative_to(out).as_posix(),'sha256':h,'development_metric':metric});atomic_json(job/f'dev_{s:06d}.json',dev)
  if s%config['training']['save_every']==0 or s in checkpoints:save(last,s)
 # Real reload and score-tensor identity, not a checkpoint filename assertion.
 chosen=min(checkpoint_rows,key=lambda r:(r['development_metric'],r['step']));atomic_json(job/'SELECTED.json',chosen)
 delta=float((next(model.parameters()).detach().cpu()-initial).abs().max());outputs=[last,last.with_suffix('.pt.sha256'),job/'SELECTED.json',out/chosen['path']]
 result=receipt(done,binding,outputs,arm=arm,seed=seed,actual_updates=max_steps-start_step,total_updates=max_steps,maximum_gradient_norm=gradmax,maximum_active_auxiliary_gradient=auxgrad,parameter_delta=delta,parameter_count=sum(p.numel() for p in model.parameters()),backbone_class=type(model).__name__,selected=chosen,seconds=time.perf_counter()-start,training_steps_sha256=filehash(job/'training.jsonl'))
 return result
