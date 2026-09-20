"""The pinned Stage6 optimizer loop with an explicit post-validation epoch stop.

Only the outer stopping boundary, fixed-epoch export and execution metadata differ.
The original training dictionary and original LR horizon remain unchanged.
"""
from dataclasses import asdict
from pathlib import Path
import math,time,traceback
import torch
from spatial_lab.core import ContractError,digest,write_json,atomic_save
from spatial_lab.binding import source_manifest,FEATURE_BINDING,LABEL_BINDING,LABEL_SPEC,dependency_contract,state_digest,verify_checkpoint
from spatial_lab.stage6_model import make
from spatial_lab.stage6_dataset import Worlds
from spatial_lab.stage6_training import inference_checkpoint,encode_step
import spatial_lab.stage6_training as original_training

loader=original_training.loader

@torch.no_grad()
def validate(model,dataset,batch,workers,device,amp):
    # Use the original validation math with the current input transport.
    previous=original_training.loader
    original_training.loader=loader
    try:return original_training.validate(model,dataset,batch,workers,device,amp)
    finally:original_training.loader=previous

def train(prepared,validation,output,training,arm,seed,device,*,execute=False,resume=False,
          development_seeds=(),allowed_configurations=None,stop_after_updates=None,stop_epoch=10,execution=None):
    from spatial_lab.work_budget import require_optimizer_authority,charge_optimizer,training_purpose
    if not execute:raise ContractError('TRAINING_REQUIRES_EXECUTE')
    require_optimizer_authority()
    if training_purpose()=='development_diagnostic' and device!='cpu':raise ContractError('LOCAL_CPU_ONLY')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(training.get('cpu_threads',1))
    if not resume and (output/'latest.pt').exists():raise ContractError('EXISTING_TRAINING_REQUIRES_RESUME')
    trainset=Worlds(prepared);valset=Worlds(validation)
    if set(trainset.world_seeds)&(set(valset.world_seeds)|set(development_seeds)):raise ContractError('WORLD_PARTITION_OVERLAP')
    if set(valset.world_seeds)&set(development_seeds):raise ContractError('VALIDATION_DEVELOPMENT_OVERLAP')
    allowed=allowed_configurations or {**trainset.configurations,**valset.configurations}
    model,cfg=make(arm,device,seed,training['model']);optimizer=torch.optim.AdamW(model.parameters(),lr=training['learning_rate'],weight_decay=training['weight_decay'])
    amp=bool(training.get('bf16',True) and device.startswith('cuda') and torch.cuda.is_bf16_supported())
    cycles={e['contexts'] for e in trainset.entries+valset.entries}
    if len(cycles)!=1:raise ContractError('BATCHED_CYCLES_MUST_MATCH')
    cycles=cycles.pop();batch=training['world_batch'];chunk=training['bptt_steps'];epochs=training['epochs'];workers=training['loader_workers']
    updates_per_epoch=math.ceil(len(trainset)/batch)*math.ceil(cycles/chunk);total=updates_per_epoch*epochs
    if type(stop_epoch) is not int or not 1 <= stop_epoch <= epochs:
        raise ContractError('INVALID_STOP_EPOCH')
    stop_updates=updates_per_epoch*stop_epoch
    # The original 20-epoch configuration remains the identity and LR clock.
    # The separate execution record describes the user-requested prefix.
    done=0;cursor={'epoch':0,'batch':0,'time':0};memory=None;ids=None;best=None;validation_rows=[]
    start=time.monotonic();timing={'data_wait_s':0.,'compute_s':0.,'checkpoint_s':0.,'validation_s':0.};examples=0
    receipt={'status':'RUNNING','phase':'LOADING','arm':arm,'model_seed':seed,'target_updates':stop_updates,'epochs':stop_epoch,'lr_schedule_target_updates':total,'lr_schedule_epochs':epochs,
             'world_batch':batch,'bptt_steps':chunk,'parameter_count':sum(p.numel() for p in model.parameters()),'precision':'bf16' if amp else 'float32'}
    def progress(**extra):
        elapsed=time.monotonic()-start
        receipt.update(optimizer_updates=done,cursor=dict(cursor),updated_ns=time.time_ns(),elapsed_s=elapsed,
              timing=dict(timing),contexts_this_attempt=examples,contexts_per_s=examples/max(elapsed,1e-9),**extra)
        if device.startswith('cuda'):receipt.update(gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write_json(output/'TRAIN_RECEIPT.json',receipt)
    def checkpoint():
        ck={'schema':'ain.spatial.numeric.priority.v6','arm':arm,'cfg':asdict(cfg),'model_cfg_digest':digest(asdict(cfg)),
            'state':model.state_dict(),'state_digest':state_digest(model.state_dict()),'source_manifest':source_manifest(),
            'feature_binding':FEATURE_BINDING,'label_binding':LABEL_BINDING,'target':LABEL_SPEC,'dependency_contract':dependency_contract(),
            'dataset_manifest':trainset.manifest,'dataset_digest':digest(trainset.manifest),'validation_manifest':valset.manifest,
            'training_law_seeds':trainset.world_seeds,'validation_law_seeds':valset.world_seeds,'development_law_seeds':list(development_seeds),
            'allowed_policy_configurations':allowed,'memory_semantics':'semantic_world_reset_chunk_bptt_feedback_reread_v6',
            'optimizer_updates':done,'trained':done>0,'training_purpose':training_purpose(),'model_seed':seed,
            'optimizer_state':optimizer.state_dict(),'training_configuration':training,'training_configuration_digest':digest(training),
            'cursor':dict(cursor),'memory':memory.detach().cpu() if memory is not None else None,'memory_ids':ids,
            'torch_rng_state':torch.get_rng_state(),'cuda_rng_states':torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
            'best_validation':best,'validation_history':validation_rows,'target_updates':total,'parameter_count':receipt['parameter_count'],
            'epoch_stop_execution':execution,'requested_stop_epoch':stop_epoch,'requested_stop_updates':stop_updates}
        started=time.monotonic();atomic_save(output/'latest.pt',lambda f:torch.save(ck,f));timing['checkpoint_s']+=time.monotonic()-started
        return ck
    try:
        if resume and (output/'latest.pt').exists():
            ck=torch.load(output/'latest.pt',map_location=device,weights_only=True);verify_checkpoint(ck)
            if (ck['arm']!=arm or ck['model_seed']!=seed or ck['dataset_manifest']!=trainset.manifest or ck['validation_manifest']!=valset.manifest or
                ck['training_configuration_digest']!=digest(training) or ck['allowed_policy_configurations']!=allowed):raise ContractError('RESUME_IDENTITY')
            model.load_state_dict(ck['state']);optimizer.load_state_dict(ck['optimizer_state']);done=ck['optimizer_updates'];cursor=ck['cursor']
            memory=ck['memory'].to(device) if ck['memory'] is not None else None;ids=ck['memory_ids'];best=ck['best_validation'];validation_rows=ck['validation_history']
            torch.set_rng_state(ck['torch_rng_state'].cpu())
            if device.startswith('cuda'):torch.cuda.set_rng_state_all([r.cpu() for r in ck['cuda_rng_states']])
            # An interruption between latest and best replacement must not lose
            # an epoch's completed selection. The exact selected weights are in latest.
            if best is not None:
                best_path=output/'best.pt'
                saved_best=torch.load(best_path,map_location='cpu',weights_only=True) if best_path.exists() else None
                if saved_best is None or saved_best.get('best_validation')!=best:
                    if best['optimizer_updates']!=done:raise ContractError('MISSING_PREVIOUS_SELECTED_CHECKPOINT')
                    atomic_save(best_path,lambda f:torch.save(inference_checkpoint(ck),f))
        if done > stop_updates or cursor['epoch'] > stop_epoch:
            raise ContractError('CHECKPOINT_PAST_REQUESTED_EPOCH')
        progress(validation=validation_rows[-1] if validation_rows else None,best_validation=best)
        model.train();last_report=time.monotonic();latest_parts={}
        while cursor['epoch']<stop_epoch:
            epoch=cursor['epoch'];order=torch.randperm(len(trainset),generator=torch.Generator().manual_seed(seed+epoch)).tolist()
            batches=loader(trainset,order[cursor['batch']*batch:],batch,workers);wait=time.monotonic()
            for worlds in batches:
                timing['data_wait_s']+=time.monotonic()-wait
                while cursor['time']<cycles:
                    if done >= stop_updates:raise ContractError('STOP_UPDATE_CURSOR_MISMATCH')
                    start_t=cursor['time'];end=min(start_t+chunk,cycles);started=time.monotonic();loss=0.;parts_sum={}
                    optimizer.zero_grad(set_to_none=True)
                    for t in range(start_t,end):
                        value,parts,memory,ids=encode_step(model,worlds,t,memory,ids,device,amp);loss=loss+value/(end-start_t)
                        for k,v in parts.items():parts_sum[k]=parts_sum.get(k,0)+v/(end-start_t)
                    if not bool(torch.isfinite(loss)):raise ContractError('NONFINITE_LOSS')
                    loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    # Registered warmup followed by cosine decay, indexed by actual optimizer updates.
                    warmup=min(training.get('warmup_updates',500),max(1,total//10));ratio=min(1.,(done+1)/warmup)
                    if done>=warmup:ratio=.1+.9*.5*(1+math.cos(math.pi*(done-warmup)/max(1,total-warmup)))
                    for group in optimizer.param_groups:group['lr']=training['learning_rate']*ratio
                    charge_optimizer(arm,seed,done+1);optimizer.step();done+=1;examples+=len(worlds)*(end-start_t)
                    memory=memory.detach();cursor['time']=end
                    if end==cycles:cursor.update(batch=cursor['batch']+1,time=0);memory=None;ids=None
                    timing['compute_s']+=time.monotonic()-started;latest_parts=parts_sum
                    due=done%training['checkpoint_interval']==0 or done==stop_updates or (stop_after_updates is not None and done>=stop_after_updates)
                    if due:checkpoint()
                    if time.monotonic()-last_report>=5 or due:
                        latest={k:float(v) for k,v in parts_sum.items()};latest['gradient_norm']=float(grad)
                        progress(phase='TRAINING',loss=latest,learning_rate=optimizer.param_groups[0]['lr'])
                        with (output/'LOSS_HISTORY.jsonl').open('a',encoding='utf-8') as f:
                            import json
                            f.write(json.dumps({'update':done,'epoch':epoch,**latest})+'\n')
                        last_report=time.monotonic()
                    if stop_after_updates is not None and done>=stop_after_updates:
                        receipt['status']='PAUSED_FOR_DIAGNOSTIC';progress();return receipt
                    if end==cycles:break
                wait=time.monotonic()
            started=time.monotonic();progress(phase='VALIDATION')
            v=validate(model,valset,batch,workers,device,amp);timing['validation_s']+=time.monotonic()-started
            validation_rows.append({'epoch':epoch+1,'optimizer_updates':done,**v});improved=best is None or v['total']<best['total']
            if improved:best={'epoch':epoch+1,'optimizer_updates':done,'total':v['total']}
            cursor={'epoch':epoch+1,'batch':0,'time':0};memory=None;ids=None;ck=checkpoint()
            if improved:atomic_save(output/'best.pt',lambda f:torch.save(inference_checkpoint(ck),f))
            progress(phase='EPOCH_COMPLETE',validation=v,best_validation=best)
        if done != stop_updates or cursor != {'epoch':stop_epoch,'batch':0,'time':0}:
            raise ContractError('STOP_EPOCH_BOUNDARY_MISMATCH')
        from epoch10_snapshot import save_snapshot
        sealed=save_snapshot(ck,output,stop_epoch,stop_updates,execution)
        receipt.update(status='COMPLETE',phase='COMPLETE',completed_epochs=stop_epoch,
                       fixed_epoch_checkpoint=sealed['checkpoint_file'],validation=sealed['validation'],
                       best_validation=best,completion_reason='USER_REQUESTED_EPOCH_LIMIT')
        progress();return receipt
    except BaseException as exc:
        receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc());progress();raise
