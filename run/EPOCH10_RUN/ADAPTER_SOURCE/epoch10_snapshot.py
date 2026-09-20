"""Exact fixed-epoch model export, independent of best-validation selection."""
from pathlib import Path
import hashlib,time
import torch
from spatial_lab.core import ContractError,atomic_save,write_json
from spatial_lab.binding import verify_checkpoint
from spatial_lab.stage6_training import inference_checkpoint

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def epoch_result(ck,epoch,updates):
    if ck.get('optimizer_updates')!=updates:
        raise ContractError('FIXED_EPOCH_WEIGHTS_UPDATE_MISMATCH')
    rows=[v for v in ck.get('validation_history',[]) if v.get('epoch')==epoch and v.get('optimizer_updates')==updates]
    if len(rows)!=1:raise ContractError('FIXED_EPOCH_VALIDATION_MISSING')
    if 'cursor' in ck and ck['cursor']!={'epoch':epoch,'batch':0,'time':0}:
        raise ContractError('FIXED_EPOCH_VALIDATION_NOT_COMMITTED')
    return rows[0]

def save_snapshot(ck,output,epoch,updates,execution):
    verify_checkpoint(ck)
    validation=epoch_result(ck,epoch,updates)
    output=Path(output);path=output/f'epoch_{epoch:03d}.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True);verify_checkpoint(saved)
        epoch_result(saved,epoch,updates)
        if saved['state_digest']!=ck['state_digest'] or saved['model_seed']!=ck['model_seed'] or saved['arm']!=ck['arm']:
            raise ContractError('EXISTING_FIXED_EPOCH_WEIGHTS_DIFFER')
    else:
        fixed={**inference_checkpoint(ck),'selection':{'kind':'fixed_completed_epoch','epoch':epoch,'optimizer_updates':updates},
               'epoch_stop_execution':execution}
        atomic_save(path,lambda stream:torch.save(fixed,stream))
    result={'schema':'ain.stage6.fixed_epoch.result.v1','status':'COMPLETE','arm':ck['arm'],'model_seed':ck['model_seed'],
            'epoch':epoch,'optimizer_updates':updates,'checkpoint_file':path.name,'checkpoint_sha256':sha(path),
            'state_digest':ck['state_digest'],'validation':validation,'best_validation':ck.get('best_validation'),
            'original_lr_schedule_target_updates':ck['target_updates'],'training_configuration':ck['training_configuration'],
            'execution':execution,'saved_ns':time.time_ns()}
    write_json(output/f'EPOCH_{epoch:03d}_RESULT.json',result)
    write_json(path.with_name(path.name+'.execution.json'),result)
    return result
