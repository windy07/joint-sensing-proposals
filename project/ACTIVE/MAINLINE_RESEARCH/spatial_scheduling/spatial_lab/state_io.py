"""Exact completed-cycle continuation, separate from transient failure receipts.

World truth is restored only into World and Evaluator. The decision interface
still receives a belief snapshot and public queues. No optimizer is involved.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .core import ContractError,digest,read_json,write_json
from .binding import source_digest,state_digest

SCHEMA='ain.stage6.completed_cycle_state.v1'

def _memory_document(priority):
    if priority is None:return None
    doc=priority.memory.snapshot()
    return {**doc,'entries':[{**r,'state':r['state'].tolist()} for r in doc['entries']]}

def save_cycle(path,seed,config,method,world,tracker,evaluator,priority,trace,energy,tx,unseen,search_age,pointing):
    def targets(mapping):return {str(i):{**t,'state':t['state'].tolist()} for i,t in mapping.items()}
    body={'schema':SCHEMA,'source_digest':source_digest(),'seed':seed,'configuration':config,'method':method,
          'checkpoint_sha256':getattr(priority,'checkpoint_sha256',None),
          'model_state_digest':state_digest(priority.model.state_dict()) if priority is not None else None,
          'world':{'epoch':world.epoch,'bias':world.bias.tolist(),'targets':targets(world.targets),'all_targets':targets(world.all_targets),
                   'pending':world.pending,'link_free_s':world.link_free_s,'processing_free_s':world.processing_free_s,
                   'evaluation_events':world.evaluation_events,'audit':world.audit},
          'tracker':{'epoch':tracker.epoch,'mean':tracker.mean.tolist(),'P':tracker.P.tolist(),'tracks':tracker.tracks,
                     'next_id':tracker.next_id,'seen':tracker.seen,'arrived':tracker.arrived,'initiated_roots':sorted(tracker.initiated_roots),
                     'last_calibration_epochs':tracker.last_calibration_epochs,
                     'linearization_anchor':tracker.linearization_anchor.tolist() if tracker.linearization_anchor is not None else None,
                     'stats':tracker.stats,'last_betas':tracker.last_betas,'nis':tracker.nis,'births':tracker.births,
                     'emission_components':[{**b,'mean':b['mean'].tolist(),'P':b['P'].tolist(),
                                             'bits':{str(i):v for i,v in b['bits'].items()}} for b in tracker.emission_components]},
          'evaluator':{k:v for k,v in evaluator.__dict__.items() if k!='config'},'memory':_memory_document(priority),
          'trace':trace,'energy_exact_J':str(energy),'tx_exact_s':str(tx),'unseen':list(map(float,unseen)),
          'search_age':list(map(int,search_age)),'pointing':list(pointing),'completed_cycles':len(trace)}
    # All integer dictionary keys are canonicalized by the JSON writer on disk;
    # compute the binding on that same representation before writing.
    from .core import clean
    body=clean(body);body['content_digest']=digest(body);write_json(path,body)
    return body['content_digest']

def restore_cycle(path,seed,config,method,world,tracker,evaluator,priority):
    d=read_json(path)
    if d.get('schema')!=SCHEMA or d.get('content_digest')!=digest({k:v for k,v in d.items() if k!='content_digest'}):raise ContractError('CYCLE_STATE_CONTENT')
    for key,value in {'source_digest':source_digest(),'seed':seed,'configuration':config,'method':method,
          'checkpoint_sha256':getattr(priority,'checkpoint_sha256',None),
          'model_state_digest':state_digest(priority.model.state_dict()) if priority is not None else None}.items():
        if d.get(key)!=value:raise ContractError('CYCLE_STATE_IDENTITY:'+key)
    w=d['world'];t=d['tracker'];n=d['completed_cycles']
    if n!=len(d['trace']) or n!=w['epoch']+1 or t['epoch']!=w['epoch'] or n>config['cycles']:raise ContractError('CYCLE_STATE_INDEX')
    for key in ('epoch','pending','link_free_s','processing_free_s','evaluation_events','audit'):setattr(world,key,deepcopy(w[key]))
    world.bias=np.asarray(w['bias'],float)
    world.all_targets={int(i):{**x,'state':np.asarray(x['state'],float)} for i,x in w['all_targets'].items()}
    world.targets={int(i):{**x,'state':np.asarray(x['state'],float)} for i,x in w['targets'].items()}
    # Preserve live-target object sharing (the same semantics as a fresh world).
    for i,tgt in world.targets.items():world.all_targets[i]=tgt
    for key in ('epoch','tracks','next_id','seen','arrived','stats','nis','births'):setattr(tracker,key,deepcopy(t[key]))
    tracker.mean=np.asarray(t['mean'],float);tracker.P=np.asarray(t['P'],float)
    tracker.last_calibration_epochs=list(t['last_calibration_epochs'])
    tracker.linearization_anchor=np.asarray(t['linearization_anchor'],float) if t['linearization_anchor'] is not None else None
    tracker.initiated_roots=set(t['initiated_roots']);tracker.last_betas={int(i):v for i,v in t['last_betas'].items()}
    tracker.emission_components=[{**b,'mean':np.asarray(b['mean'],float),'P':np.asarray(b['P'],float),
                                   'bits':{int(i):v for i,v in b['bits'].items()}} for b in t['emission_components']]
    from .core import psd
    psd(tracker.P,'RESTORED_POSTERIOR')
    for key,value in d['evaluator'].items():
        setattr(evaluator,key,{int(i):v for i,v in value.items()} if isinstance(value,dict) else deepcopy(value))
    if priority is not None:
        import torch
        mem=deepcopy(d['memory'])
        for r in mem['entries']:r['state']=torch.tensor(r['state'],dtype=torch.float32)
        priority.memory.restore(mem);priority.last_context=None;priority.out=None
    elif d['memory'] is not None:raise ContractError('CYCLE_STATE_UNEXPECTED_MEMORY')
    return d
